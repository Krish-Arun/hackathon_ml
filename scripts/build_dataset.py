"""
Build the full training table.

Runs sections 1-3 over every trip and writes one row per
(observation, candidate arc) pair with its features and its label.

Outputs
-------
data/processed/features.parquet   the training table (gitignored, regenerable)
results/build_summary.csv         per-trip stats: candidate recall, label
                                  residuals, timings
artifacts/dataset_config.json     the exact settings used, so the API can
                                  reproduce these features at inference time

THE RULE THIS SCRIPT ENFORCES:
A trip belongs entirely to one partition. Observations from one trip are
near-duplicates of each other - same roads, same vehicle, same GPS conditions
- so splitting a trip across train and test would let the model see the
answers and every score we report would be inflated. We keep the partitions
the dataset already defines and never mix them.

Run:  python scripts/build_dataset.py
"""
import sys, os, json, time, subprocess, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.io import load_trip, list_trips                 # noqa: E402
from pathsnap.candidates import generate_candidates           # noqa: E402
from pathsnap.labelling import label_trip, label_quality      # noqa: E402
from pathsnap.features import build_features, attach_labels, FEATURE_COLUMNS  # noqa: E402

DATA_ROOT = os.environ.get("PATHSNAP_DATA", str(ROOT / "Student folder" / "Student folder"))

# Candidate-generation settings. RADIUS is justified by the exploration sweep:
# the nearest arc is 4.1m away on average and 13m at p95, so 200m is generous
# while still excluding only 0.047% of observations. K caps the HMM state
# space and the table size.
CONFIG = {
    "candidate_radius_m": 200.0,
    "candidate_k": 10,
    "arc_sample_spacing_m": 25.0,
    "feature_columns": FEATURE_COLUMNS,
    "random_seed": 42,
}


def git_commit():
    """Record which code version produced this table (reproducibility)."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def process_trip(trip):
    """Candidates -> features -> labels for one trip."""
    t0 = time.time()
    cand = generate_candidates(
        trip,
        k=CONFIG["candidate_k"],
        radius=CONFIG["candidate_radius_m"],
        spacing=CONFIG["arc_sample_spacing_m"],
    )
    t_cand = time.time() - t0

    t0 = time.time()
    feats = build_features(trip, cand)
    t_feat = time.time() - t0

    t0 = time.time()
    labels = label_trip(trip)
    feats, recall = attach_labels(feats, labels)
    t_label = time.time() - t0

    q = label_quality(labels)
    stats = {
        "trip_id": trip.trip_id,
        "partition": trip.partition,
        "n_obs": trip.n_obs,
        "n_arcs": trip.n_arcs,
        "n_rows": len(feats),
        "n_positives": int(feats["label"].sum()),
        # Candidate recall is the ceiling on Top-1 accuracy: if the true arc
        # was never shortlisted, no model can possibly select it.
        "candidate_recall": recall,
        # Share of observations whose label we trust (see attach_labels).
        "frac_label_reliable": float(feats.groupby("obs_idx")["label_reliable"].first().mean()),
        "obs_with_no_candidate": int(trip.n_obs - feats["obs_idx"].nunique()),
        "label_residual_p50_m": q["residual_p50_m"],
        "label_residual_p95_m": q["residual_p95_m"],
        "route_coverage": q["route_coverage"],
        "sec_candidates": round(t_cand, 3),
        "sec_features": round(t_feat, 3),
        "sec_labels": round(t_label, 3),
    }
    return feats, stats


def main(partition, start, end):
    """
    Build one chunk of trips and write it as its own parquet part.

    Chunked for two reasons. Holding all 90 trips in memory and concatenating
    peaked high enough to get OOM-killed on this 3GB machine. And the build
    takes longer than a single shell invocation allows, so it has to be
    resumable. Each part is self-contained; scripts read the whole directory
    back as one dataset.
    """
    parts_dir = ROOT / "data" / "processed" / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    trips = list_trips(DATA_ROOT, partition)[start:end]
    out = parts_dir / f"{partition}_{start:03d}_{start+len(trips):03d}.parquet"

    writer = None
    stats = []
    total_rows = total_pos = 0
    t_start = time.time()
    try:
        for i, d in enumerate(trips, 1):
            trip = load_trip(d, partition)
            feats, s_row = process_trip(trip)
            feats["partition"] = partition

            table = pa.Table.from_pandas(feats, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out, table.schema, compression="snappy")
            writer.write_table(table)

            total_rows += len(feats)
            total_pos += int(feats["label"].sum())
            stats.append(s_row)
            del feats, table          # release before the next trip
            print(f"  [{i}/{len(trips)}] {trip.trip_id} "
                  f"obs={trip.n_obs} rows={s_row['n_rows']} "
                  f"recall={s_row['candidate_recall']:.3f} "
                  f"{s_row['sec_candidates']+s_row['sec_features']+s_row['sec_labels']:.1f}s",
                  flush=True)
    finally:
        if writer is not None:
            writer.close()

    # Append this chunk's stats to the running summary.
    sfile = ROOT / "results" / "build_summary.csv"
    sdf = pd.DataFrame(stats)
    if sfile.exists():
        sdf = pd.concat([pd.read_csv(sfile), sdf], ignore_index=True)
        sdf = sdf.drop_duplicates(subset=["trip_id", "partition"], keep="last")
    sdf.to_csv(sfile, index=False)

    cfg = dict(CONFIG)
    cfg.update(built_at=pd.Timestamp.utcnow().isoformat(), git_commit=git_commit(),
               n_rows=int(sdf.n_rows.sum()), n_trips=len(sdf))
    (ROOT / "artifacts").mkdir(exist_ok=True)
    with open(ROOT / "artifacts" / "dataset_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\nchunk done in {time.time()-t_start:.0f}s -> {out.name}")
    print(f"  rows {total_rows:,}  positives {total_pos:,} ({100*total_pos/max(total_rows,1):.1f}%)")
    print(f"  cumulative trips in summary: {len(sdf)}")
    return sdf


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="train", choices=["train", "student_test"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    a = ap.parse_args()
    main(a.partition, a.start, a.end)
