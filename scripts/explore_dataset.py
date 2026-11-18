"""
Dataset inventory and data-quality sweep (report section 2.2).

Produces results/dataset_inventory.csv with one row per trip: sizes, sampling
behaviour, geographic extent, network scale, and GPS-to-network fit. This is
the evidence behind every claim we make about the data, and it is what sets
the candidate-generation radius.

Run:  python scripts/explore_dataset.py
"""
import sys, os, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap.io import load_trip, list_trips          # noqa: E402
from pathsnap.candidates import generate_candidates    # noqa: E402

DATA_ROOT = os.environ.get(
    "PATHSNAP_DATA",
    str(Path(__file__).resolve().parents[2] / "Student folder" / "Student folder"),
)
RADIUS = 200.0   # metres, candidate search radius under test
K = 10           # candidates per observation


def summarise(trip):
    """All per-trip statistics, computed once."""
    dt = np.diff(trip.gps_t)
    lengths = trip.arc_lengths()

    cand = generate_candidates(trip, k=K, radius=RADIUS)
    nearest = np.full(trip.n_obs, np.inf)
    np.minimum.at(nearest, cand["obs_idx"], cand["dist"])
    n_cands = np.bincount(cand["obs_idx"], minlength=trip.n_obs)

    # Degree of every node, to describe connectivity. A node appearing in no
    # arc is isolated; the brief asks us to report these.
    deg = np.bincount(trip.arcs.ravel(), minlength=len(trip.node_x))

    row = {
        "trip_id": trip.trip_id,
        "partition": trip.partition,
        "n_obs": trip.n_obs,
        "n_nodes": len(trip.node_x),
        "n_arcs": trip.n_arcs,
        "n_route_arcs": len(trip.route) if trip.route is not None else np.nan,
        "has_route": trip.route is not None,
        # --- sampling behaviour ---
        "duration_s": float(trip.gps_t[-1] - trip.gps_t[0]),
        "dt_median_s": float(np.median(dt)) if len(dt) else np.nan,
        "dt_max_s": float(dt.max()) if len(dt) else np.nan,
        "n_gaps_gt_5s": int((dt > 5).sum()) if len(dt) else 0,
        # --- data quality ---
        "n_nan_coords": int(np.isnan(trip.gps_lon).sum() + np.isnan(trip.gps_lat).sum()),
        "n_duplicate_timestamps": int(len(trip.gps_t) - len(np.unique(trip.gps_t))),
        # --- geography ---
        "bbox_km_ns": float((trip.gps_lat.max() - trip.gps_lat.min()) * 110.54),
        "mean_speed_kmh": np.nan,
        # --- network ---
        "arc_len_median_m": float(np.median(lengths)),
        "arc_len_max_m": float(lengths.max()),
        "n_isolated_nodes": int((deg == 0).sum()),
        # --- GPS vs network fit: drives the radius choice ---
        "nearest_arc_p50_m": float(np.percentile(nearest[np.isfinite(nearest)], 50)),
        "nearest_arc_p95_m": float(np.percentile(nearest[np.isfinite(nearest)], 95)),
        "n_obs_no_candidate": int(np.isinf(nearest).sum()),
        "mean_candidates_per_obs": float(n_cands.mean()),
    }

    # Straight-line speed between consecutive fixes, median over the trip.
    if len(dt):
        step = np.hypot(np.diff(trip.gps_x), np.diff(trip.gps_y))
        valid = dt > 0
        if valid.any():
            row["mean_speed_kmh"] = float(np.median(step[valid] / dt[valid]) * 3.6)
    return row


def main():
    rows = []
    for partition in ("train", "student_test"):
        for d in list_trips(DATA_ROOT, partition):
            t0 = time.time()
            rows.append(summarise(load_trip(d, partition)))
            print(f"  {rows[-1]['trip_id']} ({partition})  {time.time()-t0:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parents[1] / "results" / "dataset_inventory.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(df)} trips)")
    return df


if __name__ == "__main__":
    main()
