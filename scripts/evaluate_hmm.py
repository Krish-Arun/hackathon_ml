"""
Classifier-only vs classifier+HMM/Viterbi (report section 5).

Runs the full inference pipeline over every student-test trip twice - once
taking the best candidate per observation independently, once decoding the
sequence with Viterbi - and compares them on identical trips.

The comparison the brief asks for is pointwise accuracy AND route continuity.
Continuity is where the decoder should earn its place: a classifier choosing
one observation at a time has no way of knowing whether consecutive arcs
connect.

Run:  python scripts/evaluate_hmm.py
"""
import sys, os, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.io import load_trip, list_trips              # noqa: E402
from pathsnap.candidates import generate_candidates        # noqa: E402
from pathsnap.features import build_features, FEATURE_COLUMNS  # noqa: E402
from pathsnap.labelling import label_trip                  # noqa: E402
from pathsnap.viterbi import decode_trip, classifier_only  # noqa: E402

DATA_ROOT = os.environ.get("PATHSNAP_DATA", str(ROOT / "Student folder" / "Student folder"))


def continuity(trip, pred):
    """Fraction of consecutive predictions that are physically connected."""
    v = pred[pred >= 0]
    if len(v) < 2:
        return float("nan")
    a, b = v[:-1], v[1:]
    return float(((a == b) | (trip.arcs[a, 1] == trip.arcs[b, 0])).mean())


def accuracy(pred, truth):
    m = pred >= 0
    return float((pred[m] == truth[m]).mean()) if m.any() else float("nan")


def main():
    with open(ROOT / "artifacts" / "config.json") as f:
        cfg = json.load(f)
    model = joblib.load(ROOT / "artifacts" / cfg["model_file"])

    rows = []
    for d in list_trips(DATA_ROOT, "student_test"):
        trip = load_trip(d, "student_test")
        cand = generate_candidates(trip, k=cfg["candidate_k"],
                                   radius=cfg["candidate_radius_m"],
                                   spacing=cfg["arc_sample_spacing_m"])
        feats = build_features(trip, cand)
        proba = model.predict_proba(feats[FEATURE_COLUMNS].to_numpy(np.float32))[:, 1]
        labels = label_trip(trip)

        t0 = time.time(); clf = classifier_only(trip, cand, proba); t_clf = time.time() - t0
        t0 = time.time(); hmm = decode_trip(trip, cand, proba); t_hmm = time.time() - t0

        # Restrict accuracy to reliably-labelled observations: on trips whose
        # reference route only covers part of the journey, the label itself is
        # wrong and scoring against it measures nothing.
        ok = labels["residual_m"] <= cfg["label_reliable_max_residual_m"]
        rows.append({
            "trip_id": trip.trip_id,
            "n_obs": trip.n_obs,
            "frac_reliable": float(ok.mean()),
            "clf_top1": accuracy(np.where(ok, clf, -1), labels["arc_id"]),
            "hmm_top1": accuracy(np.where(ok, hmm, -1), labels["arc_id"]),
            "clf_continuity": continuity(trip, clf),
            "hmm_continuity": continuity(trip, hmm),
            "n_unpredicted": int((hmm < 0).sum()),
            "clf_seconds": round(t_clf, 3),
            "hmm_seconds": round(t_hmm, 3),
        })
        r = rows[-1]
        print(f"  {trip.trip_id}: top1 {r['clf_top1']:.4f} -> {r['hmm_top1']:.4f}   "
              f"continuity {r['clf_continuity']:.4f} -> {r['hmm_continuity']:.4f}   "
              f"({r['hmm_seconds']:.2f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "hmm_comparison.csv", index=False)

    # Observation-weighted, so long trips are not outvoted by short ones.
    w = df.n_obs
    print("\n=== weighted over all student-test trips ===")
    print(f"  Top-1      {np.average(df.clf_top1, weights=w):.4f} -> "
          f"{np.average(df.hmm_top1, weights=w):.4f}  "
          f"({np.average(df.hmm_top1 - df.clf_top1, weights=w):+.4f})")
    print(f"  Continuity {np.average(df.clf_continuity, weights=w):.4f} -> "
          f"{np.average(df.hmm_continuity, weights=w):.4f}  "
          f"({np.average(df.hmm_continuity - df.clf_continuity, weights=w):+.4f})")
    worse = df[df.hmm_top1 < df.clf_top1]
    print(f"  trips where HMM reduced Top-1: {len(worse)} of {len(df)}")
    if len(worse):
        print(worse[["trip_id", "clf_top1", "hmm_top1",
                     "clf_continuity", "hmm_continuity"]].to_string(index=False))
    print(f"  total decode time {df.hmm_seconds.sum():.1f}s for {df.n_obs.sum():,} observations")
    return df


if __name__ == "__main__":
    main()
