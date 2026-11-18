"""
Validate the derived labels across every trip that has a reference route.

We cannot check labels against a ground truth (deriving them IS the task), so
we validate indirectly, three ways:

1. MONOTONICITY - route positions must never decrease. This is enforced by
   construction, so a failure here is a bug, not a data issue.
2. RESIDUAL DISTANCE - how far each observation sits from the arc we assigned
   it. We independently measured the nearest arc to be ~4.1m away on average.
   If our residuals match that, the labeller is picking sensible arcs; if they
   were hundreds of metres, the alignment would be scrambled.
3. ROUTE COVERAGE - what fraction of the reference route has at least one
   observation near it. Low coverage means a GPS gap skipped a stretch of road.

Writes results/label_quality.csv.
"""
import sys, os, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap.io import load_trip, list_trips           # noqa: E402
from pathsnap.labelling import label_trip, label_quality  # noqa: E402

DATA_ROOT = os.environ.get(
    "PATHSNAP_DATA",
    str(Path(__file__).resolve().parents[1] / "Student folder" / "Student folder"),
)


def main():
    rows = []
    for partition in ("train", "student_test"):
        for d in list_trips(DATA_ROOT, partition):
            trip = load_trip(d, partition)
            t0 = time.time()
            labels = label_trip(trip)
            elapsed = time.time() - t0

            q = label_quality(labels)
            q.update(
                trip_id=trip.trip_id,
                partition=partition,
                n_obs=trip.n_obs,
                n_route_arcs=len(trip.route),
                label_seconds=round(elapsed, 3),
                # Must be True for every trip - enforced by the DP.
                monotone=bool(np.all(np.diff(labels["route_pos"]) >= 0)),
                # Distinct arcs actually used as labels.
                n_distinct_label_arcs=int(len(np.unique(labels["arc_id"]))),
            )
            rows.append(q)

    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parents[1] / "results" / "label_quality.csv"
    df.to_csv(out, index=False)

    print(f"trips labelled: {len(df)}   total time: {df.label_seconds.sum():.1f}s")
    print(f"monotonicity holds on all trips: {bool(df.monotone.all())}")
    print(f"residual p50: median {df.residual_p50_m.median():.2f}m   worst trip {df.residual_p50_m.max():.1f}m")
    print(f"residual p95: median {df.residual_p95_m.median():.2f}m   worst trip {df.residual_p95_m.max():.1f}m")
    print(f"observations >50m from their label: {int(df.n_residual_gt_50m.sum())} of {int(df.n_obs.sum())}")
    print(f"route coverage: median {df.route_coverage.median():.3f}   worst {df.route_coverage.min():.3f}")
    print("\nworst 5 trips by residual p95 (labels least trustworthy here):")
    print(df.nlargest(5, "residual_p95_m")[
        ["trip_id", "partition", "n_obs", "residual_p95_m", "residual_max_m",
         "n_residual_gt_50m", "route_coverage"]].to_string(index=False))
    return df


if __name__ == "__main__":
    main()
