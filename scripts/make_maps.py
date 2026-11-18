"""
Generate the Folium maps for the report (section 6).

Produces one map per student-test trip, each showing raw GPS, the local road
network, the reference route and our prediction as separate toggleable layers.

Run:  python scripts/make_maps.py [--trips 00000000 00000009]
"""
import sys, os, json, argparse
from pathlib import Path
import numpy as np
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.io import load_trip, list_trips              # noqa: E402
from pathsnap.candidates import generate_candidates        # noqa: E402
from pathsnap.features import build_features, FEATURE_COLUMNS  # noqa: E402
from pathsnap.viterbi import decode_trip                   # noqa: E402
from pathsnap.viz.folium_maps import build_map             # noqa: E402

DATA_ROOT = os.environ.get("PATHSNAP_DATA", str(ROOT / "Student folder" / "Student folder"))
OUT = ROOT / "results" / "maps"


def main(trip_ids):
    with open(ROOT / "artifacts" / "config.json") as f:
        cfg = json.load(f)
    model = joblib.load(ROOT / "artifacts" / cfg["model_file"])
    OUT.mkdir(parents=True, exist_ok=True)

    dirs = list_trips(DATA_ROOT, "student_test")
    if trip_ids:
        dirs = [d for d in dirs if d.name in set(trip_ids)]

    for d in dirs:
        trip = load_trip(d, "student_test")
        cand = generate_candidates(trip, k=cfg["candidate_k"],
                                   radius=cfg["candidate_radius_m"],
                                   spacing=cfg["arc_sample_spacing_m"])
        feats = build_features(trip, cand)
        proba = model.predict_proba(feats[FEATURE_COLUMNS].to_numpy(np.float32))[:, 1]
        pred = decode_trip(trip, cand, proba)

        m = build_map(trip, predicted_arcs=pred, reference_arcs=trip.route)
        path = OUT / f"trip_{trip.trip_id}.html"
        m.save(str(path))
        print(f"  {trip.trip_id}: {path.stat().st_size/1000:.0f} KB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trips", nargs="*")
    main(ap.parse_args().trips)
