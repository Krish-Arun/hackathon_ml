"""
PathSnap inference API - runs on Laptop 1.

The evaluator drives the UI on Laptop 2, which calls this service. We run the
complete pipeline here and return the reconstructed route plus accuracy.

THE SEPARATION THAT MATTERS:
Ground truth is loaded on this machine and used for ONE purpose - scoring a
prediction that has already been made. It is structurally impossible for it to
influence the prediction, because `run_pipeline()` never receives it: the
route is read afterwards, inside `score()`, from a trip object the pipeline
has already finished with. No ground-truth geometry is ever sent to the UI.

Nothing is precomputed or cached per trip. Every request runs candidate
generation, feature building, the classifier and Viterbi from the raw files,
so the logs show real work for whichever trip the evaluator picks.

Run:
    uvicorn pathsnap.api.main:app --host 0.0.0.0 --port 8000
"""
import json
import logging
import os
import time
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathsnap.io import load_trip, list_trips
from pathsnap.candidates import generate_candidates
from pathsnap.features import build_features, FEATURE_COLUMNS
from pathsnap.labelling import label_trip
from pathsnap.viterbi import decode_trip, classifier_only

ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = os.environ.get("PATHSNAP_DATA", str(ROOT / "Student folder" / "Student folder"))
# Which folder holds the ten reserved evaluation trips. Defaults to the
# evaluator partition and falls back to student_test for local testing.
EVAL_PARTITION = os.environ.get("PATHSNAP_EVAL_PARTITION", "evaluator_test")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pathsnap")

app = FastAPI(title="PathSnap", version="1.0")
# The UI runs on a different machine, so the browser needs cross-origin
# permission to call this service.
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

STATE = {}


@app.on_event("startup")
def startup():
    with open(ROOT / "artifacts" / "config.json") as f:
        cfg = json.load(f)
    STATE["config"] = cfg
    STATE["model"] = joblib.load(ROOT / "artifacts" / cfg["model_file"])

    partition = EVAL_PARTITION
    try:
        dirs = list_trips(DATA_ROOT, partition)
    except FileNotFoundError:
        log.warning("partition %r not found under %s - falling back to student_test",
                    partition, DATA_ROOT)
        partition = "student_test"
        dirs = list_trips(DATA_ROOT, partition)

    STATE["partition"] = partition
    STATE["trip_dirs"] = {d.name: d for d in dirs}
    log.info("model %s loaded (trained %s, commit %s)",
             cfg["model_name"], cfg.get("exported_at", "?")[:10],
             cfg.get("git_commit", "?"))
    log.info("serving %d trips from partition %r", len(dirs), partition)


class PredictRequest(BaseModel):
    trip_id: str


def run_pipeline(trip):
    """
    Raw trip -> predicted arc per observation.

    Deliberately takes only the trip. It has no parameter through which ground
    truth could be passed, which is what makes the leakage guarantee
    structural rather than a matter of discipline.
    """
    cfg = STATE["config"]
    timings = {}

    t0 = time.time()
    cand = generate_candidates(trip, k=cfg["candidate_k"],
                               radius=cfg["candidate_radius_m"],
                               spacing=cfg["arc_sample_spacing_m"])
    timings["candidates_s"] = round(time.time() - t0, 3)

    t0 = time.time()
    feats = build_features(trip, cand)
    timings["features_s"] = round(time.time() - t0, 3)

    t0 = time.time()
    X = feats[cfg["feature_columns"]].to_numpy(np.float32)
    proba = STATE["model"].predict_proba(X)[:, 1]
    timings["classifier_s"] = round(time.time() - t0, 3)

    t0 = time.time()
    hmm = decode_trip(trip, cand, proba)
    timings["viterbi_s"] = round(time.time() - t0, 3)

    clf = classifier_only(trip, cand, proba)
    return hmm, clf, timings


def continuity(trip, pred):
    v = pred[pred >= 0]
    if len(v) < 2:
        return None
    a, b = v[:-1], v[1:]
    return float(((a == b) | (trip.arcs[a, 1] == trip.arcs[b, 0])).mean())


def score(trip, hmm, clf):
    """
    Compare a finished prediction against the reference route.

    Called only after run_pipeline has returned. Returns numbers, never
    geometry - the UI learns how well we did, not what the answer was.
    """
    if trip.route is None:
        return {"scored": False,
                "reason": "no reference route available for this trip"}

    labels = label_trip(trip)
    truth = labels["arc_id"]
    ok = labels["residual_m"] <= STATE["config"]["label_reliable_max_residual_m"]

    def acc(pred):
        m = (pred >= 0) & ok
        return float((pred[m] == truth[m]).mean()) if m.any() else None

    return {
        "scored": True,
        "top1_accuracy": acc(hmm),
        "top1_accuracy_classifier_only": acc(clf),
        "route_continuity": continuity(trip, hmm),
        "route_continuity_classifier_only": continuity(trip, clf),
        "n_observations_scored": int(((hmm >= 0) & ok).sum()),
        "n_observations_total": int(trip.n_obs),
        "frac_reference_usable": float(ok.mean()),
    }


@app.get("/health")
def health():
    cfg = STATE.get("config", {})
    return {"status": "ok" if "model" in STATE else "model not loaded",
            "model": cfg.get("model_name"),
            "git_commit": cfg.get("git_commit"),
            "partition": STATE.get("partition"),
            "n_trips": len(STATE.get("trip_dirs", {}))}


@app.get("/trips")
def trips():
    """The evaluation trips the UI offers. No route information."""
    out = []
    for tid, d in sorted(STATE["trip_dirs"].items()):
        track = d / f"{tid}.track"
        out.append({"trip_id": tid,
                    "size_kb": round(track.stat().st_size / 1000) if track.exists() else None})
    return {"partition": STATE["partition"], "trips": out}


@app.post("/predict")
def predict(req: PredictRequest):
    tid = req.trip_id
    if tid not in STATE["trip_dirs"]:
        raise HTTPException(404, f"unknown trip {tid}")

    log.info("PREDICT trip=%s  -- loading raw files", tid)
    t_start = time.time()
    trip = load_trip(STATE["trip_dirs"][tid], STATE["partition"])
    log.info("  %s: %d observations, %d arcs, %d nodes",
             tid, trip.n_obs, trip.n_arcs, len(trip.node_x))

    hmm, clf, timings = run_pipeline(trip)
    metrics = score(trip, hmm, clf)
    total = time.time() - t_start

    log.info("  %s: stages %s  total %.2fs", tid, timings, total)
    if metrics.get("scored"):
        log.info("  %s: top1 %.4f (classifier-only %.4f), continuity %.4f (%.4f)",
                 tid, metrics["top1_accuracy"], metrics["top1_accuracy_classifier_only"],
                 metrics["route_continuity"], metrics["route_continuity_classifier_only"])
    else:
        log.info("  %s: not scored (%s)", tid, metrics.get("reason"))

    # Geometry for the UI: our prediction and the raw GPS only.
    valid = hmm[hmm >= 0]
    keep = np.append(True, valid[1:] != valid[:-1]) if len(valid) else np.array([], bool)
    arc_seq = valid[keep]
    f, t = trip.arcs[arc_seq, 0], trip.arcs[arc_seq, 1]
    predicted_geom = [[[float(trip.node_lat[a]), float(trip.node_lon[a])],
                       [float(trip.node_lat[b]), float(trip.node_lon[b])]]
                      for a, b in zip(f, t)]

    step = max(1, trip.n_obs // 1500)
    gps = [[float(trip.gps_lat[i]), float(trip.gps_lon[i])]
           for i in range(0, trip.n_obs, step)]

    return {
        "trip_id": tid,
        "n_observations": int(trip.n_obs),
        "n_arcs_in_network": int(trip.n_arcs),
        "predicted_route_arc_ids": [int(a) for a in arc_seq],
        "predicted_route_length": len(arc_seq),
        "geometry": {"predicted": predicted_geom, "gps": gps},
        "metrics": metrics,
        "timings": {**timings, "total_s": round(total, 3)},
        "model": {"name": STATE["config"]["model_name"],
                  "git_commit": STATE["config"].get("git_commit")},
    }
