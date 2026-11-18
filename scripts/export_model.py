"""
Fit the selected model on all reliable training data and freeze it for the API.

MODEL CHOICE - untuned LightGBM:
  * best log loss of the five families (0.0964), by a factor of 2.5 over the
    worst, and consistently so across every fold
  * Top-1 is joint-best but that is not the deciding factor: Top-1 varies by
    0.112 across folds against 0.015 across models, so it cannot separate them
  * the Bayesian search produced no improvement (all deltas under 0.001, inside
    fold noise) while making inference 13% slower, so the simpler untuned
    configuration is preferred
  * 13.4us per row - fast enough to decode a 2,500-point trip well inside the
    time an evaluator will wait

WHAT GETS FROZEN:
Not just the model. Every setting needed to rebuild its inputs identically at
inference: candidate radius, k, arc sampling spacing, and the exact feature
column order. If the API builds features even slightly differently from
training, predictions degrade silently with no error - one config file read by
both sides is what prevents that.
"""
import sys, json, subprocess, time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.features import FEATURE_COLUMNS, LABEL_RELIABLE_MAX_RESIDUAL_M  # noqa: E402
from pathsnap.models import build_models        # noqa: E402
from pathsnap.sampling import subsample_observations  # noqa: E402

SELECTED = "lightgbm"
# The final fit uses more data than the comparison runs did: the 200k cap
# existed to make 30 cross-validation fits affordable, and no longer applies
# when we are fitting once.
FINAL_TRAIN_CAP = 900_000


def main():
    df = pd.read_parquet(ROOT / "data" / "processed" / "parts")
    train = df[(df.partition == "train") & df.label_reliable]
    tr = subsample_observations(train, FINAL_TRAIN_CAP, seed=11)
    print(f"fitting {SELECTED} on {len(tr):,} rows "
          f"({tr.trip_id.nunique()} trips, {100*tr.label.mean():.1f}% positive)")

    est, family, config_desc = build_models()[SELECTED]
    t0 = time.time()
    est.fit(tr[FEATURE_COLUMNS].to_numpy(np.float32), tr["label"].to_numpy())
    fit_s = time.time() - t0
    print(f"fitted in {fit_s:.1f}s")

    (ROOT / "artifacts").mkdir(exist_ok=True)
    joblib.dump(est, ROOT / "artifacts" / "model.joblib", compress=3)

    with open(ROOT / "artifacts" / "dataset_config.json") as f:
        data_cfg = json.load(f)

    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                         cwd=ROOT, text=True).strip()
    except Exception:
        commit = "unknown"

    config = {
        "model_file": "model.joblib",
        "model_name": SELECTED,
        "model_family": family,
        "model_config": config_desc,
        "selection_reason": (
            "Best log loss of five families (0.0964 vs 0.2428 worst), consistent "
            "across folds. Top-1 could not separate the models: fold spread 0.112 "
            "vs model spread 0.015. Bayesian tuning gave no improvement inside "
            "noise and cost 13% inference speed, so the untuned config was kept."
        ),
        # Everything the API needs to rebuild features identically.
        "candidate_radius_m": data_cfg["candidate_radius_m"],
        "candidate_k": data_cfg["candidate_k"],
        "arc_sample_spacing_m": data_cfg["arc_sample_spacing_m"],
        "feature_columns": FEATURE_COLUMNS,
        "label_reliable_max_residual_m": LABEL_RELIABLE_MAX_RESIDUAL_M,
        "n_train_rows": int(len(tr)),
        "n_train_trips": int(tr.trip_id.nunique()),
        "fit_seconds": round(fit_s, 2),
        "random_seed": 42,
        "git_commit": commit,
        "exported_at": pd.Timestamp.utcnow().isoformat(),
    }
    with open(ROOT / "artifacts" / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    size = (ROOT / "artifacts" / "model.joblib").stat().st_size / 1e6
    print(f"wrote artifacts/model.joblib ({size:.1f} MB) and artifacts/config.json")
    print(f"git commit recorded: {commit}")


if __name__ == "__main__":
    main()
