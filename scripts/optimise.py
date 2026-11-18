"""
Bayesian hyperparameter optimisation for the selected model (report section 4).

WHY BAYESIAN RATHER THAN GRID OR RANDOM:
A grid over seven hyperparameters is combinatorially hopeless, and random
search learns nothing from the trials it has already run. Optuna's TPE sampler
builds a model of which regions of the space produce good scores and
concentrates its sampling there, while still exploring enough to avoid getting
stuck in a local optimum.

OBJECTIVE - LOG LOSS, NOT TOP-1:
Deliberate, and the reason matters. Our fold analysis showed Top-1 varies by
0.112 across folds against 0.015 across models, so optimising Top-1 would be
chasing fold noise. Log loss is both the stable signal and the quantity part
two actually depends on, because the HMM consumes these probabilities
directly - a confidently wrong classifier drags the decoder onto bad roads.

The study is persisted to SQLite so trials can be added across several runs:
    python scripts/optimise.py --trials 8
"""
import sys, os, json, argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.features import FEATURE_COLUMNS       # noqa: E402
from pathsnap.splits import group_folds             # noqa: E402
from pathsnap.sampling import subsample_observations  # noqa: E402

SEED = 42
N_FOLDS = 3          # 3 not 5: the search needs many fits, and the fold-to-fold
                     # signal we are optimising is stable enough at 3.
TRAIN_CAP = 200_000
VAL_CAP = 80_000
STUDY_DB = f"sqlite:///{ROOT / 'results' / 'optuna_lightgbm.db'}"
STUDY_NAME = "lightgbm_logloss"


def load_trainable():
    df = pd.read_parquet(ROOT / "data" / "processed" / "parts")
    df = df[(df.partition == "train") & df.label_reliable]
    return df.reset_index(drop=True)


def suggest(trial):
    """
    Search space, with a justification for each range.

    num_leaves / max_depth   capacity. 63 was the untuned value; we span an
                             order of magnitude either side.
    learning_rate            log scale - the useful range spans decades.
    n_estimators             interacts with learning_rate; a low rate needs
                             more trees, so both are searched together.
    min_child_samples        the main overfitting guard on a 2M-row table.
    subsample/colsample      stochasticity; below 0.5 the model starts
                             ignoring most of only 17 features.
    reg_lambda               L2 on leaf weights, log scale.
    """
    return {
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def make_objective(df, folds):
    def objective(trial):
        params = suggest(trial)
        losses = []
        for i, (tr_idx, va_idx) in enumerate(folds):
            tr = subsample_observations(df.iloc[tr_idx], TRAIN_CAP, seed=100 + i)
            va = subsample_observations(df.iloc[va_idx], VAL_CAP, seed=200 + i)
            model = lgb.LGBMClassifier(
                **params, subsample_freq=1, random_state=SEED, n_jobs=-1, verbose=-1
            )
            model.fit(tr[FEATURE_COLUMNS].to_numpy(np.float32), tr["label"].to_numpy())
            p = model.predict_proba(va[FEATURE_COLUMNS].to_numpy(np.float32))[:, 1]
            losses.append(log_loss(va["label"].to_numpy(),
                                   np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]))
            # Let Optuna abandon clearly bad configurations early.
            trial.report(float(np.mean(losses)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(losses))
    return objective


def main(n_trials):
    df = load_trainable()
    folds = group_folds(df, n_splits=N_FOLDS)
    print(f"{len(df):,} reliable training rows, {df.trip_id.nunique()} trips, "
          f"{N_FOLDS} grouped folds")

    study = optuna.create_study(
        study_name=STUDY_NAME, storage=STUDY_DB, load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    done = len([t for t in study.trials if t.state.is_finished()])
    print(f"study has {done} finished trials; adding {n_trials}")

    t0 = time.time()
    study.optimize(make_objective(df, folds), n_trials=n_trials,
                   show_progress_bar=False)
    print(f"{n_trials} trials in {time.time()-t0:.0f}s")

    # Retain the full search record - the brief requires it.
    trials = study.trials_dataframe()
    trials.to_csv(ROOT / "results" / "optuna_trials.csv", index=False)

    print(f"\nbest log loss: {study.best_value:.5f}")
    print("best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    with open(ROOT / "artifacts" / "best_params.json", "w") as f:
        json.dump({"best_value_logloss": study.best_value,
                   "best_params": study.best_params,
                   "n_trials": len([t for t in study.trials if t.state.is_finished()]),
                   "objective": "mean log loss over 3 trip-grouped CV folds",
                   "n_folds": N_FOLDS}, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=8)
    main(ap.parse_args().trials)
