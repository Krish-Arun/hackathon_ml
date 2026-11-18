"""
Train and evaluate one baseline model.

Run one model per invocation so each stays inside a single shell call, and
append its row to results/model_comparison.csv:

    python scripts/train_baselines.py --model lightgbm

Protocol, applied identically to every model so the comparison is fair:
  * 5-fold GroupKFold over the 80 training trips, grouped by trip_id
  * training rows capped at TRAIN_CAP, validation at VAL_CAP, whole candidate
    groups only (see pathsnap.sampling for why)
  * unreliable labels excluded from TRAINING (trips whose .route covers only
    part of the journey) but RETAINED in evaluation, so reported scores are
    not flattered by quietly dropping hard cases
  * a final fit on the training partition, scored on the 10 student-test trips
"""
import sys, os, json, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.io import load_trip, list_trips          # noqa: E402
from pathsnap.models import build_models               # noqa: E402
from pathsnap.features import FEATURE_COLUMNS          # noqa: E402
from pathsnap.splits import group_folds, describe_folds  # noqa: E402
from pathsnap.sampling import subsample_observations   # noqa: E402
from pathsnap.metrics import evaluate                  # noqa: E402

DATA_ROOT = os.environ.get("PATHSNAP_DATA", str(ROOT / "Student folder" / "Student folder"))
TRAIN_CAP = 200_000
VAL_CAP = 100_000
N_FOLDS = 5


def load_table():
    df = pd.read_parquet(ROOT / "data" / "processed" / "parts")
    return df[df.partition == "train"].reset_index(drop=True), \
           df[df.partition == "student_test"].reset_index(drop=True)


def arcs_by_trip(partition):
    """Arc tables keyed by int trip_id, for the route-continuity metric."""
    out = {}
    for d in list_trips(DATA_ROOT, partition):
        trip = load_trip(d, partition)
        out[int(trip.trip_id)] = trip.arcs
    return out


def fit_and_score(est, train_df, val_df, arcs, label):
    """Fit on train_df, return metrics on val_df plus timings."""
    Xtr = train_df[FEATURE_COLUMNS].to_numpy(np.float32)
    ytr = train_df["label"].to_numpy()

    t0 = time.time()
    est.fit(Xtr, ytr)
    train_s = time.time() - t0

    Xva = val_df[FEATURE_COLUMNS].to_numpy(np.float32)
    t0 = time.time()
    proba = est.predict_proba(Xva)[:, 1]
    infer_s = time.time() - t0

    m = evaluate(val_df, proba, arcs_by_trip=arcs, label=label)
    m["train_seconds"] = round(train_s, 2)
    m["inference_seconds"] = round(infer_s, 2)
    m["inference_us_per_row"] = round(1e6 * infer_s / len(val_df), 1)
    return m, est


def run_fold(model_name, fold_id):
    """Run a single CV fold and append its row to results/cv_folds.csv.

    Folds run one per invocation because a Random Forest fit takes ~50s on
    this 2-core machine and six of them exceed a single shell call. Splitting
    them also makes the whole comparison resumable.
    """
    from sklearn.base import clone
    train_all, _ = load_table()
    est_factory, family, _ = build_models()[model_name]

    trainable = train_all[train_all.label_reliable].reset_index(drop=True)
    folds = group_folds(trainable, n_splits=N_FOLDS)
    leaks = [f for f in describe_folds(trainable, folds) if f["overlapping_trips"]]
    assert not leaks, f"trip appears on both sides of a fold: {leaks}"

    tr_idx, va_idx = folds[fold_id]
    tr = subsample_observations(trainable.iloc[tr_idx], TRAIN_CAP, seed=100 + fold_id)
    va = subsample_observations(trainable.iloc[va_idx], VAL_CAP, seed=200 + fold_id)

    m, _ = fit_and_score(clone(est_factory), tr, va, arcs_by_trip("train"), model_name)
    m["fold"] = fold_id
    m["model"] = model_name
    print(f"fold {fold_id}: top1={m['top1_accuracy']:.4f} macroF1={m['macro_f1']:.4f} "
          f"logloss={m['log_loss']:.4f} cont={m.get('route_continuity', float('nan')):.3f} "
          f"({m['train_seconds']:.0f}s fit)")

    out = ROOT / "results" / "cv_folds.csv"
    row = pd.DataFrame([m])
    if out.exists():
        prev = pd.read_csv(out)
        mask = ~((prev.model == model_name) & (prev.fold == fold_id))
        row = pd.concat([prev[mask], row], ignore_index=True)
    row.to_csv(out, index=False)


def finalise(model_name):
    """Fit on the training partition and score on the student-test trips,
    then roll the stored CV folds into results/model_comparison.csv."""
    from sklearn.base import clone
    train_all, student_test = load_table()
    est_factory, family, config_desc = build_models()[model_name]
    trainable = train_all[train_all.label_reliable].reset_index(drop=True)

    tr_final = subsample_observations(trainable, TRAIN_CAP, seed=7)
    test_m, _ = fit_and_score(clone(est_factory), tr_final, student_test,
                              arcs_by_trip("student_test"), model_name)
    print(f"student-test: top1={test_m['top1_accuracy']:.4f} "
          f"macroF1={test_m['macro_f1']:.4f} logloss={test_m['log_loss']:.4f} "
          f"cont={test_m.get('route_continuity', float('nan')):.3f}")

    cv = pd.read_csv(ROOT / "results" / "cv_folds.csv")
    cv = cv[cv.model == model_name]
    print(f"aggregating {len(cv)} stored CV folds")

    row = {
        "model": model_name, "family": family, "config": config_desc,
        "cv_top1_mean": cv.top1_accuracy.mean(), "cv_top1_std": cv.top1_accuracy.std(),
        "cv_macro_f1_mean": cv.macro_f1.mean(), "cv_macro_f1_std": cv.macro_f1.std(),
        "cv_log_loss_mean": cv.log_loss.mean(),
        "cv_route_continuity_mean": cv.route_continuity.mean(),
        "n_cv_folds": len(cv),
        "test_top1": test_m["top1_accuracy"], "test_macro_f1": test_m["macro_f1"],
        "test_precision_macro": test_m["precision_macro"],
        "test_recall_macro": test_m["recall_macro"],
        "test_log_loss": test_m["log_loss"],
        "test_route_continuity": test_m.get("route_continuity"),
        "test_candidate_recall": test_m["candidate_recall"],
        "train_seconds": test_m["train_seconds"],
        "inference_us_per_row": test_m["inference_us_per_row"],
        "n_train_rows": len(tr_final), "n_test_rows": len(student_test),
    }
    out = ROOT / "results" / "model_comparison.csv"
    df = pd.DataFrame([row])
    if out.exists():
        prev = pd.read_csv(out)
        df = pd.concat([prev[prev.model != model_name], df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"wrote {out.name}")


def main(model_name):
    train_all, student_test = load_table()
    models = build_models()
    est_factory, family, config_desc = models[model_name]

    print(f"model: {model_name}  ({family})")
    print(f"config: {config_desc}")
    print(f"train partition: {len(train_all):,} rows, {train_all.trip_id.nunique()} trips")

    # Unreliable labels are excluded from TRAINING only.
    trainable = train_all[train_all.label_reliable].reset_index(drop=True)
    print(f"after dropping unreliable labels: {len(trainable):,} rows "
          f"({100*len(trainable)/len(train_all):.1f}%)")

    arcs_train = arcs_by_trip("train")
    arcs_test = arcs_by_trip("student_test")

    # ---- cross-validation -------------------------------------------------
    folds = group_folds(trainable, n_splits=N_FOLDS)
    leaks = [f for f in describe_folds(trainable, folds) if f["overlapping_trips"]]
    assert not leaks, f"trip appears on both sides of a fold: {leaks}"

    fold_rows = []
    for i, (tr_idx, va_idx) in enumerate(folds):
        tr = subsample_observations(trainable.iloc[tr_idx], TRAIN_CAP, seed=100 + i)
        va = subsample_observations(trainable.iloc[va_idx], VAL_CAP, seed=200 + i)
        from sklearn.base import clone
        m, _ = fit_and_score(clone(est_factory), tr, va, arcs_train, model_name)
        m["fold"] = i
        fold_rows.append(m)
        print(f"  fold {i}: top1={m['top1_accuracy']:.4f} macroF1={m['macro_f1']:.4f} "
              f"logloss={m['log_loss']:.4f} cont={m.get('route_continuity', float('nan')):.3f} "
              f"({m['train_seconds']:.0f}s fit)", flush=True)

    cv = pd.DataFrame(fold_rows)

    # ---- final fit, scored on the held-out student-test trips -------------
    from sklearn.base import clone
    tr_final = subsample_observations(trainable, TRAIN_CAP, seed=7)
    test_m, fitted = fit_and_score(clone(est_factory), tr_final, student_test,
                                   arcs_test, model_name)
    print(f"  student-test: top1={test_m['top1_accuracy']:.4f} "
          f"macroF1={test_m['macro_f1']:.4f} logloss={test_m['log_loss']:.4f} "
          f"cont={test_m.get('route_continuity', float('nan')):.3f}")

    row = {
        "model": model_name,
        "family": family,
        "config": config_desc,
        # cross-validation: mean and spread across folds. Spread matters -
        # a model with a wide spread is unstable and likely to disappoint on
        # trips it has never seen.
        "cv_top1_mean": cv.top1_accuracy.mean(),
        "cv_top1_std": cv.top1_accuracy.std(),
        "cv_macro_f1_mean": cv.macro_f1.mean(),
        "cv_macro_f1_std": cv.macro_f1.std(),
        "cv_log_loss_mean": cv.log_loss.mean(),
        "cv_route_continuity_mean": cv.route_continuity.mean(),
        # student-test (development-stage held-out trips)
        "test_top1": test_m["top1_accuracy"],
        "test_macro_f1": test_m["macro_f1"],
        "test_precision_macro": test_m["precision_macro"],
        "test_recall_macro": test_m["recall_macro"],
        "test_log_loss": test_m["log_loss"],
        "test_route_continuity": test_m.get("route_continuity"),
        "test_candidate_recall": test_m["candidate_recall"],
        "train_seconds": test_m["train_seconds"],
        "inference_us_per_row": test_m["inference_us_per_row"],
        "n_train_rows": len(tr_final),
        "n_test_rows": len(student_test),
    }

    out = ROOT / "results" / "model_comparison.csv"
    df = pd.DataFrame([row])
    if out.exists():
        prev = pd.read_csv(out)
        df = pd.concat([prev[prev.model != model_name], df], ignore_index=True)
    df.to_csv(out, index=False)

    cvout = ROOT / "results" / "cv_folds.csv"
    cv["model"] = model_name
    if cvout.exists():
        prev = pd.read_csv(cvout)
        cv = pd.concat([prev[prev.model != model_name], cv], ignore_index=True)
    cv.to_csv(cvout, index=False)
    print(f"\nwrote {out.name} and {cvout.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fold", type=int, help="run just this CV fold")
    ap.add_argument("--final", action="store_true",
                    help="fit on train, score student-test, aggregate folds")
    a = ap.parse_args()
    if a.fold is not None:
        run_fold(a.model, a.fold)
    elif a.final:
        finalise(a.model)
    else:
        main(a.model)
