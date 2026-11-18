"""
Evaluation metrics.

WHY NOT PLAIN ACCURACY:
About 90% of candidate rows are negatives by construction (one correct arc per
~10 candidates). A model predicting "no" for everything scores ~90% accuracy
and identifies nothing. Every metric below is chosen to be immune to that.

THE KEY DISTINCTION - ROW LEVEL vs OBSERVATION LEVEL:
Row-level metrics (F1, log loss) score the binary task over all candidate
rows. Observation-level Top-1 scores the task we actually care about: for each
GPS observation, did the highest-scoring candidate turn out to be the right
arc? Top-1 is the headline number.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_recall_fscore_support


def top1_predictions(df, proba):
    """
    For each observation, the candidate the model scored highest.

    Returns a DataFrame indexed by (trip_id, obs_idx) with the chosen arc, the
    true arc, and whether they match.
    """
    work = pd.DataFrame({
        "trip_id": df["trip_id"].to_numpy(),
        "obs_idx": df["obs_idx"].to_numpy(),
        "arc_id": df["arc_id"].to_numpy(),
        "label": df["label"].to_numpy(),
        "proba": proba,
    })
    # idxmax on the score within each observation group.
    best = work.loc[work.groupby(["trip_id", "obs_idx"])["proba"].idxmax()]
    best = best.rename(columns={"arc_id": "pred_arc"})

    # The true arc for each observation (may be absent if the candidate set
    # missed it - those observations can never be got right).
    truth = (work[work["label"] == 1]
             .set_index(["trip_id", "obs_idx"])["arc_id"]
             .rename("true_arc"))

    best = best.set_index(["trip_id", "obs_idx"]).join(truth)
    best["correct"] = (best["pred_arc"] == best["true_arc"]).astype(int)
    return best


def route_continuity(best, arcs_by_trip):
    """
    Fraction of consecutive prediction pairs that are physically connected.

    Two consecutive predictions are "continuous" if they are the same arc, or
    the first arc's end node is the second arc's start node. A classifier
    working observation-by-observation has no notion of connectivity, so this
    is typically poor before HMM decoding and good after - which is how we
    demonstrate the decoder earned its place.

    arcs_by_trip : dict trip_id -> (n_arcs, 2) array of [from_node, to_node]
    """
    total = connected = 0
    for trip_id, grp in best.reset_index().groupby("trip_id"):
        arcs = arcs_by_trip.get(trip_id)
        if arcs is None or len(grp) < 2:
            continue
        seq = grp.sort_values("obs_idx")["pred_arc"].to_numpy()
        a, b = seq[:-1], seq[1:]
        same = a == b
        joins = arcs[a, 1] == arcs[b, 0]
        connected += int((same | joins).sum())
        total += len(a)
    return float(connected / total) if total else float("nan")


def evaluate(df, proba, arcs_by_trip=None, label="model"):
    """
    All metrics for one model's predictions on one data partition.

    `df` must carry trip_id, obs_idx, arc_id, label; `proba` is the predicted
    probability that each row is the true match.
    """
    y = df["label"].to_numpy()
    pred_binary = (proba >= 0.5).astype(int)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y, pred_binary, average="macro", zero_division=0
    )
    best = top1_predictions(df, proba)

    out = {
        "model": label,
        # --- the headline: per-observation arc selection ---
        "top1_accuracy": float(best["correct"].mean()),
        # --- row-level binary quality ---
        "macro_f1": float(f1_score(y, pred_binary, average="macro", zero_division=0)),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        # --- are the probabilities honest? matters because the HMM uses them ---
        "log_loss": float(log_loss(y, np.clip(proba, 1e-7, 1 - 1e-7), labels=[0, 1])),
        # --- ceiling: how often the true arc was even available to choose ---
        "candidate_recall": float(best["true_arc"].notna().mean()),
        "n_observations": int(len(best)),
        "n_rows": int(len(df)),
    }
    if arcs_by_trip is not None:
        out["route_continuity"] = route_continuity(best, arcs_by_trip)
    return out
