"""
Cross-validation splits.

WHY GROUP-AWARE:
Observations from the same trip are near-duplicates: same roads, same vehicle,
same GPS conditions, often consecutive seconds apart. If ordinary KFold
scattered one trip's rows across every fold, then each time we "held out" a
fold the model would already have trained on almost identical rows from that
same trip. The score would be measuring memorisation, not generalisation, and
it would be far too high.

So every split here groups by trip_id: whole trips move together, and a trip
never appears on both sides of a fold. The brief requires this explicitly.
"""
import numpy as np
from sklearn.model_selection import GroupKFold


def group_folds(df, n_splits=5):
    """
    Yield (train_idx, val_idx) positional index pairs, grouped by trip.

    Returns positional indices into `df` as it was passed in, so callers must
    not reindex between calling this and using the result.
    """
    groups = df["trip_id"].to_numpy()
    gkf = GroupKFold(n_splits=n_splits)
    # GroupKFold is deterministic and ignores y, but we pass it for the API.
    return list(gkf.split(np.zeros(len(df)), groups=groups))


def partition_masks(df):
    """Boolean masks for the dataset's own train / student_test partitions."""
    part = df["partition"].to_numpy()
    return part == "train", part == "student_test"


def describe_folds(df, folds):
    """Sanity report - confirms no trip straddles a fold boundary."""
    rows = []
    trips = df["trip_id"].to_numpy()
    for i, (tr, va) in enumerate(folds):
        train_trips = set(trips[tr])
        val_trips = set(trips[va])
        rows.append({
            "fold": i,
            "train_rows": len(tr),
            "val_rows": len(va),
            "train_trips": len(train_trips),
            "val_trips": len(val_trips),
            # Must always be 0. Anything else is a leak.
            "overlapping_trips": len(train_trips & val_trips),
            "val_positive_rate": float(df["label"].to_numpy()[va].mean()),
        })
    return rows
