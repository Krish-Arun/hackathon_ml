"""
The model zoo: five algorithm families, one interface.

Each model answers the same question - given the features of one
(observation, candidate) pair, what is the probability this is the true arc?

WHY FIVE FAMILIES:
Different algorithms fail differently, and comparing them is how we justify
the final choice with evidence. The families here are deliberately spread:
linear, instance-based, bagged trees, boosted trees, and a neural net.

SCALING:
Trees are invariant to monotone feature scaling; distance- and gradient-based
models are not (KNN would be dominated by arc_length_m, which is three orders
of magnitude larger than t_along). So those models are wrapped in a Pipeline
with StandardScaler, which also guarantees the scaler is fitted on training
folds only and never sees validation data.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

import json
from pathlib import Path

SEED = 42
_ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"


def tuned_lightgbm_params():
    """
    Best hyperparameters found by the Bayesian search (scripts/optimise.py).

    Loaded from disk rather than pasted in, so the model in the repo always
    matches the recorded search and cannot silently drift from it.
    """
    path = _ARTIFACTS / "best_params.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)["best_params"]


def build_models():
    """
    The five baselines.

    Returns dict name -> (estimator, family, short config description).
    Configurations are deliberately modest: this is the untuned comparison,
    and section 9 tunes only the winner.
    """
    tuned = tuned_lightgbm_params()
    models = {
        "logistic_regression": (
            Pipeline([
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=1000,
                    # Candidates are ~90% negative; balancing stops the model
                    # from trivially predicting "no" for everything.
                    class_weight="balanced",
                    random_state=SEED,
                )),
            ]),
            "Linear / probabilistic",
            "L2, balanced class weight, max_iter=1000, standardised",
        ),
        "knn": (
            Pipeline([
                ("scale", StandardScaler()),
                ("clf", KNeighborsClassifier(
                    n_neighbors=25,
                    # Closer neighbours count for more - appropriate when the
                    # feature space has dense and sparse regions.
                    weights="distance",
                    n_jobs=-1,
                )),
            ]),
            "Instance-based",
            "k=25, distance-weighted, standardised",
        ),
        "random_forest": (
            RandomForestClassifier(
                # 100 rather than the usual few hundred: this machine has 2
                # cores and a forest of 200 took ~95s per fit, which made the
                # 6 fits of the CV protocol impractical. Out-of-bag variance
                # at 100 trees is already small for a table this size.
                n_estimators=100,
                min_samples_leaf=20,   # guards against memorising single rows
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=SEED,
            ),
            "Ensemble / bagged trees",
            "100 trees, min_samples_leaf=20, balanced subsample",
        ),
        "lightgbm": (
            lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=63,
                min_child_samples=40,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                random_state=SEED,
                n_jobs=-1,
                verbose=-1,
            ),
            "Boosting / gradient-boosted trees",
            "300 trees, lr=0.05, 63 leaves, 0.8 subsample",
        ),
        "mlp": (
            Pipeline([
                ("scale", StandardScaler()),
                ("clf", MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    # Early stopping keeps this honest on training time and
                    # gives us a genuine epoch-wise loss curve to plot - the
                    # only model here that legitimately has one.
                    early_stopping=True,
                    n_iter_no_change=8,
                    max_iter=120,
                    random_state=SEED,
                )),
            ]),
            "Neural network",
            "64-32 hidden, ReLU, Adam, early stopping",
        ),
    }

    # The tuned variant is registered only once the search has produced
    # best_params.json, so the untuned comparison can be reproduced from a
    # clean checkout without it.
    if tuned is not None:
        models["lightgbm_tuned"] = (
            lgb.LGBMClassifier(**tuned, subsample_freq=1, random_state=SEED,
                               n_jobs=-1, verbose=-1),
            "Boosting / gradient-boosted trees (Bayesian-tuned)",
            ", ".join(f"{k}={round(v, 4) if isinstance(v, float) else v}"
                      for k, v in tuned.items()),
        )
    return models
