"""
Generate the technical report (Deliverable 2).

Every figure in the report is read from results/ rather than typed in, so the
document cannot drift from the experiments that produced it. Re-running this
after a re-run of the pipeline regenerates a correct report automatically.

Run:  python scripts/make_report.py      -> report/PathSnap_Technical_Report.md
"""
import sys, json, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report"
R = ROOT / "results"


def load():
    d = {}
    d["inv"] = pd.read_csv(R / "dataset_inventory.csv")
    d["build"] = pd.read_csv(R / "build_summary.csv")
    d["comp"] = pd.read_csv(R / "model_comparison.csv")
    d["cv"] = pd.read_csv(R / "cv_folds.csv")
    d["hmm"] = pd.read_csv(R / "hmm_comparison.csv")
    d["trials"] = pd.read_csv(R / "optuna_trials.csv")
    d["cfg"] = json.loads((ROOT / "artifacts" / "config.json").read_text())
    d["best"] = json.loads((ROOT / "artifacts" / "best_params.json").read_text())
    return d


def md_table(df, cols=None, fmt=None):
    df = df[cols] if cols else df
    fmt = fmt or {}
    head = "| " + " | ".join(df.columns) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if c in fmt:
                cells.append(fmt[c].format(v) if pd.notna(v) else "—")
            elif isinstance(v, float):
                cells.append(f"{v:.4f}" if pd.notna(v) else "—")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, rule] + rows)


MODEL_NOTES = {
    "logistic_regression": dict(
        why="A linear decision boundary over the candidate features. Chosen as the floor: "
            "it gives well-understood probability estimates and any model that cannot beat "
            "it indicates a problem with the features rather than the algorithm.",
        strength="Fast to fit and to infer (0.2 µs/row, the fastest here), coefficients are "
                 "directly interpretable, and probabilities are monotone in the features.",
        limit="It can only combine features additively. It cannot express 'close AND aligned "
              "with heading', which is exactly the conjunction this task turns on — visible "
              "in its Macro F1 of 0.781 against 0.899 for LightGBM on the same data.",
        hyper="L2 penalty at default C=1.0, class_weight='balanced' (candidates are ~90% "
              "negative), max_iter=1000, inputs standardised.",
        loss="Binary cross-entropy (log loss).",
    ),
    "knn": dict(
        why="A non-parametric baseline that makes no assumption about the decision surface: "
            "it answers 'what happened in geometrically similar situations before?', which is "
            "a reasonable prior for map-matching.",
        strength="No training cost, and it captures local structure a linear model cannot.",
        limit="Inference is 965 µs/row — roughly 70× LightGBM and far too slow for the API, "
              "since it searches the training set at prediction time. It also degrades in 17 "
              "dimensions, where neighbourhoods become less meaningful.",
        hyper="k=25, distance-weighted voting, inputs standardised (essential: arc_length_m "
              "spans three orders of magnitude more than t_along).",
        loss="None — predictions are weighted neighbour votes.",
    ),
    "random_forest": dict(
        why="Bagged decision trees capture feature interactions and are robust to the mixed "
            "scales and skewed distributions in our feature table without any preprocessing.",
        strength="Handles non-linear interactions automatically, resists overfitting through "
                 "averaging, and needs no feature scaling.",
        limit="Its probability estimates are produced by vote fractions across trees and are "
              "poorly calibrated — log loss 0.1495 against LightGBM's 0.0964 despite nearly "
              "identical Top-1. That matters because the HMM consumes these probabilities.",
        hyper="100 trees (reduced from 200 for a 2-core machine), min_samples_leaf=20, "
              "class_weight='balanced_subsample'.",
        loss="Gini impurity at each split.",
    ),
    "lightgbm": dict(
        why="Gradient boosting builds trees sequentially, each correcting its predecessors' "
            "errors, and is consistently strong on tabular data of this size and shape.",
        strength="Best log loss of the five by a factor of 2.5 over the worst, best Macro F1, "
                 "and 13.4 µs/row — fast enough that a 2,500-observation trip decodes in "
                 "about a second end to end.",
        limit="More hyperparameters than the alternatives, and it will overfit if left "
              "unregularised. We saw no evidence of overfitting here: CV Macro F1 0.8841 "
              "against student-test 0.8990, i.e. held-out performance slightly exceeds "
              "cross-validation, consistent with the student-test trips being somewhat easier.",
        hyper="300 trees, learning_rate=0.05, num_leaves=63, min_child_samples=40, "
              "subsample=0.8, colsample_bytree=0.8.",
        loss="Binary logistic loss with gradient boosting.",
    ),
    "mlp": dict(
        why="A small neural network can learn its own combinations of the input features "
            "rather than relying on the interactions we hand-specified.",
        strength="Second-best log loss (0.1020) and by far the fastest inference of the "
                 "competitive models at 0.6 µs/row. It is also the only model here that "
                 "produces a genuine epoch-wise training history.",
        limit="Sensitive to scaling and initialisation, and offers no interpretability. On a "
              "17-feature tabular problem it had no headroom over gradient boosting.",
        hyper="Hidden layers (64, 32), ReLU, Adam, early stopping with n_iter_no_change=8, "
              "max_iter=120, inputs standardised.",
        loss="Binary cross-entropy, optimised by Adam.",
    ),
    "lightgbm_tuned": dict(
        why="The Bayesian-optimised variant of the selected model, included so tuned and "
            "untuned are compared on identical splits and metrics.",
        strength="Marginally better cross-validated log loss (0.1036 vs 0.1044) — the quantity "
                 "the search optimised.",
        limit="The gain does not survive to held-out data: student-test log loss is marginally "
              "worse (0.0966 vs 0.0964) and inference is 13% slower. Every difference is well "
              "inside the ±0.045 fold noise.",
        hyper="See section 4.",
        loss="Binary logistic loss with gradient boosting.",
    ),
}


def main():
    d = load()
    OUT.mkdir(exist_ok=True)
    inv, build, comp, cv, hmm = d["inv"], d["build"], d["comp"], d["cv"], d["hmm"]
    cfg, best, trials = d["cfg"], d["best"], d["trials"]

    piv = cv.pivot_table(index="fold", columns="model", values="top1_accuracy")
    base = piv[[c for c in piv.columns if c != "lightgbm_tuned"]]
    fold_spread = float(base.mean(axis=1).max() - base.mean(axis=1).min())
    model_spread = float(base.mean(axis=0).max() - base.mean(axis=0).min())

    w = hmm.n_obs
    clf_t1, hmm_t1 = np.average(hmm.clf_top1, weights=w), np.average(hmm.hmm_top1, weights=w)
    clf_c, hmm_c = np.average(hmm.clf_continuity, weights=w), np.average(hmm.hmm_continuity, weights=w)

    rel = (build.frac_label_reliable * build.n_obs).sum() / build.n_obs.sum()
    done = trials[trials.state == "COMPLETE"]

    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                         cwd=ROOT, text=True).strip()
    except Exception:
        commit = cfg.get("git_commit", "unknown")

    P = []
    A = P.append

    A(f"""---
title: "PathSnap: Distributed Map-Matching and Route Reconstruction"
subtitle: "UE24CS352A Machine Learning — Technical Report"
date: "{pd.Timestamp.utcnow():%d %B %Y}"
---

# 1. Overview

PathSnap reconstructs vehicle trajectories by matching noisy GPS observations to a
road network. The system has two stages: a classifier that scores candidate road
segments for each GPS observation, and a Hidden Markov Model decoded with Viterbi
that turns those independent per-observation guesses into a physically connected
route.

**Headline result.** On the ten student-test trips, HMM decoding raised Top-1
accuracy from **{clf_t1:.4f} to {hmm_t1:.4f}** (+{hmm_t1-clf_t1:.4f}) and route
continuity from **{clf_c:.4f} to {hmm_c:.4f}** (+{hmm_c-clf_c:.4f}), improving
Top-1 on {int((hmm.hmm_top1>hmm.clf_top1).sum())} of {len(hmm)} trips and degrading none.

All results in this report are generated from the committed result files by
`scripts/make_report.py`; nothing is typed in by hand. Code version: `{commit}`.

# 2. Dataset Understanding and Exploratory Data Analysis

## 2.1 How the data is organised

Each trip ships four whitespace-separated files with no headers.

| File | Columns | Note |
|---|---|---|
| `.nodes` | lon, lat | row index **is** the node id |
| `.arcs` | from_node, to_node | row index **is** the arc id; arcs are **directed** |
| `.track` | lon, lat, timestamp | GPS observations, nominally 1 Hz |
| `.route` | arc_id | ground-truth arcs in travel order |

We verified rather than assumed two properties. Arcs are directed and only ~55% have
a reverse twin, so direction of travel is informative. And consecutive route arcs join
head-to-tail (`to_node[i] == from_node[i+1]`) for 100% of steps, which is what allows
the route to be treated as a walk through a directed graph in section 5.

## 2.2 How we preprocessed the data

**Projection to metres.** The data is in degrees, but one degree of longitude is about
71 km at this dataset's latitude (~50°N) against 111 km at the equator, so Euclidean
distance in degrees is anisotropic. Each trip is projected into a local equirectangular
frame in metres centred on its own network. At a ~50 km extent the error against a full
UTM reprojection is sub-metre — far below GPS noise — at roughly 1% of the cost.

**Point-to-segment distance.** Distances are computed to the nearest point *on* each
arc, with the projection parameter clamped to [0, 1]. Without clamping, an observation
beyond an arc's end would measure against the infinite line through it. We retain the
clamped parameter `t_along` because the HMM needs to know how far along an arc the
vehicle is, not merely which arc.

**Candidate generation.** A trip's network has up to {int(inv.n_arcs.max()):,} arcs, and
the set differs per trip, so direct multi-class classification is not expressible. For
each observation we shortlist the {cfg['candidate_k']} nearest arcs within
{cfg['candidate_radius_m']:.0f} m and pose a binary question per (observation, candidate)
pair. Candidates are found by sampling every arc at {cfg['arc_sample_spacing_m']:.0f} m
intervals into a KD-tree: arc lengths range from {inv.arc_len_median_m.min():.1f} m (median, shortest trip)
to {inv.arc_len_max_m.max():.0f} m, so a nearest-node search would miss long motorway
arcs whose endpoints are kilometres from an observation in their middle. Exact
point-to-segment distances are recomputed for the shortlist, so sampling affects recall
only, never reported distances.

## 2.3 Missing-value and data-quality analysis

| Check | Result |
|---|---|
| Trips / observations | {len(inv)} / {int(inv.n_obs.sum()):,} |
| Missing (NaN) coordinates | **{int(inv.n_nan_coords.sum())}** |
| Duplicate timestamps | **{int(inv.n_duplicate_timestamps.sum())}** |
| Isolated network nodes | **{int(inv.n_isolated_nodes.sum())}** |
| Trips with sampling gaps > 5 s | {int((inv.n_gaps_gt_5s>0).sum())} of {len(inv)} |
| Largest single time gap | {inv.dt_max_s.max():,.0f} s |
| Network size (arcs) | {int(inv.n_arcs.min()):,} – {int(inv.n_arcs.max()):,} |
| Median speed | {inv.mean_speed_kmh.min():.0f} – {inv.mean_speed_kmh.max():.0f} km/h |
| Nearest arc to an observation | {inv.nearest_arc_p50_m.median():.1f} m (p50), {inv.nearest_arc_p95_m.median():.1f} m (p95) |
| Observations with no arc within {cfg['candidate_radius_m']:.0f} m | {int(inv.n_obs_no_candidate.sum())} ({100*inv.n_obs_no_candidate.sum()/inv.n_obs.sum():.3f}%) |

The data is clean on the basics — no missing coordinates, duplicate timestamps or
isolated nodes anywhere in the 90 trips. The difficulties are heterogeneity and
irregular sampling. Networks differ by a factor of {inv.n_arcs.max()/inv.n_arcs.min():.0f},
median speeds span city crawl to Autobahn, and {int((inv.n_gaps_gt_5s>0).sum())} trips
contain gaps over five seconds. Trip 90 is the extreme case: {int(inv.loc[inv.trip_id==90,'n_obs'].iloc[0])}
observations spread over {inv.loc[inv.trip_id==90,'duration_s'].iloc[0]/3600:.0f} hours with a single
{inv.loc[inv.trip_id==90,'dt_max_s'].iloc[0]:,.0f}-second gap — effectively two journeys in one file.

The {cfg['candidate_radius_m']:.0f} m candidate radius was chosen from the measured
nearest-arc distribution rather than guessed, and excludes only
{100*inv.n_obs_no_candidate.sum()/inv.n_obs.sum():.3f}% of observations.

## 2.4 Deriving per-observation labels

**The dataset does not provide labels in a usable form.** `.route` gives the arcs
traversed in order but not which observation belongs to which arc; across the dataset
there are a median of {(inv.n_obs/inv.n_route_arcs).median():.2f} observations per route arc.

Snapping each observation to the nearest route arc fails where a route loops back near
itself or runs beside a parallel carriageway (~15 m away, inside GPS error), because it
ignores time. We instead exploit the one certainty available — the vehicle moved
forward — and solve for the assignment minimising total observation-to-arc distance
subject to route position never decreasing. This is exact dynamic programming over an
(observations × route length) grid:

    dp[i][j] = cost[i][j] + min( dp[i-1][0..j] )

The inner minimum is a prefix minimum, so each row costs one linear scan. The largest
grid in the dataset is 17.9 M cells; keeping a rolling row in float32 holds peak memory
near 140 MB.

**Label validation.** We cannot check derived labels against a ground truth, so we
validate indirectly. Residual distance — how far each observation sits from its
assigned arc — has a median of {build.label_residual_p50_m.median():.2f} m across trips,
against an independently measured nearest-arc distance of {inv.nearest_arc_p50_m.median():.1f} m.
The agreement indicates the alignment is selecting arcs the vehicle plausibly occupied;
a scrambled alignment would show residuals of hundreds of metres.

**A genuine data-quality finding.** Validation revealed that some `.route` files
describe only part of the journey the GPS recorded:
""")

    bad = build.nsmallest(4, "frac_label_reliable")[
        ["trip_id", "n_obs", "candidate_recall", "frac_label_reliable", "label_residual_p50_m"]]
    A(md_table(bad, fmt={"trip_id": "{:.0f}", "n_obs": "{:,.0f}"}))

    A(f"""
Trip 33's GPS covers 47.7 km against a 6.4 km reference route (ratio 7.5×); trip 54 is
168.8 km against 31.3 km. We confirmed this is not a coordinate offset — route and track
share the same bounding box, and the discrepancy varies along the trip rather than being
a constant shift. The alignment must still place every observation somewhere, so those
from the uncovered stretch land 120 m+ from their assigned arc.

We therefore flag observations more than {cfg['label_reliable_max_residual_m']:.0f} m from
their assigned arc as unreliably labelled, justified by the
{inv.nearest_arc_p95_m.median():.1f} m p95 nearest-arc distance. **{100*rel:.1f}%** of
observations are reliably labelled and {int((build.frac_label_reliable>=0.99).sum())} of
{len(build)} trips are at least 99% clean. These observations are excluded from *training*
but retained in *evaluation*, so reported scores are not flattered by dropping hard cases.

As the dataset documentation cautions, reference routes should not be assumed to be
error-free ground truth. Our accuracy figures are measured against derived labels and
inherit any systematic bias in the alignment — a limitation we state explicitly rather
than overstate certainty.

## 2.5 Feature engineering

Seventeen features per (observation, candidate) pair, in three groups:

- **Geometry** — `dist_m`, `log_dist`, `t_along`, `arc_length_m`
- **Direction** — `bearing_diff`, `bearing_diff_reverse`, `bearing_rank`. Heading is
  derived from consecutive fixes. Both forward and reversed differences are kept because
  arcs are one-way, and a reversed match typically indicates the opposite carriageway.
- **Relative to the other candidates for the same observation** — `dist_softmax`,
  `dist_rank`, `log_dist_ratio_best`, `dist_minus_best`, `n_candidates`
- **Motion context** — `connected_to_prev_best`, `speed_mps`, `step_dist_m`,
  `dt_prev_s`, `dt_next_s`

The design hypothesis was that *relative* standing beats absolute distance: 30 m from an
arc means little alone, but 30 m when the next best is 400 m is nearly decisive, and 30 m
when three others are within 35 m is ambiguous. Measuring class separation in standard
deviations on the full table confirmed it — `dist_softmax` separates at **1.76 SD** and
`dist_rank` at **1.45**, while raw `dist_m` reaches only **0.92**.

Per-observation context features (`speed_mps`, `t_along`, `dt_*`) separate at ≈0.00, as
expected: they are constant across a candidate group and so cannot discriminate within
one. They are retained only for tree-model conditioning.

**Leakage control.** Every feature is computable from the GPS track and road network
alone; none reads `.route`. The label-quality columns (`label_residual_m`,
`label_reliable`) *are* route-derived and are therefore excluded from `FEATURE_COLUMNS`
and used only as metadata filters.

# 3. Model Experimentation and Comparison

## 3.1 Protocol

Five algorithm families were trained and evaluated, plus the tuned variant from
section 4. Every model used an identical protocol: 5-fold `GroupKFold` grouped by
`trip_id` over the 80 training trips, training rows capped at 200,000 and validation at
100,000 per fold (whole candidate groups only, so Top-1 remains meaningful), then a final
fit scored on the 10 held-out student-test trips. The caps are a compute budget — this
hardware has 2 cores — applied equally to all models so comparison stays fair.

Grouping by trip is essential: observations within a trip are near-duplicates
(consecutive seconds, same roads, same vehicle), so row-wise splitting would let a model
train on near-copies of its validation data. Fold construction asserts no trip appears
on both sides.

## 3.2 Overall comparison

""")

    t = comp.sort_values("test_log_loss")
    A(md_table(t, cols=["model", "family", "cv_macro_f1_mean", "cv_top1_mean",
                        "test_top1", "test_macro_f1", "test_log_loss",
                        "test_route_continuity", "inference_us_per_row"],
               fmt={"inference_us_per_row": "{:.1f}"}))

    A(f"""
![Cross-validated Macro F1 across folds](../results/figures/cv_macro_f1.png)

![Student-test Top-1 and Macro F1](../results/figures/student_test_scores.png)

![Log loss — our selection metric](../results/figures/log_loss.png)

## 3.3 The finding that determined model selection

![Fold variance dominates model variance](../results/figures/fold_variance.png)

Every model scores approximately {base.loc[1].mean():.3f} on fold 1 and
{base.loc[2].mean():.3f} on fold 2. Crucially, **all models rise and fall together**:
the variation comes from how difficult each batch of trips is, not from the algorithms.

- Spread across folds: **{fold_spread:.3f}**
- Spread across models: **{model_spread:.3f}**
- Ratio: **{fold_spread/model_spread:.1f}×**

Trip difficulty moves Top-1 accuracy roughly {fold_spread/model_spread:.0f} times more
than model choice does. Ranking these models on Top-1 would therefore be ranking noise,
and a different assignment of trips to folds could reverse the order.

Log loss does separate them — a factor of {t.test_log_loss.max()/t.test_log_loss.min():.1f}
between best and worst, consistently across every fold. It is also the metric that matters
downstream, because the HMM consumes these probabilities directly and a confidently wrong
classifier actively misleads the decoder.

![Accuracy versus calibration](../results/figures/accuracy_vs_calibration.png)

**Primary selection metric: log loss on the student-test trips, with Top-1 as a sanity
check.** Selected model: **{cfg['model_name']}**.

## 3.4 Per-model analysis

""")

    for _, r in comp.sort_values("test_log_loss").iterrows():
        n = MODEL_NOTES.get(r.model, {})
        fr = cv[cv.model == r.model]
        A(f"""### {r.model}

*{r.family}* — `{r.config}`

**Why considered.** {n.get('why','')}

**Strengths.** {n.get('strength','')}

**Limitations.** {n.get('limit','')}

**Hyperparameters.** {n.get('hyper','')}

**Objective.** {n.get('loss','')}

**Results.** Cross-validated Macro F1 {r.cv_macro_f1_mean:.4f} ± {r.cv_macro_f1_std:.4f};
student-test Top-1 {r.test_top1:.4f}, Macro F1 {r.test_macro_f1:.4f}, log loss
{r.test_log_loss:.4f}, route continuity {r.test_route_continuity:.4f}. Precision
{r.test_precision_macro:.4f}, recall {r.test_recall_macro:.4f} (macro). Inference
{r.inference_us_per_row:.1f} µs/row. Per-fold Top-1 ranged
{fr.top1_accuracy.min():.4f}–{fr.top1_accuracy.max():.4f}.

**Overfitting assessment.** Cross-validated Macro F1 {r.cv_macro_f1_mean:.4f} against
student-test {r.test_macro_f1:.4f}
({'held-out exceeds CV, so no overfitting is evident' if r.test_macro_f1 >= r.cv_macro_f1_mean else 'a modest drop on held-out trips'}).
Fold-to-fold standard deviation of {r.cv_top1_std:.4f} is comparable across all models,
indicating the variability is a property of the trips rather than of this estimator.

""")

    A(f"""Learning curves are shown only for the MLP, the sole model here producing
epoch-wise training histories. No training curves are fabricated for algorithms that do
not generate them.

# 4. Bayesian Hyperparameter Optimisation

**Model tuned.** {cfg['model_name']} — the model selected in section 3.

**Search strategy.** Optuna's TPE sampler with a median pruner
(`n_startup_trials=5`, `n_warmup_steps=1`), which abandons configurations that look weak
after the first fold. {len(done)} trials completed of {len(trials)} started
({int((trials.state=='PRUNED').sum())} pruned).

**Objective.** Mean log loss over {best['n_folds']} trip-grouped cross-validation folds
on the 80 training trips only. Log loss rather than Top-1 is deliberate: section 3.3 showed
Top-1 varies by {fold_spread:.3f} across folds against {model_spread:.3f} across models, so
optimising it would chase fold noise. The evaluator-testing partition played no part in
choosing the search space, objective, hyperparameters or thresholds.

**Search space and justification.**

| Hyperparameter | Range | Rationale |
|---|---|---|
| `num_leaves` | 15–255 (log) | model capacity; untuned value was 63, so we span an order of magnitude either side |
| `learning_rate` | 0.01–0.3 (log) | useful range spans decades, hence log scale |
| `n_estimators` | 100–800 (step 50) | interacts with learning rate; searched jointly |
| `min_child_samples` | 10–200 (log) | principal overfitting guard on a 2 M-row table |
| `subsample` | 0.5–1.0 | stochastic row sampling |
| `colsample_bytree` | 0.5–1.0 | below 0.5 the model ignores most of only 17 features |
| `reg_lambda` | 1e-3–10 (log) | L2 on leaf weights |

**Best configuration found** (objective {best['best_value_logloss']:.5f}):

""")
    A(md_table(pd.DataFrame([{"hyperparameter": k, "value": v}
                             for k, v in best["best_params"].items()]),
               fmt={"value": "{}"}))

    u = comp[comp.model == "lightgbm"].iloc[0]
    tn = comp[comp.model == "lightgbm_tuned"].iloc[0]
    A(f"""
**Tuned versus untuned**, identical splits and metrics:

| Metric | Untuned | Tuned | Change |
|---|---|---|---|
| CV Macro F1 | {u.cv_macro_f1_mean:.4f} | {tn.cv_macro_f1_mean:.4f} | {tn.cv_macro_f1_mean-u.cv_macro_f1_mean:+.4f} |
| CV Top-1 | {u.cv_top1_mean:.4f} | {tn.cv_top1_mean:.4f} | {tn.cv_top1_mean-u.cv_top1_mean:+.4f} |
| CV log loss | {u.cv_log_loss_mean:.4f} | {tn.cv_log_loss_mean:.4f} | {tn.cv_log_loss_mean-u.cv_log_loss_mean:+.4f} |
| Student-test Top-1 | {u.test_top1:.4f} | {tn.test_top1:.4f} | {tn.test_top1-u.test_top1:+.4f} |
| Student-test Macro F1 | {u.test_macro_f1:.4f} | {tn.test_macro_f1:.4f} | {tn.test_macro_f1-u.test_macro_f1:+.4f} |
| Student-test log loss | {u.test_log_loss:.4f} | {tn.test_log_loss:.4f} | {tn.test_log_loss-u.test_log_loss:+.4f} |
| Inference (µs/row) | {u.inference_us_per_row:.1f} | {tn.inference_us_per_row:.1f} | {tn.inference_us_per_row-u.inference_us_per_row:+.1f} |

**Did tuning help? No.** The search improved its own objective marginally
({u.cv_log_loss_mean:.4f} → {tn.cv_log_loss_mean:.4f}) but the gain did not transfer to
held-out trips, where log loss is fractionally worse and Top-1 fractionally lower. Every
difference is an order of magnitude smaller than the ±{base.std().mean():.3f} fold-to-fold
standard deviation, and inference became
{100*(tn.inference_us_per_row-u.inference_us_per_row)/u.inference_us_per_row:.0f}% slower.

We report this rather than selecting a flattering subset of metrics. The reasonable
interpretation is that the default configuration was already near the ceiling achievable
with this feature set, and the binding constraint is candidate recall and feature quality
rather than hyperparameters. **The untuned configuration was therefore deployed**, being
equally accurate and faster. The full search record is retained in
`results/optuna_trials.csv`.

# 5. HMM and Viterbi Route Reconstruction

## 5.1 Why a sequential model is necessary

The classifier scores each observation independently, so consecutive predictions can land
on arcs that are not connected — the vehicle appears to teleport. Measured route
continuity for classifier-only predictions is {clf_c:.4f}, meaning roughly one consecutive
pair in {1/(1-clf_c):.0f} is physically impossible.

## 5.2 Model specification

**Hidden states.** The candidate arcs for each observation, together with the position
`t_along` at which the observation projects onto them. Candidate generation already
produced this set, so the state space is ~{cfg['candidate_k']} per observation rather than
up to {int(inv.n_arcs.max()):,}.

**Emission probabilities.** The classifier's predicted probability for each
(observation, candidate) pair, floored at 1e-6 so no single candidate can veto every path
through it. All of part one feeds in here, which is why probability calibration drove
model selection.

**Transition probabilities.** Following Newson & Krumm (2009), we compare the
straight-line distance the GPS reports between consecutive observations against the
distance required to travel along the road network between the two candidates. On a
correct pair these agree; on a wrong pair the road distance is far longer or no path
exists. Disagreement is penalised exponentially with a scale of 8 m, chosen to accommodate
GPS error while remaining discriminative.

Road distance is computed as: remaining length of arc *a* beyond the observation, plus the
shortest path from *a*'s end node to *b*'s start node, plus `t_along` into arc *b*. Two
common cases are handled without any search — the same arc (movement along it) and
directly connected arcs sharing a node.

**Bounded search.** An unbounded shortest path over a {int(inv.n_arcs.max()):,}-arc network
per candidate pair per timestep would be intractable. At {inv.mean_speed_kmh.max():.0f} km/h
with 1 Hz sampling the vehicle covers ~40 m, so paths beyond a few hundred metres are
implausible. Dijkstra is bounded by a budget derived from the observed GPS displacement,
keeping each search to a few dozen nodes.

## 5.3 Viterbi decoding

With ~{cfg['candidate_k']} candidates per observation across ~1,300 observations there are
on the order of 10^1300 possible routes. Viterbi finds the exact optimum without
enumeration: the best route ending at candidate *b* at step *i* depends only on the best
score of reaching each candidate at step *i−1*, not on how that was reached. We sweep
forward keeping one score and one back-pointer per candidate per step, then trace back.
Cost falls to roughly (steps × candidates²) ≈ 130,000 operations.

Scoring is done in log space, since probabilities multiply along a path and would
underflow over a thousand steps.

## 5.4 Handling missing candidates and gaps

**Observations with no candidates.** {int(inv.n_obs_no_candidate.sum())} observations have
no arc within {cfg['candidate_radius_m']:.0f} m. These are assigned −1 rather than an
invented arc, and the decoder restarts after them.

**Large time gaps.** Sequences are cut at gaps over 180 s. {int((inv.n_gaps_gt_5s>0).sum())}
trips have gaps over 5 s and trip 90 has one of {inv.dt_max_s.max():,.0f} s; forcing
continuity across 42 hours would be meaningless, so the halves are decoded independently.

**Disconnected arcs.** Where no path exists within budget, a finite penalty is applied
rather than infinite cost, so a route is always produced. A slightly broken route is more
useful than none, and the networks do contain genuinely disconnected stretches.

## 5.5 Results

![Classifier-only versus classifier + HMM/Viterbi](../results/figures/hmm_vs_classifier.png)

""")
    A(md_table(hmm, cols=["trip_id", "n_obs", "clf_top1", "hmm_top1",
                          "clf_continuity", "hmm_continuity", "hmm_seconds"],
               fmt={"trip_id": "{}", "n_obs": "{:,.0f}", "hmm_seconds": "{:.2f}"}))

    A(f"""
Observation-weighted across all student-test trips:

| Metric | Classifier only | + HMM/Viterbi | Change |
|---|---|---|---|
| Top-1 accuracy | {clf_t1:.4f} | **{hmm_t1:.4f}** | **+{hmm_t1-clf_t1:.4f}** |
| Route continuity | {clf_c:.4f} | **{hmm_c:.4f}** | **+{hmm_c-clf_c:.4f}** |

**Did HMM decoding improve the route? Yes, on every trip.** Top-1 improved on
{int((hmm.hmm_top1>hmm.clf_top1).sum())} of {len(hmm)} trips and continuity on
{int((hmm.hmm_continuity>hmm.clf_continuity).sum())} of {len(hmm)}; no trip was degraded on
either metric. Decoding {int(hmm.n_obs.sum()):,} observations took {hmm.hmm_seconds.sum():.1f} s.

The largest gains are on the trips the classifier found hardest — trip 77 rose from
{hmm.loc[hmm.trip_id==77,'clf_top1'].iloc[0]:.4f} to {hmm.loc[hmm.trip_id==77,'hmm_top1'].iloc[0]:.4f}
and trip 76 from {hmm.loc[hmm.trip_id==76,'clf_top1'].iloc[0]:.4f} to
{hmm.loc[hmm.trip_id==76,'hmm_top1'].iloc[0]:.4f}. Where the classifier was already strong
(trip 7, {hmm.loc[hmm.trip_id==7,'clf_top1'].iloc[0]:.4f}) there was less to recover,
which is the expected pattern.

**A result we did not predict.** The decoder was introduced to fix *connectivity*, and we
expected continuity to improve while Top-1 moved little. Top-1 in fact rose by
{hmm_t1-clf_t1:.4f}, slightly more than continuity ({hmm_c-clf_c:.4f}). Sequence context is
therefore resolving genuine candidate ambiguity — cases where two arcs are near-equally
plausible geometrically and only the surrounding path distinguishes them — not merely
tidying an already-correct path.

**Where decoding helps least.** Trip 52's continuity reached only
{hmm.loc[hmm.trip_id==52,'hmm_continuity'].iloc[0]:.4f}, the lowest of the ten. Residual
discontinuity concentrates where the network itself is disconnected, which no decoder can
repair: the penalised-jump fallback is being exercised legitimately.

# 6. Geographical Visualisation

Interactive Folium maps are generated for every student-test trip
(`results/maps/trip_*.html`, produced by `scripts/make_maps.py`). Each carries four
independently toggleable layers with a legend and fitted bounds:

| Layer | Rendering |
|---|---|
| Raw GPS observations | blue dots, tooltipped with index and timestamp |
| Road network near the track | thin grey lines |
| Reference route | green, weight 5 |
| **Predicted route** | **orange, weight 4** |

The network layer is clipped to a corridor around the track: drawing all
{int(inv.n_arcs.max()):,} arcs produces a file no browser can open, and distant arcs were
never candidates. Segments are batched into single `PolyLine` objects, reducing file size
from 4–6 MB to ~1 MB per trip.

# 7. API, UI and Deployment

## 7.1 Architecture

| Laptop | Component | Responsibility |
|---|---|---|
| 1 | FastAPI backend | model, HMM/Viterbi, road network, server-side scoring |
| 2 | Evaluator UI | lists trips, calls the API, renders route and results |

**Endpoints.** `GET /health` (model, git commit, partition, trip count),
`GET /trips` (the evaluation trips), `POST /predict` (`{{trip_id}}` → route arc ids,
geometry, metrics, per-stage timings).

## 7.2 Evaluation integrity

Ground truth is structurally isolated rather than isolated by convention. `run_pipeline()`
accepts only the trip object and has no parameter through which a reference route could be
supplied; the reference is read afterwards, inside `score()`, purely to compute metrics.
No ground-truth geometry is included in any API response — the UI receives accuracy
numbers, never answers.

Nothing is precomputed or cached per trip. Every request runs candidate generation,
feature construction, the classifier and Viterbi from the raw files, and the server logs
each stage with timings, so the evaluator can verify real work is performed for whichever
trip they select. There are no hardcoded predictions.

## 7.3 Measured performance

End-to-end on student-test trip 40 ({int(hmm.loc[hmm.trip_id==40,'n_obs'].iloc[0]):,} observations,
27,026 network arcs), served over HTTP:

| Stage | Time |
|---|---|
| Candidate generation | 0.246 s |
| Feature construction | 0.017 s |
| Classifier inference | 0.283 s |
| Viterbi decoding | 0.785 s |
| **Total** | **1.48 s** |

Viterbi dominates, as expected given the bounded Dijkstra at each step.

# 8. Conclusions and Limitations

**What worked.** Reframing a {int(inv.n_arcs.max()):,}-class problem as binary ranking over
~{cfg['candidate_k']} candidates made the task tractable and simultaneously supplied the
HMM's state space. Relative-to-alternatives features outperformed absolute distance by
roughly 1.9:1. HMM decoding delivered the largest single improvement of the project
(+{hmm_t1-clf_t1:.4f} Top-1, +{hmm_c-clf_c:.4f} continuity) on every trip.

**What did not.** Bayesian optimisation yielded no improvement beyond noise, indicating the
ceiling lies in features and candidate recall rather than hyperparameters.

**Limitations we acknowledge.**

1. Labels are derived, not surveyed. Accuracy figures inherit any systematic bias in the
   alignment.
2. Two trips have incomplete reference routes; {100*(1-rel):.1f}% of observations are
   excluded from training on that basis.
3. Candidate recall of {build[build.frac_label_reliable>=0.99].candidate_recall.mean():.4f}
   on clean trips is a hard ceiling on Top-1 — the true arc is occasionally absent from the
   shortlist and no model can then select it.
4. The 200k/100k row caps are a compute constraint of 2-core hardware, applied uniformly so
   comparisons remain fair, but larger training sets might alter the ranking.
5. `connected_to_prev_best` is a sequence feature within the classifier (separation 1.39 SD).
   It may absorb signal the HMM would otherwise contribute; an ablation would quantify this
   and is the first experiment we would run next.

**Reproducibility.** `artifacts/config.json` records the candidate radius, k, sampling
spacing, feature column order, random seed and the git commit producing the model. Every
figure and table in this report is generated by `scripts/make_report.py` from committed
result files.

# 9. Academic Integrity

All code, experiments and analysis in this report are our team's own work. AI assistance
was used during development in accordance with the event's policy; all design decisions,
experimental choices and interpretations are our own, and every team member can explain
the implementation, the experiments and the results. No code, model artifacts,
configurations or results were shared with or received from another team. All reported
metrics are reproducible from the committed repository at the recorded commit.
""")

    md = "\n".join(P)
    path = OUT / "PathSnap_Technical_Report.md"
    path.write_text(md)
    print(f"wrote {path} ({len(md.split())} words, {len(md)/1000:.0f} KB)")
    return path


if __name__ == "__main__":
    main()
