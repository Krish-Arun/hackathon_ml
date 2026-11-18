# Section 8 — Comparing models and choosing one

**Code:** `scripts/compare_models.py` → `results/model_comparison.csv` + plots

## The job

Put all five models side by side, on **identical data splits and identical
metrics**, and pick a winner for reasons we can defend.

Identical splits matters: if model A was tested on different trips than model
B, the comparison is meaningless. Same folds, same test trips, every time.

## What gets produced

The comparison table the brief specifies — one row per model, with algorithm
family, key settings, cross-validation Macro F1, Top-1 accuracy, route
continuity, student-test Macro F1, log loss, and inference time.

Plus the required charts:
- Cross-validation Macro F1 across all five models (with error bars showing
  fold-to-fold spread — a model with huge spread is fragile)
- Student-test Top-1 and Macro F1 side by side
- Log loss comparison
- Confusion matrices as diagnostics

**One rule: every chart comes from a real run.** The brief explicitly forbids
inventing training curves for algorithms that don't produce them. Random
Forest has no epochs, so it gets no learning curve. Only the MLP does.

## How to choose well

The highest number is not automatically the right answer. Things that
legitimately matter more:

- **Stability across folds.** 0.85 ± 0.01 beats 0.87 ± 0.09. The second model
  is gambling, and evaluator trips are a fresh roll of the dice.
- **Probability quality (log loss).** Part 2 consumes these probabilities. An
  over-confident model actively damages the HMM.
- **Prediction speed.** The API answers while an evaluator watches.
- **Train/test gap.** Great on training, mediocre on test = overfitting, and
  we should say so plainly rather than hide it.

> A defensible selection sentence sounds like: *"LightGBM had marginally lower
> Top-1 than Random Forest (0.91 vs 0.92) but substantially better log loss
> (0.21 vs 0.38), and since the HMM consumes these probabilities directly we
> chose LightGBM."* That is a **stronger** answer than "it had the best number".

## Viva questions this covers

- How did you select your final model?
- Why this model over the others?
- What evidence supports that choice?
