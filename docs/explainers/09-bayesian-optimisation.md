# Section 9 — Bayesian hyperparameter optimisation

**Code:** `scripts/optimise.py` (Optuna) → `results/optuna_trials.csv`

## What a hyperparameter is

Settings you choose *before* training, which the model can't learn by itself:
how deep each tree can grow, how many trees, how fast it learns. Get them
wrong and a good algorithm performs badly.

## Why not just try everything

A grid search over 5 settings with 10 values each is 100,000 combinations, and
each one needs a full cross-validation run. Completely impossible in a
hackathon.

Random search is better than it sounds, but it's memoryless — it learns nothing
from the trials it has already run.

## What Bayesian optimisation does differently

It **remembers**. After each trial it updates a belief about which regions of
the setting-space look promising, then deliberately picks the next trial to be
informative — either somewhere it expects good results, or somewhere it's
uncertain and might be surprised.

> Like finding the deepest point of a lake. Grid search measures on a fixed
> grid regardless of findings. Random search drops the weight anywhere.
> Bayesian optimisation notices the depth increasing toward the north end and
> concentrates its soundings there — while still occasionally checking
> elsewhere in case there's a second trench.

That trade-off has a name: **exploration vs exploitation**. Optuna handles it.

## How we run it

- Search space justified per parameter, not copy-pasted defaults
- Objective: the same metric we selected on, measured the same way
- **Group-aware cross-validation on the 80 training trips only** — the brief
  is explicit that evaluator data must never influence tuning
- Every trial logged to CSV (the brief requires the search record retained)
- Refit the winner, evaluate on student-test, compare against untuned

## The honest outcome

**Tuning might not help.** If the untuned model was already near the ceiling,
or the search space was poorly chosen, we may gain almost nothing. That's a
real result and we report it with the numbers. Pretending otherwise is exactly
the fabrication the policy warns about — and "tuning gained 0.3%, which is
within fold-to-fold noise" is a sophisticated answer, not a failure.

## Viva questions this covers

- Which hyperparameters did you search, over what ranges, and why those?
- What objective did you optimise, and how many trials?
- Did tuning actually help? What's your evidence?
