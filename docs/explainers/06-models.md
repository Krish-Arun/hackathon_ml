# Section 6 — The five models

**Code:** `src/pathsnap/models.py`, `scripts/train_baselines.py`

## Why five and not just the best one

The brief requires at least five models from different families — and the
reason is sound. Different model types fail in different ways. Comparing them
is how we *justify* our final choice with evidence instead of guessing, and
how we learn what the problem actually needs.

Every model answers the same question: given the clues for one (dot, road)
pair, **what's the probability this is the right road?**

## How each one thinks

**Logistic Regression** — linear and probabilistic
Draws a single straight dividing line through the clue-space: weigh each clue,
add them up, convert to a probability. Fast, gives honest probabilities, and
you can read the weights to see what it learned. This is our floor — anything
that can't beat it is broken.
*Weakness:* only straight lines. Can't learn "close AND aligned" as a combined
condition.

**k-Nearest Neighbours** — instance-based
No training at all. To classify a new pair, find the most similar pairs it has
seen before and copy their answer. "What happened in situations like this?"
*Weakness:* slow at prediction time (searches the whole training set), and
needs features scaled or whichever feature has the biggest numbers dominates.

**Random Forest** — ensemble of trees
A decision tree is a flowchart of yes/no questions ("distance under 20 m?").
One tree overfits badly. A forest builds hundreds on random subsets and lets
them vote — individual mistakes cancel out.
*Weakness:* big, and its probabilities tend to be over-confident.

**LightGBM** — boosting
Also trees, but built *one at a time*, each new tree focusing on the examples
the previous ones got wrong. Less like a committee voting, more like
successive rounds of error correction. Usually the strongest performer on
table-shaped data like ours.
*Weakness:* more dials to tune, and will happily overfit if left unchecked.

**MLP (small neural network)** — neural
Layers of simple units that learn their own combinations of the input clues,
rather than us hand-specifying them.
*Weakness:* needs scaled inputs, is sensitive to settings, and on a dataset
this size often won't beat LightGBM — which is itself a legitimate finding
worth reporting.

## What we record for every one

Configuration, random seed, cross-validation scores per fold, student-test
scores, training time, prediction time. All of it to CSV as we go, so the
report tables are generated from real runs rather than retyped by hand.

## Viva questions this covers

- Why is this algorithm suitable for this problem?
- What are its strengths and limitations *here*?
- What hyperparameters did you use and why?
- Did you see overfitting or underfitting, and what's your evidence?
