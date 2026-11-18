# Section 5 — Cross-validation design

**Code:** `src/pathsnap/splits.py`

## Why we don't just train and test once

Testing on one fixed slice tells you how you did on *that* slice. Maybe it was
easy. Maybe it was unusually hard. One number, no idea how reliable.

**Cross-validation** fixes this: split the training data into (say) 5 groups.
Train on 4, test on the 1 held out. Repeat 5 times so every group gets a turn.
Now you have 5 scores — their average is a better estimate, and how much they
*vary* tells you whether the model is stable or fragile.

## The crucial detail: group by trip

Standard cross-validation shuffles individual rows into groups. **For us that
would be cheating**, for the same reason as section 4: dots from one trip are
near-duplicates of each other.

If trip 7's dots get scattered across all 5 groups, then every time we "hold
out" a group, the model has already trained on almost-identical dots from the
same trip. The score comes out far too high.

So we use **GroupKFold with trip_id as the group** — whole trips move together,
and a trip is never split across the train and test side of a fold.

```
WRONG (shuffled rows):        RIGHT (grouped by trip):
 fold 1: dots from trips 1-80  fold 1: trips 1-16
 fold 2: dots from trips 1-80  fold 2: trips 17-32
 ...every fold sees every trip ...each fold holds out whole trips
```

The brief explicitly requires group-aware cross-validation. This is why.

## What we report

For each model: the mean score across folds **and** the spread. A model that
scores 0.85 ± 0.01 is more trustworthy than one scoring 0.87 ± 0.09 — the
second one is gambling, and will likely fail on the evaluator's trips.

## Viva questions this covers

- How did you validate your models?
- Why group-aware cross-validation?
- How stable were your models across folds?
