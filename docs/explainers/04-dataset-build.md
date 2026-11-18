# Section 4 — Building the dataset

**Code:** `scripts/build_dataset.py` → `data/processed/features.parquet`

## The job

Run sections 1–3 across all 90 trips and save one big table. One row per
(dot, candidate road) pair:

| trip_id | obs_idx | arc_id | distance | bearing_diff | dist_rank | ... | **label** |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 2641 | 3.2 | 4.1 | 0 | ... | **1** |
| 1 | 0 | 9983 | 31.7 | 88.0 | 1 | ... | 0 |
| 1 | 1 | 2641 | 2.8 | 2.2 | 0 | ... | **1** |

Roughly 227,577 dots × ~10 candidates ≈ **2.2 million rows**. Parquet because
it's compressed and loads in a second; a CSV this size would be slow and huge.

## The one rule we cannot break

**A trip belongs entirely to one side of the split.** Never half its dots in
training and half in testing.

Why it matters so much: dots from the same trip are nearly copies of each
other — same roads, same car, same GPS conditions, same weather. If the model
trains on dot 400 of trip 7 and is then tested on dot 401 of trip 7, it has
effectively seen the answer. Scores look fantastic and mean nothing.

The dataset already gives us the split (80 train / 10 student test / 10
evaluator), and we respect it exactly.

## Also saved

- A per-trip summary so we can spot trips where labelling struggled
- The exact config used (radius, k, feature list) — so this is reproducible,
  which the brief requires

## Viva questions this covers

- How did you structure your training data?
- How did you prevent information leaking between train and test?
- Is your pipeline reproducible?
