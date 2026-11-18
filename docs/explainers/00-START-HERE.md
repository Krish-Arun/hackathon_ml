# PathSnap explainers — start here

One file per section. Written for someone who has never seen map-matching
before. If you're presenting or being vivaed on a section, read that file.

## The problem in one paragraph

A car drove somewhere. A GPS logger recorded where it *thought* the car was,
once a second. GPS is wrong by a few metres, so those dots wander off the road,
cut across buildings, and sometimes jump to the street next door. We also have
a map: a big list of road segments. Our job is to take the wandering dots and
work out **which actual road segment the car was on at each moment**, then
stitch those into a sensible continuous route.

Two jobs, done in that order:

1. **Guess per dot** — the machine learning part (Part 1, sections 1–10)
2. **Make the guesses agree with each other** — the HMM part (Part 2, 11–14)

## Vocabulary you'll need

| Word | What it means here |
|---|---|
| **observation** / dot | one GPS reading: longitude, latitude, timestamp |
| **node** | a point on the road map |
| **arc** / segment | a one-way straight road piece joining two nodes. Has an ID. |
| **route** | the ground truth: the list of arcs the car actually drove, in order |
| **candidate** | an arc we think a given dot *might* be on |
| **trip** | one journey: its dots, its own map, and its route |

## Index

| File | Section |
|---|---|
| `01-data-foundation.md` | Loading data, measuring distance, shortlisting candidates |
| `02-labelling.md` | Building the answer key |
| `03-features.md` | Turning pairs into numbers |
| `04-dataset-build.md` | Running it over all 90 trips |
| `05-cross-validation.md` | How we test honestly |
| `06-models.md` | The five models and how each thinks |
| `07-metrics.md` | What we measure and why accuracy alone lies |
| `08-comparison.md` | Choosing a winner with evidence |
| `09-bayesian-optimisation.md` | Tuning the dials intelligently |
| `10-artifacts.md` | Freezing the model for the API |
| `11-hmm.md` | Scoring whole journeys |
| `12-viterbi.md` | Finding the best journey efficiently |
| `13-api.md` | The backend on Laptop 1 |
| `14-ui-and-maps.md` | The interface on Laptop 2 |
