# PathSnap — Distributed Map-Matching & Route Reconstruction

PES University · UE24CS352A Machine Learning

Reconstructs realistic vehicle trajectories by matching noisy GPS observations
to a road network: a classifier scores candidate road segments per observation,
and an HMM decoded with Viterbi turns those independent guesses into a
physically connected route.

## Results

| | Classifier only | **+ HMM / Viterbi** |
|---|---|---|
| Top-1 accuracy | 0.8300 | **0.9238** (+0.0937) |
| Route continuity | 0.9132 | **0.9802** (+0.0670) |

Observation-weighted over the 10 student-test trips. The decoder improved
Top-1 on 10 of 10 trips and degraded none. Full pipeline: ~1.1 s for a
2,500-observation trip.

**Model selection.** Five families were compared on identical trip-grouped
folds. LightGBM was selected on log loss (0.0964 against 0.2428 for the worst),
not Top-1 — Top-1 varies by 0.112 across folds against 0.015 across models, so
it cannot separate them. Log loss is also what the HMM consumes directly.
Bayesian optimisation over 20 Optuna trials produced no improvement beyond
noise, so the untuned configuration was kept.

## Quick start

```bash
pip install -r requirements.txt

# The dataset is not in this repo. Point at it (default: ../Student folder/...)
export PATHSNAP_DATA="/path/to/Student folder/Student folder"
```

Artifacts (`artifacts/model.joblib`, `artifacts/config.json`) are committed, so
the API runs without retraining. To rebuild everything from raw data see
*Reproducing* below.

### Run the system (two laptops)

`python -m uvicorn` rather than bare `uvicorn`: pip installs the launcher into
Python's `Scripts/` directory, which is not on PATH by default on Windows.

**Laptop 1 — model and API**
```bash
export PATHSNAP_EVAL_PARTITION=evaluator_test   # falls back to student_test
PYTHONPATH=src python -m uvicorn pathsnap.api.main:app --host 0.0.0.0 --port 8000
```

**Laptop 2 — evaluator UI**
```bash
python scripts/serve_ui.py --port 8080
```
Open `http://localhost:8080/?api=http://<laptop-1-ip>:8000`, pick a trip, and
the predicted route and metrics appear. Laptop 1's console logs every request
with per-stage timings.

### Endpoints

| Method | Path | Returns |
|---|---|---|
| GET | `/health` | model name, git commit, partition, trip count |
| GET | `/trips` | the evaluation trips offered to the UI |
| POST | `/predict` | `{trip_id}` → route arc ids, geometry, metrics, timings |

## Reproducing from raw data

```bash
python scripts/explore_dataset.py                                  # EDA, ~2 min
python scripts/build_dataset.py --partition train --start 0  --end 40
python scripts/build_dataset.py --partition train --start 40 --end 80
python scripts/build_dataset.py --partition student_test --start 0
python scripts/check_labels.py                                     # label validation

for m in logistic_regression knn random_forest lightgbm mlp; do
  for f in 0 1 2 3 4; do python scripts/train_baselines.py --model $m --fold $f; done
  python scripts/train_baselines.py --model $m --final
done

python scripts/optimise.py --trials 20      # Bayesian search
python scripts/export_model.py              # freeze the chosen model
python scripts/compare_models.py            # comparison table + figures
python scripts/evaluate_hmm.py              # classifier vs HMM
python scripts/make_maps.py                 # folium maps
```

The build is chunked because this hardware has 2 cores and 3 GB of RAM; each
step stays inside a few minutes and is resumable.

## Layout

```
src/pathsnap/
  geo.py          projection to metres, point-to-segment distance, bearings
  io.py           raw file loading into a Trip
  candidates.py   candidate shortlisting (densified KD-tree)
  labelling.py    per-observation labels by monotonic DP alignment
  features.py     17 features per (observation, candidate) pair
  splits.py       trip-grouped cross-validation
  sampling.py     observation-group-preserving subsampling
  models.py       the five model families + the tuned variant
  metrics.py      Top-1, Macro F1, log loss, candidate recall, continuity
  hmm.py          emission/transition model, bounded Dijkstra
  viterbi.py      decoder, gap and missing-candidate handling
  api/main.py     FastAPI backend
  ui/index.html   evaluator interface
  viz/            chart styling and Folium maps
scripts/          one runnable step each (see Reproducing)
results/          every number in the report: CSVs, figures, maps
artifacts/        model.joblib, config.json, best_params.json
docs/             explainers, build plan, viva recital, walkthrough
```

### Dataset layout

`list_trips` tolerates the redundant nesting the data ships with — both
`Student folder/Student folder/...` and `evaluator_test/evaluator_test/...` are
handled, so no files need moving before the demo.

## Notes on the data

Two findings worth knowing before reading the numbers.

**Labels are derived, not given.** `.route` lists arcs in travel order but not
which observation belongs to which arc (~5.25 observations per arc). We derive
the pairing with a monotonic DP alignment, validated by residuals of 4.0–4.65 m
median against an independently measured 4.1 m GPS error.

**Some reference routes are incomplete.** Trip 33's GPS covers 47.7 km against
a 6.4 km route; trip 54 is 168.8 km against 31.3 km. Affected observations land
120 m+ from their assigned arc, so they are flagged (`label_reliable`,
50 m threshold) and excluded from training while being retained in evaluation.
89.8% of observations are reliably labelled; 78 of 90 trips are ≥99% clean.

## Documentation

- `docs/explainers/` — one file per section, written for a newcomer
- `docs/RECITAL-sections-1-to-5.md` — viva answers with traceable numbers
- `docs/WALKTHROUGH-presentation.md` — 4-minute code walkthrough
- `docs/PLAN.md` — the build plan

## Report

`report/PathSnap_Technical_Report.pdf` (16 pages). Regenerate with:

```bash
python scripts/make_report.py     # pulls every number from results/
cd report && pandoc PathSnap_Technical_Report.md \
  -o PathSnap_Technical_Report.pdf --pdf-engine=xelatex \
  --resource-path=.:..:../results/figures \
  -V geometry:margin=2.2cm -V fontsize=10pt -V mainfont="DejaVu Serif" \
  -V monofont="DejaVu Sans Mono" --toc --toc-depth=2
```
