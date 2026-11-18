# Section 10 — Exporting the artifacts

**Code:** `scripts/export_model.py` → `artifacts/model.joblib`, `artifacts/config.json`

## The job

Freeze the trained model, plus **every setting needed to rebuild its inputs
exactly**, so the API produces identical predictions to training.

## The failure this prevents

This is the classic deployment bug. Training used candidate radius 200 m and
feature order `[distance, bearing_diff, rank, ...]`. The API gets written a
week later using radius 150 m and a slightly different column order.

Nothing crashes. No error appears. The model just quietly returns worse
answers, and you lose an hour on evaluation day wondering why your 91% became
60%.

**Fix:** one config file, written by the training run, read by the API. The
API never hardcodes a number that training also knows.

## What goes in `config.json`

- Candidate settings: radius, k, arc sampling spacing
- Exact feature column names, **in order**
- Any scaler fitted during training (fitted on training data *only*)
- HMM parameters once Part 2 tunes them
- Model type, chosen hyperparameters, random seed
- Training timestamp and the git commit it came from

That last one means any result in the report can be traced back to the exact
code that produced it — which is what "reproducible" means in the marking
criteria.

## Size note

The brief says include model weights *subject to repository size constraints*.
A LightGBM model is a few MB, which is fine. If a model turns out huge, we
commit the training script and say how to regenerate it.

## Viva questions this covers

- How do you guarantee the API behaves the same as training?
- Can you reproduce your reported results?
