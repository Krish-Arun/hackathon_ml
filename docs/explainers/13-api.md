# Section 13 — The FastAPI backend (Laptop 1)

**Code:** `src/pathsnap/api/main.py`

## The job

A small web server. It receives a trip ID, runs the complete pipeline, and
sends back the predicted route plus how well it scored.

```
Laptop 2 (UI)                    Laptop 1 (this)
    |                                  |
    |---- POST /predict {trip_id:7} -->|
    |                                  |  load trip files
    |                                  |  generate candidates
    |                                  |  build features
    |                                  |  model.predict_proba
    |                                  |  Viterbi decode
    |                                  |  score vs ground truth (server-side only)
    |<--- {route:[...], metrics:{...}}-|
```

## Endpoints

- `GET /trips` — the ten evaluation trip IDs, for the UI to display
- `POST /predict` — run the pipeline on one trip, return route + metrics
- `GET /health` — confirms the model loaded

## The rule that must not be broken

**Ground truth lives only on this laptop, and only for scoring.**

It must never enter the prediction path, and must never be sent to the UI.
The pipeline computes its route knowing nothing about the answer; only
*afterwards*, in a separate scoring function, does the server compare against
the reference and report a number.

The brief is emphatic about this, and the policy lists hardcoded predictions
as a disqualifying offence. Our structural defence: the prediction function
literally does not receive the route as an argument. It can't cheat because it
can't see it.

## What the evaluator will check

They will ask to see the **API logs on this laptop** while the UI is driven
from the other one — that's how they verify the prediction really crossed the
network and wasn't faked locally. So we log each request visibly: trip ID,
timings per stage, result summary.

It must work for **all ten** trips, not one rehearsed demo.

## Viva questions this covers

- Walk us through the API implementation.
- How do you guarantee ground truth never reaches the model or the UI?
- Show us the request logs.
