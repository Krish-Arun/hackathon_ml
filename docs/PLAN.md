# PathSnap — build plan and plain-English explainers

Two parts. Part 1 gets us a trained, selected, tuned model. Part 2 turns it
into a route and ships it behind an API.

Every section below has: what we build, why it exists, and an explainer
written for someone who has never seen map-matching before.

---

## The problem in one paragraph

A car drove somewhere. A GPS logger recorded where it *thought* the car was,
once a second. GPS is wrong by a few metres, so those dots wander off the road,
cut across buildings, and sometimes jump to the street next door. We also have
a map: a big list of road segments. Our job is to take the wandering dots and
work out **which actual road segment the car was on at each moment**, then
stitch those into a sensible continuous route.

Two jobs, and we do them in that order:
1. **Guess per dot** (this is the machine learning part — Part 1)
2. **Make the guesses agree with each other** (this is the HMM part — Part 2)

---

# PART 1 — Training and model selection

## Section 1 — Data foundation ✅ DONE

**Build:** `geo.py`, `io.py`, `candidates.py`, `scripts/explore_dataset.py`

**Explainer.** Three things had to happen before anything else.

*Measuring distance properly.* The data is in latitude/longitude degrees. But a
degree isn't a fixed distance — going one degree east near the North Pole is a
much shorter walk than one degree east at the equator. If we measured in raw
degrees, everything would be squashed. So we convert each trip into plain
metres first, like putting graph paper over that part of the world.

*Knowing how far a dot is from a road.* A road segment is a straight line
between two points. We need the shortest distance from a GPS dot to that line —
but only to the part of the line that actually exists. Without that restriction,
a dot 2 km past the end of a road would look like it was sitting on it.

*Narrowing down the options.* A trip's map can have 300,000 road segments.
We can't ask a model "which of these 300,000?" — so for each dot we shortlist
the ~10 nearest segments and only ever ask "is it this one? is it this one?"
This is the single most important design decision in the project, and it also
hands Part 2 exactly what it needs.

**Found:** 90 trips, 227,577 GPS points. No missing coordinates. Nearest road
is 4.1 m away on average. 47 trips have time gaps over 5 s. Speeds from 20 to
148 km/h.

---

## Section 2 — Labelling: inventing the answer key

**Build:** `src/pathsnap/labelling.py`

**Explainer.** Here's the catch nobody warns you about: *the dataset has no
answer key in the form we need*. We're told which roads the car drove along, in
order — but not which dot belongs to which road. There are about 5 dots per
road segment, so we have to work the pairing out ourselves.

The naive way is to snap each dot to whichever road on the route is nearest.
That breaks when the route loops back near itself, or runs alongside a parallel
carriageway — a dot near the start can get matched to a road the car reaches
twenty minutes later, and the answer key ends up scrambled.

So instead we use the one fact we know for certain: **the car went forward**.
It drove road 1, then road 2, then road 3. It never teleported backwards. So we
find the pairing that keeps every dot in order *and* keeps the total distance
as small as possible. That's a classic dynamic-programming problem — the same
shape as aligning two DNA sequences or matching audio to a transcript.

**Risk note:** every number we report downstream inherits mistakes made here.
This is the section to be most careful about.

---

## Section 3 — Feature engineering

**Build:** `src/pathsnap/features.py`

**Explainer.** The model can't look at a map. It can only look at numbers. So
for every (dot, candidate-road) pair we write down a row of clues:

- How far is the dot from this road? (closest road usually wins)
- Is the car's direction of travel lined up with this road's direction?
  (if you're heading north at 120 km/h you're not on the eastbound slip road)
- How does this road compare to the *other* options for the same dot? Being
  30 m away means nothing on its own; being 30 m away when the next best is
  400 m away means everything.
- Was the previous dot matched near this road too? (roads come in chains)

That third one matters more than people expect. Raw distance is a weak clue;
*distance relative to the alternatives* is a strong one.

---

## Section 4 — Building the dataset

**Build:** `scripts/build_dataset.py` → `data/processed/features.parquet`

**Explainer.** Run sections 1–3 over all 90 trips and save one big table:
one row per (dot, candidate road), with the clues as columns and a final
column saying "yes this was the right road" or "no it wasn't".

One rule we must not break: **a trip belongs to exactly one side of the
split.** If some dots from trip 7 are used for training and others for testing,
the model has effectively seen the answers and every score we report is a lie.

---

## Section 5 — Cross-validation design

**Build:** `src/pathsnap/splits.py`

**Explainer.** To know if a model is any good we hide some data, train on the
rest, and see how it does. Repeat a few times with different slices.

The subtlety: we must hide **whole trips**, never individual dots. Dots from
the same trip are nearly copies of each other — same roads, same car, same
weather. Testing on dots from a trip you trained on is like getting the exam
paper in advance. This is called *group-aware* cross-validation, grouping by
trip, and the brief explicitly requires it.

---

## Section 6 — Baseline model zoo (5+ models)

**Build:** `src/pathsnap/models.py`, `scripts/train_baselines.py`

**Explainer.** The brief requires at least five models from different families.
Not to show off — because different model types fail in different ways, and
comparing them is how we *justify* our final pick instead of guessing.

- **Logistic Regression** — draws one straight dividing line. Simple, fast,
  gives honest probabilities. Our floor: anything that can't beat this is broken.
- **k-Nearest Neighbours** — "what happened in similar situations before?"
- **Random Forest** — hundreds of yes/no flowcharts that vote.
- **LightGBM (boosting)** — flowcharts built one at a time, each fixing the
  last one's mistakes. Usually wins on table-shaped data like ours.
- **MLP (small neural net)** — learns its own combinations of the clues.

---

## Section 7 — Metrics and the evaluation harness

**Build:** `src/pathsnap/metrics.py`

**Explainer.** "Accuracy" alone would mislead us here, because most candidate
rows are wrong by construction (9 of 10 are). A model that says "no" to
everything scores 90% and is useless. So we measure:

- **Top-1 accuracy** — for each dot, is our highest-scoring road the right one?
  This is the one that actually matters.
- **Macro F1** — scores the rare "yes" class as heavily as the common "no".
- **Log loss** — are the model's *confidence levels* honest? Critical, because
  Part 2 feeds these probabilities into the HMM.
- **Candidate miss rate** — how often was the right road never even shortlisted?
  That's our ceiling; no model can beat it.
- **Route continuity** — do consecutive answers form a connected path, or does
  the car teleport? This is what the HMM exists to fix.

---

## Section 8 — Model comparison and selection

**Build:** `scripts/compare_models.py`, `results/model_comparison.csv`, plots

**Explainer.** Put all five models side by side on the same metrics and the
same data splits, draw the comparison charts the report requires, and pick a
winner *with a reason*. The reason is allowed to be "slightly worse Top-1 but
much better-calibrated probabilities, which the HMM needs" — in fact that's a
better answer than "highest number".

---

## Section 9 — Bayesian hyperparameter optimisation

**Build:** `scripts/optimise.py` (Optuna), `results/optuna_trials.csv`

**Explainer.** Every model has dials (tree depth, learning rate, etc.). Trying
every combination is impossible. Random guessing is wasteful.

Bayesian optimisation is smarter: it keeps a running belief about which dial
settings look promising, tries the most promising one, updates its belief, and
repeats. Like finding the deepest part of a lake by sounding where the depth
has been increasing, rather than measuring on a fixed grid.

We then report tuned vs untuned on identical splits — including the case where
tuning *didn't* help, which is a legitimate and honest finding.

---

## Section 10 — Export artifacts

**Build:** `artifacts/model.joblib`, `artifacts/config.json`

**Explainer.** Freeze the chosen model and every setting needed to reproduce
its inputs (candidate radius, k, feature order, scaler). If the API builds
features even slightly differently from training, predictions silently rot.
Locking it in one config file is what prevents that.

---

# PART 2 — Route reconstruction and delivery

## Section 11 — HMM: emissions and transitions

**Build:** `src/pathsnap/hmm.py`

**Explainer.** The classifier judges each dot alone, so it can produce nonsense
like "road A, then road Z three kilometres away, then road B". Physically
impossible — the car can't jump.

A Hidden Markov Model fixes this by scoring whole *sequences*. Two ingredients:

- **Emission** — how well does this road explain this dot? (the classifier's
  probability)
- **Transition** — how believable is moving from road A to road B in the time
  available? Connected roads: cheap. Far apart or unreachable: expensive.

So a road can win even when it isn't the single best match for its own dot, if
it makes the overall journey make sense. That's the whole idea.

---

## Section 12 — Viterbi decoding and gap handling

**Build:** `src/pathsnap/viterbi.py`

**Explainer.** There are astronomically many possible road sequences — 10
options per dot across 1,300 dots. Viterbi finds the single best one without
checking them all, by sweeping forward and keeping only the best way to reach
each road at each step, then tracing back.

We also handle the awkward cases the data actually contains: the 107 dots with
no candidate at all, and trips with enormous time gaps (trip 90 has a 42-hour
gap — effectively two separate journeys in one file).

Then we compare classifier-only vs classifier+Viterbi, which is a required
report result.

---

## Section 13 — FastAPI backend

**Build:** `src/pathsnap/api/main.py`

**Explainer.** A small web server on Laptop 1. It receives a trip ID, runs the
whole pipeline, and returns the predicted route plus accuracy. Critically, the
ground-truth answers live **only** here and are used only for scoring — they
never reach the prediction code or the UI. The brief is strict about this.

---

## Section 14 — UI, Folium maps, integration

**Build:** `src/pathsnap/ui/`, `src/pathsnap/viz/folium_maps.py`

**Explainer.** A page on Laptop 2 listing the ten evaluation trips. Pick one,
it calls the API, and draws the result on a real map: raw GPS dots, the road
network, the true route, and our predicted route in different colours with a
layer toggle. This is both the demo and the required report figure.

---

## Running thread — the report

Sections produce their own evidence as they go (CSVs, plots, maps) so the
report is assembled from real outputs rather than written from memory at the end.
