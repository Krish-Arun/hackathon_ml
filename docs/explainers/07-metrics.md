# Section 7 — Metrics: what we measure and why

**Code:** `src/pathsnap/metrics.py`

## Why plain accuracy would fool us

Our table has about 10 candidate rows per dot, and only one is correct. So
roughly **90% of all rows are "no"**.

A model that says "no" to literally everything scores **90% accuracy** and is
completely useless — it never identifies a single road.

So we need metrics that can't be gamed that way.

## What we actually measure

**Top-1 accuracy** — *the one that matters*
For each dot, take the candidate the model scored highest. Is it the right
road? This is the real question: how often do we get the road right?

**Macro F1**
Treats the rare "yes" class as equally important as the common "no" class.
Combines precision (when we say yes, how often are we right?) with recall
(of all the true matches, how many did we find?). The "macro" part is what
stops the majority class from drowning out the minority.

**Log loss** — *is the model's confidence honest?*
Doesn't just ask "right or wrong" but "how confident were you, and were you
entitled to be?" Being 99% sure and wrong is punished far more than being 55%
sure and wrong.

This matters enormously for us: **Part 2 feeds these probabilities into the
HMM.** A model that's accurate but over-confident will bully the HMM into
accepting bad roads. We may well pick a model with slightly worse Top-1 but
better-calibrated probabilities — and that would be a *better* decision, not a
compromise.

**Candidate miss rate** — *our ceiling*
How often was the correct road never even shortlisted? We measured this at
0.047% (107 dots). No model can ever beat that ceiling, so it belongs in the
report as context for every other score.

**Route continuity** — *does the answer make physical sense?*
Take consecutive predictions and ask: is road B actually reachable from road A?
A classifier working dot-by-dot has no idea, so it produces teleports. This
metric quantifies the problem the HMM exists to solve — and measuring it
*before* and *after* is how we prove the HMM earned its place.

**Training time and prediction time**
Required by the brief, and genuinely relevant: the API has to answer while an
evaluator waits.

## The headline

We select our final model primarily on **Top-1 accuracy**, with **log loss**
as the tie-breaker because of the HMM dependency. We'll state that explicitly
in the report, as the brief requires.

## Viva questions this covers

- What metrics did you use and why?
- Which was your primary selection metric, and why that one?
- Why isn't accuracy enough?
- What's your candidate miss rate?
