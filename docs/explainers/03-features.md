# Section 3 — Feature engineering

**Code:** `src/pathsnap/features.py`

## The job

A model can't look at a map. It can only look at numbers. So for every
**(dot, candidate road)** pair we write down a row of clues, and the model
learns which combinations of clues mean "yes, this is the right road".

## The clues, and why each one helps

**How far is the dot from this road?**
The most obvious clue. Usually the closest road is right — but not always,
which is exactly why we need the others.

**Does the car's direction match the road's direction?**
We can work out which way the car was heading by comparing consecutive dots.
If the car is heading north at 120 km/h, it is not on the eastbound slip road
10 m away. Arcs are one-way, so this also catches wrong-side-of-the-road
matches that pure distance can't.

**How does this road compare to the other candidates for the same dot?**
This matters more than people expect. "30 m away" means nothing by itself.
"30 m away when the next best option is 400 m away" means almost certainly
correct. "30 m away when three others are also within 35 m" means genuinely
ambiguous. So we include each candidate's distance **relative to the best
candidate for that same dot**, and its rank.

> Raw distance is a weak clue. Distance *relative to the alternatives* is a
> strong one. This is the single most valuable feature family we add.

**Was the previous dot near this road too?**
Roads come in chains. If the dot before this one matched road A, and this
candidate connects to A, that's evidence. (We keep this mild — the HMM in
Part 2 is the proper tool for sequence reasoning. We don't want the classifier
leaning so hard on sequence that the HMM has nothing left to fix.)

**Road properties** — how long the segment is, and how far along it the dot
projects. A dot projecting exactly onto an endpoint behaves differently from
one in the middle.

## The rule we must not break

Every feature must be computable **without knowing the answer**. If a feature
secretly encodes the ground truth, the model scores brilliantly in testing and
collapses on evaluation day. This is called leakage and the brief is strict
about it.

Test for any feature: *could the API compute this for a trip whose route we've
never seen?* If no, it doesn't go in.

## Viva questions this covers

- What features did you extract and why those?
- What patterns did you observe during exploration?
- How did you prevent data leakage?
