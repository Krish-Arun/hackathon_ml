# Code walkthrough — 4 minutes, 3 speakers

A presentation, not a code review. Open each file, point at the named lines,
say the lines below. Timings are generous; rehearse once and it lands at ~4 min.

**Split:** Krish does the setup + §1-2, Niharika §3, Mrudul §4-5 + close.
Whoever speaks drives the screen.

Have these open in tabs beforehand:
`src/pathsnap/candidates.py`, `labelling.py`, `features.py`, `splits.py`,
and `results/` in a file browser.

---

## [0:00–0:30] The problem and the one decision — KRISH

*Don't open anything yet. Say this to the room.*

> "Noisy GPS points, a road map, and we need to know which road segment the
> car was actually on at each moment.
>
> The first thing we had to decide was how to even pose it as a learning
> problem. A single trip's map has up to three hundred thousand road
> segments, and the set is different for every trip — so 'which of these
> three hundred thousand?' isn't a question a classifier can answer.
>
> So we reframed it."

---

## [0:30–1:15] The reframe — KRISH
**Open `src/pathsnap/candidates.py`, scroll to the module docstring (lines 1–28), then `generate_candidates` (line 62).**

> "For each GPS point we shortlist about ten nearby candidate roads, and ask a
> binary question of each pair: is this the one? An impossible 300,000-way
> choice becomes ordinary binary ranking.
>
> And it pays twice — those same candidates become the hidden states for our
> HMM in part two. One decision, two uses."

**Point at `densify_arcs`, line 30.**

> "One subtlety. We don't find candidates by nearest map-point, because arc
> lengths in this data run from under seven metres to three and a half
> kilometres. A nearest-node search would completely miss a long motorway arc
> whose endpoints are kilometres from a point sitting in its middle.
>
> So we sample every arc at twenty-five-metre intervals and index those. Then
> — line 95 — we recompute exact distances for the shortlist, so the sampling
> only ever affects which arcs we consider, never any number we report."

**Optional, if anyone looks sceptical — `geo.py` line 79, `t = np.clip(t, 0, 1)`.**

> "One line worth seeing. That clamp is what makes this a segment distance
> rather than distance to an infinite line. Without it a point two kilometres
> past the end of a road measures as if it were sitting on it."

---

## [1:15–2:15] The labels don't exist — KRISH
**Open `src/pathsnap/labelling.py`, docstring lines 1–25.**

> "Here's the part that surprised us. The dataset doesn't ship labels in a
> usable form. The route file lists which arcs were driven, in order — but
> not which GPS point belongs to which arc. There are about five points per
> arc. So the pairing is missing and we had to derive it."

> "The obvious approach is to snap each point to the nearest arc on the route.
> We rejected that, because it ignores time. Where a route loops back near
> itself, or runs beside a parallel carriageway fifteen metres away — well
> inside GPS error — a point from minute two gets matched to a road the car
> reaches at minute twenty-two."

**Scroll to `monotonic_alignment`, line 71. Point at the recurrence comment, then line 109.**

> "So we use the one thing we know for certain: the car moved forward. If this
> point is at route position j, the next one is at j or later, never earlier.
>
> We find the assignment that minimises total distance subject to that
> constraint. Dynamic programming — same family as sequence alignment.
> dp[i][j] is this cell's cost plus the best way of getting the previous point
> to position j or earlier. That inner minimum is a prefix minimum, line 109,
> so a whole row is one linear scan instead of a nested loop."

**Then line 103 — `prev = cost[0]`.**

> "And a practical note: we keep one rolling row instead of the full table.
> The biggest grid here is about eighteen million cells. Full float64 tables
> got us OOM-killed; this runs in roughly a hundred and forty megabytes."

**Now the payoff. Open `results/build_summary.csv` or just say it.**

> "Then the obvious question — how do we know the labels are right? We can't
> check against ground truth, deriving them *is* the task. So we check the
> residual: how far each point sits from the arc we gave it.
>
> On good trips that's four to four-point-six metres median. We separately
> measured GPS error at four-point-one metres. They match. If our alignment
> were scrambled, those residuals would be hundreds of metres.
>
> And that check found something. Two trips came back at a hundred and twenty
> metres. Trip thirty-three's GPS covers forty-seven kilometres while its
> reference route spans six — the route only describes part of the journey.
> We confirmed it's not a coordinate offset: same bounding box, and the error
> varies along the trip instead of being a constant shift.
>
> So we flag any point more than fifty metres from its arc as unreliable.
> Ninety percent of the dataset is clean. We train only on the reliable ones
> but keep the rest in evaluation, so our scores aren't flattered by dropping
> the hard cases."

---

## [2:15–3:00] Features — a hypothesis we tested — NIHARIKA
**Open `src/pathsnap/features.py`, `FEATURE_COLUMNS` at line 34.**

> "Seventeen features in three groups. Raw geometry. Direction agreement —
> and we keep both forward and reversed bearing difference, because arcs are
> one-way and a reversed match usually means the opposite carriageway.
>
> And the third group, which is where our actual bet was."

**Point at the comparison block, lines 139–162.**

> "How each candidate compares to the *other* candidates for the same point.
> Rank, gap to the best, log-ratio, and a softmax share across the group.
>
> The reasoning: thirty metres away means nothing on its own. Thirty metres
> when the next best is four hundred means almost certainly right. Thirty
> metres when three others are within thirty-five means genuinely ambiguous.
> Same raw number, three different situations — and only the relative
> features can tell them apart."

> "We tested that rather than assuming it. Measuring class separation in
> standard deviations: the softmax share separates at one-point-nine-six,
> distance rank at one-point-four-nine, and raw distance at one-point-zero.
> So the comparison features are worth about twice what raw distance is. The
> hypothesis held."

**Then line 217, `LABEL_RELIABLE_MAX_RESIDUAL_M`, and the docstring below it.**

> "One thing we were careful about. The label-quality columns are derived from
> the route file, so using them as features would be leakage. They're
> deliberately outside `FEATURE_COLUMNS` and used only as filters — and
> there's a check that asserts it rather than us just being careful."

---

## [3:00–3:45] Validation, and the result that changed our thinking — MRUDUL
**Open `src/pathsnap/splits.py` — it's 55 lines, show the whole thing.**

> "Five-fold cross-validation, grouped by trip. The grouping is the point.
>
> Ordinary KFold shuffles individual rows, which would scatter one trip's
> points across every fold. Points from the same trip are consecutive seconds
> on the same roads in the same car — near-duplicates. So every held-out fold
> would contain rows almost identical to ones the model trained on, and we'd
> be measuring memorisation."

**Point at `describe_folds`, line 52, `overlapping_trips`.**

> "We assert no trip appears on both sides of a fold rather than trusting it."

**Now the finding. Open `results/cv_folds.csv`.**

> "And this is the result that changed how we're selecting our model.
>
> Every model scores about point-seven-five on fold one and point-eight-seven
> on fold two. Look across the columns — they all move together. On the hard
> fold they all struggle, on the easy fold they all do well.
>
> So the thing varying isn't the model, it's how hard that batch of trips is.
> The spread across folds is twelve points. The spread across models is one
> point three. Trip difficulty moves the score nine times more than model
> choice does.
>
> Which means ranking these models on Top-1 accuracy would be measuring
> noise. That's why we're selecting on log loss instead — and log loss also
> happens to be what part two actually depends on, because the HMM consumes
> these probabilities directly. A confidently wrong model drags the decoder
> toward bad roads."

---

## [3:45–4:00] Close — MRUDUL
**Open the `results/` folder.**

> "Everything we've said is in here. `dataset_inventory.csv` is the
> exploration, `build_summary.csv` the per-trip label quality,
> `cv_folds.csv` the fold-by-fold scores. Every number in our report comes
> out of these files, and `artifacts/dataset_config.json` records the git
> commit that produced them.
>
> Next is the HMM and Viterbi decoder, which turns these per-point guesses
> into a connected route — our classifier-only continuity is around ninety
> percent, so roughly one in ten consecutive predictions is currently a
> physical impossibility. That's the gap part two closes."

---

## Questions you should expect, and the one-line answer

| Question | Answer |
|---|---|
| Why not just nearest road? | That's our baseline intuition, but parallel carriageways sit ~15m apart, inside GPS error. Raw distance only separates at 1.0 SD; relative features reach 1.96. |
| Where did labels come from? | We derived them — the dataset has no per-point mapping. Validated by residuals matching independently measured GPS error. |
| Why 200m radius? | Measured: nearest arc is 4.1m median, 13m p95. 200m excludes only 0.047% of points. |
| Why exclude those two trips? | We don't exclude trips, we exclude unreliable *points* — their reference routes only cover part of the journey, 7.5x and 5.4x ratios. |
| Is your best model significantly better? | On Top-1, no — that difference is inside fold noise. On log loss, yes, 2.5x and consistent across folds. |
| Any leakage? | Every feature is computable without the route file. Route-derived columns are excluded from FEATURE_COLUMNS and asserted in code. |
