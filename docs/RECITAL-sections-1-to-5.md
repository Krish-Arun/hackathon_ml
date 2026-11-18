# Viva recital — sections 1 to 5 (data pipeline)

Speak from this. Every number here is from our own runs and is reproducible
from the repo. Files: `results/dataset_inventory.csv`, `results/build_summary.csv`,
`results/cv_folds.csv`.

---

## The 60-second version

> "The task is map-matching: noisy GPS points have to be assigned to the road
> segments the vehicle was actually on. We do it in two stages — a classifier
> that scores candidate roads per observation, then an HMM that forces those
> guesses into a physically connected route.
>
> The dataset is 90 trips and 227,577 GPS observations. Each trip has its own
> road network — up to 307,000 segments — so we never treat it as one global
> map. Our first decision was to reframe the problem: instead of a
> 300,000-class choice, we shortlist about ten nearby candidate roads per
> observation and ask a binary question of each pair. That shortlist then
> doubles as the HMM's state space.
>
> The dataset doesn't ship per-observation labels — the route file lists arcs
> in order but not which point belongs to which arc — so we derive them with
> a monotonic dynamic-programming alignment. Validating those labels is what
> surfaced our main data-quality finding: two trips have reference routes
> covering only a fraction of the journey the GPS recorded.
>
> We then build 17 features per pair, dominated by how each candidate compares
> to the alternatives for the same observation, and validate with trip-grouped
> cross-validation."

---

## Section 1 — Data foundation

**If asked "how is the dataset organised?"**

> "Four files per trip. `.nodes` is longitude-latitude per map point, and the
> row index is the node ID. `.arcs` is from-node, to-node pairs, and the row
> index is the arc ID — arcs are directed, and only about 55% have a reverse
> twin. `.track` is the GPS: longitude, latitude, timestamp, nominally 1 Hz.
> `.route` is the ground truth, a list of arc IDs in travel order.
>
> We verified rather than assumed: consecutive route arcs join head-to-tail —
> the end node of one is the start node of the next — for 100% of steps. That
> is what lets us treat the route as a walk through a directed graph later."

**If asked "how did you preprocess?"**

> "Three things. First, we project every trip into local metres. The data is
> in degrees, but a degree of longitude is about 71 km at this latitude
> against 111 km at the equator, so Euclidean distance in degrees is
> anisotropic and wrong. We use a local equirectangular projection centred on
> each trip's own network — sub-metre error at this scale, far below GPS
> noise, and around a hundred times cheaper than a full reprojection.
>
> Second, we compute true point-to-segment distances, clamped to the segment.
> Without clamping, a point two kilometres past the end of a road would
> measure as if it were on it.
>
> Third, candidate generation. We sample points every 25 metres along every
> arc and index them in a KD-tree. We tried to avoid nearest-node lookup
> because arc lengths run from 6.7 metres to 3.6 kilometres — a nearest-node
> search would miss a long motorway arc whose endpoints are kilometres from a
> point sitting in its middle. Exact distances are recomputed for the
> shortlist, so the sampling only affects which arcs we consider, never any
> number we report."

**If asked "what are the data-quality concerns?"**

> "The data is cleaner than we expected on the basics: zero missing
> coordinates, zero duplicate timestamps, zero isolated nodes across all 90
> trips. The problems are elsewhere.
>
> Sampling is irregular — 47 of 90 trips have gaps over five seconds. Trip 90
> is the extreme: 536 observations spread over 42 hours with a single
> 153,064-second gap, so it is really two journeys in one file.
>
> The trips are also wildly heterogeneous. Networks range from 649 to 307,392
> arcs, and median speeds from 20 to 148 km/h — city crawl to Autobahn. Any
> approach tuned to one regime will fail on the other."

**If asked "how did you choose the candidate radius?"**

> "Empirically, not by guess. We measured the distance from each observation
> to its nearest arc across the whole dataset: 4.1 metres at the median, 13
> metres at p95. A 200-metre radius is generous against that, and it excludes
> only 107 observations out of 227,577 — 0.047%, all in trip 2. We keep ten
> candidates, which caps the HMM state space and the table size."

---

## Section 2 — Labelling

**If asked "where did your training labels come from?" — the key question**

> "We derived them, because the dataset doesn't provide them in usable form.
> `.route` tells us which arcs were driven in order, but not which observation
> belongs to which arc. There are about 5.25 observations per route arc,
> ranging from 1.3 to 57, so the pairing is missing.
>
> The naive approach is to snap each observation to the nearest arc on the
> route. We rejected that because it ignores time. Where a route loops back
> near itself, or runs beside a parallel carriageway about 15 metres away —
> well inside GPS error — an early observation gets matched to an arc the
> vehicle only reaches much later, and the label sequence scrambles.
>
> Instead we use the one thing we know for certain: the vehicle moved
> forward. If observation i is at route position j, observation i+1 is at j or
> later, never earlier. So we solve for the assignment that minimises total
> observation-to-arc distance subject to route position never decreasing.
> That's exact dynamic programming, same family as sequence alignment."

**If asked to explain the recurrence**

> "Build a grid: observations down, route positions across, each cell holding
> the distance between them. Then dp[i][j] equals the cost of cell i,j plus
> the minimum of dp[i-1] over everything from 0 to j — the best way to have
> got the previous observation to position j or earlier. That inner minimum is
> a prefix minimum, so a whole row is one linear scan rather than a nested
> loop, which keeps it O(observations × route length). We store which index
> achieved each prefix minimum and trace back at the end.
>
> The biggest grid in the dataset is trip 92 at 17.9 million cells. We keep a
> rolling previous row rather than the full table and use float32, which puts
> peak memory around 140 MB instead of about 500."

**If asked "how do you know the labels are right?"**

> "We can't check against a ground truth — deriving them is the task — so we
> validate three ways. Monotonicity holds by construction. Route coverage
> tells us what fraction of the route has observations near it. And the
> residual — how far each observation sits from the arc we assigned it — is
> the real test.
>
> On well-behaved trips the residual is 4.0 to 4.65 metres at the median,
> which matches the independently measured 4.1-metre GPS error almost exactly.
> If our alignment were scrambled those residuals would be hundreds of metres.
> That agreement is our evidence."

**If asked about the bad trips — volunteer this, it's a strength**

> "Validating the labels surfaced a genuine data-quality problem. Two trips
> have reference routes that describe only a fraction of the journey the GPS
> recorded. Trip 33's GPS covers 47.7 km while its route spans 6.4 km — a
> ratio of 7.5. Trip 54 is 168.8 km of GPS against a 31.3 km route.
>
> The alignment still has to place every observation somewhere, so
> observations from the uncovered stretch land on arcs 120 metres or more
> away. Those labels aren't ground truth. We confirmed it isn't a coordinate
> offset — the route and track share the same bounding box, and the error
> varies along the trip rather than being a constant shift.
>
> So we added a reliability flag: any observation more than 50 metres from its
> assigned arc is marked unreliable, justified by the 13-metre p95 nearest-arc
> distance. 89.8% of observations are reliably labelled and 78 of 90 trips are
> at least 99% clean. We train only on reliable labels but we keep the
> unreliable ones in evaluation, so our reported scores aren't flattered by
> quietly dropping the hard cases.
>
> The brief warned not to treat the reference routes as error-free ground
> truth. This is us taking that seriously."

---

## Section 3 — Feature engineering

**If asked "what features and why?"**

> "Seventeen, in three groups.
>
> Raw geometry: distance to the arc, how far along the arc the point projects,
> and arc length.
>
> Direction agreement: we derive the vehicle's heading from consecutive fixes
> and compare it to the arc's bearing. We keep both the forward difference and
> the difference against the reversed arc, because arcs are one-way — a
> reversed match usually means the opposite carriageway of a dual
> carriageway, which pure distance cannot distinguish.
>
> And the group that matters most: how each candidate compares to the other
> candidates for the same observation. Rank by distance, gap to the best,
> log-ratio to the best, and a softmax share across the group. The reasoning
> is that thirty metres away means nothing on its own, but thirty metres when
> the next best is four hundred means almost certainly correct, and thirty
> metres when three others are within thirty-five means genuinely ambiguous."

**If asked whether that actually worked — we measured it**

> "We measured each feature's class separation in standard deviations between
> true and false matches. The relative features dominate, as we predicted:
> the softmax share separates at 1.96 standard deviations and distance rank at
> 1.49, while raw distance manages only 1.00. So the comparison features are
> worth roughly twice what raw distance is.
>
> The per-observation context features — speed, time deltas, projection
> position — separate at essentially zero. That's expected, not a bug: they
> are identical across all ten candidates of an observation, so they cannot
> discriminate within a group. We kept them only because tree models can use
> them for conditioning."

**If asked about leakage**

> "Every feature is computable from the GPS track and road network alone.
> Nothing reads `.route`. The test we applied to each one: could the API
> compute this for a trip whose reference route we have never seen?
>
> The label-quality columns we derive — residual distance and the reliability
> flag — *are* route-derived, so they are deliberately excluded from the
> feature list and used only as metadata filters. We assert that in code
> rather than relying on discipline."

**One honest caveat worth raising yourself**

> "One feature, whether a candidate connects to the previous observation's
> nearest candidate, separates at 1.66 standard deviations. That's a sequence
> feature, and sequence reasoning is the HMM's job. We kept it deliberately
> weak — nearest candidate only, no decoded path — but if the HMM's
> improvement looks thin, this feature is the first thing we'd ablate."

---

## Section 4 — Dataset build

**If asked how the training data is structured**

> "One row per observation-candidate pair: 2,090,297 rows across 90 trips,
> 10.1% positive. We keep trip ID and observation index on every row so
> candidate groups stay intact — Top-1 accuracy only means anything if an
> observation's whole candidate group is present.
>
> We write each trip straight to its own parquet part rather than
> concatenating ninety frames at the end. That was a necessity, not a
> preference: the concatenating version was OOM-killed on this machine."

**If asked about train/test separation**

> "A trip belongs entirely to one partition, never split. Observations from
> one trip are near-duplicates — consecutive seconds, same roads, same
> vehicle, same conditions — so splitting one across train and test would let
> the model see the answers. We keep the partitions the dataset defines, carry
> the partition label on every row, and assert no trip crosses a fold."

**If asked about reproducibility**

> "`artifacts/dataset_config.json` records the candidate radius, k, sampling
> spacing, the exact feature column order, the random seed, the row count, and
> the git commit that produced the table. Re-running the build script at that
> commit regenerates the identical dataset."

---

## Section 5 — Cross-validation

**If asked how you validated**

> "Five-fold GroupKFold over the 80 training trips, grouped by trip ID, with
> the ten student-test trips held back entirely for development-stage
> comparison. Training rows are capped at 200,000 per fold and validation at
> 100,000, sampling whole candidate groups — applied identically to every
> model so the comparison stays fair. That cap is a compute budget: this
> machine has two cores."

**If asked why group-aware**

> "Because ordinary KFold shuffles individual rows, which would scatter one
> trip's observations across every fold. Each held-out fold would then contain
> rows nearly identical to ones the model trained on, and the score would
> measure memorisation rather than generalisation. Grouping by trip means a
> trip never appears on both sides of a fold. We assert that rather than
> assume it."

**If asked how stable the models were — the important one**

> "Not very, and that turned out to be our most useful finding. Every model
> scores around 0.75 Top-1 on fold 1 and around 0.87 on fold 2 — a standard
> deviation of about 0.045.
>
> The critical observation is that all four models move together. On the hard
> fold they all do badly; on the easy fold they all do well. So the variation
> is coming from how difficult that batch of trips is, not from the models.
>
> The spread across folds is 0.12. The spread across models is 0.013 — about
> nine times smaller. Which means Top-1 differences between these models are
> inside the noise, and selecting on Top-1 alone would be unjustifiable. That
> is why log loss becomes our discriminator: there the gap is a factor of
> two and a half and it's consistent across every fold."
