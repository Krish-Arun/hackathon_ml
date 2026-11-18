# Section 12 — Viterbi decoding and gap handling

**Code:** `src/pathsnap/viterbi.py`

## The problem

Section 11 says "score every possible journey and pick the best". With 10
candidates per dot and 1,300 dots, that's 10^1300 journeys. There are about
10^80 atoms in the observable universe. We are not enumerating them.

## The trick

Viterbi finds the exact best journey **without looking at all of them**,
using one observation:

> To know the best journey that ends on road B at dot 50, you only need the
> best way to reach each candidate at dot 49. You never need to know *how*
> you got to dot 49 — only the cost.

So we sweep forward one dot at a time. At each dot, for each candidate road,
we compute: best score of any way to arrive here = (best previous score +
transition cost) + emission cost. Keep that number, and a pointer back to
which previous road gave it.

At the end, take the best final score and **follow the pointers backwards** to
read off the whole journey.

Cost drops from 10^1300 to about (dots × candidates × candidates) — roughly
130,000 operations. Milliseconds.

> Same family of idea as the labelling DP in section 2: build the answer
> incrementally, keep only what you need, trace back at the end. If you
> understand one, you understand the other.

## The awkward real cases we must handle

Our data exploration found these, so they're not hypothetical:

**Dots with no candidates at all** (107 of them, trip 2). No road within 200 m.
We can't assign a road, so the decoder has to bridge across the gap, connecting
the roads on either side rather than crashing or guessing.

**Enormous time gaps.** Trip 90 has a 153,064-second gap — 42 hours. Trip 78
has 52,810 s. Pretending the car drove continuously would make the decoder
attempt an impossible connection. Past a threshold we **break the sequence**
and decode the two halves independently. Forcing continuity across a 42-hour
gap is physically meaningless.

**Genuinely disconnected roads.** Sometimes no legal path exists between
consecutive best candidates, because the map is incomplete. We allow a
penalised "jump" rather than returning nothing — a slightly broken route is
more useful than no route.

## What we report

Classifier-only vs classifier+Viterbi, on identical trips:
- Top-1 accuracy before and after
- **Route continuity before and after** — this is where the gain should be
  dramatic and visible
- Cases where the HMM made things *worse*, which the brief explicitly asks for

If the HMM improves continuity from (say) 60% to 98% while leaving accuracy
roughly flat, that's a complete and honest answer: it did exactly the job it
was there to do.

## Viva questions this covers

- Explain the Viterbi recurrence.
- How do you handle missing candidates and large GPS gaps?
- Did HMM decoding improve the route? Where did it not help?
