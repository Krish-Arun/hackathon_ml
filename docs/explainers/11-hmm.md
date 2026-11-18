# Section 11 — The HMM: scoring whole journeys

**Code:** `src/pathsnap/hmm.py`

## The problem with judging each dot alone

Our classifier looks at one dot at a time. It has no idea what came before or
after. So it can confidently output:

```
dot 10 -> road A       (High Street)
dot 11 -> road Z       (a lane 3 km away)
dot 12 -> road B       (High Street again)
```

Each guess was individually reasonable — maybe road Z genuinely was closest to
that one noisy dot. But as a *sequence* it's nonsense. The car would have had
to teleport 3 km and back in two seconds.

Fixing this is the entire purpose of Part 2.

## The idea

Stop scoring dots. Score **whole journeys**.

A journey is one choice of road per dot, start to finish. We give each
possible journey a score built from two ingredients:

**Emission — "how well does this road explain this dot?"**
This is our classifier's probability. Already done in Part 1.

**Transition — "how believable is driving from road A to road B?"**
New, and it's where the map does the work:
- Are A and B the same road? Very likely — cars stay on roads.
- Does B start where A ends? Likely — that's a normal junction.
- Is B reachable from A within the distance the car could cover in the time
  available? Plausible, and cheaper the better the distances agree.
- Is B unreachable, or so far the car would have broken the speed of sound?
  Effectively impossible.

The best journey is the one with the best **combined** score. Which means:

> A road can win even when it is *not* the best match for its own dot — if
> choosing it makes the whole journey make sense.

That's the whole idea of the HMM in one sentence.

## Where "hidden" and "Markov" come from

**Hidden:** the true road is never observed. We only see noisy GPS, and we
infer the road behind it.

**Markov:** we assume where you go next depends only on where you are *now*,
not your entire history. That's a simplification — real drivers follow routes
with intent — but it's what makes the problem solvable, and it works well in
practice.

## The detail that matters for this dataset

At 148 km/h with 1-second sampling, the car moves ~40 m between dots, while
the median road segment is 25 m long. **The car routinely crosses several
whole segments between consecutive GPS readings.** So "legal transition" can't
just mean "directly connected" — we need reachability within a distance budget
derived from elapsed time. Getting this wrong is the most likely way for our
HMM to underperform.

## Viva questions this covers

- What are the hidden states in your implementation?
- How are emission probabilities calculated?
- How are transition probabilities calculated?
- What happens when consecutive observations suggest disconnected roads?
