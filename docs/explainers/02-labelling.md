# Section 2 — Labelling: building the answer key

**Code:** `src/pathsnap/labelling.py`

## The catch nobody warns you about

To train a model you need examples with known right answers. You'd assume the
dataset provides them. **It doesn't — not in the form we need.**

`.route` tells us which roads the car drove, *in order*. It does not tell us
which dot belongs to which road. There are about **5 dots per road segment**
(ranging from 1.3 to 57), so the pairing is missing and we have to work it out
ourselves.

```
What we have:          What we need:
  dots:  • • • • • •     dot 1 -> road A
  route: A, B, C         dot 2 -> road A
                         dot 3 -> road B
                         dot 4 -> road B   ...etc
```

**Everything downstream inherits mistakes made here.** If the answer key is
wrong, every accuracy number we report is measuring the wrong thing. This is
the highest-risk section in the project.

## The obvious approach, and why it breaks

Obvious idea: for each dot, pick whichever road *on the route* is nearest.
Five lines of code.

It fails in two situations that genuinely occur in this data:

- **The route loops back near itself.** A dot from minute 2 sits physically
  close to a road the car reaches at minute 22. Nearest-distance matches it to
  the wrong one, and the answer key jumps around in time.
- **Parallel carriageways.** On a dual carriageway the opposite side is maybe
  15 m away — easily within GPS error. Half the dots get matched to the wrong
  direction of travel.

## What we do instead

Use the one thing we know for absolute certain: **the car moved forwards.**

It drove road 1, then road 2, then road 3. It never teleported backwards.
So the dot order and the road order must agree. If dot 50 is on road B, then
dot 51 is on road B or something later — never road A.

So we look for the pairing that:
- keeps every dot in forward order along the route, **and**
- makes the total distance (dot to its assigned road) as small as possible

Trying every possible pairing is astronomically expensive. But there's a
classic trick — **dynamic programming** — that finds the exact best answer
efficiently.

### The idea behind the DP

Build a grid: dots down the side, route roads across the top. Each cell holds
"how far is this dot from this road?"

Now walk the grid top to bottom. For each cell, ask: *if this dot ends up on
this road, what's the cheapest total cost of everything before it?* Because of
the forward rule, the previous dot must have been on this road or an earlier
one — so we only look left and up, never right. Keep a running "best so far"
as we scan, and the whole row costs one pass instead of a search.

At the bottom, pick the cheapest ending and trace the pointers back up to
recover the full assignment.

Cost is (number of dots × number of route roads) — about 2 million cells on
the biggest trip, which takes about a second.

> This is the same algorithm shape used to align two DNA sequences, or to sync
> an audio recording to its transcript. "Match two ordered sequences without
> letting them cross" is a very common problem.

## What we also record

For each dot we keep the **distance to its assigned road**. If that distance is
large, the label is suspect — maybe GPS was badly off, or the route data has a
gap. We keep it so we can report how trustworthy the answer key is, instead of
pretending it's perfect. The brief explicitly says not to overstate the
certainty of the reference routes.

## Viva questions this covers

- Where did your training labels come from?
- Why not just snap each point to the nearest road?
- What assumptions does your labelling make?
- How do you know your labels are reliable?
