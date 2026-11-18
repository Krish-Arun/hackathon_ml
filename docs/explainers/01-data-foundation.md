# Section 1 — Data foundation ✅ DONE

**Code:** `src/pathsnap/geo.py`, `io.py`, `candidates.py`, `scripts/explore_dataset.py`

## What the raw data is

Four files per trip. The thing to remember: **in two of them, the row number
IS the ID.**

| File | Columns | Key point |
|---|---|---|
| `.nodes` | lon, lat | row number = `node_id` |
| `.arcs` | from_node, to_node | row number = `arc_id`; arcs are **one-way** |
| `.track` | lon, lat, timestamp | the noisy GPS dots, about one per second |
| `.route` | arc_id | ground truth: arcs driven, in order |

So an arc is a one-way straight line between two map points, and a route is a
walk through that network. We checked: consecutive route arcs always join up
head-to-tail (the end node of one is the start node of the next), 100% of the
time. That fact is what makes Part 2 possible.

## Three problems we had to solve first

### 1. A degree is not a distance

The data is in latitude/longitude degrees. But a degree isn't a fixed length —
walking one degree east near the North Pole is much shorter than one degree
east at the equator. At our latitude (~50°N, Germany) one degree of longitude
is about 71 km, while one degree of latitude is about 111 km. If we measured
distance in raw degrees, east-west would be squashed relative to north-south
and every distance would be wrong.

**Fix:** convert each trip into plain metres, like laying graph paper over that
part of the world. We do this once per trip, centred on that trip's own map.

### 2. How far is a dot from a road?

A road segment is a straight line between two points — but a line with *ends*.
We need the distance from the dot to the nearest point **on the segment**, not
to the infinite line it sits on. Without that restriction, a dot 2 km past the
end of a road would appear to be sitting right on it.

We also record *how far along* the road the nearest point is (0 = at the start,
1 = at the end). Part 2 needs that.

### 3. 300,000 roads is too many to choose from

A single trip's map can have 307,392 arcs. You cannot ask a model "which of
these 307,392?" — and the list is different for every trip, so the question
isn't even well-formed.

**Fix — the key idea of the whole project:** for each dot, shortlist about 10
nearby roads. Then only ever ask the simple question *"is it this one? is it
this one?"* for each shortlisted pair. An impossible 300,000-way choice becomes
an ordinary yes/no problem.

This also hands Part 2 exactly what it needs, because those same candidates
become the possibilities the HMM chooses between.

### Why we sample along roads instead of using nearest map-point

Road lengths here range from 6.7 m to 3.6 km. If we found candidates by looking
for nearby *map points*, we'd completely miss a long motorway arc whose two
endpoints are kilometres away from a dot sitting in its middle.

So we place a marker every 25 m along every road, and "a marker of road A is
near this dot" means "A is a candidate". The exact distances are then
recalculated properly for the shortlist — so this shortcut only ever affects
*which roads we consider*, never any number we report.

## What we found in the data

- 90 trips, 227,577 GPS dots (80 train / 10 student test)
- **No missing coordinates, no duplicate timestamps, no isolated map points**
- Nearest road is **4.1 m away** on average — normal GPS error
- **47 of 90 trips have time gaps over 5 seconds.** Trip 90 is extreme: 536
  dots spread over 42 hours, with one 153,064-second gap. It's really two
  journeys in one file.
- Maps range from 649 to 307,392 arcs; speeds from 20 to 148 km/h
- Only **107 dots (0.047%)**, all in trip 2, have no road within 200 m

That last number is our **candidate miss rate** — the ceiling on how well any
model can possibly do, because if the right road was never shortlisted, no
model can pick it.

## Viva questions this covers

- How is the raw dataset organised? What does each file represent?
- How did you preprocess the data?
- What coordinate system did you use and why?
- What are the main data-quality concerns?
- How did you generate candidates, and how did you choose the radius?
