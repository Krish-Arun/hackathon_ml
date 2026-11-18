# Section 14 — UI and Folium maps (Laptop 2)

**Code:** `src/pathsnap/ui/`, `src/pathsnap/viz/folium_maps.py`

## The UI

A page that lists the ten evaluation trips. The evaluator picks one, it calls
the API on Laptop 1, and it displays the returned route on a map with the
accuracy numbers.

Deliberately simple. It holds **no model, no ground truth, and no prediction
logic** — it asks the API and renders the answer. That separation is a
requirement, not a style choice.

## The map

Folium turns Python into an interactive Leaflet map in the browser. Required
layers, each visually distinct:

| Layer | Shown as |
|---|---|
| Raw GPS observations | small blue dots |
| Road network nearby | thin grey lines |
| Reference route (where known) | green line |
| **Our predicted route** | **red/orange line** |

Plus a legend, a layer toggle so layers can be switched on and off, and a
sensible zoom. The point is that a human can see at a glance *where we were
right and where we drifted* — a map where you can't compare the GPS trail
against the prediction has failed at its only job.

## Why this is worth real effort

It's the most persuasive single artifact in the whole submission. A table
saying 0.91 is abstract. A map showing a noisy GPS trail that wanders across
three streets, with a clean predicted route snapped along the correct road, is
immediately convincing — and it's a required report figure.

Good one to generate for a trip where the HMM clearly rescued a bad
classifier-only result: show both, side by side. That single image justifies
all of Part 2.

## Viva questions this covers

- Walk us through the UI.
- Show the map layers and explain what each represents.
- Where does your system do well, and where does it fail?
