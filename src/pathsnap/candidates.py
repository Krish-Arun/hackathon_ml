"""
Candidate generation: for each GPS observation, which road segments could it
plausibly be on?

WHY WE NEED THIS AT ALL:
A trip's network has up to ~300k arcs. We cannot ask a classifier "which of
300k arcs is this point on?" - that is a 300k-class problem with a different
class set per trip, which no standard classifier can express. Instead we
reframe it as BINARY RANKING: generate a small candidate set per observation,
build features for each (observation, candidate) pair, and train a model to
score how likely that pair is the true match. The predicted arc is then the
highest-scoring candidate. This is also exactly the structure the HMM needs:
the candidates become the hidden states at each time step.

WHY DENSIFIED SAMPLING RATHER THAN NEAREST-NODE:
Arc lengths in this dataset are wildly uneven (median ~20 m but up to 3.6 km).
A nearest-NODE lookup would completely miss a long motorway arc whose nodes
are kilometres away from a GPS point sitting in its middle. So we sample
points ALONG every arc at a fixed spacing, index those with a KD-tree, and
treat "near a sample of arc A" as "arc A is a candidate". Exact point-to-
segment distances are then recomputed for the shortlist, so the sampling only
ever affects recall, never the reported distances.
"""
import numpy as np
from scipy.spatial import cKDTree

from .geo import point_segment_distance, bearing_deg


def densify_arcs(trip, spacing=25.0):
    """
    Sample points along every arc at roughly `spacing` metres.

    Returns
    -------
    sample_xy  : (M, 2) sample coordinates in local metres
    sample_arc : (M,)   the arc_id each sample came from

    Every arc gets at least its two endpoints, so short arcs are still indexed.
    """
    ax, ay, bx, by = trip.arc_endpoints_xy()
    lengths = np.hypot(bx - ax, by - ay)

    # Samples per arc: endpoints plus enough interior points to keep the gap
    # under `spacing`. Minimum 2 so degenerate/zero-length arcs still appear.
    n_per_arc = np.maximum(2, np.ceil(lengths / spacing).astype(np.int64) + 1)
    total = int(n_per_arc.sum())

    # Expand to one row per sample without a Python loop.
    arc_idx = np.repeat(np.arange(len(lengths), dtype=np.int64), n_per_arc)
    # Index of each sample WITHIN its own arc: 0, 1, ..., n_i-1
    starts = np.concatenate([[0], np.cumsum(n_per_arc)[:-1]])
    within = np.arange(total, dtype=np.int64) - np.repeat(starts, n_per_arc)
    # Fractional position along the arc, 0 at the from-node, 1 at the to-node.
    t = within / np.repeat(n_per_arc - 1, n_per_arc)

    sx = ax[arc_idx] + t * (bx - ax)[arc_idx]
    sy = ay[arc_idx] + t * (by - ay)[arc_idx]
    return np.column_stack([sx, sy]), arc_idx


def generate_candidates(trip, k=10, radius=200.0, spacing=25.0):
    """
    Shortlist up to `k` candidate arcs for every GPS observation.

    Parameters
    ----------
    k       : max candidates kept per observation, ranked by true perpendicular
              distance. Caps the HMM state space and the feature-table size.
    radius  : search radius in metres. Must comfortably exceed real GPS error,
              or the true arc gets excluded and no model can ever recover it
              (this is the "candidate miss rate" the brief asks us to report).
    spacing : arc sampling interval for the KD-tree index.

    Returns a dict of flat parallel arrays, one entry per (obs, candidate) pair.
    """
    sample_xy, sample_arc = densify_arcs(trip, spacing=spacing)
    tree = cKDTree(sample_xy)

    gps_xy = np.column_stack([trip.gps_x, trip.gps_y])
    ax, ay, bx, by = trip.arc_endpoints_xy()
    arc_bearing_all = bearing_deg(bx - ax, by - ay)

    # Neighbouring samples within `radius` of each observation. Returns a
    # ragged list-of-lists, so we process observation by observation.
    neighbours = tree.query_ball_point(gps_xy, r=radius, workers=-1)

    out_obs, out_arc, out_dist, out_t, out_bearing = [], [], [], [], []
    for i, idx in enumerate(neighbours):
        if len(idx) == 0:
            # No arc within the radius. We record nothing; the caller sees this
            # observation has zero candidates and the HMM must bridge the gap.
            continue
        # Several samples can belong to the same arc - collapse to arc ids.
        cand = np.unique(sample_arc[idx])

        # Exact perpendicular distance to each shortlisted SEGMENT. The
        # sampling above only decided WHICH arcs to look at; these numbers are
        # geometrically exact.
        d, t = point_segment_distance(
            trip.gps_x[i], trip.gps_y[i],
            ax[cand], ay[cand], bx[cand], by[cand],
        )

        # Keep the k nearest. argpartition is O(n) vs a full O(n log n) sort.
        if len(cand) > k:
            keep = np.argpartition(d, k)[:k]
            cand, d, t = cand[keep], d[keep], t[keep]
        order = np.argsort(d)

        out_obs.append(np.full(len(cand), i, dtype=np.int64))
        out_arc.append(cand[order])
        out_dist.append(d[order])
        out_t.append(t[order])
        out_bearing.append(arc_bearing_all[cand[order]])

    if not out_obs:
        empty_i, empty_f = np.array([], np.int64), np.array([], np.float64)
        return dict(obs_idx=empty_i, arc_id=empty_i, dist=empty_f,
                    t_along=empty_f, arc_bearing=empty_f)

    return dict(
        obs_idx=np.concatenate(out_obs),
        arc_id=np.concatenate(out_arc),
        dist=np.concatenate(out_dist),
        t_along=np.concatenate(out_t),
        arc_bearing=np.concatenate(out_bearing),
    )
