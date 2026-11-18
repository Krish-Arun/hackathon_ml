"""
Feature engineering: turning each (observation, candidate arc) pair into
numbers a classifier can learn from.

DESIGN PRINCIPLE - RELATIVE BEATS ABSOLUTE:
"This arc is 30m away" is a weak signal. "This arc is 30m away and the next
best option is 400m away" is a strong one; "30m away and three others are
within 35m" means genuinely ambiguous. So alongside the raw geometry we
compute each candidate's standing RELATIVE TO THE OTHER CANDIDATES FOR THE
SAME OBSERVATION. These comparison features are expected to carry most of the
signal, and they are cheap.

LEAKAGE RULE:
Every feature here is computable from the GPS track and the road network
alone. Nothing uses .route. The test applied to each one: could the API
compute this for a trip whose reference route we have never seen? If not, it
does not belong here.

WHAT WE DELIBERATELY DO NOT DO:
We keep sequence-awareness minimal (one mild connectivity feature). Reasoning
over the sequence is the HMM's job in Part 2; if the classifier leans hard on
sequence information, the HMM has nothing left to contribute and we cannot
show it earned its place.
"""
import numpy as np
import pandas as pd

from .geo import bearing_deg, angle_difference_deg

EPS = 1e-6

# Column order is part of the model contract - the API must build features in
# exactly this order. Exported to config.json at section 10.
FEATURE_COLUMNS = [
    # --- geometry of this candidate ---
    "dist_m",
    "log_dist",
    "t_along",
    "arc_length_m",
    # --- direction agreement ---
    "bearing_diff",
    "bearing_diff_reverse",
    # --- how this candidate compares to the others for the SAME observation ---
    "dist_minus_best",
    "log_dist_ratio_best",
    "dist_rank",
    "bearing_rank",
    "dist_softmax",
    "n_candidates",
    # --- motion context of the observation itself ---
    "speed_mps",
    "dt_prev_s",
    "dt_next_s",
    "step_dist_m",
    # --- mild sequence hint ---
    "connected_to_prev_best",
]


def vehicle_bearing_and_motion(trip):
    """
    Per-observation heading, speed and timing, derived from consecutive fixes.

    Heading at observation i is the direction from i-1 to i. The first
    observation has no predecessor, so it borrows the direction to i+1.
    A stationary vehicle has no meaningful heading; we leave the previous
    value in place rather than inventing one.
    """
    x, y, t = trip.gps_x, trip.gps_y, trip.gps_t
    n = len(t)

    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    if n > 1:
        # First observation: use the forward direction instead of zero.
        dx[0], dy[0] = x[1] - x[0], y[1] - y[0]

    step = np.hypot(dx, dy)
    dt_prev = np.diff(t, prepend=t[0])
    if n > 1:
        dt_prev[0] = t[1] - t[0]
    dt_next = np.diff(t, append=t[-1])
    if n > 1:
        dt_next[-1] = t[-1] - t[-2]

    heading = bearing_deg(dx, dy)
    # Where the vehicle barely moved the heading is noise; carry the last
    # trustworthy heading forward instead.
    unreliable = step < 1.0
    if unreliable.any():
        idx = np.arange(n)
        good = np.where(~unreliable, idx, 0)
        heading = heading[np.maximum.accumulate(good)]

    speed = np.divide(step, np.maximum(dt_prev, EPS))
    return heading, speed, step, dt_prev, dt_next


def build_features(trip, cand):
    """
    Build the feature table for one trip.

    Parameters
    ----------
    trip : Trip
    cand : dict from candidates.generate_candidates

    Returns
    -------
    pandas DataFrame, one row per (observation, candidate) pair, carrying
    trip_id / obs_idx / arc_id identifiers plus FEATURE_COLUMNS.
    """
    obs = cand["obs_idx"]
    arc = cand["arc_id"]
    dist = cand["dist"]
    n_pairs = len(obs)

    heading, speed, step, dt_prev, dt_next = vehicle_bearing_and_motion(trip)
    arc_len_all = trip.arc_lengths()

    # --- direction agreement -------------------------------------------------
    # Arcs are one-way. A candidate pointing the opposite way to travel is
    # usually the far carriageway, so we keep both the forward difference and
    # the difference against the reversed arc; the model can learn which
    # matters (one-way streets vs dual carriageways behave differently).
    bdiff = angle_difference_deg(heading[obs], cand["arc_bearing"])
    bdiff_rev = angle_difference_deg(heading[obs], (cand["arc_bearing"] + 180.0) % 360.0)

    # --- comparisons within each observation's candidate group ---------------
    # np.minimum.at scatters a groupwise min without a Python loop.
    best = np.full(trip.n_obs, np.inf)
    np.minimum.at(best, obs, dist)
    counts = np.bincount(obs, minlength=trip.n_obs).astype(np.float32)

    dist_minus_best = dist - best[obs]
    # Ratio of this candidate's distance to the best one for the same
    # observation. Taken in log space: when the best candidate sits almost
    # exactly on its arc the raw ratio explodes (we measured 36,000+), which
    # would swamp any scaled model. log1p keeps the ordering but bounds the
    # tail.
    dist_ratio_best = np.log1p(dist / (best[obs] + 1.0))

    # Rank within group. Candidates arrive already sorted by distance per
    # observation, so rank is just position within the group: subtract each
    # group's first row index from the row index.
    group_start = np.zeros(trip.n_obs, dtype=np.int64)
    first_seen = np.flatnonzero(np.diff(obs, prepend=-1) != 0)
    group_start[obs[first_seen]] = first_seen
    dist_rank = np.arange(n_pairs, dtype=np.int64) - group_start[obs]

    # Rank by bearing agreement needs an actual per-group ordering.
    order = np.lexsort((bdiff, obs))
    bearing_rank = np.empty(n_pairs, dtype=np.int64)
    bearing_rank[order] = np.arange(n_pairs) - group_start[obs[order]]

    # Softmax over negative distance within each group: a normalised
    # "share of plausibility" that sums to 1 per observation. Scale of 20m is
    # the order of GPS error, so differences below that are treated as noise.
    w = np.exp(-dist_minus_best / 20.0)
    denom = np.zeros(trip.n_obs)
    np.add.at(denom, obs, w)
    dist_softmax = w / (denom[obs] + EPS)

    # --- mild sequence hint ---------------------------------------------------
    # Does this candidate share a node with the nearest candidate of the
    # PREVIOUS observation? Roads come in chains, so this is informative, but
    # it is deliberately weak: it uses only the nearest neighbour, not a full
    # sequence model.
    best_arc = np.full(trip.n_obs, -1, dtype=np.int64)
    best_arc[obs[first_seen]] = arc[first_seen]
    prev_best = np.roll(best_arc, 1)
    prev_best[0] = -1

    connected = np.zeros(n_pairs, dtype=np.float32)
    valid = prev_best[obs] >= 0
    if valid.any():
        pb = prev_best[obs[valid]]
        cand_nodes = trip.arcs[arc[valid]]
        prev_nodes = trip.arcs[pb]
        shares = (
            (cand_nodes[:, 0] == prev_nodes[:, 1])
            | (cand_nodes[:, 0] == prev_nodes[:, 0])
            | (cand_nodes[:, 1] == prev_nodes[:, 1])
        )
        same = pb == arc[valid]
        connected[valid] = np.where(same, 2.0, shares.astype(np.float32))

    return pd.DataFrame({
        "trip_id": np.int32(int(trip.trip_id)),
        "obs_idx": obs.astype(np.int32),
        "arc_id": arc.astype(np.int32),
        "dist_m": dist.astype(np.float32),
        "log_dist": np.log1p(dist).astype(np.float32),
        "t_along": cand["t_along"].astype(np.float32),
        "arc_length_m": arc_len_all[arc].astype(np.float32),
        "bearing_diff": bdiff.astype(np.float32),
        "bearing_diff_reverse": bdiff_rev.astype(np.float32),
        "dist_minus_best": dist_minus_best.astype(np.float32),
        "log_dist_ratio_best": dist_ratio_best.astype(np.float32),
        "dist_rank": dist_rank.astype(np.int16),
        "bearing_rank": bearing_rank.astype(np.int16),
        "dist_softmax": dist_softmax.astype(np.float32),
        "n_candidates": counts[obs].astype(np.int16),
        "speed_mps": speed[obs].astype(np.float32),
        "dt_prev_s": dt_prev[obs].astype(np.float32),
        "dt_next_s": dt_next[obs].astype(np.float32),
        "step_dist_m": step[obs].astype(np.float32),
        "connected_to_prev_best": connected,
    })


# Observations whose assigned arc is further away than this are treated as
# unreliably labelled. Justification: the nearest network arc is 4.1m away at
# p50 and 13m at p95 across the dataset, so a label sitting 50m+ from its
# observation cannot be explained by GPS error - it means the reference route
# does not cover where the vehicle actually was.
LABEL_RELIABLE_MAX_RESIDUAL_M = 50.0


def attach_labels(df, labels):
    """
    Add the target column and a label-reliability flag.

    label           1 where this candidate IS the arc the labeller assigned to
                    this observation, else 0.
    label_residual_m  how far the observation sits from its assigned arc.
    label_reliable  False where that distance is implausibly large.

    WHY THE RELIABILITY FLAG EXISTS:
    Some trips' .route files describe only part of the journey the GPS
    recorded - trip 33 drove 47.7km while its route spans 6.4km. The monotonic
    alignment must still assign every observation somewhere, so observations
    from the uncovered stretch get placed on arcs 120m+ away. Those labels are
    not ground truth and training on them teaches the model nonsense.

    These columns are METADATA, never features: they are derived from .route,
    so feeding them to the model would be leakage of exactly the kind the
    brief prohibits.

    Also returns candidate recall - the fraction of observations whose true
    arc appears among its candidates. That is the ceiling on Top-1 accuracy:
    if the right arc was never shortlisted, no model can pick it.
    """
    true_arc = labels["arc_id"]
    residual = labels["residual_m"]
    obs = df["obs_idx"].to_numpy()

    df = df.copy()
    df["label"] = (df["arc_id"].to_numpy() == true_arc[obs]).astype(np.int8)
    df["label_residual_m"] = residual[obs].astype(np.float32)
    df["label_reliable"] = (residual[obs] <= LABEL_RELIABLE_MAX_RESIDUAL_M)

    hit = df.groupby("obs_idx")["label"].max()
    recall = float(hit.mean()) if len(hit) else 0.0
    return df, recall
