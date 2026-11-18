"""
Viterbi decoding: the single most likely sequence of arcs.

WHY VITERBI RATHER THAN ENUMERATION:
With ~10 candidates per observation and ~1,300 observations, there are about
10^1300 possible routes. Viterbi finds the exact best one without enumerating
them, using one observation:

    To know the best route ending on arc b at step i, you only need the best
    score of arriving at each candidate at step i-1. You never need to know
    HOW you reached step i-1, only what it cost.

So we sweep forward keeping one number per candidate per step, plus a pointer
to the predecessor that achieved it, then trace the pointers back. Cost falls
from 10^1300 to roughly (steps x candidates x candidates) - about 130,000
operations, milliseconds.

This is the same shape as the labelling DP in section 2: build the answer
incrementally, keep only what you need, trace back at the end. The difference
is what the score means - there, distance to a known route; here, emission
probability plus road-network plausibility.
"""
import numpy as np

from .hmm import (transition_log_probs, prepare_trip, segment_boundaries,
                  MIN_EMISSION, DEFAULT_BETA_M)


def group_candidates(trip, cand):
    """
    Reorganise the flat candidate arrays into per-observation groups.

    Returns a list of length n_obs; each entry is (arc_ids, t_along, row_idx)
    or None where the observation has no candidates at all (107 such
    observations exist in the dataset).
    """
    groups = [None] * trip.n_obs
    obs = cand["obs_idx"]
    if len(obs) == 0:
        return groups
    # Candidates arrive sorted by observation, so boundaries are where the
    # observation index changes.
    starts = np.flatnonzero(np.diff(obs, prepend=-1) != 0)
    ends = np.append(starts[1:], len(obs))
    for s, e in zip(starts, ends):
        groups[obs[s]] = (cand["arc_id"][s:e], cand["t_along"][s:e],
                          np.arange(s, e))
    return groups


def decode_segment(trip, adjacency, groups, indices, proba, beta):
    """
    Viterbi over one continuous run of observations.

    `indices` are observation indices in time order, all of which have
    candidates. Returns the chosen arc id for each.
    """
    first = indices[0]
    arcs0, t0, rows0 = groups[first]
    # Work in log space: probabilities multiply along a path, and summing
    # logs avoids underflow over a thousand steps.
    delta = np.log(np.maximum(proba[rows0], MIN_EMISSION))
    backptr = []
    states = [(arcs0, t0)]

    for prev_i, cur_i in zip(indices[:-1], indices[1:]):
        arcs_p, t_p, _ = groups[prev_i]
        arcs_c, t_c, rows_c = groups[cur_i]

        gps_d = float(np.hypot(trip.gps_x[cur_i] - trip.gps_x[prev_i],
                               trip.gps_y[cur_i] - trip.gps_y[prev_i]))
        T = transition_log_probs(trip, adjacency, arcs_p, t_p, arcs_c, t_c,
                                 gps_d, beta=beta)

        # score[j, k] = best way to be at candidate j, then move to k
        score = delta[:, None] + T
        best_prev = np.argmax(score, axis=0)
        delta = score[best_prev, np.arange(len(arcs_c))] + \
            np.log(np.maximum(proba[rows_c], MIN_EMISSION))

        backptr.append(best_prev)
        states.append((arcs_c, t_c))

    # Trace back from the best final state.
    k = int(np.argmax(delta))
    chosen = [k]
    for bp in reversed(backptr):
        k = int(bp[k])
        chosen.append(k)
    chosen.reverse()

    return [int(states[s][0][k]) for s, k in enumerate(chosen)]


def decode_trip(trip, cand, proba, beta=DEFAULT_BETA_M, max_gap_s=None):
    """
    Decode a whole trip into one predicted arc per observation.

    Returns an int array of length n_obs. Observations with no candidates get
    -1; the brief asks for a prediction per relevant observation, and an
    honest -1 is better than inventing an arc for a point with no road within
    200 metres.

    Sequence breaks:
      * a time gap beyond max_gap_s - not a continuous journey, so the halves
        are decoded independently rather than forcing an impossible connection
      * a run of observations with no candidates - the chain is interrupted,
        so we restart after it
    """
    adjacency = prepare_trip(trip)
    groups = group_candidates(trip, cand)
    preds = np.full(trip.n_obs, -1, dtype=np.int64)

    kwargs = {} if max_gap_s is None else {"max_gap_s": max_gap_s}
    breaks = set(segment_boundaries(trip, **kwargs))

    # Build runs of consecutive observations that have candidates and are not
    # separated by a time gap.
    run = []
    for i in range(trip.n_obs):
        starts_new = (i in breaks) or groups[i] is None
        if starts_new and run:
            _emit(trip, adjacency, groups, run, proba, beta, preds)
            run = []
        if groups[i] is not None:
            run.append(i)
    if run:
        _emit(trip, adjacency, groups, run, proba, beta, preds)
    return preds


def _emit(trip, adjacency, groups, run, proba, beta, preds):
    """Decode one run and write its predictions into `preds`."""
    if len(run) == 1:
        # Nothing to chain: fall back to the best emission alone.
        arcs, _, rows = groups[run[0]]
        preds[run[0]] = int(arcs[int(np.argmax(proba[rows]))])
    else:
        for i, arc in zip(run, decode_segment(trip, adjacency, groups, run,
                                              proba, beta)):
            preds[i] = arc


def classifier_only(trip, cand, proba):
    """
    Baseline for comparison: highest-scoring candidate per observation,
    chosen independently. This is what Top-1 accuracy measures, and what the
    HMM is meant to improve on.
    """
    groups = group_candidates(trip, cand)
    preds = np.full(trip.n_obs, -1, dtype=np.int64)
    for i, g in enumerate(groups):
        if g is None:
            continue
        arcs, _, rows = g
        preds[i] = int(arcs[int(np.argmax(proba[rows]))])
    return preds
