"""
Deriving per-observation ground truth from the route.

THE PROBLEM:
`.route` gives the arcs the vehicle traversed IN ORDER, but not which GPS
observation belongs to which arc. There are ~5 observations per route arc
(range 1.3 to 57), so the observation->arc pairing is missing and must be
reconstructed. Every metric we report inherits any error made here.

WHY NOT NEAREST-ARC:
Snapping each observation to the nearest route arc ignores time. Where a route
loops back near itself, or runs beside a parallel carriageway (~15m away, well
inside GPS error), an early observation gets matched to an arc the vehicle only
reaches much later. The label sequence then jumps around and the "route" it
implies is not a route at all.

WHAT WE DO INSTEAD:
Exploit the one thing known with certainty: the vehicle moved FORWARD along the
route. If observation i is on route position j, observation i+1 is on position
j or later, never earlier. So we find the assignment that is monotone
non-decreasing in route position AND minimises total observation-to-arc
distance. That is a shortest-path problem on a grid, solved exactly by dynamic
programming in O(n_obs * n_route).

This is the same algorithm family as sequence alignment and DTW: match two
ordered sequences without letting them cross.
"""
import numpy as np

from .geo import point_segment_distance


def route_arc_geometry(trip):
    """
    Endpoint coordinates of each arc ON THE ROUTE, in travel order.

    Returned as four flat arrays so distances to all route arcs can be
    computed in one vectorised call.
    """
    route = trip.route
    f = trip.arcs[route, 0]
    t = trip.arcs[route, 1]
    return trip.node_x[f], trip.node_y[f], trip.node_x[t], trip.node_y[t]


def observation_route_cost(trip):
    """
    Cost matrix: distance in metres from every observation to every route arc.

    Shape (n_obs, n_route). This is the grid the DP walks over.
    """
    ax, ay, bx, by = route_arc_geometry(trip)
    # Broadcast: observations down the rows, route positions across the columns.
    dist, t_along = point_segment_distance(
        trip.gps_x[:, None], trip.gps_y[:, None],
        ax[None, :], ay[None, :], bx[None, :], by[None, :],
    )
    return dist, t_along


def monotonic_alignment(cost):
    """
    Cheapest assignment of each row (observation) to a column (route position),
    subject to the column index never decreasing as we go down the rows.

    Parameters
    ----------
    cost : (n_obs, n_route) float array, cost[i, j] = distance of obs i to
           route position j.

    Returns
    -------
    assign : (n_obs,) int array of route positions, non-decreasing.

    THE RECURRENCE
    --------------
        dp[i, j] = cost[i, j] + min( dp[i-1, 0..j] )

    "The best way to have observation i on route position j is its own distance
    cost, plus the best way of getting the previous observation onto position j
    or anything before it."

    The inner `min over 0..j` is a PREFIX minimum, so a whole row costs one
    linear scan rather than a nested loop - that is what keeps this O(n*m).
    We also record WHICH previous position achieved that prefix minimum, so the
    path can be traced back at the end.
    """
    n_obs, n_route = cost.shape
    dp = np.empty((n_obs, n_route), dtype=np.float64)
    # back[i, j] = route position used by observation i-1 on the best path
    # that puts observation i at position j.
    back = np.zeros((n_obs, n_route), dtype=np.int32)

    dp[0] = cost[0]          # first observation: no history, pay its own cost
    positions = np.arange(n_route, dtype=np.int32)

    for i in range(1, n_obs):
        prev = dp[i - 1]
        # Running minimum of prev over 0..j, for every j at once.
        prefix_min = np.minimum.accumulate(prev)
        # Which index achieved it: mark the places where prev equals the
        # running min (i.e. a new best), then carry the latest such index
        # forward with a running maximum.
        is_new_best = prev <= prefix_min
        back[i] = np.maximum.accumulate(np.where(is_new_best, positions, 0))
        dp[i] = cost[i] + prefix_min

    # Trace back from the cheapest ending position.
    assign = np.empty(n_obs, dtype=np.int64)
    assign[-1] = int(np.argmin(dp[-1]))
    for i in range(n_obs - 1, 0, -1):
        assign[i - 1] = back[i, assign[i]]
    return assign


def label_trip(trip):
    """
    Assign every GPS observation of `trip` its ground-truth arc_id.

    Returns a dict:
      arc_id        (n_obs,) the true arc for each observation
      route_pos     (n_obs,) its index within trip.route (non-decreasing)
      residual_m    (n_obs,) distance from the observation to that arc.
                    Large values flag observations where the label is doubtful
                    - GPS excursions, or gaps in the reference route. We keep
                    these so label quality can be REPORTED rather than assumed.
      t_along       (n_obs,) how far along the assigned arc the observation
                    projects, 0 at the from-node and 1 at the to-node.

    Raises if the trip has no route (the evaluator partition).
    """
    if trip.route is None:
        raise ValueError(f"trip {trip.trip_id} has no .route; cannot label")

    cost, t_along = observation_route_cost(trip)
    route_pos = monotonic_alignment(cost)

    rows = np.arange(trip.n_obs)
    return {
        "arc_id": trip.route[route_pos],
        "route_pos": route_pos,
        "residual_m": cost[rows, route_pos],
        "t_along": t_along[rows, route_pos],
    }


def label_quality(labels):
    """Summary of how trustworthy a trip's derived labels look."""
    r = labels["residual_m"]
    pos = labels["route_pos"]
    return {
        "residual_p50_m": float(np.percentile(r, 50)),
        "residual_p95_m": float(np.percentile(r, 95)),
        "residual_max_m": float(r.max()),
        "n_residual_gt_50m": int((r > 50).sum()),
        # How much of the reference route the observations actually cover.
        # Well below 1.0 means whole stretches of route have no observation
        # near them, usually because of a GPS gap.
        "route_coverage": float(len(np.unique(pos)) / (pos.max() + 1)),
        "route_positions_used": int(len(np.unique(pos))),
    }
