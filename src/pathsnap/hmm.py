"""
HMM route reconstruction: emission and transition modelling.

THE PROBLEM THIS SOLVES:
The classifier scores each observation independently, so consecutive
predictions can land on arcs that are nowhere near each other. The vehicle
appears to teleport. We measured route continuity at ~0.91 for the classifier
alone, so roughly one consecutive pair in eleven is physically impossible.

THE MODEL:
  Hidden state at time i  = which candidate arc the vehicle is actually on,
                            together with where along it the observation
                            projects (t_along). Candidate generation already
                            produced this set, so the state space is ~10 per
                            observation rather than 300,000.
  Emission  P(obs | arc)  = the classifier's probability for that
                            (observation, candidate) pair. All of part one
                            feeds in here.
  Transition P(arc_b | arc_a) = how believable it is to drive from a to b in
                            the elapsed time.

TRANSITIONS - THE PART THAT USES THE MAP:
Following Newson & Krumm (2009), we compare two distances:
  * the straight-line distance the GPS says the vehicle moved
  * the distance it would have to travel ALONG THE ROAD NETWORK to get from
    candidate a to candidate b
On a correct pair these agree closely. On a wrong pair - a parallel street, or
an arc across town - the road distance is far longer than the GPS distance, or
there is no path at all. We score the disagreement with an exponential
penalty.

WHY A BOUNDED SEARCH:
Computing a true shortest path over a 300,000-arc network for every candidate
pair at every timestep would be hopeless. But we do not need unbounded paths:
at 148 km/h with 1-second sampling the vehicle covers ~40m, so any path longer
than a few hundred metres is already implausible. We run a Dijkstra bounded by
a budget derived from the observed gap, which keeps each search to a few dozen
nodes.
"""
import heapq
import numpy as np

# Scale of the exponential penalty on route-vs-GPS distance disagreement, in
# metres. Newson & Krumm fit ~4-7m for 1Hz taxi data; we allow a little more
# because this dataset mixes dense urban with 148 km/h motorway.
DEFAULT_BETA_M = 8.0

# Floor on emission probability. A candidate the classifier is certain about
# would otherwise veto every path through it, and the classifier is not
# reliable enough to be given that power.
MIN_EMISSION = 1e-6

# Cost applied when no path exists within budget. Finite, not infinite: a
# slightly broken route is more useful than no route, and the data genuinely
# contains disconnected stretches.
UNREACHABLE_PENALTY = 25.0

# Beyond this the "journey" is not continuous and we decode the two halves
# independently. Trip 90 has a 153,064s gap; forcing continuity across 42
# hours is meaningless.
MAX_GAP_SECONDS = 180.0


def build_adjacency(trip):
    """
    Outgoing arcs per node, as a flat CSR-style structure.

    Returns (indptr, arc_ids) where the arcs leaving node n are
    arc_ids[indptr[n]:indptr[n+1]]. Flat arrays rather than a dict of lists
    because this is walked tens of thousands of times per trip.
    """
    n_nodes = len(trip.node_x)
    from_node = trip.arcs[:, 0]
    counts = np.bincount(from_node, minlength=n_nodes)
    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    order = np.argsort(from_node, kind="stable")
    return indptr, order.astype(np.int64)


def bounded_dijkstra(trip, adjacency, source_node, budget_m, targets):
    """
    Shortest road distance from `source_node` to each node in `targets`,
    giving up past `budget_m`.

    Returns dict node -> distance, containing only nodes reached inside the
    budget. The budget is what makes this cheap: with a median arc of 25m, a
    200m budget visits a few dozen nodes out of up to 155,000.
    """
    indptr, arc_order = adjacency
    lengths = trip._arc_len_cache
    to_node = trip.arcs[:, 1]

    remaining = set(targets)
    dist = {source_node: 0.0}
    heap = [(0.0, source_node)]
    out = {}

    while heap and remaining:
        d, node = heapq.heappop(heap)
        if d > dist.get(node, np.inf):
            continue
        if node in remaining:
            out[node] = d
            remaining.discard(node)
            if not remaining:
                break
        if d > budget_m:
            continue
        for k in range(indptr[node], indptr[node + 1]):
            arc = arc_order[k]
            nxt = to_node[arc]
            nd = d + lengths[arc]
            if nd <= budget_m and nd < dist.get(nxt, np.inf):
                dist[nxt] = nd
                heapq.heappush(heap, (nd, nxt))
    return out


def transition_log_probs(trip, adjacency, cand_a, t_a, cand_b, t_b,
                         gps_distance_m, beta=DEFAULT_BETA_M):
    """
    Log transition score for every (candidate a -> candidate b) pair.

    Returns an (len(cand_a), len(cand_b)) array.

    The road distance from a to b is:
        (what is left of arc a beyond the observation)
      + (shortest path from a's end node to b's start node)
      + (how far into arc b the next observation sits)

    with two special cases taken first because they are both common and cheap:
      * same arc      -> the vehicle simply moved along it
      * a feeds b     -> they share a node, no search needed
    """
    lengths = trip._arc_len_cache
    arcs = trip.arcs
    n_a, n_b = len(cand_a), len(cand_b)
    out = np.full((n_a, n_b), -UNREACHABLE_PENALTY, dtype=np.float64)

    # Budget: generous multiple of the straight-line distance, since roads
    # bend, plus a floor so near-stationary steps can still move a little.
    budget = float(min(max(3.0 * gps_distance_m + 100.0, 150.0), 3000.0))

    target_nodes = {int(arcs[b, 0]) for b in cand_b}

    for ia, a in enumerate(cand_a):
        a = int(a)
        len_a = lengths[a]
        left_on_a = (1.0 - t_a[ia]) * len_a

        reach = None
        for ib, b in enumerate(cand_b):
            b = int(b)
            if a == b:
                # Stayed on the same arc: distance is the movement along it.
                road_d = abs(t_b[ib] - t_a[ia]) * len_a
            elif arcs[a, 1] == arcs[b, 0]:
                # Directly connected head-to-tail - the common case.
                road_d = left_on_a + t_b[ib] * lengths[b]
            else:
                if reach is None:
                    reach = bounded_dijkstra(trip, adjacency, int(arcs[a, 1]),
                                             budget, target_nodes)
                hop = reach.get(int(arcs[b, 0]))
                if hop is None:
                    continue            # stays at -UNREACHABLE_PENALTY
                road_d = left_on_a + hop + t_b[ib] * lengths[b]

            # Exponential penalty on disagreement between how far the GPS says
            # the vehicle moved and how far it would have to drive.
            out[ia, ib] = -abs(road_d - gps_distance_m) / beta

    return out


def prepare_trip(trip):
    """Cache per-trip structures the decoder reuses at every timestep."""
    trip._arc_len_cache = trip.arc_lengths()
    return build_adjacency(trip)


def segment_boundaries(trip, max_gap_s=MAX_GAP_SECONDS):
    """
    Indices where the observation sequence should be cut.

    A gap longer than max_gap_s is not a continuous journey - 47 of 90 trips
    have gaps over 5s and two have gaps over 14 hours. Decoding across one
    would force the model to invent an impossible connection, so we decode
    each side independently.
    """
    gaps = np.diff(trip.gps_t)
    return (np.flatnonzero(gaps > max_gap_s) + 1).tolist()
