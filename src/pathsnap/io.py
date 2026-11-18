"""
Raw dataset loading.

FILE FORMATS (verified against the raw files, not assumed from the brief):
  .track  3 cols, whitespace separated, no header: lon, lat, timestamp_seconds
  .nodes  2 cols: lon, lat.      Row index (0-based) IS the node_id.
  .arcs   2 cols: from_node_id, to_node_id. Row index (0-based) IS the arc_id.
          Arcs are DIRECTED; only ~55% have a reverse twin in the file.
  .route  1 col: chronological ground-truth arc_ids. Present for train and
          student_test, absent for the evaluator partition.

We verified on trip 00000001 that consecutive route arcs are head-to-tail
contiguous (to_node[i] == from_node[i+1]) for 100% of steps, which is what
lets the HMM treat the route as a walk through a directed graph.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .geo import projection_origin, to_local_xy


@dataclass
class Trip:
    """One trip: its GPS track, its road network, and (if known) its route."""
    trip_id: str
    partition: str

    # --- GPS observations, in time order ---
    gps_lon: np.ndarray        # (n_obs,)
    gps_lat: np.ndarray        # (n_obs,)
    gps_t: np.ndarray          # (n_obs,) seconds
    gps_x: np.ndarray          # (n_obs,) local metres, east
    gps_y: np.ndarray          # (n_obs,) local metres, north

    # --- road network ---
    node_lon: np.ndarray       # (n_nodes,)
    node_lat: np.ndarray
    node_x: np.ndarray         # (n_nodes,) local metres
    node_y: np.ndarray
    arcs: np.ndarray           # (n_arcs, 2) int, [from_node, to_node]

    # --- ground truth, None for the evaluator partition ---
    route: np.ndarray | None   # (n_route,) int arc_ids in travel order

    # --- projection origin, so predictions can be mapped back to lon/lat ---
    lon0: float
    lat0: float

    @property
    def n_obs(self) -> int:
        return len(self.gps_t)

    @property
    def n_arcs(self) -> int:
        return len(self.arcs)

    def arc_endpoints_xy(self):
        """
        Arc geometry in local metres as four flat arrays (ax, ay, bx, by).

        Returned this way (rather than as objects) because every downstream
        distance computation is vectorised over all arcs at once.
        """
        f, t = self.arcs[:, 0], self.arcs[:, 1]
        return self.node_x[f], self.node_y[f], self.node_x[t], self.node_y[t]

    def arc_lengths(self):
        """Euclidean length of every arc, in metres."""
        ax, ay, bx, by = self.arc_endpoints_xy()
        return np.hypot(bx - ax, by - ay)


def load_trip(trip_dir, partition: str) -> Trip:
    """
    Read one trip directory into a Trip.

    The projection origin is the mean of the NODE coordinates (not the GPS
    coordinates) because the network is the fixed frame we match against, and
    we want the same origin regardless of how noisy a given track is.
    """
    trip_dir = Path(trip_dir)
    trip_id = trip_dir.name

    track = np.loadtxt(trip_dir / f"{trip_id}.track", dtype=np.float64)
    nodes = np.loadtxt(trip_dir / f"{trip_id}.nodes", dtype=np.float64)
    arcs = np.loadtxt(trip_dir / f"{trip_id}.arcs", dtype=np.int64)

    # np.loadtxt collapses a 1-row file to 1-D; force 2-D so indexing is safe.
    track = np.atleast_2d(track)
    nodes = np.atleast_2d(nodes)
    arcs = np.atleast_2d(arcs)

    route_path = trip_dir / f"{trip_id}.route"
    route = None
    if route_path.exists():
        route = np.atleast_1d(np.loadtxt(route_path, dtype=np.int64))

    lon0, lat0 = projection_origin(nodes[:, 0], nodes[:, 1])
    node_x, node_y = to_local_xy(nodes[:, 0], nodes[:, 1], lon0, lat0)
    gps_x, gps_y = to_local_xy(track[:, 0], track[:, 1], lon0, lat0)

    # Enforce chronological order. The brief warns not to assume uniformity,
    # and the HMM/Viterbi stage is meaningless if observations are out of order.
    order = np.argsort(track[:, 2], kind="stable")
    track = track[order]
    gps_x, gps_y = gps_x[order], gps_y[order]

    return Trip(
        trip_id=trip_id, partition=partition,
        gps_lon=track[:, 0], gps_lat=track[:, 1], gps_t=track[:, 2],
        gps_x=gps_x, gps_y=gps_y,
        node_lon=nodes[:, 0], node_lat=nodes[:, 1],
        node_x=node_x, node_y=node_y,
        arcs=arcs, route=route, lon0=lon0, lat0=lat0,
    )


def list_trips(root, partition: str):
    """
    All trip directories in a partition, sorted by trip_id.

    Tolerates the redundant nesting the dataset ships with. The provided data
    arrives as "Student folder/Student folder/..." and the evaluator partition
    as "evaluator_test/evaluator_test/...", so if a partition directory
    contains no digit-named trip folders but does contain exactly one
    subdirectory, we descend into it rather than requiring the files be moved
    by hand on evaluation day.
    """
    p = Path(root) / partition
    if not p.is_dir():
        raise FileNotFoundError(f"partition directory not found: {p}")

    for _ in range(3):      # bounded, so a pathological tree cannot loop
        subdirs = sorted([d for d in p.iterdir() if d.is_dir()], key=lambda d: d.name)
        trips = [d for d in subdirs if d.name.isdigit()]
        if trips:
            return trips
        if len(subdirs) == 1:
            p = subdirs[0]  # descend through the redundant level
            continue
        break
    return subdirs
