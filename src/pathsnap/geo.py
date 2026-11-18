"""
Geographic helpers.

WHY A LOCAL PROJECTION:
The dataset is lon/lat (degrees). One degree of longitude is ~111 km at the
equator but only ~71 km at this dataset's latitude (~50 deg N), so Euclidean
distance in raw degrees is anisotropic and wrong. We therefore project every
trip into a LOCAL equirectangular frame in metres, centred on that trip's own
network. Over a ~50 km extent the equirectangular error vs a proper UTM
projection is well under a metre, which is far below GPS noise, and it is
~100x cheaper than a full pyproj transform. We keep ONE origin per trip so all
coordinates inside a trip are mutually consistent.
"""
import numpy as np

# Metres per degree. 111_320 m is the standard equatorial value for longitude
# (scaled by cos(lat) below); 110_540 m is the mean metres-per-degree latitude.
_M_PER_DEG_LON_EQ = 111_320.0
_M_PER_DEG_LAT = 110_540.0


def projection_origin(lon, lat):
    """Pick the projection origin as the mean of the supplied coordinates."""
    return float(np.mean(lon)), float(np.mean(lat))


def to_local_xy(lon, lat, lon0, lat0):
    """
    Project lon/lat (degrees) to local metres about (lon0, lat0).

    x runs east, y runs north. cos(lat0) shrinks the longitude axis so that a
    metre east and a metre north are the same physical length.
    """
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    x = (lon - lon0) * _M_PER_DEG_LON_EQ * np.cos(np.deg2rad(lat0))
    y = (lat - lat0) * _M_PER_DEG_LAT
    return x, y


def to_lonlat(x, y, lon0, lat0):
    """Inverse of to_local_xy - needed to draw predictions back onto a map."""
    lon = np.asarray(x) / (_M_PER_DEG_LON_EQ * np.cos(np.deg2rad(lat0))) + lon0
    lat = np.asarray(y) / _M_PER_DEG_LAT + lat0
    return lon, lat


def point_segment_distance(px, py, ax, ay, bx, by):
    """
    Perpendicular distance from points P to line SEGMENTS A->B, plus where the
    projection falls along each segment.

    Fully vectorised: px/py broadcast against ax/ay/bx/by, so this handles
    "every observation against every candidate arc" in one call.

    Returns
    -------
    dist : distance in metres from P to the nearest point ON the segment
    t    : position of that nearest point along A->B, clamped to [0, 1].
           t=0 means the arc's start node is nearest, t=1 the end node.
           We keep t because it tells us HOW FAR ALONG an arc the vehicle is,
           which the HMM transition model needs.
    """
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay

    # Squared length of each segment. Guard against zero-length (degenerate)
    # arcs, which do occur in the raw data, to avoid divide-by-zero.
    seg_len_sq = abx * abx + aby * aby
    seg_len_sq = np.where(seg_len_sq == 0.0, 1e-12, seg_len_sq)

    # Scalar projection of AP onto AB, normalised to [0,1] along the segment.
    # Clamping is what makes this a SEGMENT distance rather than an infinite
    # line distance - without it, points beyond an arc's end would get an
    # artificially small distance.
    t = (apx * abx + apy * aby) / seg_len_sq
    t = np.clip(t, 0.0, 1.0)

    # Closest point on the segment, then plain Euclidean distance to it.
    closest_x = ax + t * abx
    closest_y = ay + t * aby
    dist = np.hypot(px - closest_x, py - closest_y)
    return dist, t


def bearing_deg(dx, dy):
    """
    Compass bearing of a vector, in degrees clockwise from north (0..360).

    Used for both vehicle heading (between consecutive GPS fixes) and arc
    orientation, so we can compare the two as a feature.
    """
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def angle_difference_deg(a, b):
    """
    Smallest absolute angle between two bearings, in [0, 180].

    Plain subtraction is wrong across the 0/360 wrap (359 deg and 1 deg are 2
    deg apart, not 358). This folds the difference into [0, 180].
    """
    d = np.abs(np.asarray(a) - np.asarray(b)) % 360.0
    return np.where(d > 180.0, 360.0 - d, d)
