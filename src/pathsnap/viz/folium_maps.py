"""
Interactive geographical visualisation with Folium.

Four layers, each independently toggleable, so a reader can compare what the
GPS recorded against what we predicted:

  1. raw GPS observations      small blue dots - the noisy input
  2. nearby road network       thin grey lines - the arcs we chose among
  3. reference route           green - ground truth, where available
  4. predicted route           orange - our output

The network layer is clipped to a corridor around the track. Drawing all
307,000 arcs of a trip would produce a file no browser could open, and the
arcs far from the vehicle were never candidates anyway.
"""
import folium
import numpy as np

COLOUR_GPS = "#2a78d6"
COLOUR_NETWORK = "#b5b4ae"
COLOUR_REFERENCE = "#008300"
COLOUR_PREDICTED = "#eb6834"


def _arc_latlon(trip, arc_ids):
    """[(lat, lon), (lat, lon)] endpoint pairs for each arc."""
    f = trip.arcs[arc_ids, 0]
    t = trip.arcs[arc_ids, 1]
    return [[(trip.node_lat[a], trip.node_lon[a]),
             (trip.node_lat[b], trip.node_lon[b])] for a, b in zip(f, t)]


def nearby_arcs(trip, buffer_m=150.0, max_arcs=6000):
    """
    Arcs within roughly `buffer_m` of the track, for the context layer.

    Uses a bounding box on the projected coordinates, then caps the count so
    the HTML stays openable on a laptop.
    """
    ax, ay, bx, by = trip.arc_endpoints_xy()
    lo_x, hi_x = trip.gps_x.min() - buffer_m, trip.gps_x.max() + buffer_m
    lo_y, hi_y = trip.gps_y.min() - buffer_m, trip.gps_y.max() + buffer_m
    inside = (
        ((ax >= lo_x) & (ax <= hi_x) & (ay >= lo_y) & (ay <= hi_y)) |
        ((bx >= lo_x) & (bx <= hi_x) & (by >= lo_y) & (by <= hi_y))
    )
    idx = np.flatnonzero(inside)
    if len(idx) > max_arcs:
        idx = idx[np.linspace(0, len(idx) - 1, max_arcs).astype(int)]
    return idx


def build_map(trip, predicted_arcs=None, reference_arcs=None,
              title=None, show_network=True):
    """
    Build the Folium map for one trip.

    predicted_arcs : per-observation predicted arc ids (-1 allowed), or None
    reference_arcs : the trip's ground-truth route, or None for evaluator trips
    """
    centre = [float(np.median(trip.gps_lat)), float(np.median(trip.gps_lon))]
    # OpenStreetMap rather than CartoDB: Carto now requires an API key, and
    # the evaluator must be able to open these offline-ish without one.
    m = folium.Map(location=centre, zoom_start=13, tiles="OpenStreetMap",
                   control_scale=True)

    if show_network:
        fg = folium.FeatureGroup(name="Road network (near track)", show=True)
        # One PolyLine holding every segment, rather than thousands of
        # separate objects: same picture, a fraction of the HTML.
        folium.PolyLine(_arc_latlon(trip, nearby_arcs(trip)),
                        color=COLOUR_NETWORK, weight=1, opacity=0.6).add_to(fg)
        fg.add_to(m)

    if reference_arcs is not None and len(reference_arcs):
        fg = folium.FeatureGroup(name="Reference route (ground truth)", show=True)
        folium.PolyLine(_arc_latlon(trip, np.asarray(reference_arcs)),
                        color=COLOUR_REFERENCE, weight=5,
                        opacity=0.75).add_to(fg)
        fg.add_to(m)

    if predicted_arcs is not None:
        valid = np.asarray(predicted_arcs)
        valid = valid[valid >= 0]
        # Collapse consecutive repeats: one line per distinct arc travelled.
        if len(valid):
            keep = np.append(True, valid[1:] != valid[:-1])
            fg = folium.FeatureGroup(name="Predicted route", show=True)
            folium.PolyLine(_arc_latlon(trip, valid[keep]),
                            color=COLOUR_PREDICTED, weight=4,
                            opacity=0.9).add_to(fg)
            fg.add_to(m)

    fg = folium.FeatureGroup(name="Raw GPS observations", show=True)
    # Thin out very long tracks so the file stays manageable.
    step = max(1, trip.n_obs // 1200)
    for i in range(0, trip.n_obs, step):
        folium.CircleMarker(
            [float(trip.gps_lat[i]), float(trip.gps_lon[i])],
            radius=2, color=COLOUR_GPS, fill=True, fill_opacity=0.7, weight=0,
            tooltip=f"obs {i}  t={trip.gps_t[i]:.0f}s",
        ).add_to(fg)
    fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(_legend_html(title or trip.trip_id)))
    m.fit_bounds([[trip.gps_lat.min(), trip.gps_lon.min()],
                  [trip.gps_lat.max(), trip.gps_lon.max()]])
    return m


def _legend_html(title):
    rows = [("Raw GPS observations", COLOUR_GPS),
            ("Road network", COLOUR_NETWORK),
            ("Reference route", COLOUR_REFERENCE),
            ("Predicted route", COLOUR_PREDICTED)]
    items = "".join(
        f'<div style="margin:3px 0"><span style="display:inline-block;width:18px;'
        f'height:4px;background:{c};vertical-align:middle;margin-right:7px"></span>{n}</div>'
        for n, c in rows)
    return f'''
    <div style="position:fixed;bottom:22px;left:22px;z-index:9999;
                background:#fcfcfbee;padding:11px 14px;border-radius:8px;
                border:1px solid #e3e2de;font:13px system-ui,sans-serif;
                color:#0b0b0b;box-shadow:0 2px 8px #0002">
      <div style="font-weight:600;margin-bottom:6px">PathSnap — trip {title}</div>
      {items}
    </div>'''
