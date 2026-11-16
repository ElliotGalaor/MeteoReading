"""Dashboard page: weather reading along a GPS track (sample, uploaded, or drawn on a map)."""
from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from folium.plugins import Draw
from plotly.subplots import make_subplots
from streamlit_folium import st_folium

from meteo.geotrek import INSTANCES, fetch_gpx, nearby_treks
from meteo.gps import ACTIVITIES, Track, parse_track, passage_times, sample_points, summary, to_gpx, with_times
from meteo.route_scores import DANGER_LEVELS, Assessment, Score, assess, is_past, level
from meteo.route_weather import (
    FORECAST_FUTURE_DAYS,
    api_weight,
    clock_format,
    nearby_station_obs,
    read_route,
    track_timezone,
    whole,
)
from meteo.routing import geocode, track_from_waypoints, waypoints_from_drawings
from ui import (
    AXIS,
    COOL,
    DIVERGING,
    HEAT,
    INK,
    INK2,
    MUTED,
    RAIN,
    ROOT,
    SERIES,
    STATUS,
    WARM,
    _connect,
    chart,
    num,
    page_header,
)

SAMPLES_DIR = ROOT / "samples"
TIMEZONES = ["Europe/Paris", "UTC", "Europe/London", "Europe/Madrid", "Europe/Rome", "Europe/Berlin",
             "America/New_York", "America/Los_Angeles", "Asia/Tokyo"]


@st.cache_data(show_spinner=False)
def _parse(data: bytes, filename: str) -> tuple[pd.DataFrame, str, str | None]:
    t = parse_track(data, filename)
    return t.points, t.name, t.time_issue


@st.cache_data(show_spinner=False, ttl=86400)
def _timezone(lat: float, lon: float) -> str | None:
    return track_timezone(lat, lon)


@st.cache_data(show_spinner=False, ttl=6 * 3600, max_entries=100)
def _nearby(lat: tuple, lon: tuple, radius_km: float) -> tuple[list[dict], list[str]]:
    """Geotrek routes near a track (cached 6 hours: the parks do not change their routes every minute)."""
    found, failed = nearby_treks(lat, lon, radius_km=radius_km, limit=10)
    return [vars(s) | {"activity": s.activity} for s in found], failed


@st.cache_data(show_spinner=False, ttl=6 * 3600, max_entries=50)
def _geotrek_points(url: str) -> pd.DataFrame:
    return parse_track(fetch_gpx(url), "route.gpx").points


def _load_geotrek(route: dict) -> None:
    """Replaces the track by a Geotrek route (callback of the "Read the weather" button)."""
    try:
        points = _geotrek_points(route["gpx_url"])
    except (ValueError, RuntimeError) as exc:
        st.session_state["geotrek_error"] = f"Could not load “{route['name']}”: {exc}"
        return
    st.session_state["route_geotrek"] = route | {"points": points}
    st.session_state.pop("geotrek_error", None)
    st.toast(f"Weather reading for “{route['name']}”: see the top of the page.", icon=":material/hiking:")


def _clear_geotrek() -> None:
    st.session_state.pop("route_geotrek", None)


@st.cache_data(show_spinner=False)
def _samples() -> dict[str, str]:
    out = {}
    for p in sorted(SAMPLES_DIR.glob("*.gpx")):
        _, name, _ = _parse(p.read_bytes(), p.name)
        out[name] = str(p)
    return out


@st.cache_data(show_spinner=False, ttl=1800)
def _reading(points: pd.DataFrame, name: str):
    """Cached for 30 minutes: changing a widget and coming back does not call the API again."""
    r = read_route(Track(name=name, points=points))
    return r.points, r.source, r.timezone, r.insights, r.stats


@st.cache_data(show_spinner=False, ttl=3600)
def _stations(points: pd.DataFrame, tz: str) -> pd.DataFrame:
    with _connect() as con:
        return nearby_station_obs(con, points, tz)


@st.cache_data(show_spinner=False, ttl=3600)
def _route(waypoints: tuple, activity: str, follow: bool) -> tuple[pd.DataFrame, str, bool]:
    track, routed = track_from_waypoints(list(waypoints), activity, follow)
    return track.points, track.name, routed


@st.cache_data(show_spinner=False, ttl=86400)
def _geocode(name: str) -> list[dict]:
    return geocode(name)


def _guess_activity(name: str) -> str:
    """From whole words of the track name ("Carnac coastal ride" is not a car trip)."""
    words = set(re.findall(r"[a-zà-ÿ]+", name.lower()))
    if words & {"car", "drive", "driving", "voiture", "roadtrip"} or "road trip" in name.lower():
        return "Driving"
    if words & {"walk", "walking", "hike", "hiking", "run", "running", "rando", "randonnée", "trail", "marche"}:
        return "Walking"
    return "Cycling"


def _zoom(lat: pd.Series, lon: pd.Series, width_px: float, height_px: float) -> float:
    """Zoom at which the track fits in width_px x height_px (web maps: 512 px for 360° at zoom 0, Mercator)."""
    span_x = max(float(lon.max() - lon.min()), 1e-4)                                   # x is linear in longitude
    span_y = max(float(lat.max() - lat.min()), 1e-4) / np.cos(np.radians(lat.mean()))  # y stretched by 1/cos(lat)
    zoom = min(np.log2(width_px * 360 / (512 * span_x)), np.log2(height_px * 360 / (512 * span_y)))
    return float(np.clip(zoom, 3, 14))


def _phone() -> bool:
    """A phone, from the browser's user agent: the map is then about 350 px wide instead of 700."""
    try:
        return "Mobi" in (st.context.headers.get("User-Agent") or "")
    except Exception:
        return False


def _warehouse_last_date() -> date | None:
    try:
        with _connect() as con:
            return con.sql("SELECT max(last_date) FROM dim_station").fetchall()[0][0]
    except Exception:  # no warehouse: the page still works with the API only
        return None


# =============================================================================
def page_route():
    page_header("Route weather",
                "Draw a route on the map, import a GPS track or pick a sample: get the weather along the route, "
                "at the time you pass each point.")

    mode = st.segmented_control("Track", ["Sample track", "Draw on the map", "Upload a file"], default="Sample track",
                                required=True, key="route_mode", on_change=_clear_geotrek)
    geo = st.session_state.get("route_geotrek")   # a route picked in "Routes nearby", in place of the track
    if geo:
        points, name, time_issue = geo["points"], geo["name"], None
        left, right = st.columns([1.3, 1], gap="large")
        with left:
            st.markdown(f"**{name}**")
            link = f" · [its page]({geo['page_url']})" if geo.get("page_url") else ""
            st.caption(f"{geo['practice'] or 'Route'} from {geo['source']}, via Geotrek{link}")
            st.button("Back to my track", icon=":material/undo:", on_click=_clear_geotrek)
    elif mode == "Draw on the map":
        built = _draw_track()
        if built is None:
            return
        points, name = built
        time_issue = None
        left, right = st.columns([1.3, 1], gap="large")
        with left:
            st.markdown(f"**{name}**")
            st.download_button("Save this route as GPX", to_gpx(Track(name=name, points=points)).encode("utf-8"),
                               "route.gpx", "application/gpx+xml", icon=":material/download:",
                               help="Reuse it later with “Upload a file”, or in a GPS app.")
    else:
        left, right = st.columns([1.3, 1], gap="large")
        with left:
            try:
                if mode == "Upload a file":
                    up = st.file_uploader("GPX, KML or GeoJSON file", type=["gpx", "kml", "geojson", "json"])
                    if up is None:
                        st.info("Drop a track exported from Strava, Garmin, Komoot, Google Earth… "
                                "or draw one on the map.")
                        return
                    points, name, time_issue = _parse(up.getvalue(), up.name)
                else:
                    samples = _samples()
                    if not samples:
                        st.warning("No sample track found in the samples/ folder.")
                        return
                    names = list(samples)
                    default = next((i for i, n in enumerate(names) if n.startswith("Paris")), 0)
                    name = st.selectbox("Sample", names, index=default)
                    points, name, time_issue = _parse(open(samples[name], "rb").read(), samples[name])
            except ValueError as exc:
                st.error(f"Could not read this track: {exc}")
                return
    track = Track(name=name, points=points)
    options = list(ACTIVITIES)
    if geo:
        guessed = geo["activity"]
    elif mode == "Draw on the map":
        guessed = st.session_state.get("draw_activity", "Cycling")
    else:
        guessed = _guess_activity(track.name)

    with right:
        if track.has_time:
            timing = st.segmented_control("Timing", ["Recorded times", "Plan a new trip"], default="Recorded times",
                                            required=True)
        else:
            timing = "Plan a new trip"
            why = f"its timestamps cannot be used ({time_issue})" if time_issue else "this track has no timestamps"
            st.caption(f"{why[0].upper()}{why[1:]}: choose when you set off and how fast you go.")
        if timing != "Plan a new trip":
            if track.points["time"].max().date() > date.today() + timedelta(days=FORECAST_FUTURE_DAYS):
                st.warning(f"These recorded times are more than {FORECAST_FUTURE_DAYS + 1} days ahead, beyond the "
                           "forecasts: choose “Plan a new trip” to set a departure date.")
                return
            activity = st.selectbox("Activity", options, index=options.index(guessed),
                                    help="Sets the danger and comfort scores: the same gust is a danger on a bike, "
                                         "not in a car.")
        else:
            c1, c2 = st.columns(2)
            today = date.today()
            day = c1.date_input("Departure date", today + timedelta(days=1),
                                max_value=today + timedelta(days=FORECAST_FUTURE_DAYS),
                                help="Up to 16 days ahead (forecast). Past dates show the weather that actually happened.")
            hour = c2.time_input("Departure time", time(9, 0), step=900)
            c3, c4, c5 = st.columns([1.2, 1, 1.2])
            activity = c3.selectbox("Activity", options, index=options.index(guessed),
                                    help="Sets the speed, and the danger and comfort scores.")
            preset = ACTIVITIES[activity]
            speed = c4.number_input("Speed (km/h)", 1.0, 150.0, float(preset["speed"]), step=0.5,
                                    help="Average speed on the flat; time is added for climbs when walking or cycling.")
            # Default: the time zone where the track starts (a 09:00 departure in Tokyo is 09:00 Tokyo time)
            first = track.points.iloc[0]
            local_tz = _timezone(round(float(first["lat"]), 2), round(float(first["lon"]), 2)) or "Europe/Paris"
            zones = TIMEZONES if local_tz in TIMEZONES else [local_tz, *TIMEZONES]
            tz = c5.selectbox("Time zone", zones, index=zones.index(local_tz),
                              help="The departure time is read in this time zone; by default, the one where "
                                   "the track starts.")
            naive = pd.Timestamp(datetime.combine(day, hour))
            start = naive.tz_localize(tz, nonexistent="shift_forward", ambiguous=True)
            if start.tz_localize(None) != naive:
                st.info(f"{hour:%H:%M} does not exist on {day:%d %B} in {tz}: the clocks go forward that night. "
                        f"Departure at {start:%H:%M}.")
            track = with_times(track, passage_times(track, start, speed, preset["climb_min_per_100m"]))

    if mode == "Upload a file" and not geo:
        st.caption(f"**{track.name}**")
    s = summary(track)
    k = st.columns(4)
    k[0].metric("Distance", f"{num(s['distance_km'], 1)} km", border=True)
    k[1].metric("Ascent", f"{num(s['ascent_m'], 0)} m" if s["ascent_m"] is not None else "–", border=True)
    k[2].metric("Duration", f"{int(s['duration_h'])} h {int(s['duration_h'] % 1 * 60):02d}", border=True)
    n_points = len(sample_points(track))
    k[3].metric("Weather points", f"{n_points}", border=True,
                help=f"Points sampled along the route (at least 1 km apart). Cost: about "
                     f"{api_weight(n_points, 3):.0f} Open-Meteo calls, out of a free quota of 10,000 per day.")

    try:
        with st.spinner("Reading the weather along the route…"):
            pts, source, tz, insights, stats = _reading(track.points, track.name)
    except (ValueError, RuntimeError) as exc:
        st.error(str(exc))
        _nearby_panel(track)
        return

    local = pts["time"].dt.tz_convert(tz)
    scores = assess(pts, activity, tz)
    pts = pts.assign(local=local, local_str=local.dt.strftime(clock_format(local).replace("%a", "%a %d %b")),
                     danger=scores.points["danger"].to_numpy(), comfort=scores.points["comfort"].to_numpy())

    st.subheader("The reading")
    k = st.columns(4)
    k[0].metric("Temperature", f"{whole(stats['temp_min'])} to {whole(stats['temp_max'])} °C", border=True,
                help=f"Feels like down to {whole(stats['feels_min'])} °C.")
    rain_txt = "< 0.1 mm" if 0 < stats["rain_mm"] < 0.1 else f"{stats['rain_mm']:.1f} mm"
    k[1].metric("Rain on the way", rain_txt, border=True,
                help=f"On {stats['rain_share']:.0%} of the route (hours with ≥ 0.2 mm/h at the time you pass).")
    k[2].metric("Strongest gust", f"{stats['gust_max']:.0f} km/h", border=True)
    k[3].metric("Headwind share", f"{stats['headwind_share']:.0%}", border=True,
                help=f"Wind component against you above 5 km/h. Tailwind on {stats['tailwind_share']:.0%} of the route.")
    for line in insights:
        st.markdown(f"<div class='finding'>{line}</div>", unsafe_allow_html=True)

    _scores_panel(scores, pts, source)
    st.subheader("Along the route")
    _map(track, pts, tz)
    _profile(track, pts)
    _scores_chart(scores, pts)
    _table(pts)
    _station_panel(pts, tz)

    label = ("Open-Meteo forecast API (best available models for the area, e.g. Météo-France AROME/ARPEGE over France)"
             if source == "forecast" else "ERA5 reanalysis archive via the Open-Meteo API (what the weather actually was)")
    st.markdown(f"<div class='source'>Weather: {label}. Times in {tz}. Temperatures are adjusted to the altitude "
                f"of each point (90 m terrain model).</div>", unsafe_allow_html=True)
    _nearby_panel(track)


def _nearby_panel(track: Track) -> None:
    """Hiking, cycling and trail routes starting near the track, from the Geotrek sites of parks and départements."""
    st.subheader("Routes nearby")
    st.caption("Routes published by parks and départements on their Geotrek sites, starting near your track. "
               "Pick one to read the weather along it.")
    radius = st.select_slider("Starting within", [2, 5, 10, 20], value=10, format_func=lambda v: f"{v} km",
                              key="geotrek_radius")
    pts = track.points
    keep = np.linspace(0, len(pts) - 1, min(len(pts), 100)).astype(int)
    lat = tuple(np.round(pts["lat"].to_numpy()[keep], 3).tolist())
    lon = tuple(np.round(pts["lon"].to_numpy()[keep], 3).tolist())
    with st.spinner("Looking for routes nearby…"):
        found, failed = _nearby(lat, lon, float(radius))
    if geo := st.session_state.get("route_geotrek"):   # not the route being read
        found = [r for r in found if (r["source"], r["trek_id"]) != (geo["source"], geo["trek_id"])]
    if error := st.session_state.pop("geotrek_error", None):
        st.error(error)
    if not found:
        st.info(f"No Geotrek route starts within {radius} km of this track. The sites queried cover these areas: "
                + ", ".join(i.name for i in INSTANCES) + ".")
    else:
        table = pd.DataFrame(found)
        event = st.dataframe(
            table[["name", "practice", "length_km", "ascent_m", "distance_km", "source", "page_url"]],
            hide_index=True, on_select="rerun", selection_mode="single-row",
            key=f"geotrek_table_{hash((lat, lon, radius))}", placeholder="–",
            column_config={
                "name": st.column_config.TextColumn("Route", width="large"),
                "practice": "Practice",
                "length_km": st.column_config.NumberColumn("Length", format="%.1f km"),
                "ascent_m": st.column_config.NumberColumn("Ascent", format="%d m"),
                "distance_km": st.column_config.NumberColumn("Start", format="%.1f km away",
                                                             help="Distance from your track to the route's start"),
                "source": "Published by",
                "page_url": st.column_config.LinkColumn("Page", display_text="Open"),
            })
        rows = event.selection.rows
        if rows:
            route = found[rows[0]]
            st.button(f"Read the weather along “{route['name']}”", type="primary", icon=":material/hiking:",
                      on_click=_load_geotrek, args=(route,))
        else:
            st.caption("Select a route in the table (box on the left of the row) to read the weather along it.")
    if failed:
        st.caption("Not reached this time: " + ", ".join(failed) + ".")


DANGER_STATUS = {"Low": "good", "Moderate": "warning", "High": "serious", "Severe": "critical"}
COMFORT_STATUS = {"Very pleasant": "good", "Pleasant": "good", "Mixed": "warning", "Unpleasant": "serious",
                  "Very unpleasant": "critical"}
METHOD = """
**Danger** (0 = nothing to worry about, 100 = do not go). At each point, every hazard (thunderstorm, ice, snow,
heavy rain, gusts, heat, cold, poor visibility, darkness) gets a score from 0 to 100, rising between a threshold
where it starts to matter and one where it is severe. The thresholds depend on the activity: a 60 km/h gust
scores 50 on a bike and 0 in a car. Hazards met at the same time add up like independent risks, and the route's
score is 70% its worst moment and 30% the average over the time on the way.

**Comfort** (100 = ideal for the activity). Points are taken off at each point for the temperature outside the
comfortable range of the activity, muggy air, rain, wind (the headwind on a bike), snow, strong sun, darkness
and fog, then averaged over the time on the way.

**Advice**: simple rules on the conditions met, where and when, worded for the activity. They do not replace the
official warnings. Details and thresholds: `meteo/route_scores.py`.
"""


def _card(title: str, score: Score, status: str, why: str) -> str:
    color = STATUS[status]
    return (f"<div class='score-card'><div class='score-label'>{title}</div>"
            f"<div class='score-value'>{score.value}<small> / 100</small>"
            f"<span class='chip'><i style='background:{color}'></i>{score.level}</span></div>"
            f"<div class='meter'><div style='width:{max(score.value, 2)}%;background:{color}'></div></div>"
            f"<div class='score-why'>{why}</div></div>")


def _scores_panel(scores: Assessment, pts: pd.DataFrame, source: str) -> None:
    st.subheader(f"Safety and comfort · {scores.activity.lower()}")
    clock = dict(zip(pts["time"], pts["local_str"], strict=True))
    danger, comfort = scores.danger, scores.comfort
    if danger.factors:
        why = "Worst moments: " + " · ".join(f"{f.name.lower()} {f.value:.0f} at km {f.km:.0f} ({clock[f.time]})"
                                              for f in danger.factors[:3])
    else:
        why = "No significant hazard on the way."
    c1, c2 = st.columns(2)
    c1.markdown(_card("Danger score", danger, DANGER_STATUS[danger.level], why), unsafe_allow_html=True)
    if comfort.factors:
        why = "Points lost on average: " + " · ".join(f"{f.name.lower()} −{f.value:.0f}" for f in comfort.factors[:3])
    else:
        why = "Nothing spoils the trip."
    c2.markdown(_card("Comfort score", comfort, COMFORT_STATUS[comfort.level], why), unsafe_allow_html=True)
    with st.expander("How are the scores computed?"):
        st.markdown(METHOD)

    st.markdown("<div class='chart-title'>What to plan for</div>", unsafe_allow_html=True)
    if is_past(pts):
        st.caption("A past trip: the advice says what those conditions called for.")
    for a in scores.advice:
        st.markdown(f"<div class='advice {a.kind}'>{a.icon} <b>{a.title}</b><br><span>{a.text}</span></div>",
                    unsafe_allow_html=True)


def _scores_chart(scores: Assessment, pts: pd.DataFrame) -> None:
    sp = scores.points
    h, pen = sp.filter(like="h_"), sp.filter(like="p_")
    top_h = np.where(h.max(axis=1) >= 10, h.idxmax(axis=1).str[2:].str.lower(), "none")
    top_p = np.where(pen.max(axis=1) >= 3, pen.idxmax(axis=1).str[2:].str.lower(), "none")
    d_level = [level(v, DANGER_LEVELS) for v in sp["danger"]]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                        subplot_titles=["Danger (0 = none, 100 = do not go)", "Comfort (100 = ideal)"])
    fig.add_scatter(x=sp["dist_km"], y=sp["danger"], mode="lines+markers", name="Danger", showlegend=False,
                    line=dict(color=AXIS, width=2),
                    marker=dict(size=10, color=[STATUS[DANGER_STATUS[lv]] for lv in d_level],
                                line=dict(width=2, color="#fcfcfb")),
                    customdata=np.stack([pts["local_str"], d_level, top_h], axis=1),
                    hovertemplate="km %{x:.1f} · %{customdata[0]}<br>danger %{y:.0f} (%{customdata[1]})"
                                  "<br>main hazard: %{customdata[2]}<extra></extra>", row=1, col=1)
    for y, name in [(25, "moderate"), (50, "high"), (75, "severe")]:
        fig.add_hline(y=y, line=dict(color=MUTED, width=1, dash="dot"), row=1, col=1,
                      annotation_text=name, annotation_position="top left",
                      annotation_font=dict(color=MUTED, size=11))
    fig.add_scatter(x=sp["dist_km"], y=sp["comfort"], mode="lines+markers", name="Comfort", showlegend=False,
                    line=dict(color=SERIES[0], width=2), marker=dict(size=8),
                    customdata=np.stack([pts["local_str"], top_p], axis=1),
                    hovertemplate="km %{x:.1f} · %{customdata[0]}<br>comfort %{y:.0f}"
                                  "<br>main discomfort: %{customdata[1]}<extra></extra>", row=2, col=1)
    fig.update_yaxes(range=[0, 104], tickvals=[0, 25, 50, 75, 100], row=1, col=1)
    fig.update_yaxes(range=[0, 104], tickvals=[0, 25, 50, 75, 100], row=2, col=1)
    fig.update_xaxes(title_text="distance (km)", row=2, col=1)
    fig.update_annotations(selector=dict(text="Danger (0 = none, 100 = do not go)"), font=dict(size=13, color=INK2),
                           x=0, xanchor="left")
    fig.update_annotations(selector=dict(text="Comfort (100 = ideal)"), font=dict(size=13, color=INK2),
                           x=0, xanchor="left")
    chart(fig, 460, "Danger and comfort along the route",
          "Danger dots are coloured by level · hover for the main hazard or discomfort at each point",
          margin=dict(t=34))


def _draw_track() -> tuple[pd.DataFrame, str] | None:
    """Map where the user clicks waypoints; returns the resulting track (points, name)."""
    c1, c2, c3 = st.columns([2, 1.1, 1.2], vertical_alignment="bottom")
    place = c1.text_input("Go to a place", placeholder="Town, village, mountain…", key="draw_place")
    activity = c2.selectbox("Route for", list(ACTIVITIES), index=1, key="draw_activity")
    follow = c3.toggle("Follow roads and paths", value=True, key="draw_follow",
                       help="Snap the line to roads and paths suited to the activity (OpenStreetMap routing). "
                            "Off: straight lines between your points, e.g. off-trail.")
    center, zoom = (46.6, 2.6), 6
    if place:
        try:
            hits = _geocode(place)
        except OSError:   # requests' errors (connection, timeout, HTTP error) are OSErrors
            hits = None
            c1.caption("The place search is not answering: try again in a moment.")
        if hits:
            center, zoom = (hits[0]["lat"], hits[0]["lon"]), 12
            c1.caption(f"Centred on {hits[0]['label']}")
        elif hits is not None:
            c1.caption("Place not found: try the name of a town, village or mountain.")
    m = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap", control_scale=True)
    Draw(draw_options={"polyline": {"shapeOptions": {"color": SERIES[0], "weight": 4}, "metric": True},
                       "polygon": False, "rectangle": False, "circle": False, "circlemarker": False, "marker": False},
         edit_options={"edit": True, "remove": True}).add_to(m)
    out = st_folium(m, key="draw_map", height=460, use_container_width=True, center=center, zoom=zoom,
                    returned_objects=["all_drawings"])
    waypoints = waypoints_from_drawings((out or {}).get("all_drawings"))
    if len(waypoints) < 2:
        st.info("Pick the line tool (top left of the map), click your waypoints, then click the last point again "
                "to finish. The edit and bin tools below it let you move or delete the line.")
        return None
    with st.spinner("Building the route…"):
        try:
            points, name, routed = _route(tuple(waypoints), activity, follow)
        except ValueError as exc:
            st.error(str(exc))
            return None
    if follow and not routed:
        st.warning("The routing service did not answer: your points are joined by straight lines.")
    return points, name


def _map(track: Track, pts: pd.DataFrame, tz: str) -> None:
    choice = st.segmented_control("Colour the route by", ["Temperature", "Rain", "Headwind", "Danger", "Comfort"],
                                  default="Temperature", required=True)
    choice = choice or "Temperature"
    col, scale, unit = {"Temperature": ("temperature_2m", HEAT, "°C"), "Rain": ("precipitation", RAIN, "mm/h"),
                        "Headwind": ("headwind", DIVERGING, "km/h"), "Danger": ("danger", HEAT, "/ 100"),
                        "Comfort": ("comfort", RAIN, "/ 100")}[choice]
    vals = pts[col].fillna(0)
    marker = dict(size=13, color=vals, colorscale=scale, colorbar=dict(title=dict(text=unit, side="top"), thickness=12, len=0.6))
    if choice == "Headwind":
        lim = max(abs(vals).max(), 5)
        marker.update(cmin=-lim, cmax=lim)
    elif choice == "Rain":
        marker.update(cmin=0, cmax=max(vals.max(), 2))
    elif choice in ("Danger", "Comfort"):
        marker.update(cmin=0, cmax=100)
    full = track.points
    fig = go.Figure()
    fig.add_trace(go.Scattermap(lat=full["lat"], lon=full["lon"], mode="lines", line=dict(width=3, color=INK2),
                                hoverinfo="skip", name="Track"))
    custom = np.stack([pts["dist_km"], pts["local_str"], pts["icon"] + " " + pts["condition"], pts["temperature_2m"],
                       pts["precipitation"].fillna(0), pts["wind_from"], pts["wind_speed_10m"], pts["headwind"],
                       pts["danger"], pts["comfort"]], axis=1)
    fig.add_trace(go.Scattermap(
        lat=pts["lat"], lon=pts["lon"], mode="markers", marker=marker, customdata=custom, name="Weather",
        hovertemplate="<b>km %{customdata[0]:.1f}</b> · %{customdata[1]}<br>%{customdata[2]}<br>"
                      "%{customdata[3]:.1f} °C · rain %{customdata[4]:.1f} mm/h<br>"
                      "wind from %{customdata[5]} %{customdata[6]:.0f} km/h (headwind %{customdata[7]:+.0f})<br>"
                      "danger %{customdata[8]:.0f} · comfort %{customdata[9]:.0f}<extra></extra>"))
    ends = pts.iloc[[0, -1]]
    fig.add_trace(go.Scattermap(lat=ends["lat"], lon=ends["lon"], mode="markers+text", text=["Start", "Finish"],
                                textposition="top right", marker=dict(size=9, color=INK), hoverinfo="skip",
                                textfont=dict(color=INK, size=13)))
    # Room for the track and its Start / Finish labels
    zoom = _zoom(full["lat"], full["lon"], 190 if _phone() else 520, 330)   # minus the colour bar
    fig.update_layout(map=dict(style="carto-positron", zoom=zoom,
                               center=dict(lat=float(full["lat"].mean()), lon=float(full["lon"].mean()))),
                      showlegend=False)
    chart(fig, 460, f"{choice} along the route", "Hover over a dot for the full reading at that point",
          margin=dict(l=0, r=0, b=0))


def _profile(track: Track, pts: pd.DataFrame) -> None:
    has_ele = pts["ele_profile"].notna().any()
    rows = ["Elevation (m)", "Temperature (°C)", "Rain (mm/h)", "Wind along the route (km/h)"][0 if has_ele else 1:]
    fig = make_subplots(rows=len(rows), cols=1, shared_xaxes=True, vertical_spacing=0.06, subplot_titles=rows)
    r = 1
    hover_time = pts["local_str"]
    if has_ele:
        full = track.points
        y = full["ele"] if track.has_elevation else np.interp(full["dist_km"], pts["dist_km"], pts["ele_profile"])
        fig.add_scatter(x=full["dist_km"], y=y, mode="lines", line=dict(color=MUTED, width=1.5), fill="tozeroy",
                        fillcolor="rgba(195,194,183,0.35)", name="Elevation", showlegend=False,
                        hovertemplate="km %{x:.1f}: %{y:.0f} m<extra></extra>", row=r, col=1)
        lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
        fig.update_yaxes(range=[max(lo - (hi - lo) * 0.15 - 10, 0) if lo >= 0 else lo - 10, hi + (hi - lo) * 0.1 + 10],
                         row=r, col=1)
        r += 1
    fig.add_scatter(x=pts["dist_km"], y=pts["temperature_2m"], mode="lines+markers", name="Temperature",
                    line=dict(color=SERIES[1], width=2.2), marker=dict(size=7), customdata=hover_time,
                    hovertemplate="km %{x:.1f} · %{customdata}: %{y:.1f} °C<extra>temperature</extra>", row=r, col=1)
    fig.add_scatter(x=pts["dist_km"], y=pts["apparent_temperature"], mode="lines+markers", name="Feels like",
                    line=dict(color=SERIES[0], width=2.2), marker=dict(size=7), customdata=hover_time,
                    hovertemplate="km %{x:.1f} · %{customdata}: %{y:.1f} °C<extra>feels like</extra>", row=r, col=1)
    r += 1
    prob = pts["precipitation_probability"]
    prob_txt = [f"{p:.0f}%" if pd.notna(p) else "n/a" for p in prob]
    fig.add_bar(x=pts["dist_km"], y=pts["precipitation"].fillna(0), name="Rain", marker_color=SERIES[0],
                showlegend=False, customdata=np.stack([hover_time, prob_txt], axis=1),
                hovertemplate="km %{x:.1f} · %{customdata[0]}: %{y:.1f} mm/h (probability %{customdata[1]})<extra></extra>",
                row=r, col=1)
    fig.update_yaxes(rangemode="tozero", row=r, col=1)
    r += 1
    fig.add_bar(x=pts["dist_km"], y=pts["headwind"], name="Headwind (+) / tailwind (−)", showlegend=False,
                marker_color=[WARM if v >= 0 else COOL for v in pts["headwind"]], customdata=hover_time,
                hovertemplate="km %{x:.1f} · %{customdata}: %{y:+.0f} km/h<extra>head/tail wind</extra>", row=r, col=1)
    fig.add_scatter(x=pts["dist_km"], y=pts["wind_gusts_10m"], mode="lines", name="Gusts", line=dict(color=INK, width=1.5),
                    customdata=hover_time, hovertemplate="km %{x:.1f} · %{customdata}: %{y:.0f} km/h<extra>gusts</extra>",
                    row=r, col=1)
    fig.update_xaxes(title_text="distance (km)", row=r, col=1)
    fig.update_annotations(font=dict(size=13, color=INK2), x=0, xanchor="left")
    fig.update_layout(bargap=0.3, hovermode="closest")
    chart(fig, 230 * len(rows), "Profile of the route",
          "Each panel shares the same distance axis · red bars: wind against you, blue: wind behind you")


def _table(pts: pd.DataFrame) -> None:
    tbl = pd.DataFrame({
        "km": pts["dist_km"].round(1), "time": pts["local_str"], "elevation_m": pts["ele_profile"].round(0),
        "conditions": pts["icon"] + " " + pts["condition"], "temp_c": pts["temperature_2m"].round(1),
        "feels_like_c": pts["apparent_temperature"].round(1), "rain_mm_h": pts["precipitation"].round(1),
        "rain_prob_pct": pts["precipitation_probability"].round(0), "wind": pts["wind_from"] + " " + pts["wind_speed_10m"].round(0).astype(int).astype(str),
        "headwind_kmh": pts["headwind"].round(0), "gusts_kmh": pts["wind_gusts_10m"].round(0),
        "danger": pts["danger"].round(0), "comfort": pts["comfort"].round(0),
    })
    with st.expander("Point-by-point reading", expanded=False):
        st.dataframe(tbl, hide_index=True, column_config={
            "km": st.column_config.NumberColumn("km", format="%.1f"), "time": "Time",
            "elevation_m": st.column_config.NumberColumn("Elevation", format="%d m"), "conditions": "Conditions",
            "temp_c": st.column_config.NumberColumn("Temp.", format="%.1f °C"),
            "feels_like_c": st.column_config.NumberColumn("Feels like", format="%.1f °C"),
            "rain_mm_h": st.column_config.NumberColumn("Rain", format="%.1f mm/h"),
            "rain_prob_pct": st.column_config.NumberColumn("Rain prob.", format="%d%%"),
            "wind": "Wind (from, km/h)",
            "headwind_kmh": st.column_config.NumberColumn("Headwind", format="%+d km/h"),
            "gusts_kmh": st.column_config.NumberColumn("Gusts", format="%d km/h"),
            "danger": st.column_config.NumberColumn("Danger", format="%d", help="0 = none, 100 = do not go"),
            "comfort": st.column_config.NumberColumn("Comfort", format="%d", help="100 = ideal for the activity")})
        st.download_button("Download as CSV", tbl.to_csv(index=False).encode("utf-8"), "route_weather.csv", "text/csv",
                           icon=":material/download:")


def _station_panel(pts: pd.DataFrame, tz: str) -> None:
    last = _warehouse_last_date()
    day = pts["local"].dt.date.max()
    if last is None:
        return
    with st.expander("Météo-France stations near the route (observed)", expanded=False):
        if day > last:
            st.caption("Station observations are shown for past trips only (the warehouse ends on "
                       f"{last:%d %B %Y}).")
            return
        obs = _stations(pts[["dist_km", "lat", "lon", "time"]], tz)
        if obs.empty:
            st.caption("No station of the downloaded départements within 5 km of the route had data that day.")
            return
        st.caption("Daily values measured by stations within 5 km of the route, on the day you passed: "
                   "a ground truth to compare with the model reading above.")
        st.dataframe(obs, hide_index=True, column_config={
            "route_km": st.column_config.NumberColumn("Route km", format="%.1f"), "station_name": "Station",
            "altitude": st.column_config.NumberColumn("Alt.", format="%d m"),
            "distance_km": st.column_config.NumberColumn("Distance", format="%.1f km"),
            "day": st.column_config.DateColumn("Day", format="D MMM YYYY"),
            "tn": st.column_config.NumberColumn("Min temp.", format="%.1f °C"),
            "tx": st.column_config.NumberColumn("Max temp.", format="%.1f °C"),
            "rr": st.column_config.NumberColumn("Rain", format="%.1f mm"),
            "fxi": st.column_config.NumberColumn("Max gust", format="%.1f m/s")})
