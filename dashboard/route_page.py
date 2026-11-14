"""Dashboard page: weather reading along a GPS track (sample, uploaded, or drawn on a map)."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from folium.plugins import Draw
from plotly.subplots import make_subplots
from streamlit_folium import st_folium

from meteo.gps import ACTIVITIES, Track, parse_track, passage_times, sample_points, summary, to_gpx, with_times
from meteo.route_weather import FORECAST_FUTURE_DAYS, api_weight, nearby_station_obs, read_route
from meteo.routing import geocode, track_from_waypoints, waypoints_from_drawings
from ui import COOL, DIVERGING, HEAT, INK, INK2, MUTED, RAIN, ROOT, SERIES, WARM, _connect, chart, num, page_header

SAMPLES_DIR = ROOT / "samples"
TIMEZONES = ["Europe/Paris", "UTC", "Europe/London", "Europe/Madrid", "Europe/Rome", "Europe/Berlin",
             "America/New_York", "America/Los_Angeles", "Asia/Tokyo"]


@st.cache_data(show_spinner=False)
def _parse(data: bytes, filename: str) -> tuple[pd.DataFrame, str]:
    t = parse_track(data, filename)
    return t.points, t.name


@st.cache_data(show_spinner=False)
def _samples() -> dict[str, str]:
    out = {}
    for p in sorted(SAMPLES_DIR.glob("*.gpx")):
        _, name = _parse(p.read_bytes(), p.name)
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
    n = name.lower()
    if any(w in n for w in ("car", "drive", "driving", "voiture", "road trip")):
        return "Driving"
    if any(w in n for w in ("walk", "hike", "run", "rando", "trail", "marche")):
        return "Walking"
    return "Cycling"


def _zoom(lat: pd.Series, lon: pd.Series) -> float:
    extent = max(lat.max() - lat.min(), (lon.max() - lon.min()) * np.cos(np.radians(lat.mean())), 0.005)
    return float(np.clip(np.log2(360 / extent) - 1.3, 3, 14))


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

    mode = st.segmented_control("Track", ["Sample track", "Draw on the map", "Upload a file"], default="Sample track")
    if mode == "Draw on the map":
        built = _draw_track()
        if built is None:
            return
        points, name = built
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
                    points, name = _parse(up.getvalue(), up.name)
                else:
                    samples = _samples()
                    if not samples:
                        st.warning("No sample track found in the samples/ folder.")
                        return
                    names = list(samples)
                    default = next((i for i, n in enumerate(names) if n.startswith("Paris")), 0)
                    name = st.selectbox("Sample", names, index=default)
                    points, name = _parse(open(samples[name], "rb").read(), samples[name])
            except ValueError as exc:
                st.error(f"Could not read this track: {exc}")
                return
    track = Track(name=name, points=points)

    with right:
        if track.has_time:
            timing = st.segmented_control("Timing", ["Recorded times", "Plan a new trip"], default="Recorded times")
        else:
            timing = "Plan a new trip"
            st.caption("This track has no timestamps: choose when you set off and how fast you go.")
        if timing == "Plan a new trip":
            c1, c2 = st.columns(2)
            today = date.today()
            day = c1.date_input("Departure date", today + timedelta(days=1),
                                max_value=today + timedelta(days=FORECAST_FUTURE_DAYS),
                                help="Up to 16 days ahead (forecast). Past dates show the weather that actually happened.")
            hour = c2.time_input("Departure time", time(9, 0), step=900)
            c3, c4, c5 = st.columns([1.2, 1, 1.2])
            options = list(ACTIVITIES)
            activity = c3.selectbox("Activity", options, index=options.index(_guess_activity(track.name)))
            preset = ACTIVITIES[activity]
            speed = c4.number_input("Speed (km/h)", 1.0, 150.0, float(preset["speed"]), step=0.5,
                                    help="Average speed on the flat; time is added for climbs when walking or cycling.")
            tz = c5.selectbox("Time zone", TIMEZONES)
            start = pd.Timestamp(datetime.combine(day, hour), tz=tz)
            track = with_times(track, passage_times(track, start, speed, preset["climb_min_per_100m"]))

    if mode == "Upload a file":
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
        return

    local = pts["time"].dt.tz_convert(tz)
    multi_day = local.dt.date.nunique() > 1
    pts = pts.assign(local=local, local_str=local.dt.strftime("%a %d %b %H:%M" if multi_day else "%H:%M"))

    st.subheader("The reading")
    k = st.columns(4)
    k[0].metric("Temperature", f"{stats['temp_min']:.0f} to {stats['temp_max']:.0f} °C", border=True,
                help=f"Feels like down to {stats['feels_min']:.0f} °C.")
    rain_txt = "< 0.1 mm" if 0 < stats["rain_mm"] < 0.1 else f"{stats['rain_mm']:.1f} mm"
    k[1].metric("Rain on the way", rain_txt, border=True,
                help=f"On {stats['rain_share']:.0%} of the route (hours with ≥ 0.2 mm/h at the time you pass).")
    k[2].metric("Strongest gust", f"{stats['gust_max']:.0f} km/h", border=True)
    k[3].metric("Headwind share", f"{stats['headwind_share']:.0%}", border=True,
                help=f"Wind component against you above 5 km/h. Tailwind on {stats['tailwind_share']:.0%} of the route.")
    for line in insights:
        st.markdown(f"<div class='finding'>{line}</div>", unsafe_allow_html=True)

    _map(track, pts, tz)
    _profile(track, pts)
    _table(pts)
    _station_panel(pts, tz)

    label = ("Open-Meteo forecast API (best available models for the area, e.g. Météo-France AROME/ARPEGE over France)"
             if source == "forecast" else "ERA5 reanalysis archive via the Open-Meteo API (what the weather actually was)")
    st.markdown(f"<div class='source'>Weather: {label}. Times in {tz}. Temperatures are adjusted to the altitude "
                f"of each point (90 m terrain model).</div>", unsafe_allow_html=True)


def _draw_track() -> tuple[pd.DataFrame, str] | None:
    """Map where the user clicks waypoints; returns the resulting track (points, name)."""
    c1, c2, c3 = st.columns([2, 1.1, 1.2], vertical_alignment="bottom")
    place = c1.text_input("Go to a place", placeholder="Town, mountain, address…", key="draw_place")
    activity = c2.selectbox("Route for", list(ACTIVITIES), index=1, key="draw_activity")
    follow = c3.toggle("Follow roads and paths", value=True, key="draw_follow",
                       help="Snap the line to roads and paths suited to the activity (OpenStreetMap routing). "
                            "Off: straight lines between your points, e.g. off-trail.")
    center, zoom = (46.6, 2.6), 6
    if place:
        hits = _geocode(place)
        if hits:
            center, zoom = (hits[0]["lat"], hits[0]["lon"]), 12
            c1.caption(f"Centred on {hits[0]['label']}")
        else:
            c1.caption("Place not found.")
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
    choice = st.segmented_control("Colour the route by", ["Temperature", "Rain", "Headwind"], default="Temperature")
    choice = choice or "Temperature"
    col, scale, unit = {"Temperature": ("temperature_2m", HEAT, "°C"), "Rain": ("precipitation", RAIN, "mm/h"),
                        "Headwind": ("headwind", DIVERGING, "km/h")}[choice]
    vals = pts[col].fillna(0)
    marker = dict(size=13, color=vals, colorscale=scale, colorbar=dict(title=dict(text=unit, side="top"), thickness=12, len=0.6))
    if choice == "Headwind":
        lim = max(abs(vals).max(), 5)
        marker.update(cmin=-lim, cmax=lim)
    elif choice == "Rain":
        marker.update(cmin=0, cmax=max(vals.max(), 2))
    full = track.points
    fig = go.Figure()
    fig.add_trace(go.Scattermap(lat=full["lat"], lon=full["lon"], mode="lines", line=dict(width=3, color=INK2),
                                hoverinfo="skip", name="Track"))
    custom = np.stack([pts["dist_km"], pts["local_str"], pts["icon"] + " " + pts["condition"], pts["temperature_2m"],
                       pts["precipitation"].fillna(0), pts["wind_from"], pts["wind_speed_10m"], pts["headwind"]], axis=1)
    fig.add_trace(go.Scattermap(
        lat=pts["lat"], lon=pts["lon"], mode="markers", marker=marker, customdata=custom, name="Weather",
        hovertemplate="<b>km %{customdata[0]:.1f}</b> · %{customdata[1]}<br>%{customdata[2]}<br>"
                      "%{customdata[3]:.1f} °C · rain %{customdata[4]:.1f} mm/h<br>"
                      "wind from %{customdata[5]} %{customdata[6]:.0f} km/h (headwind %{customdata[7]:+.0f})<extra></extra>"))
    ends = pts.iloc[[0, -1]]
    fig.add_trace(go.Scattermap(lat=ends["lat"], lon=ends["lon"], mode="markers+text", text=["Start", "Finish"],
                                textposition="top right", marker=dict(size=9, color=INK), hoverinfo="skip",
                                textfont=dict(color=INK, size=13)))
    fig.update_layout(map=dict(style="carto-positron", zoom=_zoom(full["lat"], full["lon"]),
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
            "gusts_kmh": st.column_config.NumberColumn("Gusts", format="%d km/h")})
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
