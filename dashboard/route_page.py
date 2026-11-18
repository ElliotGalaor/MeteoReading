"""Dashboard page: weather reading along a GPS track (sample, uploaded, or drawn on a map)."""
from __future__ import annotations

import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time, timedelta

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from folium.plugins import Draw
from plotly.subplots import make_subplots
from streamlit_folium import st_folium

from meteo.departure import DARK_SLACK_H, best, best_window, plan_track, sweep, wind_aware
from meteo.ensemble import HEAVY_RAIN
from meteo.ensemble import fetch as fetch_ensemble
from meteo.ensemble import spread as ensemble_spread
from meteo.ensemble import weight as ens_weight
from meteo.geotrek import INSTANCES, PASS_BAND_KM, fetch_route_file, nearby_treks
from meteo.gps import (
    ACTIVITIES,
    Track,
    haversine_km,
    main_track,
    moving_hours,
    parse_track,
    read_tracks,
    sample_points,
    summary,
    to_gpx,
)
from meteo.links import tracks_from_link
from meteo.osm_routes import nearby_relations, relation_track
from meteo.route_scores import (
    AQI_MODERATE,
    DANGER_LEVELS,
    OZONE_ALERT,
    OZONE_INFO,
    PM10_ALERT,
    PM10_INFO,
    PROFILES,
    Assessment,
    Score,
    assess,
    is_past,
    level,
)
from meteo.route_timing import moving_hours as timing_hours
from meteo.route_weather import (
    FORECAST_FUTURE_DAYS,
    POLLEN,
    api_weight,
    clock_format,
    clock_text,
    localize,
    nearby_station_obs,
    read_route,
    summarize,
    track_timezone,
    whole,
)
from meteo.routing import geocode, track_from_waypoints, waypoints_from_drawings
from meteo.vigilance import OFFICIAL_PAGE, at_points, departments
from meteo.vigilance import api_key as vigilance_key
from meteo.vigilance import covered_until as vigilance_until
from meteo.vigilance import fetch as fetch_vigilance
from meteo.vigilance import parse as parse_vigilance
from share import from_link, publish, restore
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
    csv_bytes,
    date_text,
    fmt_date,
    is_dark,
    is_narrow,
    is_phone,
    lang,
    main_width,
    num,
    number,
    page_header,
    pct,
    plural,
    styled,
    t,
    themed,
)

SAMPLES_DIR = ROOT / "samples"
TIMEZONES = ["Europe/Paris", "UTC", "Europe/London", "Europe/Madrid", "Europe/Rome", "Europe/Berlin",
             "America/New_York", "America/Los_Angeles", "Asia/Tokyo"]


MAX_UPLOAD_MB = 20   # checked here, under the server's limit (.streamlit/config.toml), to say why a file is refused


@st.cache_data(show_spinner=False)
def _parse(data: bytes, filename: str) -> list[tuple[pd.DataFrame, str, str | None, bool]]:
    """The tracks of a file: (points, name, why its timestamps are not used, points out of order)."""
    return [(tr.points, tr.name, tr.time_issue, tr.looks_scrambled) for tr in read_tracks(data, filename)]


def _pick(tracks: list[tuple], key: str) -> tuple[pd.DataFrame, str, str | None]:
    """The only track of a file, or the one picked when it holds several (default: the longest in order)."""
    if len(tracks) == 1:
        return tracks[0][:3]
    objs = [Track(name=n, points=p) for p, n, _, _ in tracks]
    default = objs.index(main_track(objs))

    def label(i: int) -> str:
        p, n, _, scrambled = tracks[i]
        return f"{n} · {num(p['dist_km'].iloc[-1], 1)} km" + (t("route.out_of_order") if scrambled else "")
    i = st.selectbox(t("route.pick_track", n=len(tracks)), range(len(tracks)), index=default, format_func=label,
                     key=key, help=t("route.pick_track_help"))
    return tracks[i][:3]


@st.cache_data(show_spinner=False, ttl=86400)
def _timezone(lat: float, lon: float) -> str | None:
    return track_timezone(lat, lon)


PAGE = 10                              # routes listed at a time
NEARBY, LINK = "Nearby route", "From a link"
MODES = ["Sample track", "Draw on the map", "Upload a file", LINK]   # codes; shown with t("route.mode_…")
MODE_KEYS = {"Sample track": "sample", "Draw on the map": "draw", "Upload a file": "upload", LINK: "link",
             NEARBY: "nearby"}
SHARED_MODES = {"sample": "Sample track", "link": LINK}   # the track choices a link can carry


@st.cache_data(show_spinner=False, ttl=6 * 3600, max_entries=100)
def _nearby(lat: tuple, lon: tuple, radius_km: float) -> tuple[list[dict], list[str]]:
    """Geotrek and OpenStreetMap routes near a track, queried in parallel (cached 6 hours: routes rarely change)."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        geotrek = pool.submit(nearby_treks, lat, lon, radius_km=radius_km)
        osm = pool.submit(nearby_relations, lat, lon, radius_km=radius_km)
        (found, failed), (more, failed_osm) = geotrek.result(), osm.result()
    rows = [vars(s) | {"activity": s.activity, "nearness_km": s.nearness_km} for s in found + more]
    return rows, failed + failed_osm


@st.cache_data(show_spinner=False, ttl=6 * 3600, max_entries=50)
def _geotrek_points(url: str) -> pd.DataFrame:
    return parse_track(*fetch_route_file(url)).points


@st.cache_data(show_spinner=False, ttl=6 * 3600, max_entries=50)
def _osm_points(relation_id: int, name: str) -> pd.DataFrame:
    return relation_track(relation_id, name).points


def _route_points(route: dict) -> pd.DataFrame:
    if route["provider"] == "OpenStreetMap":
        return _osm_points(route["trek_id"], route["name"])
    return _geotrek_points(route["gpx_url"])


@st.cache_data(show_spinner=False, ttl=3600, max_entries=30)
def _link_tracks(url: str) -> tuple[list[tuple[pd.DataFrame, str, str | None, bool]], str | None]:
    """The tracks behind a link, and the activity its site gives (a Geotrek practice), if any."""
    tracks = tracks_from_link(url)
    return [(tr.points, tr.name, tr.time_issue, tr.looks_scrambled) for tr in tracks], tracks[0].activity


def _load_route(route: dict) -> None:
    """Replaces the track by a nearby route (callback of the "Read the weather" buttons)."""
    try:
        points = _route_points(route)
    except (ValueError, RuntimeError) as exc:
        st.session_state["nearby_error"] = t("route.load_error", name=route["name"], error=exc)
        return
    if st.session_state.get("route_mode") != NEARBY:
        st.session_state["route_prev_mode"] = st.session_state.get("route_mode", "Sample track")
        # The link box is not drawn while a nearby route is read, so Streamlit forgets it: kept here for "Back" (#76)
        if st.session_state.get("route_mode") == LINK:
            st.session_state["route_prev_link"] = st.session_state.get("route_link", "")
        else:
            st.session_state.pop("route_prev_link", None)
    st.session_state["route_geotrek"] = route | {"points": points}
    st.session_state["route_mode"] = NEARBY          # the Track control shows what is being read
    st.session_state["route_scroll_top"] = True       # and the page goes back up to the reading
    st.session_state.pop("nearby_error", None)
    st.session_state.pop("nearby_selected", None)


def _back_to_track() -> None:
    st.session_state.pop("route_geotrek", None)
    st.session_state["route_mode"] = st.session_state.pop("route_prev_mode", "Sample track")
    if (link := st.session_state.pop("route_prev_link", None)) is not None:
        st.session_state["route_link"] = link


def _on_mode() -> None:
    if st.session_state.get("route_mode") != NEARBY:
        st.session_state.pop("route_geotrek", None)


@st.cache_data(show_spinner=False)
def _samples() -> dict[str, str]:
    out = {}
    for p in sorted(SAMPLES_DIR.glob("*.gpx")):
        name = _parse(p.read_bytes(), p.name)[0][1]
        out[name] = str(p)
    return out


@st.cache_data(show_spinner=False, ttl=1800)
def _reading(points: pd.DataFrame, name: str):
    """Cached for 30 minutes: changing a widget and coming back does not call the API again. The reading keeps the
    forecast fetched, so the same trip can be read at other times (wind-aware pace, other departures) for free."""
    return read_route(Track(name=name, points=points))


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


WALKING_WORDS = {"walk", "walking", "hike", "hiking", "run", "running", "trail", "rando", "randonnee", "marche",
                 "pied", "pedestre", "pieton", "sentier", "balade", "promenade", "gr", "grp", "pr", "trek", "raquette"}
CYCLING_WORDS = {"bike", "cycling", "ride", "vtt", "vttae", "velo", "cyclo", "cyclotourisme", "gravel", "veloroute"}
DRIVING_WORDS = {"car", "drive", "driving", "voiture", "roadtrip", "auto"}


def _guess_activity(name: str) -> str:
    """From whole words of the track name, accents ignored ("Carnac coastal ride" is not a car trip, "GR 3 à pied" is a
    walk). Cycling when nothing says otherwise."""
    plain = unicodedata.normalize("NFKD", name.lower()).encode("ascii", "ignore").decode()
    words = set(re.findall(r"[a-z]+", plain))
    if words & DRIVING_WORDS or "road trip" in plain:
        return "Driving"
    if words & WALKING_WORDS:
        return "Walking"
    return "Cycling"


def _zoom(lat: pd.Series, lon: pd.Series, width_px: float, height_px: float) -> float:
    """Zoom at which the track fits in width_px x height_px (web maps: 512 px for 360° at zoom 0, Mercator)."""
    span_x = max(float(lon.max() - lon.min()), 1e-4)                                   # x is linear in longitude
    span_y = max(float(lat.max() - lat.min()), 1e-4) / np.cos(np.radians(lat.mean()))  # y stretched by 1/cos(lat)
    zoom = min(np.log2(width_px * 360 / (512 * span_x)), np.log2(height_px * 360 / (512 * span_y)))
    return float(np.clip(zoom, 3, 14))


def _map_px() -> float:
    """The width the maps fit the track into (room left for the colour bar and the labels): about half the main column
    on a narrow layout, where the map takes the whole column, else 520 px (#81)."""
    if is_narrow():
        width = main_width()
        return max(190.0, 0.55 * width) if width else 190.0
    return 520.0


def _warehouse_last_date() -> date | None:
    try:
        with _connect() as con:
            return con.sql("SELECT max(last_date) FROM dim_station").fetchall()[0][0]
    except Exception:  # no warehouse: the page still works with the API only
        return None


# =============================================================================
def page_route():
    page_header(t("route.title"), t("route.lead"), tab=t("app.nav_route"))

    if st.session_state.pop("route_scroll_top", False):   # after loading a nearby route, back to the reading
        st.html("<script>(document.querySelector('[data-testid=\"stMain\"]') || document.scrollingElement)"
                ".scrollTo({top: 0, behavior: 'smooth'}); window.scrollTo(0, 0);</script>", unsafe_allow_javascript=True)
    geo = st.session_state.get("route_geotrek")   # a route picked in "Routes nearby", in place of the track
    if st.session_state.get("route_mode") == NEARBY and not geo:
        st.session_state["route_mode"] = "Sample track"
    # A shared link: a sample or a public route link (never an uploaded or drawn track), and the plan
    shared = from_link("track")
    if shared in SHARED_MODES:
        st.session_state.pop("route_geotrek", None)
        geo = None
        st.session_state["route_mode"] = SHARED_MODES[shared]
    restore("route_link", "url")
    if (act := from_link("activity")) in ACTIVITIES:
        st.session_state["route_link_activity"] = act
    today = date.today()
    restore("route_day", "date", parse=date.fromisoformat, valid=lambda v: v <= today + timedelta(days=FORECAST_FUTURE_DAYS))
    restore("route_hour", "time", parse=time.fromisoformat)
    if (sp := from_link("speed")) is not None:
        st.session_state["route_link_speed"] = sp
    link_activity = None   # the activity a linked route's site gives
    mode = st.segmented_control(t("route.track"), MODES + ([NEARBY] if geo else []), default="Sample track",
                                required=True, key="route_mode", on_change=_on_mode,
                                format_func=lambda m: t(f"route.mode_{MODE_KEYS[m]}"))
    if mode == NEARBY and geo:
        points, name, time_issue = geo["points"], geo["name"], None
        left, right = st.columns([1.3, 1], gap="large")
        with left:
            st.markdown(f"**{name}**")
            link = t("route.its_page", url=geo["page_url"]) if geo.get("page_url") else ""
            via = "OpenStreetMap" if geo["provider"] == "OpenStreetMap" else t("route.via_geotrek", source=geo["source"])
            st.caption(t("route.from_source", practice=geo["practice"] or t("route.route"), via=via, link=link))
            st.button(t("route.back"), icon=":material/undo:", on_click=_back_to_track)
    elif mode == LINK:
        left, right = st.columns([1.3, 1], gap="large")
        with left:
            url = st.text_input(t("route.link_label"), key="route_link",
                                placeholder="https://rando.ecrins-parcnational.fr/trek/995106", help=t("route.link_help"))
            if not url.strip():
                st.info(t("route.link_info"))
                publish({"track": "link"})   # no route shown: the address no longer names one
                return
            try:
                found, link_activity = _link_tracks(url.strip())
                points, name, time_issue = _pick(found, f"pick_link_{abs(hash(url.strip()))}")
            except (ValueError, RuntimeError) as exc:
                st.error(t("route.link_error", error=exc))
                return
            st.markdown(f"**{name}**")
    elif mode == "Draw on the map":
        built = _draw_track()
        if built is None:
            return
        points, name = built
        time_issue = None
        left, right = st.columns([1.3, 1], gap="large")
        with left:
            st.markdown(f"**{name}**")
            st.download_button(t("route.save_gpx"), to_gpx(Track(name=name, points=points)).encode("utf-8"),
                               "route.gpx", "application/gpx+xml", icon=":material/download:",
                               help=t("route.save_gpx_help"))
    else:
        left, right = st.columns([1.3, 1], gap="large")
        with left:
            try:
                if mode == "Upload a file":
                    up = st.file_uploader(t("route.upload_label"), type=["gpx", "kml", "geojson", "json"],
                                          help=t("route.upload_help", mb=MAX_UPLOAD_MB))
                    if up is None:
                        st.info(t("route.upload_info"))
                        return
                    if up.size > MAX_UPLOAD_MB * 1024 * 1024:
                        st.error(t("route.upload_too_big", name=up.name, size=num(up.size / 1024 / 1024, 1),
                                   mb=MAX_UPLOAD_MB))
                        return
                    points, name, time_issue = _pick(_parse(up.getvalue(), up.name), f"pick_{up.name}_{up.size}")
                else:
                    samples = _samples()
                    if not samples:
                        st.warning(t("route.no_samples"))
                        return
                    names = list(samples)
                    default = next((n for n in names if n.startswith("Paris")), names[0])
                    restore("route_sample", "sample", default, valid=lambda v: v in names)
                    name = st.selectbox(t("route.sample"), names, key="route_sample")
                    points, name, time_issue = _pick(_parse(open(samples[name], "rb").read(), samples[name]),
                                                     f"pick_{name}")
            except ValueError as exc:
                st.error(t("route.track_error", error=exc))
                return
    track = Track(name=name, points=points)
    options = list(ACTIVITIES)
    if mode == NEARBY and geo:
        guessed = geo["activity"]
    elif mode == "Draw on the map":
        guessed = st.session_state.get("draw_activity", "Cycling")
    elif mode == LINK and link_activity:
        guessed = link_activity
    else:
        guessed = _guess_activity(track.name)
    guessed = st.session_state.get("route_link_activity", guessed)

    plan = None   # a planned trip: (base track without times, departure, flat speed, hours per day)
    with right:
        if track.has_time:
            timing = st.segmented_control(t("route.timing"), ["Recorded times", "Plan a new trip"],
                                          default="Recorded times", required=True, key="route_timing",
                                          format_func=lambda v: t("route.recorded" if v == "Recorded times"
                                                                  else "route.plan"))
        else:
            timing = "Plan a new trip"
            why = (t("route.times_unusable", why=t(f"route.time_issue_{time_issue}")) if time_issue
                   else t("route.no_times"))
            st.caption(t("route.set_off", why=why[:1].upper() + why[1:]))
        if timing != "Plan a new trip":
            if track.points["time"].max().date() > date.today() + timedelta(days=FORECAST_FUTURE_DAYS):
                st.warning(t("route.times_too_far", days=FORECAST_FUTURE_DAYS + 1))
                return
            activity = st.selectbox(t("seasons.activity"), options, index=options.index(guessed),
                                    key=f"route_activity_rec_{abs(hash(track.name))}",
                                    format_func=lambda a: t(f"activity.{a}"), help=t("route.activity_help"))
        else:
            c1, c2 = st.columns(2)
            today = date.today()
            st.session_state.setdefault("route_day", today + timedelta(days=1))
            st.session_state.setdefault("route_hour", time(9, 0))
            if st.session_state["route_day"] > today + timedelta(days=FORECAST_FUTURE_DAYS):
                st.session_state["route_day"] = today + timedelta(days=FORECAST_FUTURE_DAYS)
            day = c1.date_input(t("route.departure_date"), max_value=today + timedelta(days=FORECAST_FUTURE_DAYS),
                                key="route_day", format="DD/MM/YYYY", help=t("route.departure_date_help"))
            hour = c2.time_input(t("route.departure_time"), step=900, key="route_hour")
            c3, c4, c5 = st.columns([1.2, 1, 1.2])
            activity = c3.selectbox(t("seasons.activity"), options, index=options.index(guessed),
                                    key=f"route_activity_plan_{abs(hash(track.name))}",
                                    format_func=lambda a: t(f"activity.{a}"), help=t("route.activity_help_plan"))
            preset = ACTIVITIES[activity]
            speed_key = f"route_speed_{activity}"
            if (sp := st.session_state.pop("route_link_speed", None)) is not None:
                try:
                    if 1 <= float(sp) <= 150:
                        st.session_state[speed_key] = float(sp)
                except ValueError:
                    pass
            st.session_state.setdefault(speed_key, float(preset["speed"]))
            speed = c4.number_input(t("route.speed"), 1.0, 150.0, step=0.5, key=speed_key, format="%g",
                                    help=t("route.speed_help"))
            # Default: the time zone where the track starts (a 09:00 departure in Tokyo is 09:00 Tokyo time)
            first = track.points.iloc[0]
            local_tz = _timezone(round(float(first["lat"]), 2), round(float(first["lon"]), 2)) or "Europe/Paris"
            zones = TIMEZONES if local_tz in TIMEZONES else [local_tz, *TIMEZONES]
            tz = c5.selectbox(t("route.timezone"), zones, index=zones.index(local_tz), help=t("route.timezone_help"),
                              key=f"route_tz_{abs(hash(track.name))}")
            naive = pd.Timestamp(datetime.combine(day, hour))
            start = naive.tz_localize(tz, nonexistent="shift_forward", ambiguous=True)
            if start.tz_localize(None) != naive:
                st.info(t("route.clocks", hour=f"{hour:%H:%M}", day=fmt_date(day, "day_month_long"), tz=tz,
                          start=f"{start:%H:%M}"))
            # Long routes: split into days, each starting at the departure time (the stages of a multi-day route)
            hours_per_day = None
            total_h = float(timing_hours(track, activity, speed)[-1])
            if total_h > 10:
                if st.toggle(t("route.split", hours=num(total_h, 0)), value=True, key="route_split",
                             help=t("route.split_help")):
                    default_h = {"Walking": 6.0, "Cycling": 5.0, "Driving": 8.0}[activity]
                    hours_per_day = st.slider(t("route.hours_per_day"), 2.0, 12.0, default_h, 0.5,
                                              key="route_hours_per_day")
                    days = int(np.ceil(total_h / hours_per_day - 1e-9))
                    st.caption(t("route.split_caption", days=days, hour=f"{hour:%H:%M}")
                               + (t("route.split_beyond") if days > 16 else ""))
            plan = (track, start, speed, hours_per_day)
            track = plan_track(track, start, activity, speed, hours_per_day)

    _share_route(mode, geo, activity, plan)
    if mode == "Upload a file":
        st.caption(f"**{track.name}**")
    try:
        with st.spinner(t("route.reading_spinner")):
            reading = _reading(track.points, track.name)
    except (ValueError, RuntimeError) as exc:
        _track_kpis(track, activity, plan)
        st.error(str(exc))
        _nearby_panel(track, activity)
        return
    still = track
    if plan is not None:   # on a bike, the pace with the forecast wind (same forecast, read again)
        reading, track = wind_aware(reading, track, plan[1], activity, plan[2], plan[3])
    _track_kpis(track, activity, plan, still)

    # The words of the reading in the visitor's language (the reading itself may come from the cache)
    pts, source, tz = localize(reading.points), reading.source, reading.timezone
    stats, insights = summarize(pts, tz)
    vig = _vigilance(pts, source)
    if vig["warnings"] is not None:
        pts = at_points(pts, vig["deps"], vig["warnings"])
    local = pts["time"].dt.tz_convert(tz)
    scores = assess(pts, activity, tz)
    clock = clock_format(local).replace("%a", "%a %d %b")
    pts = pts.assign(local=local, local_str=[clock_text(ts, clock) for ts in local],
                     danger=scores.points["danger"].to_numpy(), comfort=scores.points["comfort"].to_numpy())

    st.subheader(t("route.reading"))
    k = st.columns(4)
    k[0].metric(t("route.kpi_temp"), t("route.kpi_temp_value", low=whole(stats["temp_min"]), high=whole(stats["temp_max"])),
                border=True, help=t("route.kpi_temp_help", feels=whole(stats["feels_min"])))
    rain_txt = f"< {num(0.1, 1)} mm" if 0 < stats["rain_mm"] < 0.1 else f"{num(stats['rain_mm'], 1)} mm"
    k[1].metric(t("route.kpi_rain"), rain_txt, border=True, help=t("route.kpi_rain_help", share=pct(stats["rain_share"])))
    k[2].metric(t("route.kpi_gust"), f"{num(stats['gust_max'], 0)} km/h", border=True)
    k[3].metric(t("route.kpi_headwind"), pct(stats["headwind_share"]), border=True,
                help=t("route.kpi_headwind_help", share=pct(stats["tailwind_share"])))
    _vigilance_notes(pts, vig)
    for line in insights:
        st.markdown(f"<div class='finding'>{line}</div>", unsafe_allow_html=True)
    if pts["precip_15min"].any():
        st.caption(t("route.minutely"))

    ens = None
    if source == "forecast":
        n_days = (pts["time"].max().date() - pts["time"].min().date()).days + 1
        if st.toggle(t("route.ensemble"), key="route_ensemble",
                     help=t("route.ensemble_help", calls=num(ens_weight(len(pts), n_days), 0))):
            try:
                with st.spinner(t("route.ensemble_spinner")):
                    ens = _ensemble(pts[["dist_km", "time", "lat", "lon"]], PROFILES[activity].gusts[0])
            except (ValueError, RuntimeError, KeyError) as exc:
                st.caption(t("route.ensemble_error", error=exc))

    _scores_panel(scores, pts, source, ens)
    if plan is not None and source == "forecast":
        _when_to_leave(plan, activity, tz, still.points, track.name, vig, scores)
    st.subheader(t("route.along"))
    _map(track, pts, tz)
    _profile(track, pts, ens)
    _scores_chart(scores, pts)
    _table(pts, ens)
    _station_panel(pts, tz)

    label = t("route.source_forecast" if source == "forecast" else "route.source_archive")
    extra = t("route.source_air") if pts["european_aqi"].notna().any() else ""
    st.markdown(f"<div class='source'>{t('route.source', label=label, extra=extra, tz=tz)}</div>", unsafe_allow_html=True)
    _nearby_panel(track, activity)


def _share_route(mode: str, geo: dict | None, activity: str, plan) -> None:
    """The address of the page: a sample or a public route link, and the plan (an uploaded or drawn track is never
    put in it)."""
    if mode == "Sample track":
        values = {"track": "sample", "sample": st.session_state.get("route_sample")}
    elif mode == LINK:
        values = {"track": "link", "url": st.session_state.get("route_link", "").strip() or None}
    elif mode == NEARBY and geo and geo.get("page_url"):
        values = {"track": "link", "url": geo["page_url"]}
    else:
        publish({})
        return
    values["activity"] = activity
    if plan is not None:
        values.update(date=st.session_state["route_day"].isoformat(), time=f"{st.session_state['route_hour']:%H:%M}",
                      speed=f"{plan[2]:g}")
    publish(values)


def _fmt_hours(h: float) -> str:
    return (t("route.days_hours", d=int(h // 24), h=int(h % 24)) if h >= 24
            else f"{int(h)} h {int(round(h % 1 * 60)) % 60:02d}")


def _track_kpis(track: Track, activity: str, plan, still: Track | None = None) -> None:
    """Distance, ascent, duration (with the other estimates of a planned trip) and weather points."""
    s = summary(track)
    k = st.columns(4)
    k[0].metric(t("route.distance"), f"{num(s['distance_km'], 1)} km", border=True)
    k[1].metric(t("route.ascent"), f"{num(s['ascent_m'], 0)} m" if s["ascent_m"] is not None else "–", border=True)
    dur = s["duration_h"]   # departure to arrival, nights included on a multi-day plan
    help_ = None
    if plan is not None:
        base, _, speed, _ = plan
        moving = float(timing_hours(track if still is None else still, activity, speed)[-1])
        old = float(moving_hours(base, speed, ACTIVITIES[activity]["climb_min_per_100m"])[-1])
        lines = [t("route.moving", time=_fmt_hours(moving))]
        if still is not None and track is not still:
            wind_h = float((track.points["time"].iloc[-1] - track.points["time"].iloc[0]).total_seconds() / 3600)
            still_h = float((still.points["time"].iloc[-1] - still.points["time"].iloc[0]).total_seconds() / 3600)
            lines.append(t("route.with_wind", wind=_fmt_hours(wind_h), still=_fmt_hours(still_h)))
        lines.append(t("route.old_estimate", time=_fmt_hours(old)))
        help_ = t("route.duration_help_bike" if activity == "Cycling" else "route.duration_help", lines="; ".join(lines))
    k[2].metric(t("route.duration"), _fmt_hours(dur), border=True, help=help_)
    st.markdown(f"<div class='source'><a href='#routes-nearby'>{t('route.nearby_link')}</a></div>",
                unsafe_allow_html=True)
    n_points = len(sample_points(track))
    k[3].metric(t("route.points"), f"{n_points}", border=True,
                help=t("route.points_help", calls=num(api_weight(n_points, 3), 0)))
    if still is not None and track is not still:
        moving_wind = float((track.points["time"].iloc[-1] - track.points["time"].iloc[0]).total_seconds() / 3600)
        moving_still = float((still.points["time"].iloc[-1] - still.points["time"].iloc[0]).total_seconds() / 3600)
        if abs(moving_wind - moving_still) >= 0.1:
            st.caption(t("route.bike_wind", wind=_fmt_hours(moving_wind), still=_fmt_hours(moving_still)))


@st.cache_data(show_spinner=False, ttl=1800)
def _warnings() -> tuple[pd.DataFrame | None, str | None]:
    """Current Météo-France vigilance (cached 30 minutes), or why there is none."""
    key = vigilance_key()
    if not key:
        return None, "not connected"
    try:
        return parse_vigilance(fetch_vigilance(key)), None
    except RuntimeError as exc:
        return None, str(exc)


@st.cache_data(show_spinner=False, ttl=86400)
def _departments(points: pd.DataFrame) -> list[dict]:
    return departments(points)


def _vigilance(pts: pd.DataFrame, source: str) -> dict:
    """The warnings that may apply to a forecast reading in France: {warnings, deps, status, until}."""
    out = {"warnings": None, "deps": [], "status": None, "until": None}
    if source != "forecast":
        return out
    warnings, status = _warnings()
    out["status"] = status
    if warnings is None:
        return out
    out["deps"] = _departments(pts[["dist_km", "lat", "lon"]].round(4))
    out["warnings"], out["until"] = warnings, vigilance_until(warnings)
    return out


def _vigilance_notes(pts: pd.DataFrame, vig: dict) -> None:
    link = f"[vigilance.meteofrance.fr]({OFFICIAL_PAGE})"
    if vig["status"] == "not connected":
        # No Météo-France key on this server (#73): say so rather than nothing, for a route in France or Corsica
        if (pts["lat"].between(41.3, 51.1) & pts["lon"].between(-5.2, 9.6)).any():
            st.caption(t("route.vig_off", link=link))
        return
    if vig["warnings"] is None and vig["status"] is None:
        return
    if vig["warnings"] is None:
        st.caption(t("route.vig_error", status=vig["status"], link=link))
        return
    if not vig["deps"]:
        return   # outside France: no Météo-France vigilance
    texts = pts.loc[pts["vigilance_color"] >= 2, ["vigilance_color", "vigilance_text"]].drop_duplicates("vigilance_text")
    for colour, text in texts.sort_values("vigilance_color", ascending=False).itertuples(index=False):
        (st.error if colour >= 4 else st.warning if colour == 3 else st.info)(t("route.vig_details", text=text,
                                                                                 link=link), icon=":material/warning:")
    until = vig["until"]
    if until is not None and pts["time"].max() > until:
        st.caption(t("route.vig_until", until=f"{fmt_date(until, 'weekday')} {until:%H:%M}", link=link))
    elif texts.empty:
        st.caption(t("route.vig_green", link=link))


@st.cache_data(show_spinner=False, ttl=1800)
def _ensemble(points: pd.DataFrame, gust_danger: float):
    first, last = points["time"].min().date(), points["time"].max().date()
    frames, idx = fetch_ensemble(points, first, last)
    return ensemble_spread(frames, idx, points, gust_danger)


@st.cache_data(show_spinner=False, ttl=1800, max_entries=20)
def _sweep(points: pd.DataFrame, name: str, activity: str, speed: float, hours_per_day: float | None, tz: str,
           days: tuple, vig_key: str, now: pd.Timestamp):
    """Every departure of the days, read from the forecast fetched for the planned trip (no new API call)."""
    fetched = _reading(points, name)
    warnings, _ = _warnings()
    deps = _departments(fetched.points[["dist_km", "lat", "lon"]].round(4)) if vig_key and warnings is not None else []

    def decorate(df):
        return at_points(df, deps, warnings) if deps else df
    return sweep(fetched, Track(name=name, points=points), list(days), tz, activity, speed, hours_per_day,
                 decorate=decorate, now=now)


def _use_departure(when: pd.Timestamp) -> None:
    st.session_state["route_day"] = when.date()
    st.session_state["route_hour"] = when.time().replace(second=0, microsecond=0)


def _when_to_leave(plan, activity: str, tz: str, points: pd.DataFrame, name: str, vig: dict, scores: Assessment) -> None:
    """Danger and comfort for every departure from 05:00 to 20:00, from the forecast already fetched."""
    _, start, speed, hours_per_day = plan
    st.markdown(f"<div class='chart-title'>{t('route.leave_title')}</div>", unsafe_allow_html=True)
    last_day = date.today() + timedelta(days=FORECAST_FUTURE_DAYS)
    day0 = start.tz_convert(tz).date()
    choices = [d for d in (day0, day0 + timedelta(days=1)) if d <= last_day]
    pick = st.segmented_control(t("route.leave_day"), choices, default=day0, required=True, key=f"leave_day_{day0}",
                                format_func=lambda d: fmt_date(d, "weekday")) or day0
    now = pd.Timestamp.now(tz="UTC").ceil("30min")   # the next half hour: never a departure already past (#77)
    with st.spinner(t("route.leave_spinner")):
        sw = _sweep(points, name, activity, speed, hours_per_day, tz, (pick,),
                    "on" if vig["warnings"] is not None else "", now)
    if sw.empty:
        st.caption(t("route.leave_none"))
        return
    best_row = best(sw)
    a, b = best_window(sw, best_row)
    dep = sw["departure"].dt.tz_convert(tz)
    fig = go.Figure()
    fig.add_vrect(x0=a.tz_convert(tz) - pd.Timedelta(minutes=10), x1=b.tz_convert(tz) + pd.Timedelta(minutes=10),
                  fillcolor="rgba(195,194,183,0.35)", line_width=0, layer="below")
    arrival = [clock_text(ts, "%a %H:%M") for ts in sw["arrival"].dt.tz_convert(tz)]
    custom = np.stack([arrival, [t(f"level.{v}") for v in sw["danger_level"]],
                       [t(f"level.{v}") for v in sw["comfort_level"]], [num(v, 1) for v in sw["rain_mm"]],
                       (sw["headwind_share"] * 100).round(0), [num(v, 1) for v in sw["dark_hours"]]], axis=1)
    fig.add_scatter(x=dep, y=sw["comfort"], mode="lines+markers", name=t("route.comfort"),
                    line=dict(color=SERIES[0], width=2.2), marker=dict(size=7), customdata=custom,
                    hovertemplate=t("route.leave_hover_comfort"))
    fig.add_scatter(x=dep, y=sw["danger"], mode="lines+markers", name=t("route.danger"), line=dict(color=AXIS, width=2),
                    marker=dict(size=8, color=[STATUS[DANGER_STATUS[lv]] for lv in sw["danger_level"]],
                                line=dict(width=1.5, color="#fcfcfb")),
                    customdata=custom, hovertemplate=t("route.leave_hover_danger"))
    fig.add_scatter(x=[best_row["departure"].tz_convert(tz)], y=[best_row["comfort"]], mode="markers",
                    name=t("route.leave_best"), marker=dict(symbol="star", size=16, color=SERIES[0],
                                                            line=dict(width=1.5, color="#fcfcfb")), hoverinfo="skip")
    fig.update_layout(yaxis=dict(range=[0, 104], title=t("route.score")), xaxis=dict(tickformat="%H:%M"),
                      hovermode="x unified")
    window = (f"{a.tz_convert(tz):%H:%M}" if a == b
              else t("route.leave_window", a=f"{a.tz_convert(tz):%H:%M}", b=f"{b.tz_convert(tz):%H:%M}"))
    leave = best_row["departure"].tz_convert(tz)
    chart(fig, 330, t("route.leave_chart", day=fmt_date(pick, "weekday_day"), time=f"{leave:%H:%M}"),
          t("route.leave_sub", window=window))
    if sw["dark_hours"].min() > DARK_SLACK_H:
        st.caption(t("route.leave_dark", hours=num(best_row["dark_hours"], 1)))
    same = best_row["departure"] == start
    if same:
        st.caption(t("route.leave_same"))
    else:
        st.button(t("route.leave_button", time=f"{leave:%H:%M}", day=fmt_date(pick, "weekday"),
                    comfort=best_row["comfort"], danger=best_row["danger"], now_comfort=scores.comfort.value,
                    now_danger=scores.danger.value), on_click=_use_departure, args=(leave,), icon=":material/schedule:")


PRACTICE_GROUPS = [   # Geotrek names practices in French, OpenStreetMap in English: one set of filters for both
    # (codes; shown with t("route.practice_…"))
    ("Mountain bike", ("vtt", "mountain", "enduro")), ("Cycling", ("velo", "cyclo", "gravel", "cycling", "bike")),
    ("Trail running", ("trail",)), ("Snowshoe", ("raquette", "snowshoe")), ("Horse riding", ("cheval", "equestre", "horse")),
    ("Walking", ("pied", "pedestre", "hiking", "walk", "rando", "marche", "accessible")),
]


def practice_group(practice: str) -> str:
    plain = unicodedata.normalize("NFKD", (practice or "").lower()).encode("ascii", "ignore").decode()
    return next((group for group, words in PRACTICE_GROUPS if any(w in plain for w in words)), "Other")


def _duration(hours) -> str:
    if hours is None or pd.isna(hours):
        return ""
    return f"{num(hours, 1).removesuffix('.0').removesuffix(',0')} h" if hours < 24 else plural("common.n_days", round(hours / 24))


def _same_route(route: dict, track: Track) -> bool:
    """The suggestion is the track itself (a Geotrek GPX uploaded, then suggested back)."""
    if route.get("start_lat") is None or not route.get("length_km"):
        return False
    first = track.points.iloc[0]
    start_gap = float(haversine_km(first["lat"], first["lon"], route["start_lat"], route["start_lon"]))
    return start_gap < 0.2 and abs(route["length_km"] - track.distance_km) <= 0.05 * max(track.distance_km, 0.1)


def _select_from_table(key: str, view: list[dict]) -> None:
    rows = st.session_state[key].selection.rows
    if rows:
        st.session_state["nearby_selected"] = (view[rows[0]]["provider"], view[rows[0]]["trek_id"])


def _select_from_map(key: str, view: list[dict]) -> None:
    points = st.session_state[key].selection.points
    if points and "customdata" in points[0]:
        r = view[int(points[0]["customdata"][0])]
        st.session_state["nearby_selected"] = (r["provider"], r["trek_id"])


def _show_more(key: str) -> None:
    st.session_state[key] = st.session_state.get(key, PAGE) + PAGE


def _nearby_map(track: Track, view: list[dict], selected: dict | None, key: str) -> None:
    """The track, the starts of the listed routes (numbered as in the list) and the selected route's line."""
    pts = track.points
    fig = go.Figure()
    fig.add_trace(go.Scattermap(lat=pts["lat"], lon=pts["lon"], mode="lines", line=dict(width=4, color=INK),
                                name=t("route.your_track"), hoverinfo="skip"))
    if selected:
        try:
            line = _route_points(selected)
            fig.add_trace(go.Scattermap(lat=line["lat"], lon=line["lon"], mode="lines", name=selected["name"],
                                        line=dict(width=4, color=SERIES[1]), hoverinfo="skip"))
        except (ValueError, RuntimeError):
            pass
    starts = [(i, r) for i, r in enumerate(view) if r.get("start_lat") is not None]
    if starts:
        bike = [r["activity"] == "Cycling" for _, r in starts]
        fig.add_trace(go.Scattermap(
            lat=[r["start_lat"] for _, r in starts], lon=[r["start_lon"] for _, r in starts], mode="markers+text",
            text=[str(i + 1) for i, _ in starts], textposition="top center", textfont=dict(color=INK, size=12),
            marker=dict(size=13, color=[SERIES[2] if b else SERIES[0] for b in bike]),
            customdata=[[i, r["name"], r["practice"]] for i, r in starts], name=t("route.starts"),
            hovertemplate=t("route.starts_hover")))
    lat = list(pts["lat"]) + [r["start_lat"] for _, r in starts]
    lon = list(pts["lon"]) + [r["start_lon"] for _, r in starts]
    zoom = _zoom(pd.Series(lat), pd.Series(lon), _map_px(), 330)
    fig.update_layout(map=dict(style="carto-positron", zoom=zoom,
                               center=dict(lat=float(np.mean([min(lat), max(lat)])), lon=float(np.mean([min(lon), max(lon)])))),
                      showlegend=False, margin=dict(l=0, r=0, t=0, b=0), height=380)
    st.plotly_chart(themed(fig), theme=None, key=key, on_select=lambda: _select_from_map(key, view), selection_mode="points",
                    config={"displaylogo": False, "displayModeBar": False, "locale": lang()})


def _nearby_panel(track: Track, activity: str) -> None:
    """Routes near the track: Geotrek sites of parks and départements, and waymarked OpenStreetMap routes."""
    st.subheader(t("route.nearby"), anchor="routes-nearby")
    st.caption(t("route.nearby_caption"))
    radii = [round(2 + i / 10, 1) for i in range(481)]   # 2.0 to 50.0 km, by 0.1 km
    if st.session_state.get("nearby_radius") not in radii:
        st.session_state["nearby_radius"] = 10.0
    radius = st.select_slider(t("route.radius"), radii, key="nearby_radius", format_func=lambda v: f"{num(v, 1)} km",
                              help=t("route.radius_help", km=num(PASS_BAND_KM, 0)))
    pts = track.points
    keep = np.linspace(0, len(pts) - 1, min(len(pts), 100)).astype(int)
    lat = tuple(np.round(pts["lat"].to_numpy()[keep], 3).tolist())
    lon = tuple(np.round(pts["lon"].to_numpy()[keep], 3).tolist())
    with st.spinner(t("route.nearby_spinner")):
        rows, failed = _nearby(lat, lon, round(float(radius), 1))
    if error := st.session_state.pop("nearby_error", None):
        st.error(error)
    geo = st.session_state.get("route_geotrek")
    rows = [r for r in rows if not (geo and (r["provider"], r["trek_id"]) == (geo["provider"], geo["trek_id"]))
            and not _same_route(r, track)]
    if not rows:
        st.info(t("route.nearby_none", km=num(radius, 1)))
        with st.expander(t("route.covered_title")):
            st.markdown(t("route.covered", sites=", ".join(i.name for i in INSTANCES)))
        if failed:
            st.caption(t("route.not_reached", sites=", ".join(failed)))
        return

    for r in rows:
        r["group"] = practice_group(r["practice"])
    groups = [g for g, _ in PRACTICE_GROUPS + [("Other", ())] if any(r["group"] == g for r in rows)]
    matching = {"Walking": {"Walking", "Trail running", "Snowshoe"}, "Cycling": {"Cycling", "Mountain bike"},
                "Driving": set()}[activity]
    default = [g for g in groups if g in matching]
    c1, c2 = st.columns([3, 1.2], vertical_alignment="bottom")
    chosen = c1.pills(t("route.practice"), groups, selection_mode="multi", default=default or groups,
                      key=f"nearby_practice_{activity}_{abs(hash(tuple(groups)))}",
                      format_func=lambda g: t(f"route.practice_{g.replace(' ', '_')}"),
                      help=t("route.practice_help", activity=t(f"activity.{activity}").lower()))
    # Only the sorts the routes have data for: OpenStreetMap routes have no start, length or ascent (#83)
    known = {"Nearest": any(r["distance_km"] is not None or r["passes_near"] for r in rows),
             "Shortest": any(r["length_km"] for r in rows), "Longest": any(r["length_km"] for r in rows),
             "Least ascent": any(r["ascent_m"] is not None for r in rows)}
    sorts = [o for o, ok in known.items() if ok]
    if st.session_state.get("nearby_sort") not in sorts:
        st.session_state.pop("nearby_sort", None)
    order = (c2.selectbox(t("route.sort"), sorts, key="nearby_sort",
                          format_func=lambda o: t(f"route.sort_{o.replace(' ', '_')}")) if sorts else None)
    rows = [r for r in rows if r["group"] in (chosen or groups)]
    big = 1e9
    sorters = {"Nearest": lambda r: (r["nearness_km"], r["name"]),
               "Shortest": lambda r: (r["length_km"] if r["length_km"] else big, r["name"]),
               "Longest": lambda r: (-(r["length_km"] or 0), r["name"]),
               "Least ascent": lambda r: (r["ascent_m"] if r["ascent_m"] is not None else big, r["name"])}
    rows.sort(key=sorters[order] if order else (lambda r: r["name"]))
    shown_key = f"nearby_shown_{abs(hash((lat, lon, round(float(radius), 1))))}"
    view = rows[: st.session_state.get(shown_key, PAGE)]
    numbered = any(r.get("start_lat") is not None for r in view)
    st.caption(t("route.showing" if numbered else "route.showing_plain", shown=len(view), total=len(rows)) if rows
               else t("route.no_practice"))
    sel_key = st.session_state.get("nearby_selected")
    selected = next((r for r in view if (r["provider"], r["trek_id"]) == sel_key), None)
    if view:
        _nearby_map(track, view, selected, key=f"nearby_map_{shown_key}_{abs(hash(tuple(r['trek_id'] for r in view)))}")
    if is_narrow():   # cards: a table needs a horizontal scroll in a narrow column
        for i, r in enumerate(view):
            with st.container(border=True):
                st.markdown(f"**{i + 1}. {r['name']}**")
                facts = [r["practice"], r["difficulty"], f"{num(r['length_km'], 1)} km" if r["length_km"] else "",
                         t("route.m_up", m=num(r["ascent_m"], 0)) if r["ascent_m"] is not None else "", _where(r),
                         r["source"]]
                st.caption(" · ".join(f for f in facts if f))
                st.button(t("route.read_it"), key=f"nearby_card_{r['provider']}_{r['trek_id']}",
                          on_click=_load_route, args=(r,), icon=":material/hiking:")
    elif view:
        table = pd.DataFrame([{**r, "rank": i + 1, "where": _where(r), "duration": _duration(r["duration_h"])}
                              for i, r in enumerate(view)])
        key = f"nearby_table_{shown_key}_{abs(hash(tuple(r['trek_id'] for r in view)))}"
        cols = ["rank", "name", "practice", "difficulty", "route_type", "length_km", "ascent_m", "duration", "where",
                "source", "page_url"]
        st.dataframe(
            styled(table[cols], {"length_km": number(1, "km"), "ascent_m": number(0, "m")}),
            hide_index=True, on_select=lambda: _select_from_table(key, view), selection_mode="single-row", key=key,
            placeholder="–",
            column_config={
                "rank": st.column_config.NumberColumn("#", width="small"),
                "name": st.column_config.TextColumn(t("route.col_route"), width="large"),
                "practice": t("route.practice"), "difficulty": t("route.col_difficulty"),
                "route_type": t("route.col_type"), "length_km": t("common.length"), "ascent_m": t("route.ascent"),
                "duration": t("route.duration"), "where": t("route.col_where"), "source": t("route.col_source"),
                "page_url": st.column_config.LinkColumn(t("route.col_page"), display_text=t("route.col_open")),
            })
    if len(rows) > len(view):
        st.button(t("route.show_more", n=min(PAGE, len(rows) - len(view))), on_click=_show_more, args=(shown_key,),
                  icon=":material/expand_more:")
    if selected:
        st.button(t("route.read_selected", name=selected["name"]), type="primary", icon=":material/hiking:",
                  on_click=_load_route, args=(selected,))
    elif view and not is_narrow():
        st.caption(t("route.select_hint"))
    if failed:
        st.caption(t("route.not_reached", sites=", ".join(failed)))


def _where(r: dict) -> str:
    if r["distance_km"] is None:
        return t("route.where_area")
    start = t("route.where_start", km=num(r["distance_km"], 1))
    return t("route.where_passes", start=start) if r["passes_near"] and r["distance_km"] > PASS_BAND_KM else start


DANGER_STATUS = {"Low": "good", "Moderate": "warning", "High": "serious", "Severe": "critical"}
COMFORT_STATUS = {"Very pleasant": "good", "Pleasant": "good", "Mixed": "warning", "Unpleasant": "serious",
                  "Very unpleasant": "critical"}
def _method() -> str:
    """How the scores are computed (the thresholds come from meteo/route_scores.py)."""
    return t("route.method", aqi=AQI_MODERATE, o3a=OZONE_INFO, o3b=OZONE_ALERT, pma=PM10_INFO, pmb=PM10_ALERT)


def _card(title: str, score: Score, status: str, why: str) -> str:
    color = STATUS[status]
    return (f"<div class='score-card'><div class='score-label'>{title}</div>"
            f"<div class='score-value'>{score.value}<small> / 100</small>"
            f"<span class='chip'><i style='background:{color}'></i>{t(f'level.{score.level}')}</span></div>"
            f"<div class='meter'><div style='width:{max(score.value, 2)}%;background:{color}'></div></div>"
            f"<div class='score-why'>{why}</div></div>")


def _scores_panel(scores: Assessment, pts: pd.DataFrame, source: str, ens=None) -> None:
    st.subheader(t("route.scores_title", activity=t(f"activity.{scores.activity}").lower()))
    clock = dict(zip(pts["time"], pts["local_str"], strict=True))
    danger, comfort = scores.danger, scores.comfort
    if danger.factors:
        why = t("route.worst", moments=" · ".join(
            t("route.worst_item", name=t(f"hazard.{f.name}").lower(), value=num(f.value, 0), km=num(f.km, 0),
              time=clock[f.time]) for f in danger.factors[:3]))
    else:
        why = t("route.no_hazard")
    if ens is not None:
        gust, heavy = PROFILES[scores.activity].gusts[0], HEAVY_RAIN
        names = {"gust": t("route.ens_gust", kmh=num(gust, 0)), "heavy": t("route.ens_heavy", mm=num(heavy, 0))}
        shares = [t("route.ens_share", name=names[k], share=pct(share)) for k, share in ens.hazard_shares.items()
                  if np.isfinite(share) and 0.05 <= share < 0.95]
        if shares:
            why += t("route.ens_shares", shares=", ".join(shares))
    c1, c2 = st.columns(2)
    c1.markdown(_card(t("route.danger_score"), danger, DANGER_STATUS[danger.level], why), unsafe_allow_html=True)
    if comfort.factors:
        why = t("route.points_lost", items=" · ".join(f"{t(f'penalty.{f.name}').lower()} −{num(f.value, 0)}"
                                                      for f in comfort.factors[:3]))
    else:
        why = t("route.nothing_spoils")
    c2.markdown(_card(t("route.comfort_score"), comfort, COMFORT_STATUS[comfort.level], why), unsafe_allow_html=True)
    if ens is not None:
        rain = t("route.ens_rain", share=pct(ens.rain_anywhere)) if np.isfinite(ens.rain_anywhere) else ""
        confidence = t(f"route.confidence_{ens.confidence}")
        st.markdown(f"<div class='finding'>{t('route.ens_finding', confidence=confidence, n=ens.n_members, spread=num(ens.temp_spread, 1), rain=rain)}</div>",
                    unsafe_allow_html=True)
    e1, e2 = st.columns(2)
    with e1.expander(t("route.method_title")):
        st.markdown(_method())
    with e2.expander(t("route.times_title")):
        st.markdown(t("route.times_method"))

    st.markdown(f"<div class='chart-title'>{t('route.plan_for')}</div>", unsafe_allow_html=True)
    if is_past(pts):
        st.caption(t("route.past_trip"))
    for a in scores.advice:
        st.markdown(f"<div class='advice {a.kind}'>{a.icon} <b>{a.title}</b><br><span>{a.text}</span></div>",
                    unsafe_allow_html=True)


def _scores_chart(scores: Assessment, pts: pd.DataFrame) -> None:
    sp = scores.points
    h, pen = sp.filter(like="h_"), sp.filter(like="p_")
    none = t("route.none")
    top_h = [t(f"hazard.{c[2:]}").lower() if v >= 10 else none for c, v in zip(h.idxmax(axis=1), h.max(axis=1), strict=True)]
    top_p = [t(f"penalty.{c[2:]}").lower() if v >= 3 else none
             for c, v in zip(pen.idxmax(axis=1), pen.max(axis=1), strict=True)]
    d_level = [level(v, DANGER_LEVELS) for v in sp["danger"]]
    titles = [t("route.danger_axis"), t("route.comfort_axis")]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12, subplot_titles=titles)
    fig.add_scatter(x=sp["dist_km"], y=sp["danger"], mode="lines+markers", name=t("route.danger"), showlegend=False,
                    line=dict(color=AXIS, width=2),
                    marker=dict(size=10, color=[STATUS[DANGER_STATUS[lv]] for lv in d_level],
                                line=dict(width=2, color="#fcfcfb")),
                    customdata=np.stack([pts["local_str"], [t(f"level.{lv}") for lv in d_level], top_h], axis=1),
                    hovertemplate=t("route.danger_hover"), row=1, col=1)
    for y, code in [(25, "Moderate"), (50, "High"), (75, "Severe")]:
        fig.add_hline(y=y, line=dict(color=MUTED, width=1, dash="dot"), row=1, col=1,
                      annotation_text=t(f"level.{code}").lower(), annotation_position="top left",
                      annotation_font=dict(color=MUTED, size=11))
    fig.add_scatter(x=sp["dist_km"], y=sp["comfort"], mode="lines+markers", name=t("route.comfort"), showlegend=False,
                    line=dict(color=SERIES[0], width=2), marker=dict(size=8),
                    customdata=np.stack([pts["local_str"], top_p], axis=1),
                    hovertemplate=t("route.comfort_hover"), row=2, col=1)
    fig.update_yaxes(range=[0, 104], tickvals=[0, 25, 50, 75, 100], row=1, col=1)
    fig.update_yaxes(range=[0, 104], tickvals=[0, 25, 50, 75, 100], row=2, col=1)
    fig.update_xaxes(title_text=t("route.distance_axis"), row=2, col=1)
    for title in titles:
        fig.update_annotations(selector=dict(text=title), font=dict(size=13, color=INK2), x=0, xanchor="left")
    chart(fig, 460, t("route.scores_chart"), t("route.scores_chart_sub"), margin=dict(t=34),
          source=t("source.route_csv", models=t("route.models")))


def _draw_track() -> tuple[pd.DataFrame, str] | None:
    """Map where the user clicks waypoints; returns the resulting track (points, name)."""
    c1, c2, c3 = st.columns([2, 1.1, 1.2], vertical_alignment="bottom")
    place = c1.text_input(t("route.place"), placeholder=t("route.place_hint"), key="draw_place")
    activity = c2.selectbox(t("route.route_for"), list(ACTIVITIES), index=1, key="draw_activity",
                            format_func=lambda a: t(f"activity.{a}"))
    follow = c3.toggle(t("route.follow"), value=True, key="draw_follow", help=t("route.follow_help"))
    center, zoom = (46.6, 2.6), 6
    if place:
        try:
            hits = _geocode(place)
        except OSError:   # requests' errors (connection, timeout, HTTP error) are OSErrors
            hits = None
            c1.caption(t("route.place_down"))
        if hits:
            center, zoom = (hits[0]["lat"], hits[0]["lon"]), 12
            c1.caption(t("route.place_found", label=hits[0]["label"]))
        elif hits is not None:
            c1.caption(t("route.place_none"))
    # On a phone one finger scrolls the page past the map; two fingers move and zoom it (and a tap adds a point)
    m = folium.Map(location=center, zoom_start=zoom, control_scale=True, dragging=not is_phone(),
                   tiles="CartoDB dark_matter" if is_dark() else "OpenStreetMap")
    Draw(draw_options={"polyline": {"shapeOptions": {"color": SERIES[0], "weight": 4}, "metric": True},
                       "polygon": False, "rectangle": False, "circle": False, "circlemarker": False, "marker": False},
         edit_options={"edit": True, "remove": True}).add_to(m)
    out = st_folium(m, key="draw_map", height=460, use_container_width=True, center=center, zoom=zoom,
                    returned_objects=["all_drawings"])
    waypoints = waypoints_from_drawings((out or {}).get("all_drawings"))
    if is_phone():
        st.caption(t("route.draw_phone"))
    if len(waypoints) < 2:
        st.info(t("route.draw_info"))
        return None
    with st.spinner(t("route.draw_spinner")):
        try:
            points, name, routed = _route(tuple(waypoints), activity, follow)
        except ValueError as exc:
            st.error(str(exc))
            return None
    if follow and not routed:
        st.warning(t("route.draw_straight"))
    return points, name


def _map(track: Track, pts: pd.DataFrame, tz: str) -> None:
    choice = st.segmented_control(t("route.colour_by"), ["Temperature", "Rain", "Headwind", "Danger", "Comfort"],
                                  default="Temperature", required=True, key="route_colour",
                                  format_func=lambda c: t(f"route.colour_{c}"))
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
                                hoverinfo="skip", name=t("route.track")))
    extra = [(t("route.map_aqi", aqi=num(a, 0)) if pd.notna(a) else "") + (f"<br>⚠️ {v}" if v else "")
             for a, v in zip(pts["european_aqi"], pts.get("vigilance_text", pd.Series([""] * len(pts))), strict=True)]
    custom = np.stack([pts["dist_km"], pts["local_str"], pts["icon"] + " " + pts["condition"], pts["temperature_2m"],
                       pts["precipitation"].fillna(0), pts["wind_from"], pts["wind_speed_10m"], pts["headwind"],
                       pts["danger"], pts["comfort"], extra], axis=1)
    fig.add_trace(go.Scattermap(
        lat=pts["lat"], lon=pts["lon"], mode="markers", marker=marker, customdata=custom, name=t("route.weather"),
        hovertemplate=t("route.map_hover")))
    ends = pts.iloc[[0, -1]]
    fig.add_trace(go.Scattermap(lat=ends["lat"], lon=ends["lon"], mode="markers+text",
                                text=[t("route.map_start"), t("route.map_finish")],
                                textposition="top right", marker=dict(size=9, color=INK), hoverinfo="skip",
                                textfont=dict(color=INK, size=13)))
    # Room for the track and its Start / Finish labels
    zoom = _zoom(full["lat"], full["lon"], _map_px(), 330)   # minus the colour bar
    fig.update_layout(map=dict(style="carto-positron", zoom=zoom,
                               center=dict(lat=float(full["lat"].mean()), lon=float(full["lon"].mean()))),
                      showlegend=False)
    chart(fig, 460, t("route.map_title", what=t(f"route.colour_{choice}")), t("route.map_sub"),
          margin=dict(l=0, r=0, b=0))


# Six pollens, three series colours: the colour and the dash follow the species, never its rank among those shown
# (trees solid, herbs dashed)
POLLEN_STYLE = {"alder_pollen": (0, "solid"), "birch_pollen": (1, "solid"), "olive_pollen": (2, "solid"),
                "grass_pollen": (0, "dash"), "mugwort_pollen": (1, "dash"), "ragweed_pollen": (2, "dash")}
assert set(POLLEN_STYLE) == set(POLLEN)


def _profile(track: Track, pts: pd.DataFrame, ens=None) -> None:
    has_ele = pts["ele_profile"].notna().any()
    has_air = pts["european_aqi"].notna().any()
    pollen_cols = [c for c in POLLEN_STYLE if pts[c].fillna(0).gt(0).any()]   # none in the air: no toggle (#83)
    show_pollen = bool(pollen_cols) and st.toggle(t("route.pollen"), key="route_pollen", help=t("route.pollen_help"))
    rain_row = t("route.row_rain")
    rows = [t("route.row_ele"), t("route.row_temp"), rain_row, t("route.row_wind")][0 if has_ele else 1:]
    if ens is not None:
        rows.insert(rows.index(rain_row) + 1, t("route.row_chance"))
    if has_air:
        rows.append(t("route.row_air"))
    if show_pollen:
        rows.append(t("route.row_pollen"))
    fig = make_subplots(rows=len(rows), cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=rows)
    r = 1
    hover_time = pts["local_str"]
    if has_ele:
        full = track.points
        y = full["ele"] if track.has_elevation else np.interp(full["dist_km"], pts["dist_km"], pts["ele_profile"])
        fig.add_scatter(x=full["dist_km"], y=y, mode="lines", line=dict(color=MUTED, width=1.5), fill="tozeroy",
                        fillcolor="rgba(195,194,183,0.35)", name=t("route.elevation"), showlegend=False,
                        hovertemplate=t("route.ele_hover"), row=r, col=1)
        lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
        fig.update_yaxes(range=[max(lo - (hi - lo) * 0.15 - 10, 0) if lo >= 0 else lo - 10, hi + (hi - lo) * 0.1 + 10],
                         row=r, col=1)
        r += 1
    if ens is not None:
        e = ens.points
        fig.add_scatter(x=e["dist_km"], y=e["t_p90"], mode="lines", line=dict(width=0), hoverinfo="skip",
                        showlegend=False, row=r, col=1)
        fig.add_scatter(x=e["dist_km"], y=e["t_p10"], mode="lines", line=dict(width=0), fill="tonexty",
                        fillcolor="rgba(195,194,183,0.35)", name=t("route.ens_band"), showlegend=False,
                        customdata=e["t_p90"], hovertemplate=t("route.ens_band_hover"), row=r, col=1)
    fig.add_scatter(x=pts["dist_km"], y=pts["temperature_2m"], mode="lines+markers", name=t("route.colour_Temperature"),
                    line=dict(color=SERIES[1], width=2.2), marker=dict(size=7), customdata=hover_time,
                    hovertemplate=t("route.temp_hover"), row=r, col=1)
    fig.add_scatter(x=pts["dist_km"], y=pts["apparent_temperature"], mode="lines+markers", name=t("route.feels"),
                    line=dict(color=SERIES[0], width=2.2), marker=dict(size=7), customdata=hover_time,
                    hovertemplate=t("route.feels_hover"), row=r, col=1)
    r += 1
    prob = pts["precipitation_probability"]
    prob_txt = [pct(p / 100) if pd.notna(p) else t("route.na") for p in prob]
    step = np.where(pts["precip_15min"], t("route.step_15"), t("route.step_hour"))
    fig.add_bar(x=pts["dist_km"], y=pts["precipitation"].fillna(0), name=t("route.colour_Rain"), marker_color=SERIES[0],
                showlegend=False, customdata=np.stack([hover_time, prob_txt, step], axis=1),
                hovertemplate=t("route.rain_hover"), row=r, col=1)
    fig.update_yaxes(rangemode="tozero", row=r, col=1)
    r += 1
    if ens is not None:
        fig.add_scatter(x=ens.points["dist_km"], y=ens.points["rain_chance"] * 100, mode="lines",
                        name=t("pred.rain_chance"), line=dict(color=SERIES[0], width=2), fill="tozeroy",
                        fillcolor="rgba(195,194,183,0.35)", showlegend=False, hovertemplate=t("route.chance_hover"),
                        row=r, col=1)
        fig.update_yaxes(range=[0, 104], row=r, col=1)
        r += 1
    fig.add_bar(x=pts["dist_km"], y=pts["headwind"], name=t("route.headwind_name"), showlegend=False,
                marker_color=[WARM if v >= 0 else COOL for v in pts["headwind"]], customdata=hover_time,
                hovertemplate=t("route.headwind_hover"), row=r, col=1)
    fig.add_scatter(x=pts["dist_km"], y=pts["wind_gusts_10m"], mode="lines", name=t("route.gusts"),
                    line=dict(color=INK, width=1.5), customdata=hover_time, hovertemplate=t("route.gusts_hover"),
                    row=r, col=1)
    if ens is not None:
        fig.add_scatter(x=ens.points["dist_km"], y=ens.points["gust_p90"], mode="lines", name=t("route.gusts_p90"),
                        line=dict(color=INK, width=1, dash="dot"), hovertemplate=t("route.gusts_p90_hover"),
                        row=r, col=1)
    moderate = t("route.aqi_moderate")
    if has_air:
        r += 1
        aqi = pts["european_aqi"]
        fig.add_bar(x=pts["dist_km"], y=aqi, name=t("route.air"), showlegend=False,
                    marker_color=[STATUS["good"] if v <= 20 else STATUS["warning"] if v <= 60 else STATUS["serious"]
                                  if v <= 80 else STATUS["critical"] for v in aqi.fillna(0)],
                    customdata=np.stack([hover_time, pts["ozone"].round(0), pts["pm2_5"].round(0),
                                         pts["pm10"].round(0)], axis=1),
                    hovertemplate=t("route.air_hover"), row=r, col=1)
        fig.add_hline(y=AQI_MODERATE, line=dict(color=MUTED, width=1, dash="dot"), row=r, col=1,
                      annotation_text=moderate, annotation_position="top left",
                      annotation_font=dict(color=MUTED, size=11))
        fig.update_yaxes(rangemode="tozero", row=r, col=1)
    if show_pollen:
        r += 1
        for c in pollen_cols:
            colour, dash = POLLEN_STYLE[c]
            fig.add_scatter(x=pts["dist_km"], y=pts[c], mode="lines", name=t(f"route.pollen_{c}"), legend="legend2",
                            line=dict(color=SERIES[colour], width=1.8, dash=dash),
                            hovertemplate=t("route.pollen_hover", name=t(f"route.pollen_{c}")), row=r, col=1)
        fig.update_yaxes(rangemode="tozero", row=r, col=1)
        # The pollen legend sits in the pollen panel, not at the top of a chart over a thousand pixels tall
        top = fig.layout["yaxis" if r == 1 else f"yaxis{r}"].domain[1]
        fig.update_layout(legend2=dict(x=1, xanchor="right", y=top, yanchor="top", bgcolor="rgba(0,0,0,0)",
                                       font=dict(size=11)), legend=dict(visible=False))
    fig.update_xaxes(title_text=t("route.distance_axis"), row=r, col=1)
    fig.update_annotations(selector=dict(text=moderate), x=None)
    for title in rows:
        fig.update_annotations(selector=dict(text=title), font=dict(size=13, color=INK2), x=0, xanchor="left")
    fig.update_layout(bargap=0.3, hovermode="closest", showlegend=show_pollen)
    chart(fig, 220 * len(rows), t("route.profile"),
          t("route.profile_sub") + (t("route.profile_sub_ens") if ens is not None else ""), source=t("source.route_csv", models=t("route.models")))


def _table(pts: pd.DataFrame, ens=None) -> None:
    tbl = pd.DataFrame({
        "km": pts["dist_km"].round(1), "time": pts["local_str"], "elevation_m": pts["ele_profile"].round(0),
        "conditions": pts["icon"] + " " + pts["condition"], "temp_c": pts["temperature_2m"].round(1),
        "feels_like_c": pts["apparent_temperature"].round(1), "rain_mm_h": pts["precipitation"].round(1),
        "rain_prob_pct": pts["precipitation_probability"].round(0), "wind": pts["wind_from"] + " " + pts["wind_speed_10m"].round(0).astype(int).astype(str),
        "headwind_kmh": pts["headwind"].round(0), "gusts_kmh": pts["wind_gusts_10m"].round(0),
        "danger": pts["danger"].round(0), "comfort": pts["comfort"].round(0),
    })
    if pts["european_aqi"].notna().any():
        tbl["air_index"] = pts["european_aqi"].round(0)
        tbl["ozone"] = pts["ozone"].round(0)
    if ens is not None:
        tbl["rain_chance_pct"] = (ens.points["rain_chance"] * 100).round(0).to_numpy()
    if "vigilance_text" in pts and pts["vigilance_text"].ne("").any():
        tbl["vigilance"] = pts["vigilance_text"]
    share = number(0, "%" if t("fmt.decimal") == "." else "\u202f%")
    formats = {"km": number(1), "elevation_m": number(0, "m"), "temp_c": number(1, "°C"),
               "feels_like_c": number(1, "°C"), "rain_mm_h": number(1, "mm/h"), "rain_prob_pct": share,
               "headwind_kmh": number(0, "km/h", sign=True), "gusts_kmh": number(0, "km/h"), "danger": number(0),
               "comfort": number(0), "air_index": number(0), "ozone": number(0, "µg/m³"), "rain_chance_pct": share}
    labels = {"km": "km", "time": t("route.col_time"), "elevation_m": t("route.elevation"),
              "conditions": t("route.col_conditions"), "temp_c": t("route.col_temp"), "feels_like_c": t("route.feels"),
              "rain_mm_h": t("route.colour_Rain"), "rain_prob_pct": t("route.col_rain_prob"),
              "wind": t("route.col_wind"), "headwind_kmh": t("route.colour_Headwind"), "gusts_kmh": t("route.gusts"),
              "danger": t("route.danger"), "comfort": t("route.comfort"), "air_index": t("route.col_air"),
              "ozone": t("route.col_ozone"), "rain_chance_pct": t("route.col_rain_chance"), "vigilance": "Vigilance"}
    helps = {"danger": t("route.help_danger"), "comfort": t("route.help_comfort"), "air_index": t("route.help_air"),
             "rain_chance_pct": t("route.help_chance")}
    with st.expander(t("route.table_title"), expanded=False):
        st.dataframe(styled(tbl, formats), hide_index=True,
                     column_config={c: st.column_config.Column(label, help=helps.get(c)) for c, label in labels.items()})
        st.download_button(t("chart.download_table"), csv_bytes(tbl.rename(columns=labels), t("route.table_title"),
                                                                source=t("source.route_csv", models=t("route.models"))),
                           "route_weather.csv", "text/csv", icon=":material/download:")


def _station_panel(pts: pd.DataFrame, tz: str) -> None:
    last = _warehouse_last_date()
    day = pts["local"].dt.date.max()
    if last is None:
        return
    with st.expander(t("route.stations_title"), expanded=False):
        if day > last:
            st.caption(t("route.stations_past", date=fmt_date(last)))
            return
        obs = _stations(pts[["dist_km", "lat", "lon", "time"]], tz)
        obs["fxi"] = obs["fxi"] * 3.6   # m/s in the warehouse, km/h like the reading above
        if obs.empty:
            st.caption(t("route.stations_none"))
            return
        st.caption(t("route.stations_caption"))
        st.dataframe(styled(obs, {"route_km": number(1), "altitude": number(0, "m"), "distance_km": number(1, "km"),
                                  "day": date_text(), "tn": number(1, "°C"), "tx": number(1, "°C"),
                                  "rr": number(1, "mm"), "fxi": number(0, "km/h")}),
                     hide_index=True, column_config={
                         "route_km": t("route.col_route_km"), "station_name": t("common.station"),
                         "altitude": t("quality.col_alt"), "distance_km": t("city.col_distance"),
                         "day": t("pred.col_day"), "tn": t("route.col_tn"), "tx": t("route.col_tx"),
                         "rr": t("route.colour_Rain"), "fxi": t("quality.var_FXI")})
