"""Shared building blocks of the dashboard: data access, Plotly theme, formatting."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from meteo.config import load_config  # noqa: E402
from meteo.fmt import num, pvalue, signed  # noqa: E402,F401
from meteo.palette import (  # noqa: E402,F401
    AXIS,
    COOL,
    DARK_AXIS,
    DARK_COOL,
    DARK_DIVERGING_STOPS,
    DARK_GRID,
    DARK_HEAT_STOPS,
    DARK_INK,
    DARK_INK2,
    DARK_MUTED,
    DARK_NEUTRAL,
    DARK_RAIN_STOPS,
    DARK_SERIES,
    DARK_SURFACE,
    DARK_WARM,
    DIVERGING_STOPS,
    GRID,
    HEAT_STOPS,
    INK,
    INK2,
    MUTED,
    NEUTRAL,
    RAIN_STOPS,
    SERIES,
    STATUS,
    SURFACE,
    WARM,
)

CFG = load_config(ROOT / "config.yaml")
A = CFG.analysis
START, END = A["start_year"], A["end_year"]
REF = tuple(A["reference_period"])
OLD = tuple(A["old_period"])
RECENT = tuple(A["recent_period"])
TH = CFG.thresholds


# --- Data ----------------------------------------------------------------------
def warehouse_ready() -> bool:
    return CFG.warehouse.exists() and (CFG.reports_dir / "results.json").exists()


def _connect() -> duckdb.DuckDBPyConnection:
    """Read-only, with no access to files on disk: the dashboard cannot change anything."""
    return duckdb.connect(str(CFG.warehouse), read_only=True, config={"enable_external_access": False})


@st.cache_data(show_spinner=False)
def query(sql: str) -> pd.DataFrame:
    with _connect() as con:
        return con.sql(sql).df()


@st.cache_data(show_spinner=False)
def table(name: str) -> pd.DataFrame:
    df = query(f"SELECT * FROM {name}")
    for col in ("date", "start_date", "end_date", "first_date", "last_date"):
        if col in df:
            df[col] = pd.to_datetime(df[col])
    return df


@st.cache_data(show_spinner=False)
def results() -> dict:
    return json.loads((CFG.reports_dir / "results.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def era5() -> pd.DataFrame | None:
    p = CFG.reports_dir / "tables" / "era5_validation.csv"
    return pd.read_csv(p) if p.exists() else None


# --- Stable colours per city (the colour follows the city, not its rank) ---------
def city_colors(selected: list[str], key: str) -> dict[str, str]:
    slots: dict[str, int] = st.session_state.setdefault(f"slots_{key}", {})
    for c in list(slots):
        if c not in selected:
            del slots[c]
    for c in selected:
        if c not in slots:
            slots[c] = min(set(range(len(SERIES))) - set(slots.values()))
    return {c: SERIES[slots[c]] for c in selected}


# --- Plotly theme ----------------------------------------------------------------
FONT = "Segoe UI, system-ui, -apple-system, Helvetica, Arial, sans-serif"
pio.templates["meteo"] = go.layout.Template(layout=dict(
    font=dict(family=FONT, color=INK2, size=13),
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
    colorway=SERIES,
    separators=".,",
    xaxis=dict(gridcolor=GRID, linecolor=AXIS, showline=True, zeroline=False, ticks="", automargin=True,
               tickfont=dict(color=INK2), title=dict(font=dict(color=INK2))),
    yaxis=dict(gridcolor=GRID, linecolor=AXIS, showline=False, zeroline=False, ticks="", automargin=True,
               tickfont=dict(color=INK2), title=dict(font=dict(color=INK2))),
    hoverlabel=dict(bgcolor="#ffffff", bordercolor=AXIS, font=dict(family=FONT, color=INK, size=13)),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, title_text="", font=dict(color=INK)),
    margin=dict(l=64, r=16, t=40, b=44),
))
pio.templates.default = "meteo"

PLOTLY_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}


def colorscale(stops: list[str]) -> list[list]:
    return [[i / (len(stops) - 1), c] for i, c in enumerate(stops)]


DIVERGING = colorscale(DIVERGING_STOPS)
HEAT = colorscale(HEAT_STOPS)
RAIN = colorscale(RAIN_STOPS)


def is_phone() -> bool:
    """A phone, from the browser's user agent (the layout cannot measure the screen)."""
    try:
        return "Mobi" in (st.context.headers.get("User-Agent") or "")
    except Exception:
        return False


def chart(fig: go.Figure, height: int = 420, title: str | None = None, subtitle: str | None = None,
          margin: dict | None = None) -> None:
    """Title and subtitle in HTML above the chart: they wrap on small screens."""
    if title:
        sub = f"<div class='chart-sub'>{subtitle}</div>" if subtitle else ""
        st.markdown(f"<div class='chart-title'>{title}</div>{sub}", unsafe_allow_html=True)
    has_legend = any(t.showlegend is not False for t in fig.data) and len(fig.data) > 1
    fig.update_layout(height=height, margin={"t": 40 if has_legend else 12, **(margin or {})})
    # Without hover, Plotly keeps its toolbar visible on touch screens, over legends and labels
    st.plotly_chart(themed(fig), theme=None,
                    config=PLOTLY_CONFIG | ({"displayModeBar": False} if is_phone() else {}))


def signed_color(values) -> list[str]:
    return [WARM if v >= 0 else COOL for v in values]


# --- Page style --------------------------------------------------------------------
LIGHT = {"surface": SURFACE, "ink": INK, "ink2": INK2, "muted": MUTED, "grid": GRID, "series": SERIES[0],
         "card_border": "rgba(11,11,11,0.10)", "meter": "#ecebe6"}
DARK = {"surface": DARK_SURFACE, "ink": DARK_INK, "ink2": DARK_INK2, "muted": DARK_MUTED, "grid": DARK_GRID,
        "series": DARK_SERIES[0], "card_border": "rgba(240,239,236,0.14)", "meter": "#2f2e2b"}


def is_dark() -> bool:
    """Whether the page is shown in dark mode: Streamlit follows the device's setting (or the one picked in its
    menu), and tells the script through st.context.theme."""
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


def theme() -> dict:
    return DARK if is_dark() else LIGHT


def css(p: dict) -> str:
    return f"""
<style>
  .block-container {{ padding-top: 2.2rem; max-width: 1280px; }}
  h1, h2, h3 {{ color: {p['ink']}; letter-spacing: -0.01em; }}
  [data-testid="stMetricValue"] {{ font-size: 1.9rem; color: {p['ink']}; }}
  [data-testid="stMetricLabel"] p {{ color: {p['ink2']}; font-size: 0.9rem; }}
  .kpi-note {{ color: {p['ink2']}; font-size: 0.82rem; margin-top: -0.6rem; }}
  .lead {{ color: {p['ink2']}; font-size: 1.05rem; margin-top: -0.6rem; margin-bottom: 1.2rem; }}
  .finding {{ border-left: 3px solid {p['series']}; padding: 0.35rem 0 0.35rem 0.8rem; margin: 0.45rem 0;
              color: {p['ink']}; }}
  .finding b {{ color: {p['ink']}; }}
  .source {{ color: {p['muted']}; font-size: 0.78rem; }}
  .stTextArea textarea {{ font-family: ui-monospace, "Cascadia Code", Consolas, monospace; font-size: 0.88rem; }}
  .chart-title {{ color: {p['ink']}; font-weight: 600; font-size: 1.08rem; margin-top: 0.8rem; line-height: 1.35; }}
  .chart-sub {{ color: {p['ink2']}; font-size: 0.88rem; margin-top: 0.1rem; line-height: 1.4; }}
  .score-card {{ border: 1px solid {p['card_border']}; border-radius: 8px; padding: 0.85rem 1rem 0.75rem;
                 background: {p['surface']}; height: 100%; }}
  .score-label {{ color: {p['ink2']}; font-size: 0.9rem; }}
  .score-value {{ color: {p['ink']}; font-size: 2.1rem; font-weight: 600; line-height: 1.15; }}
  .score-value small {{ color: {p['muted']}; font-size: 0.95rem; font-weight: 400; }}
  .chip {{ display: inline-flex; align-items: center; gap: 0.4rem; color: {p['ink']}; font-size: 0.95rem;
           font-weight: 500; margin-left: 0.6rem; vertical-align: 0.35rem; }}
  .chip i {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
  .meter {{ height: 8px; border-radius: 4px; background: {p['meter']}; margin: 0.55rem 0 0.55rem; }}
  .meter div {{ height: 100%; border-radius: 4px; }}
  .score-why {{ color: {p['ink2']}; font-size: 0.86rem; line-height: 1.45; }}
  .advice {{ border-left: 3px solid {p['muted']}; padding: 0.3rem 0 0.3rem 0.8rem; margin: 0.4rem 0; color: {p['ink']}; }}
  .advice.danger {{ border-left-color: {STATUS['critical']}; }}
  .advice.comfort {{ border-left-color: {p['series']}; }}
  .advice span {{ color: {p['ink2']}; }}
</style>
"""


CSS = css(LIGHT)


# Light-mode colours of the figures -> their dark-mode steps (the figures are built with the light palette)
_DARK_COLOURS = {c.lower(): d for c, d in {
    SURFACE: DARK_SURFACE, INK: DARK_INK, INK2: DARK_INK2, MUTED: DARK_MUTED, GRID: DARK_GRID, AXIS: DARK_AXIS,
    NEUTRAL: DARK_NEUTRAL, "#ffffff": "#262624", **dict(zip(SERIES, DARK_SERIES, strict=True)), WARM: DARK_WARM,
    COOL: DARK_COOL, "rgba(195,194,183,0.35)": "rgba(143,141,135,0.28)",
}.items()}


def _rgb(hex_colour: str) -> tuple[int, int, int]:
    return tuple(int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))


def rgba(hex_colour: str, alpha: float) -> str:
    """A palette colour made transparent (for fills): in dark mode it takes its dark step, same transparency."""
    return "rgba({},{},{},{})".format(*_rgb(hex_colour), alpha)


_DARK_RGB = {_rgb(c): _rgb(d) for c, d in _DARK_COLOURS.items() if c.startswith("#") and d.startswith("#")}
_RGBA = re.compile(r"rgba\((\d+),(\d+),(\d+),([\d.]+)\)")
_DARK_SCALES = [(HEAT_STOPS, DARK_HEAT_STOPS), (RAIN_STOPS, DARK_RAIN_STOPS), (DIVERGING_STOPS, DARK_DIVERGING_STOPS)]
_DARK_MAPS = {"carto-positron": "carto-darkmatter", "light": "dark"}


def _scale(value):
    """A Plotly colour scale ([[0, c0], …, [1, cn]]) of the light palette -> its dark version, else None."""
    if (isinstance(value, (list, tuple)) and len(value) >= 2
            and all(isinstance(v, (list, tuple)) and len(v) == 2 and isinstance(v[1], str) for v in value)):
        colours = [v[1].lower() for v in value]
        for light, dark in _DARK_SCALES:
            if colours == [c.lower() for c in light]:
                return [[v[0], d] for v, d in zip(value, dark, strict=True)]
    return None


def _to_dark(obj, key: str = ""):
    if isinstance(obj, dict):
        return {k: _to_dark(v, k) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        if key == "colorscale" and (scale := _scale(obj)) is not None:
            return scale
        return [_to_dark(v, key) for v in obj]
    if isinstance(obj, str):
        if key == "style" and obj in _DARK_MAPS:
            return _DARK_MAPS[obj]
        colour = obj.lower().replace(" ", "")
        if colour in _DARK_COLOURS:
            return _DARK_COLOURS[colour]
        if (m := _RGBA.fullmatch(colour)) and (rgb := _DARK_RGB.get(tuple(int(x) for x in m.groups()[:3]))):
            return "rgba({},{},{},{})".format(*rgb, m[4])
        return obj
    return obj


def themed(fig: go.Figure) -> go.Figure:
    """The figure in the page's theme: as built (light palette), or with the dark steps of every UI colour, series,
    colour scale and map style when the page is in dark mode."""
    if not is_dark():
        return fig
    return go.Figure(_to_dark(fig.to_dict()))


def page_header(title: str, lead: str, tab: str | None = None) -> None:
    """Title and lead of a page, and its "Copy link" button; the browser tab is named after the page (`tab`, or the
    title)."""
    from share import copy_link  # dashboard/share.py (imported here: ui is also imported from outside the app)

    st.set_page_config(page_title=f"{tab or title} · Météo Analytics")
    st.markdown(css(theme()), unsafe_allow_html=True)
    st.title(title)
    st.markdown(f"<div class='lead'>{lead}</div>", unsafe_allow_html=True)
    copy_link()


def source_note(extra: str = "") -> None:
    st.markdown(f"<div class='source'>Source: Météo-France daily climatological data "
                f"(data.gouv.fr, Etalab 2.0 open licence){extra}.</div>", unsafe_allow_html=True)


def rolling(series: pd.Series, window: int = 10) -> pd.Series:
    return series.rolling(window, center=True, min_periods=max(3, window // 2 + 1)).mean()


def doy_climatology(daily: pd.DataFrame, col: str, ref: tuple[int, int], half_window: int = 15) -> pd.DataFrame:
    """Smoothed daily normal (mean, 10th and 90th percentiles) over a ±15-day window."""
    d = daily[daily["date"].dt.year.between(*ref)][["date", col]].dropna()
    d = d[~((d["date"].dt.month == 2) & (d["date"].dt.day == 29))]
    doy = d["date"].dt.day_of_year - ((d["date"].dt.is_leap_year) & (d["date"].dt.month > 2)).astype(int)
    values = [[] for _ in range(365)]
    for day, v in zip(doy.to_numpy() - 1, d[col].to_numpy(), strict=True):
        values[day].append(v)
    rows = []
    for i in range(365):
        window = np.concatenate([values[(i + k) % 365] for k in range(-half_window, half_window + 1)])
        rows.append((i + 1, window.mean(), np.percentile(window, 10), np.percentile(window, 90)))
    return pd.DataFrame(rows, columns=["doy", "mean", "p10", "p90"])
