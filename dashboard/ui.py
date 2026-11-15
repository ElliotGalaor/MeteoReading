"""Shared building blocks of the dashboard: data access, Plotly theme, formatting."""
from __future__ import annotations

import json
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
    st.plotly_chart(fig, theme=None, config=PLOTLY_CONFIG | ({"displayModeBar": False} if is_phone() else {}))


def signed_color(values) -> list[str]:
    return [WARM if v >= 0 else COOL for v in values]


# --- Page style --------------------------------------------------------------------
CSS = f"""
<style>
  .block-container {{ padding-top: 2.2rem; max-width: 1280px; }}
  h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}
  [data-testid="stMetricValue"] {{ font-size: 1.9rem; color: {INK}; }}
  [data-testid="stMetricLabel"] p {{ color: {INK2}; font-size: 0.9rem; }}
  .kpi-note {{ color: {INK2}; font-size: 0.82rem; margin-top: -0.6rem; }}
  .lead {{ color: {INK2}; font-size: 1.05rem; margin-top: -0.6rem; margin-bottom: 1.2rem; }}
  .finding {{ border-left: 3px solid {SERIES[0]}; padding: 0.35rem 0 0.35rem 0.8rem; margin: 0.45rem 0;
              color: {INK}; }}
  .finding b {{ color: {INK}; }}
  .source {{ color: {MUTED}; font-size: 0.78rem; }}
  .stTextArea textarea {{ font-family: ui-monospace, "Cascadia Code", Consolas, monospace; font-size: 0.88rem; }}
  .chart-title {{ color: {INK}; font-weight: 600; font-size: 1.08rem; margin-top: 0.8rem; line-height: 1.35; }}
  .chart-sub {{ color: {INK2}; font-size: 0.88rem; margin-top: 0.1rem; line-height: 1.4; }}
  .score-card {{ border: 1px solid rgba(11,11,11,0.10); border-radius: 8px; padding: 0.85rem 1rem 0.75rem;
                 background: {SURFACE}; height: 100%; }}
  .score-label {{ color: {INK2}; font-size: 0.9rem; }}
  .score-value {{ color: {INK}; font-size: 2.1rem; font-weight: 600; line-height: 1.15; }}
  .score-value small {{ color: {MUTED}; font-size: 0.95rem; font-weight: 400; }}
  .chip {{ display: inline-flex; align-items: center; gap: 0.4rem; color: {INK}; font-size: 0.95rem;
           font-weight: 500; margin-left: 0.6rem; vertical-align: 0.35rem; }}
  .chip i {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
  .meter {{ height: 8px; border-radius: 4px; background: #ecebe6; margin: 0.55rem 0 0.55rem; }}
  .meter div {{ height: 100%; border-radius: 4px; }}
  .score-why {{ color: {INK2}; font-size: 0.86rem; line-height: 1.45; }}
  .advice {{ border-left: 3px solid {MUTED}; padding: 0.3rem 0 0.3rem 0.8rem; margin: 0.4rem 0; color: {INK}; }}
  .advice.danger {{ border-left-color: {STATUS['critical']}; }}
  .advice.comfort {{ border-left-color: {SERIES[0]}; }}
  .advice span {{ color: {INK2}; }}
</style>
"""


def page_header(title: str, lead: str, tab: str | None = None) -> None:
    """Title and lead of a page; the browser tab is named after the page (`tab`, or the title)."""
    st.set_page_config(page_title=f"{tab or title} · Météo Analytics")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title(title)
    st.markdown(f"<div class='lead'>{lead}</div>", unsafe_allow_html=True)


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
