"""Shared building blocks of the dashboard: data access, Plotly theme, formatting."""
from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
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

from meteo import i18n  # noqa: E402
from meteo.config import load_config  # noqa: E402
from meteo.fmt import num, pct, percent, pvalue, signed  # noqa: E402,F401
from meteo.i18n import fmt_date, lang, listing, ordinal, plural, t, words  # noqa: E402,F401
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




def _session_lang() -> str | None:
    """The visitor's language, kept in the session state by dashboard/language.py (None outside a page run)."""
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is None:
        return None
    return st.session_state.get("lang")


i18n.set_provider(_session_lang)


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


def slug(text: str | None) -> str:
    """A file name from a chart title: "Hot days (TX ≥ 30 °C)" -> "hot_days_tx_30_c"."""
    plain = unicodedata.normalize("NFKD", re.sub(r"<[^>]+>", "", text or "chart")).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", plain.lower()).strip("_")[:60] or "chart"


def plotly_config(title: str | None = None) -> dict:
    """Plotly's options: toolbar, the page's language (month names, decimal comma), image file name. On a phone the
    toolbar keeps only its camera, hidden: the "Save image" button under the chart presses it (dashboard/gestures.py).
    """
    config = PLOTLY_CONFIG | {"locale": lang(), "locales": i18n.plotly_locale(),
                              "toImageButtonOptions": {"filename": slug(title), "scale": 2}}
    if is_phone():
        config |= {"modeBarButtons": [["toImage"]]}
    return config


# Date axes: day before month ("4 Oct", "4 oct."), the way both languages write dates, at every zoom level
DATE_TICKS = [dict(dtickrange=[None, 86_400_000], value="%H:%M<br>%-d %b"),
              dict(dtickrange=[86_400_000, "M1"], value="%-d %b"),
              dict(dtickrange=["M1", "M12"], value="%b %Y"), dict(dtickrange=["M12", None], value="%Y")]


def _date_axes(fig: go.Figure) -> None:
    """Applies DATE_TICKS to the x axes of a figure drawn against dates, unless their ticks are already set."""
    def dated(tr) -> bool:
        x = getattr(tr, "x", None)
        if x is None or not len(x):
            return False
        first = next((v for v in x if v is not None), None)
        return isinstance(first, (pd.Timestamp, np.datetime64)) or hasattr(first, "isoformat")
    if any(dated(tr) for tr in fig.data):
        fig.for_each_xaxis(lambda ax: ax.update(tickformatstops=DATE_TICKS)
                           if ax.tickvals is None and ax.tickformat is None else None)


def chart(fig: go.Figure, height: int = 420, title: str | None = None, subtitle: str | None = None,
          margin: dict | None = None, data: pd.DataFrame | None = None, source: str | None = None) -> None:
    """Title and subtitle in HTML above the chart (they wrap on small screens), the chart, then the download of its
    data. `data` replaces the table read from the figure; `source` replaces the Météo-France source in the file."""
    if title:
        sub = f"<div class='chart-sub'>{subtitle}</div>" if subtitle else ""
        st.markdown(f"<div class='chart-title'>{title}</div>{sub}", unsafe_allow_html=True)
    has_legend = any(tr.showlegend is not False for tr in fig.data) and len(fig.data) > 1
    fig.update_layout(height=height, margin={"t": 40 if has_legend else 12, **(margin or {})},
                      separators=t("fmt.plotly_separators"))
    _date_axes(fig)
    st.plotly_chart(themed(fig), theme=None, config=plotly_config(title))
    downloads(fig, title, subtitle, data, source)


# --- Data behind the charts ---------------------------------------------------------
def _axis_title(fig: go.Figure, axis: str, ref: str | None) -> str | None:
    """Title of the axis a trace uses ("y", "y2"…): its unit, most of the time."""
    name = f"{axis}axis{(ref or axis)[1:]}"
    try:
        title = fig.layout[name].title.text
    except (KeyError, AttributeError, TypeError, ValueError):
        return None
    return re.sub(r"<[^>]+>", "", title) if title else None


def _x_name(values: list, title: str | None) -> str:
    if title:
        return title
    sample = next((v for v in values if v is not None), None)
    if isinstance(sample, (pd.Timestamp, np.datetime64)) or hasattr(sample, "isoformat"):
        return t("chart.csv_date")
    if isinstance(sample, (int, np.integer)) and 1800 <= int(sample) <= 2200:
        return t("chart.csv_year")
    return t("chart.csv_category")


def _column(name: str, unit: str | None) -> str:
    return f"{name} ({unit})" if unit and unit not in name else name


def _x_values(x: list) -> list:
    """The x of a trace as plain values: dates without their time zone (Plotly draws the wall-clock time), so
    that series given with and without a zone line up."""
    first = next((v for v in x if v is not None), None)
    if isinstance(first, (pd.Timestamp, np.datetime64)) or hasattr(first, "isoformat"):
        dates = pd.Series(pd.to_datetime(pd.Series(x).astype(object).map(
            lambda v: v.replace(tzinfo=None) if getattr(v, "tzinfo", None) else v)))
        return list(dates)
    return x


def _numeric(v) -> bool:
    return isinstance(v, (int, float, np.number)) and not isinstance(v, bool) and np.isfinite(v)


def figure_frame(fig: go.Figure) -> pd.DataFrame | None:
    """The values drawn in a figure, as a table: one column per named series on a shared x (dates, years,
    categories), or the grid of a heatmap. Helper traces are left out: unnamed lines, legend-only entries, text
    labels, and markers that only highlight points of another series. A trace can name its column with
    meta={"csv": "…"}, a heatmap its first column with meta={"row": "…"}. Maps give None: their page offers their
    data otherwise."""
    traces = list(fig.data)
    heat = [tr for tr in traces if tr.type == "heatmap"]
    if heat:
        tr = heat[0]
        grid = pd.DataFrame(np.asarray(tr.z, dtype=float), index=list(tr.y), columns=[str(c) for c in tr.x])
        meta = tr.meta if isinstance(tr.meta, dict) else {}
        grid.index.name = meta.get("row") or _axis_title(fig, "y", tr.yaxis) or t("chart.csv_row")
        return grid.round(3).reset_index()
    kept, highlights = [], []
    for i, tr in enumerate(traces):
        if tr.type not in ("scatter", "bar", "scattergl") or tr.x is None or tr.y is None or getattr(tr, "mode", None) == "text":
            continue
        horizontal = getattr(tr, "orientation", None) == "h"
        x, y = (list(tr.y), list(tr.x)) if horizontal else (list(tr.x), list(tr.y))
        x = _x_values(x)
        if not y or all(v is None for v in y) or any(v is None for v in x):
            continue   # a legend-only entry, or segments drawn between points
        meta = tr.meta if isinstance(tr.meta, dict) else {}
        name = meta.get("csv") or tr.name
        nxt = traces[i + 1] if i + 1 < len(traces) else None
        prev = traces[i - 1] if i > 0 else None
        if not meta.get("csv"):
            if not tr.name:
                if nxt is not None and getattr(nxt, "fill", None) == "tonexty" and nxt.name:     # upper edge of a band
                    name = t("chart.csv_band_high", name=nxt.name)
                elif getattr(tr, "hoverinfo", None) == "skip":
                    continue                                                    # a helper line
                else:
                    name = t("chart.csv_value")
            elif getattr(tr, "fill", None) == "tonexty" and prev is not None and not prev.name:   # lower edge of a band
                name = t("chart.csv_band_low", name=tr.name)
        unit = _axis_title(fig, "x" if horizontal else "y", tr.xaxis if horizontal else tr.yaxis)
        entry = (x, y, _column(name, unit), tr)
        (highlights if tr.type == "scatter" and getattr(tr, "mode", None) == "markers" else kept).append(entry)
    seen = {(str(xv), round(float(yv), 6)) for x, y, _, _ in kept for xv, yv in zip(x, y, strict=False) if _numeric(yv)}
    for x, y, col, tr in highlights:
        points = [(str(xv), round(float(yv), 6)) for xv, yv in zip(x, y, strict=False) if _numeric(yv)]
        if points and not all(p in seen for p in points):
            kept.append((x, y, col, tr))
    if not kept:
        return None
    names, frames = set(), []
    for x, y, col, tr in kept:
        while col in names:
            col += " "
        names.add(col)
        frame = pd.DataFrame({"x": x, col: pd.to_numeric(pd.Series(y), errors="coerce").to_numpy()})
        err = tr.error_y if tr.type == "bar" else None
        if err is not None and err.array is not None:
            high = np.asarray(err.array, dtype=float)
            low = np.asarray(err.arrayminus if err.arrayminus is not None else err.array, dtype=float)
            frame[t("chart.csv_low", name=col)] = frame[col] - low
            frame[t("chart.csv_high", name=col)] = frame[col] + high
        frames.append(frame.drop_duplicates("x"))
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="x", how="outer", sort=False)
    if len(frames) > 1:
        try:
            out = out.sort_values("x", kind="stable")
        except TypeError:
            pass
    first = kept[0][3]
    x_title = None if getattr(first, "orientation", None) == "h" else _axis_title(fig, "x", first.xaxis)
    out = out.rename(columns={"x": _x_name(kept[0][0], x_title)})
    for c in out.columns[1:]:
        out[c] = out[c].round(3)
    return out.reset_index(drop=True)


def csv_bytes(frame: pd.DataFrame, title: str | None, subtitle: str | None = None, source: str | None = None) -> bytes:
    """The table as a CSV file with its title, source and licence on top (lines starting with #). Commas and points
    in English, semicolons and decimal commas in French, as spreadsheets expect in each language; UTF-8 with a byte
    order mark, so that spreadsheets read the accents."""
    def plain(text: str | None) -> str:
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", text or "")).strip()
    lines = [t("chart.csv_title", title=plain(title) or t("chart.csv_untitled"))]
    if subtitle:
        lines.append(plain(subtitle))
    lines.append(t("chart.csv_source", source=plain(source) if source else t("source.meteo_france")))
    try:   # the page's address with its choices (dashboard/share.py), to find the same view again
        from urllib.parse import urlencode
        url = st.context.url
        params = [(k, v) for k in st.query_params for v in st.query_params.get_all(k)]
        url = url + ("?" + urlencode(params) if params and url else "")
    except Exception:
        url = None
    if url:
        lines.append(t("chart.csv_page", url=url))
    lines.append(t("chart.csv_downloaded", date=pd.Timestamp.today().strftime("%Y-%m-%d")))
    table = frame.copy()
    for c in table.columns:
        if pd.api.types.is_datetime64_any_dtype(table[c]):
            timed = bool((table[c].dt.hour + table[c].dt.minute).fillna(0).any())
            table[c] = table[c].dt.strftime("%Y-%m-%d %H:%M" if timed else "%Y-%m-%d")
    body = table.to_csv(index=False, sep=t("fmt.csv_separator"), decimal=t("fmt.csv_decimal"), lineterminator="\n")
    return ("﻿" + "".join(f"# {line}\n" for line in lines) + body).encode("utf-8")


# The icon is a ligature of Streamlit's own icon font (st.html keeps no SVG)
SAVE_IMAGE = ("<button class='save-chart-image' type='button'><span class='save-icon' aria-hidden='true'>image</span>"
              "{label}</button>")


def downloads(fig: go.Figure, title: str | None, subtitle: str | None = None, data: pd.DataFrame | None = None,
              source: str | None = None) -> None:
    """Under a chart: "Download data (CSV)", and on a phone "Save image" (the toolbar is hidden there)."""
    frame = data if data is not None else figure_frame(fig)
    has_data = frame is not None and not frame.empty
    phone = is_phone()
    if not has_data and not phone:
        return
    row = st.container(horizontal=True, gap="small", vertical_alignment="center")
    if has_data:
        payload = csv_bytes(frame, title, subtitle, source)
        row.download_button(t("chart.download_csv"), payload, f"{slug(title)}.csv", "text/csv",
                            icon=":material/download:", type="tertiary", on_click="ignore",
                            key=f"csv_{hashlib.md5(payload).hexdigest()[:16]}")
    if phone:
        row.html(SAVE_IMAGE.format(label=t("chart.save_image")))


def styled(df: pd.DataFrame, formats: dict):
    """A table whose numbers and dates read in the page's language (12,5 °C, 17 nov. 2026) and still sort as numbers
    and dates. `formats` maps a column to a function of its value (see `number` and `date_text`)."""
    return df.style.format({c: f for c, f in formats.items() if c in df.columns}, na_rep="–")


def number(digits: int = 1, unit: str = "", sign: bool = False, scale: float = 1.0):
    """Formatter of a table column: 12.5 -> "12.5 °C" ("12,5 °C" in French)."""
    def fmt(v) -> str:
        if v is None or pd.isna(v):
            return "–"
        text = (signed if sign else num)(float(v) * scale, digits)
        return f"{text} {unit}" if unit else text
    return fmt


def date_text(style: str = "short"):
    """Formatter of a date column: "17 Nov 2026" ("17 nov. 2026")."""
    return lambda v: fmt_date(v, style) if v is not None and not pd.isna(v) else "–"


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
  .save-chart-image {{ display: inline-flex; align-items: center; gap: 0.5rem; font: inherit; font-size: 0.875rem;
                       background: none; border: none; color: {p['ink']}; padding: 0.3rem 0.4rem; cursor: pointer; }}
  .save-chart-image .save-icon {{ font-family: "Material Symbols Rounded"; font-size: 20px; line-height: 1;
                                  color: {p['ink2']}; font-feature-settings: "liga"; }}
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
    """The source line under a page; `extra` adds other sources ("; ERA5: …"), already translated."""
    st.markdown(f"<div class='source'>{t('source.note', source=t('source.meteo_france'), extra=extra)}</div>",
                unsafe_allow_html=True)


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
