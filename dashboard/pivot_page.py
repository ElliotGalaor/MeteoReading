"""Dashboard page: pivot tables on the warehouse, built by picking from lists (no SQL typed by the user)."""
from __future__ import annotations

import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap

from meteo.pivot import TOTAL, Measure, PivotResult, PivotSpec, datasets, filter_values, run_pivot, year_range
from ui import HEAT_STOPS, RAIN_STOPS, SURFACE, TH, _connect, num, page_header

CATALOG = datasets(TH)
NONE = "(none)"
EXAMPLES = {
    "Warming by decade": dict(dataset="yearly", rows=["city"], column="decade", measure="tmean", agg="Mean"),
    "Hot days per year": dict(dataset="yearly", rows=["city"], column="decade", measure="hot_days", agg="Mean"),
    "Rainfall by month": dict(dataset="monthly", rows=["city"], column="month", measure="rr", agg="Mean"),
    "Heatwaves by month": dict(dataset="heatwaves", rows=["decade"], column="month", measure="episodes", agg="Count"),
    "Station network": dict(dataset="stations", rows=["altitude"], column="dep", measure="stations", agg="Count"),
    "25 million records": dict(dataset="records", rows=["dep"], column="decade", measure="tx", agg="Mean"),
}
# Empty cells take the page colour: by default a colour map paints them in transparent black, which pandas writes
# as opaque black, the darkest colour of both scales
COLOURS = {"heat": LinearSegmentedColormap.from_list("heat", HEAT_STOPS).with_extremes(bad=SURFACE),
           "blue": LinearSegmentedColormap.from_list("blue", RAIN_STOPS).with_extremes(bad=SURFACE)}
# Light text only on cells dark enough for it: at this luminance both text colours reach about 4.3:1
TEXT_THRESHOLD = 0.17


@st.cache_data(show_spinner=False)
def _filter_values(dataset: str, key: str) -> list[str]:
    with _connect() as con:
        return filter_values(con, dataset, key, CATALOG)


@st.cache_data(show_spinner=False)
def _year_range(dataset: str) -> tuple[int, int] | None:
    with _connect() as con:
        return year_range(con, dataset, CATALOG)


@st.cache_data(show_spinner=False, max_entries=200)
def _run(dataset: str, rows: tuple, column: str | None, measure: str, agg: str, filters: tuple,
         years: tuple | None, totals: bool) -> PivotResult:
    spec = PivotSpec(dataset, list(rows), column, measure, agg, {k: list(v) for k, v in filters}, years, totals)
    with _connect() as con:
        return run_pivot(con, spec, CATALOG)


def _apply_example(name: str) -> None:
    """Sets the widgets of the page to one of the examples (before the next run draws them)."""
    ex = EXAMPLES[name]
    ds = ex["dataset"]
    state = st.session_state
    state["pv_dataset"] = ds
    state[f"pv_rows_{ds}"] = ex["rows"]
    state[f"pv_col_{ds}"] = ex["column"] or NONE
    state[f"pv_measure_{ds}"] = ex["measure"]
    state[f"pv_agg_{ds}_{ex['measure']}"] = ex["agg"]
    for key in [k for k in state if k.startswith((f"pv_f_{ds}_", f"pv_years_{ds}"))]:
        del state[key]


def _on_example() -> None:
    name = st.session_state.get("pv_example")
    if name:
        _apply_example(name)
        st.session_state["pv_example"] = None   # a shortcut, not a mode: the widgets can be changed from there


def _style(result: PivotResult, measure: Measure, colour: bool):
    t = result.table
    values = [c for c in t.columns if c not in result.row_labels]
    body_cols = [c for c in values if c != TOTAL]
    body_rows = t.index[t[result.row_labels[0]] != TOTAL]
    styler = t.style.format(f"{{:,.{result.decimals}f}}", subset=values, na_rep="–")
    if colour and body_cols and len(body_rows):
        styler = styler.background_gradient(cmap=COLOURS[measure.scale], axis=None, text_color_threshold=TEXT_THRESHOLD,
                                            subset=pd.IndexSlice[body_rows, body_cols])
    return styler


def page_pivot():
    page_header("Pivot table", "Cross any dimension of the warehouse with any measure, by picking from lists: the "
                               "query is assembled on the server from a fixed catalog, nothing typed is ever run.")
    st.pills("Examples", list(EXAMPLES), key="pv_example", on_change=_on_example)

    keys = list(CATALOG)
    ds_key = st.selectbox("Dataset", keys, key="pv_dataset", format_func=lambda k: CATALOG[k].label)
    ds = CATALOG[ds_key]
    st.caption(ds.description)
    dims, measures = ds.dimensions, ds.measures

    c1, c2, c3, c4 = st.columns([1.6, 1.1, 1.6, 1])
    # Defaults go through the session state, which the examples also write to (not through default= / index=)
    st.session_state.setdefault(f"pv_rows_{ds_key}", [next(iter(dims))])
    rows = c1.multiselect("Rows", list(dims), max_selections=2, key=f"pv_rows_{ds_key}",
                          format_func=lambda k: dims[k].label, help="One or two dimensions.")
    col_options = [NONE] + [k for k in dims if k not in rows]
    column = c2.selectbox("Columns", col_options, key=f"pv_col_{ds_key}",
                          format_func=lambda k: k if k == NONE else dims[k].label)
    measure_key = c3.selectbox("Measure", list(measures), key=f"pv_measure_{ds_key}",
                               format_func=lambda k: measures[k].label)
    measure = measures[measure_key]
    st.session_state.setdefault(f"pv_agg_{ds_key}_{measure_key}", measure.agg)
    agg = c4.selectbox("Aggregation", list(measure.aggs), key=f"pv_agg_{ds_key}_{measure_key}")

    filterable = [k for k, d in dims.items() if d.filterable]
    bounds = _year_range(ds_key)
    f_cols = st.columns([1] * len(filterable) + ([1] if bounds else []) + ([1] if len(filterable) + bool(bounds) < 2 else []))
    filters = {}
    for col, key in zip(f_cols, filterable, strict=False):
        chosen = col.multiselect(f"{dims[key].label} filter", _filter_values(ds_key, key), key=f"pv_f_{ds_key}_{key}",
                                 placeholder="All")
        if chosen:
            filters[key] = tuple(chosen)
    years = None
    if bounds:
        lo, hi = f_cols[len(filterable)].slider("Years", bounds[0], bounds[1], bounds, key=f"pv_years_{ds_key}")
        years = (lo, hi) if (lo, hi) != bounds else None
    t1, t2, _ = st.columns([1, 1, 3])
    totals = t1.toggle("Totals", value=True, help="Computed on the underlying values: the total of a median is the "
                                                  "median of all the values.")
    colour = t2.toggle("Colour scale", value=True)

    if not rows:
        st.info("Pick at least one dimension for the rows.")
        return
    try:
        result = _run(ds_key, tuple(rows), None if column == NONE else column, measure_key, agg,
                      tuple(sorted(filters.items())), years, totals)
    except ValueError as exc:
        st.warning(str(exc))
        return
    if result.long.empty:
        st.info("No data for this selection.")
        return

    detail = (result.long.filter(regex=r"^g\d+$") == 0).all(axis=1)
    by = " and ".join(dims[k].label.lower() for k in rows + ([] if column == NONE else [column]))
    unit = "" if agg == "Count" else f" ({measure.unit})"
    what = measure.label if agg == "Count" else f"{measure.label}{unit} · {agg.lower()}"
    st.markdown(f"<div class='chart-title'>{what} by {by}</div>", unsafe_allow_html=True)
    st.caption(f"{num(int(detail.sum()), 0)} cells from {num(int(result.long.loc[detail, 'n'].sum()), 0)} values, "
               f"computed in {num(result.elapsed * 1000, 0)} ms")
    height = min(36 * (len(result.table) + 1) + 4, 640)
    st.dataframe(_style(result, measure, colour), hide_index=True, height=height, placeholder="–")
    st.download_button("Download as CSV", result.table.to_csv(index=False).encode("utf-8"), "pivot_table.csv",
                       "text/csv", icon=":material/download:")
    with st.expander("The query that ran"):
        st.caption("Assembled on the server from the catalog in meteo/pivot.py; the filter values are bound to the "
                   "? placeholders, never written into the SQL.")
        st.code(result.sql, language="sql")
        st.caption(f"Bound values: {result.params}" if result.params else "No bound value (no filter).")
