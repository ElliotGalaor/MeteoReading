"""Dashboard page: pivot tables on the warehouse, built by picking from lists (no SQL typed by the user)."""
from __future__ import annotations

import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap

from meteo.pivot import TOTAL, Measure, PivotResult, PivotSpec, datasets, filter_values, run_pivot, year_range
from share import from_link, publish, restore
from ui import (
    DARK_HEAT_STOPS,
    DARK_RAIN_STOPS,
    DARK_SURFACE,
    HEAT_STOPS,
    RAIN_STOPS,
    SURFACE,
    TH,
    _connect,
    csv_bytes,
    is_dark,
    lang,
    listing,
    num,
    page_header,
    plural,
    t,
)

STRUCTURE = datasets(TH)   # SQL and keys only (the same in every language); labels: catalog()
NONE = "(none)"
EXAMPLES = {   # code -> choices; named t("pivot.ex_…")
    "warming": dict(dataset="yearly", rows=["city"], column="decade", measure="tmean", agg="Mean"),
    "hot_days": dict(dataset="yearly", rows=["city"], column="decade", measure="hot_days", agg="Mean"),
    "rain_month": dict(dataset="monthly", rows=["city"], column="month", measure="rr", agg="Mean"),
    "heatwaves": dict(dataset="heatwaves", rows=["decade"], column="month", measure="episodes", agg="Count"),
    "stations": dict(dataset="stations", rows=["altitude"], column="dep", measure="stations", agg="Count"),
    "records": dict(dataset="records", rows=["dep"], column="decade", measure="tx", agg="Mean"),
}
# Empty cells take the page colour: by default a colour map paints them in transparent black, which pandas writes
# as opaque black, the darkest colour of both scales
COLOURS = {"heat": LinearSegmentedColormap.from_list("heat", HEAT_STOPS).with_extremes(bad=SURFACE),
           "blue": LinearSegmentedColormap.from_list("blue", RAIN_STOPS).with_extremes(bad=SURFACE)}
# Dark mode: low values recede into the dark surface, strong ones are bright
DARK_COLOURS = {"heat": LinearSegmentedColormap.from_list("heat_dark", DARK_HEAT_STOPS).with_extremes(bad=DARK_SURFACE),
                "blue": LinearSegmentedColormap.from_list("blue_dark", DARK_RAIN_STOPS).with_extremes(bad=DARK_SURFACE)}
# Light text only on cells dark enough for it: at this luminance both text colours reach about 4.3:1
TEXT_THRESHOLD = 0.17


def catalog():
    """The catalog with its labels in the visitor's language."""
    return datasets(TH)


@st.cache_data(show_spinner=False)
def _filter_values(dataset: str, key: str) -> list[str]:
    with _connect() as con:
        return filter_values(con, dataset, key, STRUCTURE)


@st.cache_data(show_spinner=False)
def _year_range(dataset: str) -> tuple[int, int] | None:
    with _connect() as con:
        return year_range(con, dataset, STRUCTURE)


@st.cache_data(show_spinner=False, max_entries=200)
def _run(dataset: str, rows: tuple, column: str | None, measure: str, agg: str, filters: tuple,
         years: tuple | None, totals: bool, language: str) -> PivotResult | str:
    """The result, or the reason it was refused (cached as well, so a refused query is not run again); per language,
    as the table holds the labels."""
    spec = PivotSpec(dataset, list(rows), column, measure, agg, {k: list(v) for k, v in filters}, years, totals)
    try:
        with _connect() as con:
            return run_pivot(con, spec, catalog())
    except ValueError as exc:
        return str(exc)


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
    # Reset the filters by giving the widgets a value: deleting their keys left the old values on screen, and the
    # browser sent them back at the next change
    for key, dim in STRUCTURE[ds].dimensions.items():
        if dim.filterable:
            state[f"pv_f_{ds}_{key}"] = []
    bounds = _year_range(ds)
    if bounds:
        state[f"pv_years_{ds}"] = bounds


def _on_example() -> None:
    name = st.session_state.get("pv_example")
    if name:
        _apply_example(name)
        st.session_state["pv_example"] = None   # a shortcut, not a mode: the widgets can be changed from there


def _style(result: PivotResult, measure: Measure, colour: bool):
    table = result.table
    values = [c for c in table.columns if c not in result.row_labels]
    body_cols = [c for c in values if c != TOTAL]
    body_rows = table.index[table[result.row_labels[0]] != TOTAL]
    styler = table.style.format(lambda v: num(v, result.decimals), subset=values, na_rep="–")
    if colour and body_cols and len(body_rows):
        cmaps = DARK_COLOURS if is_dark() else COLOURS
        styler = styler.background_gradient(cmap=cmaps[measure.scale], axis=None, text_color_threshold=TEXT_THRESHOLD,
                                            subset=pd.IndexSlice[body_rows, body_cols])
    return styler


def page_pivot():
    page_header(t("pivot.title"), t("pivot.lead"), tab=t("app.nav_pivot"))
    st.pills(t("pivot.examples"), list(EXAMPLES), key="pv_example", on_change=_on_example,
             format_func=lambda k: t(f"pivot.ex_{k}"))

    cat = catalog()
    keys = list(cat)
    restore("pv_dataset", "dataset", valid=lambda v: v in cat)
    ds_key = st.selectbox(t("pivot.dataset"), keys, key="pv_dataset", format_func=lambda k: cat[k].label)
    ds = cat[ds_key]
    st.caption(ds.description)
    dims, measures = ds.dimensions, ds.measures
    # A shared link: its choices for this dataset (anything unknown is ignored)
    restore(f"pv_rows_{ds_key}", "rows", multi=True, valid=lambda v: 0 < len(v) <= 2 and set(v) <= set(dims))
    restore(f"pv_col_{ds_key}", "column", valid=lambda v: v in dims)
    restore(f"pv_measure_{ds_key}", "measure", valid=lambda v: v in measures)

    c1, c2, c3, c4 = st.columns([1.6, 1.1, 1.6, 1])
    # Defaults go through the session state, which the examples also write to (not through default= / index=)
    st.session_state.setdefault(f"pv_rows_{ds_key}", [next(iter(dims))])
    rows = c1.multiselect(t("pivot.rows"), list(dims), max_selections=2, key=f"pv_rows_{ds_key}",
                          format_func=lambda k: dims[k].label, help=t("pivot.rows_help"))
    col_options = [NONE] + [k for k in dims if k not in rows]
    column = c2.selectbox(t("pivot.columns"), col_options, key=f"pv_col_{ds_key}",
                          format_func=lambda k: t("pivot.none") if k == NONE else dims[k].label)
    measure_key = c3.selectbox(t("pivot.measure"), list(measures), key=f"pv_measure_{ds_key}",
                               format_func=lambda k: measures[k].label)
    measure = measures[measure_key]
    restore(f"pv_agg_{ds_key}_{measure_key}", "agg", measure.agg, valid=lambda v: v in measure.aggs)
    agg = c4.selectbox(t("pivot.aggregation"), list(measure.aggs), key=f"pv_agg_{ds_key}_{measure_key}",
                       format_func=lambda a: t(f"pivot.agg_{a}"))

    filterable = [k for k, d in dims.items() if d.filterable]
    bounds = _year_range(ds_key)
    f_cols = st.columns([1] * len(filterable) + ([1] if bounds else []) + ([1] if len(filterable) + bool(bounds) < 2 else []))
    filters = {}
    for col, key in zip(f_cols, filterable, strict=False):
        restore(f"pv_f_{ds_key}_{key}", f"f_{key}", multi=True,
                valid=lambda v, k=key: set(v) <= set(_filter_values(ds_key, k)))
        chosen = col.multiselect(t("pivot.filter", dim=dims[key].label), _filter_values(ds_key, key),
                                 key=f"pv_f_{ds_key}_{key}", placeholder=t("pivot.all"))
        if chosen:
            filters[key] = tuple(chosen)
    years = None
    if bounds:
        first, last = from_link("from"), from_link("to")
        if first or last:
            try:
                lo, hi = sorted((int(first or bounds[0]), int(last or bounds[1])))
                if bounds[0] <= lo <= hi <= bounds[1]:
                    st.session_state[f"pv_years_{ds_key}"] = (lo, hi)
            except ValueError:
                pass
        st.session_state.setdefault(f"pv_years_{ds_key}", bounds)
        years = f_cols[len(filterable)].slider(t("pivot.years"), bounds[0], bounds[1], key=f"pv_years_{ds_key}",
                                               help=t("pivot.years_help"))
    t1, t2, _ = st.columns([1, 1, 3])
    totals = t1.toggle(t("pivot.totals"), value=True, help=t("pivot.totals_help"))
    colour = t2.toggle(t("pivot.colour"), value=True)
    publish({"dataset": ds_key, "rows": rows, "column": None if column == NONE else column, "measure": measure_key,
             "agg": None if agg == measure.agg else agg, **{f"f_{k}": list(v) for k, v in filters.items()},
             "from": years[0] if years and years[0] != bounds[0] else None,
             "to": years[1] if years and years[1] != bounds[1] else None})

    if not rows:
        st.info(t("pivot.pick_rows"))
        return
    result = _run(ds_key, tuple(rows), None if column == NONE else column, measure_key, agg,
                  tuple(sorted(filters.items())), tuple(years) if years else None, totals, lang())
    if isinstance(result, str):
        st.warning(result)
        return
    if result.long.empty:
        st.info(t("pivot.no_data"))
        return

    detail = (result.long.filter(regex=r"^g\d+$") == 0).all(axis=1)
    picked = rows + ([] if column == NONE else [column])
    by = listing([dims[k].label.lower() for k in picked])
    if measure.aggs == ("Count",):        # the measure is itself a count (heatwaves, stations)
        what = measure.label
    elif agg == "Count":                  # how many rows have a value: say what is counted
        what = t("pivot.count_of", rows=ds.row_name, measure=measure.label.lower())
    else:
        what = f"{measure.label} ({measure.unit}) · {t(f'pivot.agg_{agg}').lower()}"
    st.markdown(f"<div class='chart-title'>{t('pivot.what_by', what=what, by=by)}</div>", unsafe_allow_html=True)
    st.caption(t("pivot.cells", cells=plural("pivot.n_cells", int(detail.sum())),
                 values=plural("pivot.n_values", int(result.long.loc[detail, "n"].sum())),
                 ms=num(result.elapsed * 1000, 0)))
    if "decade" in picked and agg in ("Count", "Sum") and years:
        lo, hi = years
        partial = [t("pivot.partial_decade", decade=d, n=min(hi, d + 9) - max(lo, d) + 1)
                   for d in sorted({lo // 10 * 10, hi // 10 * 10}) if max(lo, d) > d or min(hi, d + 9) < d + 9]
        if partial:
            st.caption(t("pivot.partial", decades=listing(partial)))
    height = min(36 * (len(result.table) + 1) + 4, 640)
    st.dataframe(_style(result, measure, colour), hide_index=True, height=height, placeholder="–")
    csv = result.table.copy()
    for c in [c for c in csv.columns if c not in result.row_labels]:
        csv[c] = csv[c].round(result.decimals).astype("Int64" if result.decimals == 0 else float)
    st.download_button(t("chart.download_table"), csv_bytes(csv, f"{what} · {by}"), "pivot_table.csv", "text/csv",
                       icon=":material/download:")
    with st.expander(t("pivot.query_title")):
        st.caption(t("pivot.query_caption"))
        st.code(result.sql, language="sql")
        st.caption(t("pivot.bound", values=result.params) if result.params else t("pivot.no_bound"))
