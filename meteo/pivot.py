"""Pivot tables on the warehouse, with no SQL coming from the user.

The pivot page of the dashboard only lets the user pick names from the catalog below: a dataset, the dimensions
in rows and in columns, a measure, an aggregation, and filter values. The query is assembled here from those
whitelisted SQL fragments, and the filter values are bound as parameters (?), never pasted into the SQL text.
Any name that is not in the catalog is refused with a ValueError. The dashboard's connection is also read-only,
with no access to files (dashboard/ui.py): two locks rather than one.

Totals come from GROUPING SETS: they are computed on the underlying rows, so the total of a median is the median
of all the values, not the median of the medians.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from .fmt import num
from .i18n import month_name, t

SEASONS = ["Winter", "Spring", "Summer", "Autumn"]   # codes, named t("season.Winter")…
TEMP_AGGS = ("Mean", "Median", "Min", "Max", "Std dev", "Count")
AMOUNT_AGGS = ("Mean", "Median", "Min", "Max", "Sum", "Std dev", "Count")
AGGREGATIONS = {"Mean": "avg({})", "Median": "median({})", "Min": "min({})", "Max": "max({})", "Sum": "sum({})",
                "Std dev": "stddev_samp({})", "Count": "count({})"}
MAX_CELLS = 20_000
# The cap is on the cells, not on the totals: with at most MAX_CELLS cells, totals add at most one per row and
# column, plus the grand total, so a complete answer has fewer rows than this
ROW_LIMIT = 2 * MAX_CELLS + 2
TOTAL = "Total"   # the same word in English and French


@dataclass(frozen=True)
class Dimension:
    label: str
    sql: str                                   # expression on the dataset's table, sorting in the natural order
    labels: Callable[[object], str] | None = None   # how a value is shown (month 1 -> "Jan" / "janv.")
    filterable: bool = False                   # offered as a filter (a category with few values)
    season_sql: str | None = None              # expression used when the season is also picked (winter year)


@dataclass(frozen=True)
class Measure:
    label: str
    sql: str
    unit: str = ""
    agg: str = "Mean"                          # default aggregation
    aggs: tuple[str, ...] = TEMP_AGGS          # aggregations that make sense (no sum of temperatures)
    decimals: int = 1
    scale: str = "blue"                        # colour scale of the table: "heat" or "blue"


@dataclass(frozen=True)
class Dataset:
    label: str
    table: str
    description: str
    dimensions: dict[str, Dimension]
    measures: dict[str, Measure]
    year_sql: str | None = None                # expression of the year, for the year range filter
    season_year_sql: str | None = None         # the same, with December counted in the next year's winter
    span_sql: str | None = None                # first and last day of the data behind it: complete years only
    row_name: str = "rows"                     # what one row is, for counts ("years", "days"…)


@dataclass
class PivotSpec:
    dataset: str
    rows: list[str]
    column: str | None
    measure: str
    agg: str
    filters: dict[str, list] = field(default_factory=dict)
    years: tuple[int, int] | None = None
    totals: bool = True


@dataclass
class PivotResult:
    table: pd.DataFrame        # rows: the row dimensions as columns; then one column per value of the column dimension
    row_labels: list[str]      # names of the row dimension columns in `table`
    long: pd.DataFrame         # one row per cell, as returned by the query (with the number of values n)
    sql: str
    params: list
    elapsed: float
    unit: str
    decimals: int


# --- Catalog ---------------------------------------------------------------------------------
# The labels are read from the language catalogs (meteo/locales) when datasets() is called: the dashboard builds the
# catalog in the visitor's language; the keys (dataset, dimension, measure, aggregation) never change.
def _city() -> Dimension:
    return Dimension(t("pivot.dim_city"), "city", filterable=True)


def _dep() -> Dimension:
    return Dimension(t("pivot.dim_dep"), "dep", filterable=True)


def _decade_label(v) -> str:
    return t("pivot.decade_label", decade=int(v))


def _decade(year: str, season_year: str | None = None) -> Dimension:
    return Dimension(t("pivot.dim_decade"), f"({year}) // 10 * 10", labels=_decade_label,
                     season_sql=f"({season_year}) // 10 * 10" if season_year else None)


def _season_year(year: str, month: str) -> str:
    """Year of the season: December belongs to the winter of the next year (winter 1963 = Dec 1962 to Feb 1963)."""
    return f"({year}) + CASE WHEN {month} = 12 THEN 1 ELSE 0 END"


def _month(month: str) -> Dimension:
    return Dimension(t("pivot.dim_month"), month, labels=lambda v: month_name(int(v), short=True))


def _season(month: str) -> Dimension:
    """Meteorological seasons: winter is December, January and February."""
    sql = (f"CASE WHEN {month} IN (12, 1, 2) THEN 1 WHEN {month} IN (3, 4, 5) THEN 2 "
           f"WHEN {month} IN (6, 7, 8) THEN 3 ELSE 4 END")
    return Dimension(t("pivot.dim_season"), sql, labels=lambda v: t(f"season.{SEASONS[int(v) - 1]}"))


def _calendar(year: str, month: str | None) -> dict[str, Dimension]:
    if not month:
        return {"year": Dimension(t("pivot.dim_year"), year), "decade": _decade(year)}
    sy = _season_year(year, month)
    return {"year": Dimension(t("pivot.dim_year"), year, season_sql=sy), "decade": _decade(year, sy),
            "season": _season(month), "month": _month(month)}


CITY_SPAN = "SELECT min(date), max(date) FROM city_daily"   # the city tables are all built from city_daily


def datasets(th: dict | None = None) -> dict[str, Dataset]:
    """The catalog, in the current language; `th` holds the thresholds of config.yaml, used in the labels of the
    indicators."""
    th = {"hot_day": 30, "very_hot_day": 35, "tropical_night": 20, "frost_day": 0, "rain_day": 1, "heavy_rain": 20,
          "heatwave_min_days": 3} | (th or {})
    days_unit = t("common.days_unit")

    def temp(key: str, col: str, valid: str | None = None, **kw) -> Measure:
        sql = f"CASE WHEN {valid} THEN {col} END" if valid else col
        return Measure(t(f"pivot.m_{key}", **th), sql, "°C", scale="heat", **kw)

    def rain(key: str, col: str, valid: str | None = None, **kw) -> Measure:
        sql = f"CASE WHEN {valid} THEN {col} END" if valid else col
        return Measure(t(f"pivot.m_{key}", **th), sql, "mm", aggs=AMOUNT_AGGS, **kw)

    def days(key: str, col: str, valid: str, scale: str = "blue", **kw) -> Measure:
        return Measure(t(f"pivot.m_{key}", **th), f"CASE WHEN {valid} THEN {col} END", days_unit, aggs=AMOUNT_AGGS,
                       scale=scale, **kw)

    return {
        "yearly": Dataset(
            t("pivot.ds_yearly"), "city_yearly", t("pivot.ds_yearly_desc"),
            {"city": _city()} | _calendar("year", None),
            {
                "tmean": temp("mean_temp", "tmean", "valid_temp"),
                "tx_max": temp("hottest_day", "tx_max", "valid_temp", agg="Max"),
                "tn_min": temp("coldest_night", "tn_min", "valid_temp", agg="Min"),
                "rr": rain("annual_rain", "rr", "valid_rain", decimals=0),
                "hot_days": days("hot_days", "hot_days", "valid_temp", "heat"),
                "very_hot_days": days("very_hot_days", "very_hot_days", "valid_temp", "heat"),
                "tropical_nights": days("tropical_nights", "tropical_nights", "valid_temp", "heat"),
                "frost_days": days("frost_days", "frost_days", "valid_temp"),
                "rain_days": days("rain_days", "rain_days", "valid_rain"),
                "heavy_rain_days": days("heavy_rain_days", "heavy_rain_days", "valid_rain"),
                "longest_dry_spell": days("longest_dry", "longest_dry_spell", "valid_rain", agg="Max"),
                "heatwave_days": days("heatwave_days", "heatwave_days", "valid_temp", "heat"),
            },
            year_sql="year", span_sql=CITY_SPAN, row_name=t("pivot.rows_years"),
        ),
        "monthly": Dataset(
            t("pivot.ds_monthly"), "city_monthly", t("pivot.ds_monthly_desc"),
            {"city": _city()} | _calendar("year", "month"),
            {
                "tmean": temp("mean_temp", "tmean", "valid_temp"),
                "tn": temp("mean_min", "tn", "valid_temp"),
                "tx": temp("mean_max", "tx", "valid_temp"),
                "rr": rain("monthly_rain", "rr", "valid_rain", decimals=0),
                "rain_days": days("rain_days", "rain_days", "valid_rain"),
            },
            year_sql="year", season_year_sql=_season_year("year", "month"), span_sql=CITY_SPAN,
            row_name=t("pivot.rows_months"),
        ),
        "daily": Dataset(
            t("pivot.ds_daily"), "city_daily", t("pivot.ds_daily_desc"),
            {"city": _city()} | _calendar("year(date)", "month(date)"),
            {
                "tmean": temp("mean_temp", "tmean"),
                "tn": temp("min_temp", "tn"),
                "tx": temp("max_temp", "tx"),
                "rr": rain("daily_rain", "rr"),
                # Stored in m/s by Météo-France, shown in km/h like the rest of the site
                "ffm": Measure(t("pivot.m_mean_wind"), "3.6 * ffm", "km/h", decimals=0),
                "fxi": Measure(t("pivot.m_gust"), "3.6 * fxi", "km/h", agg="Max", decimals=0),
            },
            year_sql="year(date)", season_year_sql=_season_year("year(date)", "month(date)"), span_sql=CITY_SPAN,
            row_name=t("pivot.rows_days"),
        ),
        "heatwaves": Dataset(
            t("pivot.ds_heatwaves"), "city_heatwaves", t("pivot.ds_heatwaves_desc", **th),
            {"city": _city()} | _calendar("year(start_date)", "month(start_date)"),
            {
                "episodes": Measure(t("pivot.m_n_heatwaves"), "start_date", t("pivot.unit_episodes"), agg="Count",
                                    aggs=("Count",), decimals=0, scale="heat"),
                "n_days": Measure(t("pivot.m_length"), "n_days", days_unit, aggs=AMOUNT_AGGS, scale="heat"),
                "tx_peak": temp("peak", "tx_peak", agg="Max"),
            },
            year_sql="year(start_date)", season_year_sql=_season_year("year(start_date)", "month(start_date)"),
            span_sql=CITY_SPAN, row_name=t("pivot.rows_heatwaves"),
        ),
        "stations": Dataset(
            t("pivot.ds_stations"), "dim_station", t("pivot.ds_stations_desc"),
            {"dep": _dep(),
             "altitude": Dimension(t("pivot.dim_altitude"), "altitude // 250 * 250",
                                   labels=lambda v: f"{int(v)}-{int(v) + 249} m"),
             "opened": Dimension(t("pivot.dim_opened"), "year(first_date) // 10 * 10", labels=_decade_label),
             "closed": Dimension(t("pivot.dim_closed"), "year(last_date) // 10 * 10", labels=_decade_label)},
            {
                "stations": Measure(t("pivot.m_n_stations"), "station_id", t("pivot.unit_stations"), agg="Count",
                                    aggs=("Count",), decimals=0),
                "altitude": Measure(t("pivot.m_altitude"), "altitude", "m", decimals=0),
                "completeness_temp": Measure(t("pivot.m_completeness"), "100 * completeness_temp", "%", decimals=0),
                "n_rows": Measure(t("pivot.m_daily_records"), "n_rows", t("pivot.unit_rows"), agg="Sum",
                                  aggs=AMOUNT_AGGS, decimals=0),
            },
            row_name=t("pivot.rows_stations"),
        ),
        "records": Dataset(
            t("pivot.ds_records"), "obs_daily", t("pivot.ds_records_desc"),
            {"dep": _dep()} | _calendar("year(date)", "month(date)"),
            {
                "tn": temp("min_temp", "tn"),
                "tx": temp("max_temp", "tx"),
                "tm": temp("mean_temp", "tm"),
                "rr": rain("daily_rain", "rr"),
                "fxi": Measure(t("pivot.m_gust"), "3.6 * fxi", "km/h", agg="Max", decimals=0),
            },
            year_sql="year(date)", season_year_sql=_season_year("year(date)", "month(date)"),
            span_sql="SELECT min(date), max(date) FROM obs_daily", row_name=t("pivot.rows_records"),
        ),
    }


DATASETS = datasets()


# --- Query -------------------------------------------------------------------------------------
def _pick(options: dict, key, what: str):
    if not isinstance(key, str) or key not in options:
        raise ValueError(t("pivot.err_unknown", what=t(f"pivot.what_{what}"), key=repr(key)))
    return options[key]


def build_query(spec: PivotSpec, catalog: dict[str, Dataset] = DATASETS) -> tuple[str, list]:
    """SQL text made only of catalog fragments, and the values to bind to its ? placeholders."""
    ds = _pick(catalog, spec.dataset, "dataset")
    if not 1 <= len(spec.rows) <= 2:
        raise ValueError(t("pivot.err_rows"))
    keys = [*spec.rows] + ([spec.column] if spec.column else [])
    if len(set(keys)) != len(keys):
        raise ValueError(t("pivot.err_twice"))
    # With the season, the year and decade are those of the season (December goes with the next winter)
    season = "season" in keys
    dims = [_pick(ds.dimensions, k, "dimension") for k in keys]
    exprs = [d.season_sql if season and d.season_sql else d.sql for d in dims]
    measure = _pick(ds.measures, spec.measure, "measure")
    if spec.agg not in measure.aggs:
        raise ValueError(t("pivot.err_agg", agg=repr(spec.agg), measure=measure.label,
                           aggs=", ".join(t(f"pivot.agg_{a}") for a in measure.aggs)))

    where, params = [], []
    for key, values in spec.filters.items():
        dim = _pick(ds.dimensions, key, "filter")
        if not dim.filterable:
            raise ValueError(t("pivot.err_filter", dim=dim.label))
        if values:
            where.append(f"{dim.sql} IN ({', '.join('?' for _ in values)})")
            params.extend(str(v) for v in values)
    if spec.years:
        if ds.year_sql is None:
            raise ValueError(t("pivot.err_years", dataset=ds.label))
        lo, hi = (int(y) for y in spec.years)   # anything that is not a number stops here
        year_sql = ds.season_year_sql if season and ds.season_year_sql else ds.year_sql
        where.append(f"{year_sql} BETWEEN ? AND ?")
        params += [lo, hi]

    n_rows = len(spec.rows)
    if spec.totals:
        rows = list(range(n_rows))
        # cells, total of each row, total of each column, grand total
        sets = [rows + [n_rows], rows, [n_rows], []] if spec.column else [rows, []]
        group = "GROUP BY GROUPING SETS (" + ", ".join(
            "(" + ", ".join(exprs[i] for i in s) + ")" for s in sets) + ")"
    else:
        group = "GROUP BY " + ", ".join(exprs)
    select = [f"{e} AS d{i}" for i, e in enumerate(exprs)] + [f"GROUPING({e}) AS g{i}" for i, e in enumerate(exprs)]
    value = AGGREGATIONS[spec.agg].format(measure.sql)
    sql = (f"SELECT {', '.join(select)},\n       {value} AS value, count({measure.sql}) AS n\n"
           f"FROM {ds.table}\n" + (f"WHERE {' AND '.join(where)}\n" if where else "") + f"{group}\n"
           f"ORDER BY {', '.join(f'g{i}, d{i}' for i in range(len(keys)))}\nLIMIT {ROW_LIMIT}")
    return sql, params


def _labels(long: pd.DataFrame, i: int, dim: Dimension, blank_total: bool) -> pd.Series:
    """Display labels of dimension i ("Total" on total rows), ordered by value with the total last."""
    show = dim.labels or (lambda v: str(v))

    def label(v, g) -> str:
        if g:
            return "" if blank_total else TOTAL
        return "–" if pd.isna(v) else show(v)

    out = [label(v, g) for v, g in zip(long[f"d{i}"], long[f"g{i}"], strict=True)]
    order = long[[f"g{i}", f"d{i}"]].drop_duplicates().sort_values([f"g{i}", f"d{i}"], na_position="last")
    categories = list(dict.fromkeys(label(v, g) for g, v in order.itertuples(index=False)))
    return pd.Series(pd.Categorical(out, categories=categories, ordered=True), index=long.index)


def run_pivot(con, spec: PivotSpec, catalog: dict[str, Dataset] = DATASETS) -> PivotResult:
    sql, params = build_query(spec, catalog)
    t0 = time.perf_counter()
    long = con.execute(sql, params).df()
    elapsed = time.perf_counter() - t0
    cells = int((long.filter(regex=r"^g\d+$") == 0).all(axis=1).sum())
    if len(long) >= ROW_LIMIT or cells > MAX_CELLS:
        raise ValueError(t("pivot.err_cells", cells=num(MAX_CELLS, 0)))
    ds = catalog[spec.dataset]
    measure = ds.measures[spec.measure]
    keys = [*spec.rows] + ([spec.column] if spec.column else [])
    dims = [ds.dimensions[k] for k in keys]
    row_labels = [d.label for d in dims[:len(spec.rows)]]
    decimals = 0 if spec.agg == "Count" else measure.decimals
    if not (long["n"] > 0).any():
        # No value at all (the grand total of an empty selection is still one row): nothing to show
        return PivotResult(pd.DataFrame(columns=row_labels), row_labels, long.iloc[0:0], sql, params, elapsed,
                           measure.unit, decimals)
    frame = pd.DataFrame({"value": long["value"]})
    for i, dim in enumerate(dims):
        # On the column-total row, only the first row dimension says "Total"
        frame[dim.label] = _labels(long, i, dim, blank_total=0 < i < len(spec.rows))
    if spec.column:
        col_label = dims[-1].label
        table = frame.pivot(index=row_labels, columns=col_label, values="value")
        table.columns = [str(c) for c in table.columns]
        table = table.reset_index()
    else:
        name = (measure.label if measure.label not in row_labels
                else f"{measure.label} · {t(f'pivot.agg_{spec.agg}').lower()}")
        table = frame[row_labels + ["value"]].rename(columns={"value": name})
    for name in row_labels:
        table[name] = table[name].astype(str)
    return PivotResult(table, row_labels, long, sql, params, elapsed, measure.unit, decimals)


def filter_values(con, dataset: str, key: str, catalog: dict[str, Dataset] = DATASETS) -> list[str]:
    """Values offered for a filter (the query only uses catalog fragments)."""
    dim = _pick(_pick(catalog, dataset, "dataset").dimensions, key, "filter")
    if not dim.filterable:
        raise ValueError(t("pivot.err_filter", dim=dim.label))
    rows = con.execute(f"SELECT DISTINCT {dim.sql} FROM {catalog[dataset].table} WHERE {dim.sql} IS NOT NULL ORDER BY 1")
    return [str(r[0]) for r in rows.fetchall()]


def year_range(con, dataset: str, catalog: dict[str, Dataset] = DATASETS) -> tuple[int, int] | None:
    ds = _pick(catalog, dataset, "dataset")
    if ds.year_sql is None:
        return None
    lo, hi = con.execute(f"SELECT min({ds.year_sql}), max({ds.year_sql}) FROM {ds.table}").fetchall()[0]
    if lo is None:
        return None
    lo, hi = int(lo), int(hi)
    if ds.span_sql:
        # Complete years only: a partial year (the current one) would look warmer or drier than the others
        first, last = con.execute(ds.span_sql).fetchall()[0]
        if first is not None:
            lo = max(lo, first.year if (first.month, first.day) == (1, 1) else first.year + 1)
            hi = min(hi, last.year if (last.month, last.day) == (12, 31) else last.year - 1)
    return (lo, hi) if lo <= hi else None
