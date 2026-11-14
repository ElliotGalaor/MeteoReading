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

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SEASONS = ["Winter", "Spring", "Summer", "Autumn"]
TEMP_AGGS = ("Mean", "Median", "Min", "Max", "Std dev", "Count")
AMOUNT_AGGS = ("Mean", "Median", "Min", "Max", "Sum", "Std dev", "Count")
AGGREGATIONS = {"Mean": "avg({})", "Median": "median({})", "Min": "min({})", "Max": "max({})", "Sum": "sum({})",
                "Std dev": "stddev_samp({})", "Count": "count({})"}
MAX_CELLS = 20_000
TOTAL = "Total"


@dataclass(frozen=True)
class Dimension:
    label: str
    sql: str                                   # expression on the dataset's table, sorting in the natural order
    labels: Callable[[object], str] | None = None   # how a value is shown (month 1 -> "Jan")
    filterable: bool = False                   # offered as a filter (a category with few values)


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
def _city() -> Dimension:
    return Dimension("City", "city", filterable=True)


def _dep() -> Dimension:
    return Dimension("Département", "dep", filterable=True)


def _decade(year: str) -> Dimension:
    return Dimension("Decade", f"({year}) // 10 * 10", labels=lambda v: f"{int(v)}s")


def _month(month: str) -> Dimension:
    return Dimension("Month", month, labels=lambda v: MONTHS[int(v) - 1])


def _season(month: str) -> Dimension:
    """Meteorological seasons: winter is December, January and February."""
    sql = (f"CASE WHEN {month} IN (12, 1, 2) THEN 1 WHEN {month} IN (3, 4, 5) THEN 2 "
           f"WHEN {month} IN (6, 7, 8) THEN 3 ELSE 4 END")
    return Dimension("Season", sql, labels=lambda v: SEASONS[int(v) - 1])


def _calendar(year: str, month: str | None) -> dict[str, Dimension]:
    dims = {"year": Dimension("Year", year), "decade": _decade(year)}
    if month:
        dims |= {"season": _season(month), "month": _month(month)}
    return dims


def datasets(th: dict | None = None) -> dict[str, Dataset]:
    """The catalog; `th` holds the thresholds of config.yaml, used in the labels of the indicators."""
    th = {"hot_day": 30, "very_hot_day": 35, "tropical_night": 20, "frost_day": 0, "rain_day": 1, "heavy_rain": 20,
          "heatwave_min_days": 3} | (th or {})

    def temp(label: str, col: str, valid: str | None = None, **kw) -> Measure:
        sql = f"CASE WHEN {valid} THEN {col} END" if valid else col
        return Measure(label, sql, "°C", scale="heat", **kw)

    def rain(label: str, col: str, valid: str | None = None, **kw) -> Measure:
        sql = f"CASE WHEN {valid} THEN {col} END" if valid else col
        return Measure(label, sql, "mm", aggs=AMOUNT_AGGS, **kw)

    def days(label: str, col: str, valid: str, scale: str = "blue", **kw) -> Measure:
        return Measure(label, f"CASE WHEN {valid} THEN {col} END", "days", aggs=AMOUNT_AGGS, scale=scale, **kw)

    return {
        "yearly": Dataset(
            "Yearly indicators by city", "city_yearly",
            "One row per city and per year, 1951-2025. Incomplete years are left out of each measure.",
            {"city": _city()} | _calendar("year", None),
            {
                "tmean": temp("Mean temperature", "tmean", "valid_temp"),
                "tx_max": temp("Hottest day (TX max)", "tx_max", "valid_temp", agg="Max"),
                "tn_min": temp("Coldest night (TN min)", "tn_min", "valid_temp", agg="Min"),
                "rr": rain("Annual rainfall", "rr", "valid_rain", decimals=0),
                "hot_days": days(f"Hot days (TX ≥ {th['hot_day']} °C)", "hot_days", "valid_temp", "heat"),
                "very_hot_days": days(f"Very hot days (TX ≥ {th['very_hot_day']} °C)", "very_hot_days", "valid_temp",
                                      "heat"),
                "tropical_nights": days(f"Tropical nights (TN ≥ {th['tropical_night']} °C)", "tropical_nights",
                                        "valid_temp", "heat"),
                "frost_days": days(f"Frost days (TN < {th['frost_day']} °C)", "frost_days", "valid_temp"),
                "rain_days": days(f"Rain days (RR ≥ {th['rain_day']} mm)", "rain_days", "valid_rain"),
                "heavy_rain_days": days(f"Heavy rain days (RR ≥ {th['heavy_rain']} mm)", "heavy_rain_days",
                                        "valid_rain"),
                "longest_dry_spell": days("Longest dry spell", "longest_dry_spell", "valid_rain", agg="Max"),
                "heatwave_days": days("Heatwave days", "heatwave_days", "valid_temp", "heat"),
            },
            year_sql="year",
        ),
        "monthly": Dataset(
            "Monthly values by city", "city_monthly",
            "One row per city and per month. Incomplete months are left out of each measure.",
            {"city": _city()} | _calendar("year", "month"),
            {
                "tmean": temp("Mean temperature", "tmean", "valid_temp"),
                "tn": temp("Mean minimum temperature", "tn", "valid_temp"),
                "tx": temp("Mean maximum temperature", "tx", "valid_temp"),
                "rr": rain("Monthly rainfall", "rr", "valid_rain", decimals=0),
                "rain_days": days(f"Rain days (RR ≥ {th['rain_day']} mm)", "rain_days", "valid_rain"),
            },
            year_sql="year",
        ),
        "daily": Dataset(
            "Daily values by city", "city_daily",
            "One row per city and per day, 1951-2025 (about 550,000 rows).",
            {"city": _city()} | _calendar("year(date)", "month(date)"),
            {
                "tmean": temp("Mean temperature", "tmean"),
                "tn": temp("Minimum temperature", "tn"),
                "tx": temp("Maximum temperature", "tx"),
                "rr": rain("Daily rainfall", "rr"),
                "ffm": Measure("Mean wind speed", "ffm", "m/s"),
                "fxi": Measure("Strongest gust", "fxi", "m/s", agg="Max"),
            },
            year_sql="year(date)",
        ),
        "heatwaves": Dataset(
            "Heatwaves", "city_heatwaves",
            f"One row per heat episode: at least {th['heatwave_min_days']} consecutive days with "
            f"TX ≥ {th['hot_day']} °C, dated by its first day.",
            {"city": _city()} | _calendar("year(start_date)", "month(start_date)"),
            {
                "episodes": Measure("Number of heatwaves", "start_date", "episodes", agg="Count", aggs=("Count",),
                                    decimals=0, scale="heat"),
                "n_days": Measure("Length", "n_days", "days", aggs=AMOUNT_AGGS, scale="heat"),
                "tx_peak": temp("Peak temperature", "tx_peak", agg="Max"),
            },
            year_sql="year(start_date)",
        ),
        "stations": Dataset(
            "Weather stations", "dim_station",
            "Every station of the downloaded départements, open or closed, with its completeness.",
            {"dep": _dep(),
             "altitude": Dimension("Altitude", "altitude // 250 * 250", labels=lambda v: f"{int(v)}-{int(v) + 249} m"),
             "opened": Dimension("First record (decade)", "year(first_date) // 10 * 10", labels=lambda v: f"{int(v)}s"),
             "closed": Dimension("Last record (decade)", "year(last_date) // 10 * 10", labels=lambda v: f"{int(v)}s")},
            {
                "stations": Measure("Number of stations", "station_id", "stations", agg="Count", aggs=("Count",),
                                    decimals=0),
                "altitude": Measure("Altitude", "altitude", "m", decimals=0),
                "completeness_temp": Measure("Temperature completeness", "100 * completeness_temp", "%", decimals=0),
                "n_rows": Measure("Daily records", "n_rows", "rows", agg="Sum", aggs=AMOUNT_AGGS, decimals=0),
            },
        ),
        "records": Dataset(
            "All daily records (25 million rows)", "obs_daily",
            "Every validated daily record of every station, by département: DuckDB aggregates the 25 million rows "
            "in well under a second.",
            {"dep": _dep()} | _calendar("year(date)", "month(date)"),
            {
                "tn": temp("Minimum temperature", "tn"),
                "tx": temp("Maximum temperature", "tx"),
                "tm": temp("Mean temperature", "tm"),
                "rr": rain("Daily rainfall", "rr"),
                "fxi": Measure("Strongest gust", "fxi", "m/s", agg="Max"),
            },
            year_sql="year(date)",
        ),
    }


DATASETS = datasets()


# --- Query -------------------------------------------------------------------------------------
def _pick(options: dict, key, what: str):
    if not isinstance(key, str) or key not in options:
        raise ValueError(f"Unknown {what}: {key!r}")
    return options[key]


def build_query(spec: PivotSpec, catalog: dict[str, Dataset] = DATASETS) -> tuple[str, list]:
    """SQL text made only of catalog fragments, and the values to bind to its ? placeholders."""
    ds = _pick(catalog, spec.dataset, "dataset")
    if not 1 <= len(spec.rows) <= 2:
        raise ValueError("Pick one or two dimensions for the rows")
    keys = [*spec.rows] + ([spec.column] if spec.column else [])
    if len(set(keys)) != len(keys):
        raise ValueError("A dimension can only be used once")
    exprs = [_pick(ds.dimensions, k, "dimension").sql for k in keys]
    measure = _pick(ds.measures, spec.measure, "measure")
    if spec.agg not in measure.aggs:
        raise ValueError(f"{spec.agg!r} does not apply to {measure.label}: choose one of {', '.join(measure.aggs)}")

    where, params = [], []
    for key, values in spec.filters.items():
        dim = _pick(ds.dimensions, key, "filter")
        if not dim.filterable:
            raise ValueError(f"{dim.label} cannot be filtered")
        if values:
            where.append(f"{dim.sql} IN ({', '.join('?' for _ in values)})")
            params.extend(str(v) for v in values)
    if spec.years:
        if ds.year_sql is None:
            raise ValueError(f"{ds.label} has no year to filter on")
        lo, hi = (int(y) for y in spec.years)   # anything that is not a number stops here
        where.append(f"{ds.year_sql} BETWEEN ? AND ?")
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
           f"ORDER BY {', '.join(f'g{i}, d{i}' for i in range(len(keys)))}\nLIMIT {MAX_CELLS + 1}")
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
    if len(long) > MAX_CELLS:
        raise ValueError(f"More than {MAX_CELLS:,} cells: add a filter or choose broader dimensions")
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
        table = frame[row_labels + ["value"]].rename(columns={"value": measure.label})
    for name in row_labels:
        table[name] = table[name].astype(str)
    return PivotResult(table, row_labels, long, sql, params, elapsed, measure.unit, decimals)


def filter_values(con, dataset: str, key: str, catalog: dict[str, Dataset] = DATASETS) -> list[str]:
    """Values offered for a filter (the query only uses catalog fragments)."""
    dim = _pick(_pick(catalog, dataset, "dataset").dimensions, key, "filter")
    if not dim.filterable:
        raise ValueError(f"{dim.label} cannot be filtered")
    rows = con.execute(f"SELECT DISTINCT {dim.sql} FROM {catalog[dataset].table} WHERE {dim.sql} IS NOT NULL ORDER BY 1")
    return [str(r[0]) for r in rows.fetchall()]


def year_range(con, dataset: str, catalog: dict[str, Dataset] = DATASETS) -> tuple[int, int] | None:
    ds = _pick(catalog, dataset, "dataset")
    if ds.year_sql is None:
        return None
    lo, hi = con.execute(f"SELECT min({ds.year_sql}), max({ds.year_sql}) FROM {ds.table}").fetchall()[0]
    return int(lo), int(hi)
