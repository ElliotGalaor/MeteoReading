"""Pivot tables on the warehouse (meteo/pivot.py): no SQL from the user, correct totals, natural order."""
from __future__ import annotations

import duckdb
import pytest

from meteo.pivot import PivotSpec, build_query, filter_values, run_pivot, year_range


@pytest.fixture
def con():
    c = duckdb.connect()
    c.execute("""
        CREATE TABLE city_yearly AS SELECT * FROM (VALUES
            ('A', 1990, 10.0, true), ('A', 1991, 11.0, true), ('A', 1992, 30.0, true), ('A', 1993, 99.0, false),
            ('B', 1990, 12.0, true), ('B', 1991, 13.0, true)
        ) t(city, year, tmean, valid_temp)""")
    c.execute("""
        CREATE TABLE city_monthly AS SELECT * FROM (VALUES
            ('A', 2000, 12, 5.0, true), ('A', 2000, 1, 4.0, true), ('A', 2000, 6, 20.0, true)
        ) t(city, year, month, tmean, valid_temp)""")
    return c


def spec(**kw) -> PivotSpec:
    base = dict(dataset="yearly", rows=["city"], column="decade", measure="tmean", agg="Mean")
    return PivotSpec(**(base | kw))


# --- Nothing from the user reaches the SQL text ---------------------------------------------
@pytest.mark.parametrize("kw", [
    {"dataset": "yearly; DROP TABLE city_yearly"},
    {"rows": ["city) OR 1=1 --"]},
    {"column": "decade, (SELECT 1)"},
    {"measure": "tmean FROM city_yearly; --"},
    {"agg": "Mean); DROP TABLE city_yearly; --"},
    {"filters": {"city = city OR 1": ["A"]}},
    {"years": ("1990; DROP TABLE city_yearly", 2000)},
])
def test_anything_outside_the_catalog_is_refused(kw):
    with pytest.raises(ValueError):
        build_query(spec(**kw))


def test_filter_values_are_bound_not_pasted(con):
    sneaky = "A' OR '1'='1"
    sql, params = build_query(spec(filters={"city": [sneaky]}))
    assert sneaky not in sql and "?" in sql and params == [sneaky]
    assert run_pivot(con, spec(filters={"city": [sneaky]})).long.empty   # matches no city, not every city


def test_only_category_dimensions_can_be_filtered():
    with pytest.raises(ValueError, match="cannot be filtered"):
        build_query(spec(filters={"year": ["1990"]}))


def test_meaningless_or_malformed_requests_are_refused():
    with pytest.raises(ValueError, match="does not apply"):
        build_query(spec(agg="Sum"))               # a sum of temperatures means nothing
    with pytest.raises(ValueError, match="only be used once"):
        build_query(spec(rows=["city"], column="city"))
    with pytest.raises(ValueError, match="one or two"):
        build_query(spec(rows=[]))


# --- Results ------------------------------------------------------------------------------------
def test_totals_are_computed_on_the_values_not_on_the_cells(con):
    r = run_pivot(con, spec(column=None, agg="Median"))
    table = r.table.set_index("City")["Mean temperature"]
    assert table["A"] == 11 and table["B"] == 12.5     # 1993 is an incomplete year: left out
    assert table["Total"] == 12                        # median of 10, 11, 12, 13, 30 (not of 11 and 12.5)


def test_pivot_layout_with_row_and_column_totals(con):
    r = run_pivot(con, spec(years=(1990, 1991)))
    assert list(r.table.columns) == ["City", "1990s", "Total"]
    assert r.table["City"].tolist() == ["A", "B", "Total"]
    assert r.table.set_index("City").loc["Total", "Total"] == pytest.approx(11.5)
    assert r.params == [1990, 1991]


def test_months_keep_their_calendar_order_and_the_total_comes_last(con):
    r = run_pivot(con, spec(dataset="monthly", column="month"))
    assert list(r.table.columns) == ["City", "Jan", "Jun", "Dec", "Total"]


def test_filter_choices_and_year_bounds(con):
    assert filter_values(con, "yearly", "city") == ["A", "B"]
    assert year_range(con, "yearly") == (1990, 1993)
    assert year_range(con, "stations") is None


def test_empty_cells_keep_the_page_colour(con, monkeypatch):
    """A colour map paints missing values in transparent black, which pandas writes as opaque black (#000000)."""
    monkeypatch.syspath_prepend("dashboard")
    import pivot_page

    r = run_pivot(con, spec(column="year", years=(1990, 1992)))     # city B has no value for 1992
    styler = pivot_page._style(r, pivot_page.CATALOG["yearly"].measures["tmean"], colour=True)
    styler._compute()
    cols = list(r.table.columns)
    empty = dict(styler.ctx[(1, cols.index("1992"))])
    assert empty["background-color"] == pivot_page.SURFACE
    assert dict(styler.ctx[(0, cols.index("1992"))])["background-color"] not in ("#000000", pivot_page.SURFACE)
