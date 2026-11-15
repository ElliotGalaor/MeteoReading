"""Command line (meteo/__main__.py): plain errors, SQL scripts split correctly, only the rows shown fetched."""
from __future__ import annotations

from types import SimpleNamespace

import duckdb
import pytest

from meteo.__main__ import cmd_query, main, split_sql


def test_semicolons_in_strings_and_comments_do_not_split():
    script = "SELECT 'a;b' AS x; -- note; here\nSELECT 2 /* c;d */ AS y;\n-- only a comment\n"
    assert split_sql(script) == ["SELECT 'a;b' AS x", "-- note; here\nSELECT 2 /* c;d */ AS y"]
    assert split_sql("SELECT 'it''s; fine'") == ["SELECT 'it''s; fine'"]


@pytest.fixture
def warehouse(tmp_path):
    path = tmp_path / "w.duckdb"
    with duckdb.connect(str(path)) as con:
        con.execute("CREATE TABLE t AS SELECT range AS i FROM range(1000)")
    return SimpleNamespace(warehouse=path)


def query(warehouse, sql, max_rows=30):
    cmd_query(warehouse, SimpleNamespace(sql=sql, max_rows=max_rows))


def test_query_shows_only_max_rows(warehouse, capsys):
    query(warehouse, "-- First rows\nSELECT * FROM t ORDER BY i", max_rows=3)
    out = capsys.readouterr().out
    assert "=== First rows" in out and "more rows: only the first 3" in out
    assert out.count("\n") < 10


@pytest.mark.parametrize(("sql", "message"), [
    ("SELECT * FROM missing_table", "error in query 1"),
    ("SELEC 1", "error in query 1"),
    ("CREATE TABLE u (x INT)", "read-only"),                 # the warehouse is opened read-only
    ("sql/does_not_exist.sql", "file not found"),
])
def test_query_errors_are_plain_messages(warehouse, sql, message):
    with pytest.raises(SystemExit, match=message):
        query(warehouse, sql)


def test_route_rejects_a_zero_speed_and_a_missing_file(capsys, tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["route", "track.gpx", "--speed", "0"])
    assert exc.value.code == 2 and "greater than 0" in capsys.readouterr().err
    with pytest.raises(SystemExit, match="file not found"):
        main(["route", str(tmp_path / "missing.gpx"), "--start", "2026-11-20 09:00"])


def test_every_command_has_a_description(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    for name in ("download", "build", "analyze", "era5", "report", "all", "query", "route"):
        line = next(x for x in out.splitlines() if x.strip().startswith(name + " ") or x.strip() == name)
        assert line.strip() != name, f"{name} has no description"
