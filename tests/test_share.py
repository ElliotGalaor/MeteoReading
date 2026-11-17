"""Shareable links (#66): a link opens the page with its choices; a bad value falls back to the default."""
from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard"
PAGE = f"""
import sys
sys.path.insert(0, {str(DASHBOARD)!r})
import streamlit as st
from share import publish, restore

CITIES = ["Paris", "Lyon", "Brest"]
restore("city", "city", "Paris", valid=lambda v: v in CITIES)
restore("cities", "cities", ["Paris"], multi=True, valid=lambda v: 0 < len(v) <= 2 and set(v) <= set(CITIES))
restore("year", "year", 2022, parse=int, valid=lambda v: 1951 <= v <= 2025)
city = st.selectbox("City", CITIES, key="city")
cities = st.multiselect("Cities", CITIES, key="cities")
year = st.number_input("Year", 1951, 2025, key="year")
publish({{"city": city, "cities": cities, "year": year}})
"""


def page(tmp_path: Path, **params) -> AppTest:
    script = tmp_path / "page.py"
    script.write_text(PAGE, encoding="utf-8")
    at = AppTest.from_file(str(script))
    for k, v in params.items():
        at.query_params[k] = v
    return at.run()


def test_a_link_opens_the_page_with_its_choices(tmp_path):
    at = page(tmp_path, city="Lyon", cities=["Brest", "Lyon"], year="2003")
    assert not at.exception
    assert at.selectbox[0].value == "Lyon"
    assert at.multiselect[0].value == ["Brest", "Lyon"]
    assert at.number_input[0].value == 2003


def test_bad_values_fall_back_to_the_defaults(tmp_path):
    at = page(tmp_path, city="Atlantis", cities=["Paris", "Lyon", "Brest"], year="nineteen")
    assert not at.exception
    assert at.selectbox[0].value == "Paris" and at.multiselect[0].value == ["Paris"]
    assert at.number_input[0].value == 2022


def test_the_choices_are_written_to_the_address(tmp_path):
    at = page(tmp_path)
    at.selectbox[0].select("Brest").run()
    assert at.query_params["city"] == "Brest"
    assert at.query_params["year"] == "2022"
