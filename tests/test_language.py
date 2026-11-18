"""Language switch (#68): a link chooses the language, switching keeps the page's choices and its link."""
from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard"
PAGE = f"""
import sys
sys.path.insert(0, {str(DASHBOARD)!r})
import streamlit as st
import language
from share import publish, restore
from ui import t, num
language.setup()
CITIES = ["Paris", "Lyon", "Brest"]
restore("city", "city", "Paris", valid=lambda v: v in CITIES)
restore("mode", "display", "Values", valid=lambda v: v in ("Values", "Anomaly"))
city = st.selectbox(t("common.city"), CITIES, key="city")
mode = st.segmented_control(t("city.display"), ["Values", "Anomaly"], key="mode", required=True,
                            format_func=lambda v: t(f"city.{{v}}"))
publish({{"city": city, "display": mode}})
st.markdown(t("app.nav_city") + " · " + num(1234.5, 1))
with st.sidebar:
    language.picker()
"""


def page(tmp_path: Path, **params) -> AppTest:
    script = tmp_path / "page.py"
    script.write_text(PAGE, encoding="utf-8")
    at = AppTest.from_file(str(script))
    for k, v in params.items():
        at.query_params[k] = v
    return at.run()


def test_a_link_opens_the_page_in_french(tmp_path):
    at = page(tmp_path, lang="fr", city="Lyon")
    assert not at.exception
    assert at.selectbox[0].label == "Ville" and at.selectbox[0].value == "Lyon"
    assert at.markdown[0].value == "Explorer une ville · 1 234,5"


def test_switching_language_keeps_the_choices_and_the_link(tmp_path):
    at = page(tmp_path, lang="en")
    at.selectbox[0].select("Brest").run()
    at.button_group[0].set_value("Anomaly").run()
    at.button_group[1].set_value("fr").run()           # the language switch of the sidebar
    assert not at.exception
    assert at.selectbox[0].label == "Ville" and at.selectbox[0].value == "Brest"
    assert at.session_state["mode"] == "Anomaly"
    assert at.query_params["city"] == "Brest" and at.query_params["display"] == "Anomaly"
    at.button_group[1].set_value("en").run()
    assert at.selectbox[0].label == "City" and at.selectbox[0].value == "Brest"
    assert at.markdown[0].value == "Explore a city · 1,234.5"


def test_an_unknown_language_falls_back_to_the_browser_one(tmp_path):
    at = page(tmp_path, lang="de")
    assert not at.exception
    assert at.session_state["lang"] in ("en", "fr")
