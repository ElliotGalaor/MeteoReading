"""The layout follows the width of the main column measured in the browser, not the user agent (#81)."""
from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard"
PAGE = f"""
import sys
sys.path.insert(0, {str(DASHBOARD)!r})
import streamlit as st
import ui
ui.measure_layout()
st.markdown(f"narrow={{ui.is_narrow()}} cards={{not ui.is_narrow(880)}} width={{ui.main_width()}}")
"""


def page(tmp_path: Path, width: int | None = None) -> AppTest:
    script = tmp_path / "page.py"
    script.write_text(PAGE, encoding="utf-8")
    at = AppTest.from_file(str(script))
    if width:
        at.session_state["main_width"] = width
    return at.run()


def test_before_the_browser_has_measured_a_computer_gets_the_wide_layout(tmp_path):
    at = page(tmp_path)
    assert not at.exception
    assert at.markdown[0].value == "narrow=False cards=True width=None"


def test_a_narrow_main_column_gets_the_narrow_layout(tmp_path):
    assert page(tmp_path, 560).markdown[0].value == "narrow=True cards=False width=560"     # a tablet, sidebar open
    assert page(tmp_path, 800).markdown[0].value == "narrow=False cards=False width=800"    # cards need 880 px
    assert page(tmp_path, 1120).markdown[0].value == "narrow=False cards=True width=1120"
