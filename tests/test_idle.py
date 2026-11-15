"""Idle tabs pause themselves (dashboard/idle.py) so that Cloud Run can scale to zero."""
from __future__ import annotations

import pytest


@pytest.fixture
def idle(monkeypatch):
    monkeypatch.syspath_prepend("dashboard")
    import idle

    return idle


@pytest.mark.parametrize(("value", "minutes"), [(None, 0), ("0", 0), ("20", 20), ("-5", 0), ("soon", 0)])
def test_idle_minutes_from_the_environment(idle, monkeypatch, value, minutes):
    if value is None:
        monkeypatch.delenv("IDLE_PAUSE_MINUTES", raising=False)
    else:
        monkeypatch.setenv("IDLE_PAUSE_MINUTES", value)
    assert idle.idle_minutes() == minutes


def test_the_script_goes_to_the_paused_page_after_the_delay(idle):
    js = idle.SCRIPT % {"ms": 20 * 60_000}
    assert "const limit = 1200000;" in js and "/paused.html?from=" in js
    assert "window.__idlePause" in js            # installed once per tab, whatever the reruns
