"""Pause an idle tab, so that the server can scale to zero.

Streamlit keeps a WebSocket open for as long as a tab is open, and Cloud Run bills an instance while a request (the
WebSocket) is open: a forgotten tab would keep an instance running day and night, and could trip the budget guard.
After IDLE_PAUSE_MINUTES without a click, key or scroll, the tab goes to a static "paused" page served by Caddy
(docker/paused.html), which closes the connection; one click brings the visitor back to the same page.

IDLE_PAUSE_MINUTES is set in the image (Dockerfile); unset or 0, as when running locally, nothing is added.
"""
from __future__ import annotations

import os

import streamlit as st

SCRIPT = """
<script>
(() => {
  if (window.__idlePause) return;            // the page reruns: install once
  window.__idlePause = true;
  const limit = %(ms)d;
  let last = Date.now();
  const active = () => { last = Date.now(); };
  for (const e of ["pointerdown", "pointermove", "keydown", "wheel", "touchstart", "scroll"]) {
    window.addEventListener(e, active, {passive: true, capture: true});
  }
  document.addEventListener("visibilitychange", () => { if (!document.hidden) active(); });
  setInterval(() => {
    if (Date.now() - last > limit) {
      window.location.replace("/paused.html?from=" + encodeURIComponent(location.pathname + location.search));
    }
  }, 30000);
})();
</script>
"""


def idle_minutes() -> int:
    try:
        return max(int(os.environ.get("IDLE_PAUSE_MINUTES", "0")), 0)
    except ValueError:
        return 0


def pause_when_idle() -> None:
    """Adds the idle timer to the page (once per tab), if IDLE_PAUSE_MINUTES is set."""
    minutes = idle_minutes()
    if minutes:
        st.html(SCRIPT % {"ms": minutes * 60_000}, unsafe_allow_javascript=True)
