"""Touch gestures on the charts: one finger scrolls the page, two fingers pinch to zoom the chart.

On a touch screen Plotly turns any drag on a chart into a zoom box, so a finger that lands on a chart cannot scroll
the page past it. This script, installed once per tab, keeps touches on the plot area away from Plotly:

- one finger scrolls the page, as anywhere else; a tap still shows the values (the browser turns it into a click);
- two fingers zoom the chart, the point under the fingers staying under them, and move it once zoomed;
- a double tap goes back to the whole chart (Plotly's double click).

The pinch is applied with Plotly's own wheel zoom (a zoom around the pointer, deltaY = 200 ln(range factor), at most
20 per event). That zoom stays off for the mouse wheel, which scrolls the page: it is switched on only while the
script sends its own wheel events. Mouse and trackpad on a computer are unchanged (drag to zoom a box).

Maps (Plotly's MapLibre maps) get MapLibre's own "cooperative gestures" on touch screens: one finger scrolls the
page and shows "Use two fingers to move the map", two fingers move and zoom the map. On a computer they keep their
usual gestures (that mode would ask for Ctrl to zoom with the wheel). The drawing map (Leaflet) is set apart in the
route page: it does not follow one finger on a phone.
"""
from __future__ import annotations

import streamlit as st

SCRIPT = """
<style id="chart-gestures">
  /* No double-tap zoom of the page on the plot area: a double tap resets the chart instead */
  .js-plotly-plot .draglayer, .js-plotly-plot .draglayer * { touch-action: manipulation; }
  /* This block shows nothing: no gap for it above the page */
  [data-testid="stElementContainer"]:has(#chart-gestures) { display: none; }
</style>
<script>
(() => {
  if (window.__chartGestures) return;        // the page reruns: install once
  window.__chartGestures = true;

  const STEP = 20;                           // Plotly's largest wheel step
  let pinch = null;                          // {gd, rect, x, y, d}: the chart and the last midpoint and spread
  let next = null, frame = 0;

  const chartOf = (el) => (el instanceof Element && el.closest(".js-plotly-plot .draglayer"))
    ? el.closest(".js-plotly-plot") : null;
  const spread = (t) => ({
    x: (t[0].clientX + t[1].clientX) / 2,
    y: (t[0].clientY + t[1].clientY) / 2,
    d: Math.hypot(t[0].clientX - t[1].clientX, t[0].clientY - t[1].clientY),
  });

  // Plotly's wheel zoom around the point (x, y) of the screen: ranges times `factor`
  function zoom(p, factor, x, y) {
    const flags = p.gd._context && p.gd._context._scrollZoom;
    if (!flags || !p.rect.isConnected) return;
    const was = flags.cartesian;
    flags.cartesian = true;
    try {
      let delta = 200 * Math.log(factor);
      while (Math.abs(delta) > 1e-9) {
        const step = Math.max(-STEP, Math.min(STEP, delta));
        p.rect.dispatchEvent(new WheelEvent("wheel", {deltaY: step, clientX: x, clientY: y, cancelable: true}));
        delta -= step;
      }
    } finally {
      flags.cartesian = was;
    }
  }

  function apply(now) {
    const p = pinch;
    // 1. the change of spread zooms around the last midpoint
    if (now.d > 0 && p.d > 0 && Math.abs(Math.log(p.d / now.d)) > 1e-4) zoom(p, p.d / now.d, p.x, p.y);
    // 2. the shift of the midpoint moves the chart: one wheel step out around a far point, then back in around
    //    the midpoint, shifts the content by (k - 1) times the gap between the two points
    const dx = now.x - p.x, dy = now.y - p.y;
    if (Math.hypot(dx, dy) >= 0.5) {
      const k = Math.exp(STEP / 200);
      zoom(p, k, now.x + dx / (k - 1), now.y + dy / (k - 1));
      zoom(p, 1 / k, now.x, now.y);
    }
    Object.assign(p, now);
  }

  window.addEventListener("touchstart", (e) => {
    const gd = chartOf(e.target);
    if (!gd) return;
    e.stopPropagation();                     // Plotly does not see the touch: no zoom box, the page scrolls
    const t = e.touches;
    const rect = gd.querySelector(".draglayer .nsewdrag");
    if (t.length === 2 && rect && gd._context && gd._context._scrollZoom
        && chartOf(t[0].target) === gd && chartOf(t[1].target) === gd) {
      pinch = {gd, rect, ...spread(t)};
      if (e.cancelable) e.preventDefault();  // no zoom of the whole page
    }
  }, {capture: true, passive: false});

  window.addEventListener("touchmove", (e) => {
    if (!pinch) return;
    if (e.cancelable) e.preventDefault();
    if (e.touches.length < 2) return;
    next = spread(e.touches);
    if (!frame) frame = requestAnimationFrame(() => {
      frame = 0;
      if (pinch && next) apply(next);
      next = null;
    });
  }, {capture: true, passive: false});

  const end = (e) => { if (pinch && e.touches.length < 2) pinch = next = null; };
  window.addEventListener("touchend", end, {capture: true});
  window.addEventListener("touchcancel", end, {capture: true});
  // Safari's own pinch events: no zoom of the whole page while a chart is pinched
  for (const type of ["gesturestart", "gesturechange"]) {
    window.addEventListener(type, (e) => { if (pinch) e.preventDefault(); }, {capture: true, passive: false});
  }

  // Maps on a touch screen: one finger scrolls the page, two fingers move the map (MapLibre's cooperative gestures)
  const touch = window.matchMedia && window.matchMedia("(pointer: coarse)").matches;
  function cooperate() {
    document.querySelectorAll(".js-plotly-plot").forEach((gd) => {
      const fl = gd._fullLayout, ids = fl && fl._subplots && fl._subplots.map;
      (ids || []).forEach((id) => {
        const map = fl[id] && fl[id]._subplot && fl[id]._subplot.map;
        const coop = map && map.cooperativeGestures;
        if (coop && !coop.isEnabled()) coop.enable();
      });
    });
  }
  if (touch) {
    let queued = false;
    new MutationObserver(() => {
      if (queued) return;
      queued = true;
      requestAnimationFrame(() => { queued = false; cooperate(); });
    }).observe(document.body, {childList: true, subtree: true});
    cooperate();
  }
})();
</script>
"""


def chart_gestures() -> None:
    """Adds the touch gestures of the charts to the page (installed once per tab)."""
    st.html(SCRIPT, unsafe_allow_javascript=True)
