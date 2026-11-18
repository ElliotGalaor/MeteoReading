"""Shareable links: the main choices of each page are kept in the address (st.query_params).

- `restore` before a widget: when the address carries a value this page did not write itself (a link was opened),
  the widget starts from it; an unknown or invalid value is ignored and the default is used.
- `publish` after the widgets: the current choices are written to the address, so copying it shares them. Only
  public values go there (cities, years, sample names, public route links): never an uploaded or drawn track.
- `copy_link` is the "Copy link" button of the page header.

The language (`lang`, dashboard/language.py) belongs to every page: `publish` leaves it in the address.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import streamlit as st

from meteo.i18n import t

KEEP = {"lang"}   # parameters of every page
COPY_LINK = """
<style>
  [data-testid="stElementContainer"]:has(#copy-link) { margin-top: -0.9rem; margin-bottom: -0.4rem; }
  #copy-link { font: inherit; font-size: 0.82rem; padding: 0.15rem 0.6rem; border-radius: 6px; cursor: pointer;
               border: 1px solid rgba(128,128,128,0.35); background: transparent; color: inherit; opacity: 0.8; }
  #copy-link:hover { opacity: 1; }
</style>
<button id="copy-link" type="button" title="%(tip)s">%(copy)s</button>
<script>
(() => {
  const b = document.getElementById("copy-link");
  if (!b || b.dataset.ready) return;
  b.dataset.ready = "1";
  b.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(location.href); b.textContent = %(copied)s; }
    catch (e) { window.prompt(%(prompt)s, location.href); }
    setTimeout(() => { b.textContent = %(copy_js)s; }, 2000);
  });
})();
</script>
"""


def _marker(name: str) -> str:
    return f"_link_{name}"


def from_link(name: str, multi: bool = False):
    """The raw value of `name` in the address when it was not written by this page (a link was opened), else None."""
    raw = st.query_params.get_all(name) if multi else st.query_params.get(name)
    if not raw or raw == st.session_state.get(_marker(name)):
        return None
    st.session_state[_marker(name)] = raw
    return raw


def restore(key: str, name: str, default: Any = None, parse: Callable[[str], Any] = str,
            valid: Callable[[Any], bool] = lambda v: True, multi: bool = False) -> None:
    """Sets the widget `key` from the link's `name` parameter when a link was opened, else to `default` the first
    time. Values that do not parse or are not valid are ignored."""
    raw = from_link(name, multi)
    if raw is not None:
        try:
            value = [parse(x) for x in raw] if multi else parse(raw)
            if valid(value):
                st.session_state[key] = value
                return
        except (ValueError, TypeError, KeyError):
            pass
    if default is not None:
        st.session_state.setdefault(key, default)


def publish(values: dict[str, Any]) -> None:
    """Writes the page's choices to the address; parameters of other pages are removed, None leaves one out."""
    for name in list(st.query_params):
        if name in KEEP:
            continue
        if name not in values or values[name] is None:
            del st.query_params[name]
            st.session_state.pop(_marker(name), None)
    for name, v in values.items():
        if v is None or (isinstance(v, (list, tuple)) and not v):
            st.query_params.pop(name, None)
            continue
        text = [str(x) for x in v] if isinstance(v, (list, tuple)) else str(v)
        if (st.query_params.get_all(name) if isinstance(text, list) else st.query_params.get(name)) != text:
            st.query_params[name] = text
        st.session_state[_marker(name)] = text


def copy_link() -> None:
    import html
    import json
    st.html(COPY_LINK % {"tip": html.escape(t("share.copy_tip")), "copy": html.escape(t("share.copy")),
                         "copied": json.dumps(t("share.copied")), "prompt": json.dumps(t("share.prompt")),
                         "copy_js": json.dumps(t("share.copy"))}, unsafe_allow_javascript=True)
