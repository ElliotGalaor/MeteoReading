"""The visitor's language: English or French.

- First visit: the browser's language (French if it prefers French, else English).
- A link with ?lang=fr or ?lang=en opens the page in that language.
- The choice in the sidebar is kept for the session; it goes in the address only when it differs from the browser's
  language, so a copied link shows the page as it was seen. The page and its choices stay as they are: widgets keep
  their values (their keys do not change with the language) and the options they hold are codes, not words.
"""
from __future__ import annotations

import streamlit as st

from meteo.i18n import LANGS, t

NAMES = {"en": "English", "fr": "Français"}
MARKER = "_link_lang"


def browser_language() -> str:
    try:
        locale = st.context.locale or ""
    except Exception:
        locale = ""
    return "fr" if locale.lower().startswith("fr") else "en"


def setup() -> None:
    """Sets the session's language before anything is drawn (to be called first in every run)."""
    raw = st.query_params.get("lang")
    if raw in LANGS and raw != st.session_state.get(MARKER):
        st.session_state["lang"] = raw          # a link was opened
        st.session_state[MARKER] = raw
    if st.session_state.get("lang") not in LANGS:
        st.session_state["lang"] = browser_language()


def picker() -> None:
    """The language switch of the sidebar, and the address that goes with it."""
    st.segmented_control(t("app.language"), list(LANGS), format_func=NAMES.__getitem__, key="lang",
                         required=True, label_visibility="collapsed")
    code = st.session_state.get("lang") or browser_language()
    if code == browser_language():
        st.query_params.pop("lang", None)
        st.session_state.pop(MARKER, None)
    elif st.query_params.get("lang") != code:
        st.query_params["lang"] = code
        st.session_state[MARKER] = code
    else:
        st.session_state[MARKER] = code
