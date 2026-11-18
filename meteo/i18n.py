"""The dashboard's two languages, English and French: text, numbers and dates.

The text lives in one catalog per language, `meteo/locales/en.toml` and `meteo/locales/fr.toml`, under the same keys
(a test checks it). `t("overview.title")` returns the text of the current language; `{name}` in it is replaced by
the keyword argument `name`. Plotly's own `%{y:.1f}` placeholders are left alone, so hover templates can live in the
catalogs too.

The current language comes from a provider the dashboard installs (the visitor's choice, kept in Streamlit's
session state). Without one, as in the command line, the report and the tests, everything is in English.

French typography: a narrow no-break space goes before ; : ! ? and inside « », numbers take a decimal comma and a
narrow no-break space between thousands (12 345,6), and dates read "17 nov. 2026", "1er mai".
"""
from __future__ import annotations

import re
import tomllib
from collections.abc import Callable
from functools import cache
from pathlib import Path

import pandas as pd

LANGS = ("en", "fr")
DEFAULT = "en"
LOCALES = Path(__file__).with_name("locales")
NNBSP = " "                                     # narrow no-break space

_provider: Callable[[], str | None] | None = None
_FIELD = re.compile(r"(?<!%)\{(\w+)\}")             # {name}, not Plotly's %{name}
_FR_PUNCT = re.compile(r"[  ]([:;!?»])")
_FR_OPEN = re.compile(r"«[  ]")


def set_provider(provider: Callable[[], str | None] | None) -> None:
    """Installs the function that tells the current language (None: English)."""
    global _provider
    _provider = provider


def lang() -> str:
    if _provider is not None:
        try:
            code = _provider()
        except Exception:   # outside a Streamlit run, for instance
            code = None
        if code in LANGS:
            return code
    return DEFAULT


def _flatten(table: dict, prefix: str = ""):
    for key, value in table.items():
        if isinstance(value, dict):
            yield from _flatten(value, f"{prefix}{key}.")
        else:
            yield f"{prefix}{key}", value


@cache
def catalog(code: str) -> dict[str, str | list[str]]:
    """Every entry of a language's catalog, by dotted key ("route.title")."""
    with open(LOCALES / f"{code}.toml", "rb") as f:
        return dict(_flatten(tomllib.load(f)))


def placeholders(text: str) -> set[str]:
    return set(_FIELD.findall(text))


def french_spacing(text: str) -> str:
    return _FR_OPEN.sub("«" + NNBSP, _FR_PUNCT.sub(NNBSP + r"\1", text))


def _entry(key: str, code: str):
    value = catalog(code).get(key)
    if value is None:
        value = catalog(DEFAULT).get(key)
    if value is None:
        raise KeyError(f"No text for {key!r}")
    return value


def t(key: str, /, **values) -> str:
    """The text of `key` in the current language, with its {placeholders} filled from `values`."""
    code = lang()
    text = _entry(key, code)
    if values:
        text = _FIELD.sub(lambda m: str(values[m[1]]) if m[1] in values else m[0], text)
    return french_spacing(text) if code == "fr" else text


def words(key: str) -> list[str]:
    """A list from the catalog (month names, for instance)."""
    return list(_entry(key, lang()))


def has(key: str) -> bool:
    return key in catalog(DEFAULT)


def plural(key: str, n: float, /, **values) -> str:
    """`key.one` or `key.other` by the number `n` (French: 0 and 1 take the singular), with {n} filled in."""
    from .fmt import num
    one = abs(n) < 2 if lang() == "fr" else n == 1
    return t(f"{key}.{'one' if one else 'other'}", n=num(n, 0) if float(n).is_integer() else num(n, 1), **values)


def ordinal(n: int) -> str:
    """1st, 2nd, 3rd, 11th… / 1er, 2e, 3e…"""
    n = int(n)
    if lang() == "fr":
        return "1er" if n == 1 else f"{n}e"
    suffix = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def listing(items: list[str]) -> str:
    """"a", "a and b", "a, b and c" / "a, b et c"."""
    items = [str(i) for i in items]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + t("fmt.and") + items[-1]


# --- Dates ---------------------------------------------------------------------------------
def month_name(month: int, short: bool = False) -> str:
    return words("fmt.months_short" if short else "fmt.months")[month - 1]


def weekday_name(day, short: bool = False) -> str:
    """Name of the day of the week of a date (Monday first in the catalogs)."""
    return words("fmt.weekdays_short" if short else "fmt.weekdays")[pd.Timestamp(day).day_of_week]


def _day(d: pd.Timestamp) -> str:
    return "1er" if lang() == "fr" and d.day == 1 else str(d.day)


def fmt_date(value, style: str = "long") -> str:
    """A date in the current language.

    long: 7 November 2026 · short: 7 Nov 2026 · day_month: 7 Nov · day_month_long: 7 November ·
    weekday: Sat 7 Nov · weekday_long: Saturday 7 November 2026 · month_year: November 2026
    """
    if value is None or pd.isna(value):
        return "–"
    d = pd.Timestamp(value)
    parts = {"day": _day(d), "year": d.year, "month": month_name(d.month), "mon": month_name(d.month, True),
             "weekday": weekday_name(d), "wd": weekday_name(d, True)}
    return t(f"fmt.date_{style}", **parts)


def fmt_time(value, with_day: bool = False) -> str:
    """14:30, or Sat 14:30 when a trip spans several days."""
    d = pd.Timestamp(value)
    return f"{weekday_name(d, True)} {d:%H:%M}" if with_day else f"{d:%H:%M}"


def plotly_locale() -> dict:
    """Plotly's locale for the current language: month and day names, decimal comma, toolbar tooltips."""
    code = lang()
    days = words("fmt.weekdays")
    short_days = words("fmt.weekdays_short")
    return {code: {
        "dictionary": dict(zip(words("fmt.plotly_terms_en"), words("fmt.plotly_terms"), strict=True)),
        "format": {
            "days": days[-1:] + days[:-1], "shortDays": short_days[-1:] + short_days[:-1],   # Sunday first
            "months": words("fmt.months"), "shortMonths": words("fmt.months_short"),
            "date": t("fmt.plotly_date"), "decimal": t("fmt.decimal"), "thousands": t("fmt.thousands"),
        },
    }}
