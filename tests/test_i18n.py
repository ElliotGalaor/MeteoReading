"""The two languages (#68): the same keys and placeholders in both catalogs, French numbers and dates."""
from __future__ import annotations

import pandas as pd
import pytest

from meteo import i18n
from meteo.fmt import num, pct, signed
from meteo.i18n import catalog, fmt_date, ordinal, placeholders, plural, t


@pytest.fixture
def french():
    previous = i18n._provider
    i18n.set_provider(lambda: "fr")
    yield
    i18n.set_provider(previous)


def test_both_catalogs_have_the_same_keys():
    en, fr = catalog("en"), catalog("fr")
    assert sorted(set(en) - set(fr)) == []
    assert sorted(set(fr) - set(en)) == []


def test_both_catalogs_fill_the_same_placeholders():
    en, fr = catalog("en"), catalog("fr")
    for key, text in en.items():
        if isinstance(text, str):
            assert placeholders(text) == placeholders(fr[key]), key
        else:
            assert len(text) == len(fr[key]), key


def test_plotly_placeholders_are_kept():
    assert "%{y:.1f}" in t("city.hover_value", city="Paris", unit="°C")
    assert t("city.hover_value", city="Paris", unit="°C").startswith("Paris: ")


def test_english_without_a_provider():
    assert t("app.nav_city") == "Explore a city"
    assert num(25026059, 0) == "25,026,059" and num(-1.25, 1) == "−1.2"


def test_french_text_numbers_and_dates(french):
    assert t("app.nav_city") == "Explorer une ville"
    assert num(25026059, 0) == "25 026 059"
    assert num(12.345, 1) == "12,3" and signed(1.5, 1) == "+1,5"
    assert pct(0.42) == "42 %"
    assert fmt_date(pd.Timestamp("2026-11-17"), "short") == "17 nov. 2026"
    assert fmt_date(pd.Timestamp("2026-05-01"), "long") == "1er mai 2026"
    assert fmt_date(pd.Timestamp("2026-11-17"), "weekday") == "mar. 17 nov."
    assert ordinal(1) == "1er" and ordinal(40) == "40e"
    assert plural("common.n_days", 1) == "1 jour" and plural("common.n_days", 0) == "0 jour"


def test_french_punctuation_gets_a_narrow_space(french):
    text = t("overview.lead_latest", date="17 nov. 2026")
    assert "mesure : 17" in text


def test_english_dates_and_ordinals():
    assert fmt_date(pd.Timestamp("2026-11-07"), "short") == "7 Nov 2026"
    assert fmt_date(pd.Timestamp("2026-11-07"), "weekday_long") == "Saturday 7 November 2026"
    assert [ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 40)] == \
        ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "40th"]
    assert plural("common.n_days", 1) == "1 day" and plural("common.n_days", 3) == "3 days"
    # A value that rounds to zero has no minus sign (#82)
    assert (num(-0.04, 1), signed(-0.04, 1), num(-0.4, 0), signed(-0.6, 0)) == ("0.0", "+0.0", "0", "−1")


def test_plotly_locale_names_sunday_first(french):
    locale = i18n.plotly_locale()["fr"]
    assert locale["format"]["days"][0] == "dimanche" and locale["format"]["decimal"] == ","
    assert locale["dictionary"]["Download plot as a png"].startswith("Télécharger")


def test_every_key_the_code_asks_for_exists():
    """Literal keys in the code (t("…"), tr("…"), plural("…")) are all in the catalogs."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    en = catalog("en")
    used = set()
    for path in [*root.glob("dashboard/*.py"), *root.glob("meteo/*.py")]:
        text = path.read_text(encoding="utf-8")
        used |= set(re.findall(r"\b(?:t|tr)\(\"([a-z_]+\.[A-Za-z0-9_]+)\"", text))
        used |= {f"{k}.one" for k in re.findall(r"\bplural\(\"([a-z_]+\.[A-Za-z0-9_]+)\"", text)}
    missing = sorted(k for k in used if k not in en)
    assert missing == []
