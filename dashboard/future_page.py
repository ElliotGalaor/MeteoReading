"""Dashboard page: each city in a warmer France, at the official French warming levels (TRACC)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from meteo.climate_ahead import LEVEL_NOW, LEVELS, MODELS, REGIONS, RESULT_FILE, UNITS
from meteo.i18n import plural
from share import publish, restore
from ui import CFG, OLD, RECENT, SERIES, TH, WARM, chart, is_narrow, num, page_header, source_note, t

CITIES = [c.name for c in CFG.cities]


@st.cache_data(show_spinner=False)
def _table() -> pd.DataFrame | None:
    path = CFG.reports_dir / "tables" / RESULT_FILE
    return pd.read_csv(path, dtype={"period": str}) if path.exists() else None


def _number(v: float, unit: str, digits: int = 0) -> str:
    return num(v, 1) if unit == "°C" else num(v, digits)


def _value(v: float, unit: str, digits: int = 0) -> str:
    """With its unit: "12.4 °C", "640 mm", "1 day", "3 days" (a day count agrees with its number)."""
    if unit == "days":
        return plural("common.n_days", round(float(v), digits))
    return f"{_number(v, unit, digits)} {unit}"


def _spread(low: float, high: float, unit: str) -> str:
    """The models' range at the last warming level. The annual mean warming is the TRACC level itself (the models
    only split it between seasons): no range. A count whose range rounds to one number gets a decimal (#80)."""
    if unit == "°C":
        return t("ahead.set_by_level") if _number(low, unit) == _number(high, unit) else             t("ahead.range", low=_number(low, unit), high=_value(high, unit))
    digits = 0 if _number(low, unit) != _number(high, unit) else 1
    if _number(low, unit, digits) == _number(high, unit, digits):
        return ""
    return t("ahead.range", low=_number(low, unit, digits), high=_value(high, unit, digits))


def _period(code: str) -> str:
    """"2050" -> "+2.7 °C, 2050" ("+2,7 °C en 2050"); the measured periods stay as they are."""
    if code in LEVELS:
        return t("ahead.level", warming=num(LEVELS[code], 1).removesuffix(".0").removesuffix(",0"), year=code)
    return code


def _indicator(code: str) -> str:
    return t("ahead.rr") if code == "rr" else t(f"ind.{code}", **TH)


def page_future():
    page_header(t("ahead.title"), t("ahead.lead"), tab=t("app.nav_ahead"))
    st.warning(t("ahead.warning"), icon=":material/info:")
    tbl = _table()
    if tbl is None:
        st.info(t("ahead.no_table"))
        return
    restore("future_city", "city", "Paris", valid=lambda v: v in CITIES)
    city = st.selectbox(t("common.city"), CITIES, key="future_city")
    publish({"city": city})
    c = tbl[tbl["city"] == city]
    periods = list(dict.fromkeys(c["period"]))
    old_label, recent_label = f"{OLD[0]}-{OLD[1]}", f"{RECENT[0]}-{RECENT[1]}"
    last_level = list(LEVELS)[-1]
    indicators = list(dict.fromkeys(c["indicator"]))
    per_row = 1 if is_narrow() else 2
    for i in range(0, len(indicators), per_row):
        cols = st.columns(per_row, gap="large")
        for col, code in zip(cols, indicators[i:i + per_row], strict=False):
            unit = UNITS.get(code, "days")
            d = c[c["indicator"] == code].set_index("period").loc[periods]
            labels = [_period(p) for p in d.index]
            observed = d.index.isin([old_label, recent_label])
            fig = go.Figure()
            fig.add_bar(x=[lb for lb, o in zip(labels, observed, strict=True) if o], y=d["median"][observed],
                        marker_color=SERIES[0], name=t("ahead.measured"), width=0.55,
                        hovertemplate=t("ahead.hover_measured"))
            f = d[~observed]
            fig.add_bar(x=[lb for lb, o in zip(labels, observed, strict=True) if not o], y=f["median"],
                        marker_color=WARM, name=t("ahead.scenario"), width=0.55,
                        error_y=dict(type="data", symmetric=False, array=(f["high"] - f["median"]).to_numpy(),
                                     arrayminus=(f["median"] - f["low"]).to_numpy(), color="#52514e", thickness=1.5,
                                     width=6),
                        customdata=f[["low", "high"]].to_numpy(), hovertemplate=t("ahead.hover_scenario"))
            unit_text = unit if unit != "days" else t("ahead.days_per_year")
            fig.update_layout(showlegend=False, bargap=0.3, xaxis=dict(type="category", categoryorder="array",
                                                                       categoryarray=labels),
                              yaxis=dict(title=unit_text, rangemode="tozero" if unit != "°C" else "normal"))
            if unit == "°C":
                lo, hi = float(d["low"].min()), float(d["high"].max())
                fig.update_yaxes(range=[lo - 1.5, hi + 1])
            now, end = d.loc[recent_label, "median"], d.loc[last_level]
            spread = _spread(end["low"], end["high"], unit)
            with col:
                chart(fig, 300, _indicator(code),
                      t("ahead.sub", recent=recent_label, now=_value(now, unit), level=_period(last_level),
                        value=_value(end["median"], unit), spread=spread), source=t("source.ahead_csv"))

    region = c["region"].iloc[0]
    with st.expander(t("ahead.method_title"), expanded=False):
        steps = ", ".join(t("ahead.step", warming=num(v - LEVEL_NOW, 1), year=k) for k, v in LEVELS.items())
        st.markdown(t("ahead.method", now=num(LEVEL_NOW, 1), steps=steps,
                      models=", ".join(m.replace("_", "-") for m in MODELS), city=city,
                      region=t(f"ahead.region_{region}"), lat=num(REGIONS[region][0], 1),
                      lon=num(REGIONS[region][1], 1), recent=recent_label))
    source_note(t("source.ahead"))
