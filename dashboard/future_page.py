"""Dashboard page: each city in a warmer France, at the official French warming levels (TRACC)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from meteo.climate_ahead import LEVEL_NOW, LEVELS, MODELS, REGIONS, RESULT_FILE, UNITS
from share import publish, restore
from ui import CFG, OLD, RECENT, SERIES, WARM, chart, is_phone, num, page_header, source_note

CITIES = [c.name for c in CFG.cities]


@st.cache_data(show_spinner=False)
def _table() -> pd.DataFrame | None:
    path = CFG.reports_dir / "tables" / RESULT_FILE
    return pd.read_csv(path) if path.exists() else None


def _value(v: float, unit: str) -> str:
    return f"{num(v, 1)} °C" if unit == "°C" else (f"{num(v, 0)} mm" if unit == "mm" else f"{num(v, 0)} days")


def page_future():
    page_header("Climate ahead", "What each city's climate would look like in a France 2 °C, 2.7 °C and 4 °C warmer "
                "than before the industrial era: the warming levels the French State plans for in 2030, 2050 and "
                "2100.", tab="Climate ahead")
    st.warning("**Scenarios, not forecasts.** These figures show the days measured recently, made warmer by the "
               "amount the official trajectory (TRACC) plans for, split between seasons as seven climate models do. "
               "The range across the models is shown, never a single number.", icon=":material/info:")
    t = _table()
    if t is None:
        st.info("This view needs the climate model patterns: run `python -m meteo climate` once, then "
                "`python -m meteo analyze`.")
        return
    restore("future_city", "city", "Paris", valid=lambda v: v in CITIES)
    city = st.selectbox("City", CITIES, key="future_city")
    publish({"city": city})
    c = t[t["city"] == city]
    periods = list(dict.fromkeys(c["period"]))
    old_label, recent_label = f"{OLD[0]}-{OLD[1]}", f"{RECENT[0]}-{RECENT[1]}"
    indicators = list(dict.fromkeys(c["indicator"]))
    per_row = 1 if is_phone() else 2
    for i in range(0, len(indicators), per_row):
        cols = st.columns(per_row, gap="large")
        for col, name in zip(cols, indicators[i:i + per_row], strict=False):
            unit = UNITS.get(name, "days")
            d = c[c["indicator"] == name].set_index("period").loc[periods]
            observed = d.index.isin([old_label, recent_label])
            fig = go.Figure()
            fig.add_bar(x=d.index[observed], y=d["median"][observed], marker_color=SERIES[0], name="Measured",
                        width=0.55, hovertemplate="%{x}: %{y:.1f}<extra>measured</extra>")
            f = d[~observed]
            fig.add_bar(x=f.index, y=f["median"], marker_color=WARM, name="Scenario (median of the models)", width=0.55,
                        error_y=dict(type="data", symmetric=False, array=(f["high"] - f["median"]).to_numpy(),
                                     arrayminus=(f["median"] - f["low"]).to_numpy(), color="#52514e", thickness=1.5,
                                     width=6),
                        customdata=f[["low", "high"]].to_numpy(),
                        hovertemplate="%{x}: %{y:.1f} (models: %{customdata[0]:.1f} to %{customdata[1]:.1f})"
                                      "<extra>scenario</extra>")
            fig.update_layout(showlegend=False, bargap=0.3, xaxis=dict(type="category"),
                              yaxis=dict(title=unit if unit != "days" else "days / year",
                                         rangemode="tozero" if unit != "°C" else "normal"))
            if unit == "°C":
                lo, hi = float(d["low"].min()), float(d["high"].max())
                fig.update_yaxes(range=[lo - 1.5, hi + 1])
            now, end = d.loc[recent_label, "median"], d.loc[list(LEVELS)[-1]]
            # The annual mean warming is the TRACC level itself: the models only split it between seasons
            spread = (f"({_value(end['low'], unit)} to {_value(end['high'], unit)})"
                      if _value(end["low"], unit) != _value(end["high"], unit) else "(set by the warming level)")
            with col:
                chart(fig, 300, name, f"{recent_label}: {_value(now, unit)} · at +4 °C: {_value(end['median'], unit)} "
                      f"{spread}")

    region = c["region"].iloc[0]
    with st.expander("Method and limits", expanded=False):
        steps = ", ".join(f"+{num(v - LEVEL_NOW, 1)} °C ({k.split(', ')[1]})" for k, v in LEVELS.items())
        st.markdown(f"""
**Warming levels.** The *trajectoire de réchauffement de référence pour l'adaptation au changement climatique*
(TRACC), the reference the French State asks planners to use, sets mainland France at +2 °C in 2030, +2.7 °C in 2050
and +4 °C in 2100 compared with the pre-industrial climate. France had warmed by about +{num(LEVEL_NOW, 1)} °C by
2020 (Météo-France), so the warming still to come is {steps}.

**Seasons, days and nights.** Warming is not the same in summer and winter, by day and by night, and rainfall
changes with it. These splits come from seven high-resolution climate models (CMIP6 HighResMIP:
{", ".join(m.replace("_", "-") for m in MODELS)}), through the Open-Meteo Climate API, at the point of France
closest to {city} ({region}: {REGIONS[region][0]}° N, {REGIONS[region][1]}° E). In each model, the change of each
season's mean maximum and minimum temperature between 1995-2006 and 2039-2050 is divided by the change of the annual
mean temperature, and so is the change of each season's rainfall.

**Days.** Every day measured in {city} in {recent_label} is shifted by that model's change times the warming still
to come, and the indicators are counted again. The bars show the median of the seven models, the lines the lowest
and highest.

**Limits.**
- Day-to-day variability stays the one measured: heatwaves get hotter, not longer or more frequent, so the counts of
  hot days and tropical nights are rather low estimates.
- The model patterns come from four points of France, not from each city; they are those of the HighResMIP
  scenario, scaled to the TRACC levels.
- The TRACC is a planning trajectory, not a forecast: the warming actually reached will depend on emissions.
- To compare with Météo-France's own indicators for each commune, see Climadiag Commune (meteofrance.com).
""")
    source_note("; climate models: CMIP6 HighResMIP via the Open-Meteo Climate API (CC BY 4.0); warming levels: "
                "TRACC, French Ministry for the Ecological Transition")
