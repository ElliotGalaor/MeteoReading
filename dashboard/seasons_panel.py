"""City page: the best weeks of the year for walking or cycling, and the dates of the warm and cold seasons."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from meteo.route_scores import PROFILES, WIND_COMFORT_KMH
from meteo.seasons import (
    ACTIVITIES,
    SUMMER_TX,
    WINTER_TX,
    best_weeks,
    season,
    shift_days,
    week_label,
)
from ui import OLD, RECENT, SERIES, TH, chart, num, table

PERIODS = {f"{OLD[0]}-{OLD[1]}": OLD, f"{RECENT[0]}-{RECENT[1]}": RECENT}
PERIODS_LABELS = list(PERIODS)


@st.cache_data(show_spinner=False)
def _weeks(city: str, activity: str) -> tuple[pd.DataFrame, bool]:
    d = table("city_daily")
    return best_weeks(d[d["city"] == city], activity, PERIODS, TH["rain_day"])


@st.cache_data(show_spinner=False)
def _seasons(city: str) -> dict:
    d = table("city_daily")
    d = d[d["city"] == city]
    return {(label, warm): season(d, p, warm) for label, p in PERIODS.items() for warm in (True, False)}


def _season_line(name: str, threshold: str, then, now) -> str:
    old, new = PERIODS_LABELS
    if then is None and now is None:
        return f"<b>{name}</b> ({threshold}): none, in either period."
    if now is None:
        return f"<b>{name}</b> ({threshold}): {then.label} in {old}, none any more in {new}."
    if then is None:
        return f"<b>{name}</b> ({threshold}): none in {old}, {now.label} in {new} ({now.days} days)."
    text = f"<b>{name}</b> ({threshold}): {now.label} in {new} ({now.days} days), {then.label} in {old} ({then.days} days)"
    shift = shift_days(then, now)
    if shift:
        earlier, later = shift
        parts = []
        if earlier:
            parts.append(f"starts {abs(earlier)} days {'earlier' if earlier > 0 else 'later'}")
        if later:
            parts.append(f"ends {abs(later)} days {'later' if later > 0 else 'earlier'}")
        if parts:
            text += ": it " + " and ".join(parts)
    return text + "."


def seasons_panel(cities: list[str]) -> None:
    st.subheader("Best weeks for outdoor activities")
    c1, c2 = st.columns([1.2, 2])
    activity = c1.segmented_control("Activity", ACTIVITIES, default="Walking", required=True, key="weeks_activity")
    activity = activity or "Walking"
    city = cities[0] if len(cities) == 1 else c2.selectbox("City", cities, key="weeks_city")
    weeks, use_wind = _weeks(city, activity)
    if weeks.empty:
        st.info("Not enough data for this city.")
        return
    old, new = PERIODS_LABELS
    fig = go.Figure()
    for label, colour, width in ((old, SERIES[0], 1.8), (new, SERIES[1], 2.6)):
        w = weeks[weeks["period"] == label].sort_values("week")
        fig.add_scatter(x=[week_label(v) for v in w["week"]], y=w["share"] * 100, mode="lines", name=label,
                        line=dict(color=colour, width=width, shape="spline", smoothing=0.6),
                        customdata=w["n_days"], hovertemplate=f"week of %{{x}}: %{{y:.0f}}% of days comfortable "
                                                              f"({label})<extra></extra>")
    fig.update_layout(hovermode="x unified", yaxis=dict(title="% of days", rangemode="tozero"),
                      xaxis=dict(type="category", nticks=13))
    low, high = PROFILES[activity].comfort_range
    rules = (f"daytime temperature {low}-{high} °C, under {TH['rain_day']} mm of rain"
             + (f", mean wind under {WIND_COMFORT_KMH} km/h" if use_wind else ""))
    chart(fig, 380, f"Comfortable days for {activity.lower()} in {city}, week by week",
          f"Share of days with {rules} · {old} vs {new}")

    pv = weeks.pivot(index="week", columns="period", values="share").dropna()
    if not pv.empty:
        best = pv[new].idxmax()
        diff = (pv[new] - pv[old]) * 100
        gain, loss = diff.idxmax(), diff.idxmin()
        line = (f"Best week now: <b>{week_label(best)}</b>, {num(pv.loc[best, new] * 100, 0)}% of days comfortable "
                f"({num(pv.loc[best, old] * 100, 0)}% in {old}).")
        if diff[loss] <= -5:
            line += (f" Biggest loss: week of {week_label(loss)}, {num(pv.loc[loss, old] * 100, 0)}% → "
                     f"{num(pv.loc[loss, new] * 100, 0)}%.")
        if diff[gain] >= 5:
            line += (f" Biggest gain: week of {week_label(gain)}, {num(pv.loc[gain, old] * 100, 0)}% → "
                     f"{num(pv.loc[gain, new] * 100, 0)}%.")
        st.markdown(f"<div class='finding'>{line}</div>", unsafe_allow_html=True)

    s = _seasons(city)
    st.markdown("<div class='chart-title'>Seasons, then and now</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='finding'>{_season_line('Warm season', f'normal maximum ≥ {num(SUMMER_TX, 0)} °C', s[(old, True)], s[(new, True)])}</div>",
                unsafe_allow_html=True)
    st.markdown(f"<div class='finding'>{_season_line('Cold season', f'normal maximum ≤ {num(WINTER_TX, 0)} °C', s[(old, False)], s[(new, False)])}</div>",
                unsafe_allow_html=True)
    with st.expander("How are comfortable days and seasons defined?"):
        walk, bike = PROFILES["Walking"].comfort_range, PROFILES["Cycling"].comfort_range
        st.markdown(f"""
- **Daytime temperature** = 0.75 × maximum + 0.25 × minimum, about the mean of 10:00 to 18:00. It must lie in the
  activity's comfortable range: **{walk[0]}-{walk[1]} °C for walking, {bike[0]}-{bike[1]} °C for cycling**, the
  ranges where the route comfort score (`meteo/route_scores.py`) takes off no points for temperature.
- **Dry**: less than {TH['rain_day']} mm of rain that day.
- **Wind**: mean wind under {WIND_COMFORT_KMH} km/h, where the walking comfort score starts to take points off. A daily
  mean has no direction, so cycling uses the same limit rather than the headwind. Left out when a period has wind
  values on fewer than 80% of its days{'' if use_wind else ' (the case here)'}.
- Sunshine and humidity are not in every station's files: they are not used.
- **Seasons**: the days whose normal maximum, smoothed over ±15 days, is at least {num(SUMMER_TX, 0)} °C (warm) or at
  most {num(WINTER_TX, 0)} °C (cold).
""")
