"""Dashboard page: a bulletin of the coming days projected from past measurements (not a weather forecast)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from meteo.projection import YEARS, Projection, project, skill
from ui import (
    CFG,
    INK2,
    MUTED,
    SERIES,
    SURFACE,
    TH,
    chart,
    is_phone,
    num,
    page_header,
    rgba,
    source_note,
    table,
    theme,
)

CITIES = [c.name for c in CFG.cities]
DAYS = 7                       # days of the bulletin, today included
MAX, MIN = SERIES[1], SERIES[0]  # orange for the maximum, blue for the minimum (fixed, whatever the city)
SHOWN_RISK = 0.10              # frost and hot-day chances are shown from 10%


@st.cache_data(show_spinner=False, ttl=3600)
def _projection(city: str, until: pd.Timestamp) -> Projection:
    d = table("city_daily")
    return project(d[d["city"] == city], until, TH)


@st.cache_data(show_spinner=False)
def _skill(city: str) -> pd.DataFrame:
    d = table("city_daily")
    return skill(d[d["city"] == city], "tx")


def _rain_words(p: float) -> tuple[str, str]:
    """Icon and words of a chance of rain (the projection knows rain, not clouds or sunshine)."""
    if p >= 0.6:
        return ":material/rainy:", "Rain likely"
    if p >= 0.3:
        return ":material/cloud:", "Rain possible"
    return ":material/partly_cloudy_day:", "Rain unlikely"


def _pct(p: float) -> str:
    return f"{num(p * 100, 0)}%"


def _when(day: pd.Series, today: pd.Timestamp) -> str:
    ahead = (day["date"] - today).days
    return "today" if ahead == 0 else "tomorrow" if ahead == 1 else f"in {ahead} days"


def _risks(day: pd.Series) -> list[str]:
    risks = []
    if day["p_frost"] >= SHOWN_RISK:
        risks.append(f":material/ac_unit: Frost {_pct(day['p_frost'])}")
    if day["p_hot"] >= SHOWN_RISK:
        risks.append(f":material/local_fire_department: {TH['hot_day']} °C or more {_pct(day['p_hot'])}")
    return risks


def _card(day: pd.Series, today: pd.Timestamp) -> None:
    st.markdown(f"**{day['date'].strftime('%a %d %b')}**  \n:gray[{_when(day, today)}]")
    st.markdown(f"<div class='pj-temp'><b>{num(day['tx'], 0)}°</b><span> / {num(day['tn'], 0)}°</span></div>"
                f"<div class='pj-range'>max {num(day['tx_low'], 0)} to {num(day['tx_high'], 0)}°<br>"
                f"min {num(day['tn_low'], 0)} to {num(day['tn_high'], 0)}°</div>", unsafe_allow_html=True)
    icon, words = _rain_words(day["p_rain"])
    st.markdown(f"{icon} {words}  \n:gray[{_pct(day['p_rain'])} chance, ~{num(day['rain_mm'], 0)} mm]")
    if risks := _risks(day):
        st.markdown("  \n".join(risks))


def _compact(bulletin: pd.DataFrame, today: pd.Timestamp) -> None:
    """On a phone the cards would stack into a long column: one line per day instead."""
    lines = []
    for _, day in bulletin.iterrows():
        icon, words = _rain_words(day["p_rain"])
        lines.append(f"**{day['date'].strftime('%a %d %b')}** :gray[{_when(day, today)}]  \n"
                     f"**{num(day['tx'], 0)}°** / {num(day['tn'], 0)}° · {icon} {words}, {_pct(day['p_rain'])}"
                     + "".join(f" · {r}" for r in _risks(day)))
    with st.container(border=True):
        st.markdown("\n\n".join(lines))


def _mark(fig: go.Figure, x, text: str, left: bool = False) -> None:
    """A vertical line with its label inside the top of the plot (above it, the legend would cover it), on the
    right of the line or on its left (two lines close together on a phone)."""
    fig.add_vline(x=x, line=dict(color=MUTED, width=1))
    fig.add_annotation(x=x, y=1, yref="paper", yanchor="top", text=text, showarrow=False,
                       xanchor="right" if left else "left", xshift=-4 if left else 4, font=dict(color=INK2, size=12))


def _temperature_chart(p: Projection, today: pd.Timestamp, city: str) -> None:
    rec, days = p.recent, p.days
    # The projected lines start from the last measurement, so that they continue the measured ones
    last = rec.iloc[[-1]].assign(tx_low=rec["tx"].iloc[-1], tx_high=rec["tx"].iloc[-1],
                                 tn_low=rec["tn"].iloc[-1], tn_high=rec["tn"].iloc[-1])
    proj = pd.concat([last, days], ignore_index=True)
    fig = go.Figure()
    for var, colour, name in (("tx", MAX, "Maximum"), ("tn", MIN, "Minimum")):
        fig.add_scatter(x=proj["date"], y=proj[f"{var}_high"], mode="lines", line=dict(width=0), hoverinfo="skip",
                        showlegend=False, legendgroup=var)
        fig.add_scatter(x=proj["date"], y=proj[f"{var}_low"], mode="lines", line=dict(width=0), fill="tonexty",
                        fillcolor=rgba(colour, 0.16), hoverinfo="skip", showlegend=False, legendgroup=var)
        fig.add_scatter(x=rec["date"], y=rec[var], mode="lines+markers", name=name, legendgroup=var,
                        line=dict(color=colour, width=2), marker=dict(size=6, color=colour),
                        customdata=rec[f"{var}_normal"],
                        hovertemplate=f"{name}: %{{y:.1f}} °C, measured (normal %{{customdata:.1f}})<extra></extra>")
        fig.add_scatter(x=days["date"], y=days[var], mode="lines", showlegend=False, legendgroup=var,
                        line=dict(color=colour, width=2, dash="dash"),
                        customdata=np.stack([days[f"{var}_low"], days[f"{var}_high"], days[f"{var}_normal"]], axis=1),
                        hovertemplate=f"{name}: %{{y:.1f}} °C, projected (%{{customdata[0]:.0f}} to "
                                      f"%{{customdata[1]:.0f}}, normal %{{customdata[2]:.1f}})<extra></extra>")
        fig.add_scatter(x=pd.concat([rec["date"], days["date"]]), y=pd.concat([rec[f"{var}_normal"], days[f"{var}_normal"]]),
                        mode="lines", line=dict(color=MUTED, width=1.2, dash="dot"), hoverinfo="skip",
                        name="Normal of the day", legendgroup="normal", showlegend=var == "tx")
        fig.add_annotation(x=days["date"].iloc[-1], y=days[var].iloc[-1], text=name, showarrow=False, xanchor="left",
                           xshift=6, font=dict(color=INK2, size=12))
    fig.add_scatter(x=[None], y=[None], mode="lines", line=dict(color=MUTED, width=2, dash="dash"),
                    name="Projected (band: 80% of past cases)", hoverinfo="skip")
    _mark(fig, p.anchor, "last measurement", left=True)
    _mark(fig, today, "today")
    half_day = pd.Timedelta(hours=12)
    fig.update_layout(yaxis_title="°C", hovermode="x unified", margin=dict(r=72), xaxis=dict(
        range=[rec["date"].iloc[0] - half_day, days["date"].iloc[-1] + half_day], showspikes=True, spikecolor=MUTED,
        spikethickness=1, spikedash="solid", spikemode="across"))
    chart(fig, 440, f"Temperature in {city}: the last three weeks, then the projection",
          "Solid: measured. Dashed: projected, with the range of 80% of past cases. Dotted: the normal of each day, "
          f"including the warming trend of the last {YEARS} years")


def _rain_chart(p: Projection, today: pd.Timestamp) -> None:
    days = p.days
    fig = go.Figure()
    fig.add_bar(x=days["date"], y=days["p_rain"] * 100, name="Chance of rain", marker=dict(color=SERIES[0],
                cornerradius=4, line=dict(color=SURFACE, width=2)), customdata=days["rain_mm"],
                hovertemplate="%{x|%a %d %b}: %{y:.0f}% chance of rain, about %{customdata:.0f} mm if it does"
                              "<extra></extra>")
    fig.add_scatter(x=days["date"], y=days["p_rain_normal"] * 100, mode="markers", name="Usual at this time of year",
                    marker=dict(symbol="line-ew-open", size=22, color=INK2, line=dict(width=2)),
                    hovertemplate="usually %{y:.0f}%<extra></extra>")
    _mark(fig, today - pd.Timedelta(hours=12), "today")
    fig.update_layout(yaxis=dict(title="% chance", range=[0, 100], ticksuffix="%"), hovermode="x unified", bargap=0.35)
    chart(fig, 300, f"Chance of a rainy day (≥ {TH['rain_day']} mm)",
          "After a day like the last measured one (rainy or dry), in the same season of past years; the line is "
          "the usual chance for the day. Hover over a bar for the typical amount")


def _skill_chart(s: pd.DataFrame, lead_today: int) -> None:
    fig = go.Figure()
    fig.add_scatter(x=s["lead"], y=s["normal"], mode="lines+markers", name="Normal of the day alone",
                    line=dict(color=MUTED, width=2), marker=dict(size=8, color=MUTED, line=dict(color=SURFACE, width=2)),
                    hovertemplate="%{x} days ahead: %{y:.1f} °C off on average<extra>normal alone</extra>")
    fig.add_scatter(x=s["lead"], y=s["projection"], mode="lines+markers", name="Projection",
                    line=dict(color=SERIES[0], width=2),
                    marker=dict(size=8, color=SERIES[0], line=dict(color=SURFACE, width=2)),
                    hovertemplate="%{x} days ahead: %{y:.1f} °C off on average<extra>projection</extra>")
    if lead_today <= s["lead"].max():
        _mark(fig, lead_today, "today")
    fig.update_layout(xaxis=dict(title="days after the last measurement", dtick=1),
                      yaxis=dict(title="average error (°C)", rangemode="tozero"), hovermode="closest")
    useful = s[s["projection"] < 0.95 * s["normal"]]["lead"]
    horizon = int(useful.max()) if len(useful) else 0
    beats = (f"It beats the normal alone by 5% or more up to day {horizon}" if horizon else
             "Here it never beats the normal alone by 5%")
    now = (f"; today is day {lead_today}, so today's values are hardly more than the normal" if lead_today > horizon
           else f"; today is day {lead_today}, still within that reach")
    chart(fig, 320, "How far ahead does it help?",
          f"Average error on the maximum temperature, checked on the last two years with a model trained on the "
          f"{YEARS} years before them. {beats}{now}")


def _kept(p: Projection, lead: int) -> str:
    row = p.days[p.days["lead"] == lead]
    return _pct(max(float(row["tx_kept"].iloc[0]), 0)) if len(row) else "–"


def _table(p: Projection) -> None:
    days = p.days.copy()
    days["p_rain"] *= 100
    days["p_rain_normal"] *= 100
    days["p_frost"] *= 100
    days["p_hot"] *= 100
    pct = st.column_config.NumberColumn
    st.dataframe(days, hide_index=True, column_order=["date", "tx", "tx_low", "tx_high", "tn", "tn_low", "tn_high",
                                                      "p_rain", "p_rain_normal", "rain_mm", "p_frost", "p_hot"],
                 column_config={
                     "date": st.column_config.DateColumn("Day", format="ddd D MMM"),
                     "tx": pct("Max", format="%.1f °C"), "tx_low": pct("Max, low", format="%.1f °C"),
                     "tx_high": pct("Max, high", format="%.1f °C"),
                     "tn": pct("Min", format="%.1f °C"), "tn_low": pct("Min, low", format="%.1f °C"),
                     "tn_high": pct("Min, high", format="%.1f °C"),
                     "p_rain": pct("Rain", format="%.0f%%"), "p_rain_normal": pct("Rain, usual", format="%.0f%%"),
                     "rain_mm": pct("Amount if rain", format="%.0f mm"), "p_frost": pct("Frost", format="%.0f%%"),
                     "p_hot": pct(f"≥ {TH['hot_day']} °C", format="%.0f%%")})


def page_projection():
    page_header("The coming days, projected", tab="Prediction", lead=
                "A bulletin of the next days built only from past measurements: what the season usually brings, "
                "and how long a warm, cold or wet spell usually lasts.")
    st.warning(
        "**A projection, not a forecast.** These values come from the past alone: the normal of each day, plus the "
        "gap of the last measured day, which fades over the next days at the pace seen in past years. They know "
        "nothing of the weather on its way (fronts, storms, heat waves announced). For the actual forecast, see "
        "[Météo-France](https://meteofrance.com/).", icon=":material/info:")
    p_ = theme()
    st.markdown(f"<style>.pj-temp {{ font-size: 1.7rem; color: {p_['ink']}; line-height: 1.2; margin-top: 0.2rem; }}"
                f".pj-temp span {{ color: {p_['ink2']}; font-size: 1.25rem; }}"
                f".pj-range {{ color: {p_['ink2']}; font-size: 0.82rem; margin-bottom: 0.6rem; }}</style>",
                unsafe_allow_html=True)

    city = st.selectbox("City", CITIES, index=CITIES.index("Paris"))
    today = pd.Timestamp.today().normalize()
    p = _projection(city, today + pd.Timedelta(days=DAYS - 1))
    gap = (today - p.anchor).days
    past = f", including the {gap - 1} already past" if gap > 2 else (", including yesterday" if gap == 2 else "")
    st.caption(f"Last measurement: {p.anchor.strftime('%A %d %B %Y').replace(' 0', ' ')}. Météo-France publishes its "
               f"data a few days late and this site reloads it every Monday: every day after it is projected{past}.")
    if gap > 30:
        st.info("The last measurement is over a month old: the projection is then the normal of each day.")

    bulletin = p.days[p.days["date"] >= today].head(DAYS)
    if is_phone():
        _compact(bulletin, today)
    else:
        for col, (_, day) in zip(st.columns(len(bulletin)), bulletin.iterrows(), strict=True):
            with col.container(border=True):
                _card(day, today)

    _temperature_chart(p, today, city)
    _rain_chart(p, today)
    _skill_chart(_skill(city), gap)

    with st.expander("The projected values"):
        _table(p)
        st.caption("Low and high: the range of 80% of past cases. Frost and hot day: the share of past cases that "
                   "would have crossed the threshold.")
    with st.expander("How it is computed"):
        st.markdown(f"""
- **Normal of the day**: the seasonal cycle plus a linear trend, fitted on the last {YEARS} years of daily values and
  extended to the projected day, so that it includes recent warming.
- **The gap fades**: for each number of days ahead, a regression of the gap to the normal on the gap of the last
  measured day, over the days of the same season (±45 days) in those {YEARS} years. In {city} at this time of year,
  {_kept(p, 1)} of the gap on the maximum temperature is still there the next day, {_kept(p, 3)} after three days and
  {_kept(p, 7)} after a week.
- **Range**: the 10th to 90th percentiles of that regression's errors in the past.
- **Rain**: among the days of the same season that were rainy (or dry) like the last measured day, the share
  followed by a rainy day that many days later; the amount is the median of the rainy days at that time of year.
- **Its limits**: one station per city, no knowledge of the atmosphere, and data a few days old, so that the
  projection of today is close to the normal. It describes what past years make likely, nothing more.
""")
    source_note()
