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
from ui import OLD, RECENT, SERIES, TH, chart, num, pct, t, table

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
    head = f"<b>{name}</b> ({threshold})"
    if then is None and now is None:
        return t("seasons.none_either", head=head)
    if now is None:
        return t("seasons.gone", head=head, then=then.label, old=old, new=new)
    if then is None:
        return t("seasons.new", head=head, old=old, now=now.label, new=new, days=now.days)
    text = t("seasons.both", head=head, now=now.label, new=new, days_now=now.days, then=then.label, old=old,
             days_then=then.days)
    shift = shift_days(then, now)
    if shift:
        earlier, later = shift
        parts = []
        if earlier:
            parts.append(t("seasons.starts_earlier" if earlier > 0 else "seasons.starts_later", n=abs(earlier)))
        if later:
            parts.append(t("seasons.ends_later" if later > 0 else "seasons.ends_earlier", n=abs(later)))
        if parts:
            text += t("seasons.shift", parts=t("fmt.and").join(parts))
    return text + "."


def seasons_panel(cities: list[str]) -> None:
    st.subheader(t("seasons.title"))
    c1, c2 = st.columns([1.2, 2])
    activity = c1.segmented_control(t("seasons.activity"), ACTIVITIES, default="Walking", required=True,
                                    key="weeks_activity", format_func=lambda a: t(f"activity.{a}"))
    activity = activity or "Walking"
    city = cities[0] if len(cities) == 1 else c2.selectbox(t("common.city"), cities, key="weeks_city")
    weeks, use_wind = _weeks(city, activity)
    if weeks.empty:
        st.info(t("seasons.no_data"))
        return
    old, new = PERIODS_LABELS
    fig = go.Figure()
    for label, colour, width in ((old, SERIES[0], 1.8), (new, SERIES[1], 2.6)):
        w = weeks[weeks["period"] == label].sort_values("week")
        fig.add_scatter(x=[week_label(v) for v in w["week"]], y=w["share"] * 100, mode="lines", name=label,
                        line=dict(color=colour, width=width, shape="spline", smoothing=0.6),
                        customdata=w["n_days"], hovertemplate=t("seasons.hover", period=label))
    fig.update_layout(hovermode="x unified", yaxis=dict(title=t("seasons.axis"), rangemode="tozero"),
                      xaxis=dict(type="category", nticks=13))
    low, high = PROFILES[activity].comfort_range
    rules = (t("seasons.rules", low=low, high=high, rain=num(TH["rain_day"], 0))
             + (t("seasons.rules_wind", wind=WIND_COMFORT_KMH) if use_wind else ""))
    chart(fig, 380, t("seasons.chart_title", activity=t(f"seasons.for_{activity}"), city=city),
          t("seasons.chart_sub", rules=rules, old=old, new=new))

    pv = weeks.pivot(index="week", columns="period", values="share").dropna()
    if not pv.empty:
        best = pv[new].idxmax()
        diff = (pv[new] - pv[old]) * 100
        gain, loss = diff.idxmax(), diff.idxmin()
        line = t("seasons.best", week=week_label(best), now=pct(pv.loc[best, new]), then=pct(pv.loc[best, old]),
                 old=old)
        if diff[loss] <= -5:
            line += t("seasons.loss", week=week_label(loss), then=pct(pv.loc[loss, old]), now=pct(pv.loc[loss, new]))
        if diff[gain] >= 5:
            line += t("seasons.gain", week=week_label(gain), then=pct(pv.loc[gain, old]), now=pct(pv.loc[gain, new]))
        st.markdown(f"<div class='finding'>{line}</div>", unsafe_allow_html=True)

    s = _seasons(city)
    st.markdown(f"<div class='chart-title'>{t('seasons.seasons_title')}</div>", unsafe_allow_html=True)
    warm = _season_line(t("seasons.warm"), t("seasons.warm_rule", tx=num(SUMMER_TX, 0)), s[(old, True)], s[(new, True)])
    cold = _season_line(t("seasons.cold"), t("seasons.cold_rule", tx=num(WINTER_TX, 0)), s[(old, False)],
                        s[(new, False)])
    st.markdown(f"<div class='finding'>{warm}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='finding'>{cold}</div>", unsafe_allow_html=True)
    with st.expander(t("seasons.how_title")):
        walk, bike = PROFILES["Walking"].comfort_range, PROFILES["Cycling"].comfort_range
        st.markdown(t("seasons.how", walk_low=walk[0], walk_high=walk[1], bike_low=bike[0], bike_high=bike[1],
                      rain=num(TH["rain_day"], 0), wind=WIND_COMFORT_KMH,
                      here="" if use_wind else t("seasons.how_here"), warm=num(SUMMER_TX, 0), cold=num(WINTER_TX, 0)))
