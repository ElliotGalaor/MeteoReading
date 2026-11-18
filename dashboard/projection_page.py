"""Dashboard page: a bulletin of the coming days projected from past measurements (not a weather forecast)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from meteo.projection import YEARS, Projection, project, skill
from share import publish, restore
from ui import (
    CFG,
    INK2,
    MUTED,
    SERIES,
    SURFACE,
    TH,
    chart,
    date_text,
    fmt_date,
    is_phone,
    num,
    number,
    page_header,
    pct,
    plural,
    rgba,
    source_note,
    styled,
    t,
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
        return ":material/rainy:", t("pred.rain_likely")
    if p >= 0.3:
        return ":material/cloud:", t("pred.rain_possible")
    return ":material/partly_cloudy_day:", t("pred.rain_unlikely")


def _when(day: pd.Series, today: pd.Timestamp) -> str:
    ahead = (day["date"] - today).days
    return t("pred.today") if ahead == 0 else t("pred.tomorrow") if ahead == 1 else plural("pred.in_days", ahead)


def _risks(day: pd.Series) -> list[str]:
    risks = []
    if day["p_frost"] >= SHOWN_RISK:
        risks.append(t("pred.risk_frost", pct=pct(day["p_frost"])))
    if day["p_hot"] >= SHOWN_RISK:
        risks.append(t("pred.risk_hot", hot_day=TH["hot_day"], pct=pct(day["p_hot"])))
    return risks


def _card(day: pd.Series, today: pd.Timestamp) -> None:
    st.markdown(f"**{fmt_date(day['date'], 'weekday')}**  \n:gray[{_when(day, today)}]")
    st.markdown(f"<div class='pj-temp'><b>{num(day['tx'], 0)}°</b><span> / {num(day['tn'], 0)}°</span></div>"
                f"<div class='pj-range'>{t('pred.range_max', low=num(day['tx_low'], 0), high=num(day['tx_high'], 0))}"
                f"<br>{t('pred.range_min', low=num(day['tn_low'], 0), high=num(day['tn_high'], 0))}</div>",
                unsafe_allow_html=True)
    icon, words = _rain_words(day["p_rain"])
    st.markdown(f"{icon} {words}  \n:gray[{t('pred.chance', pct=pct(day['p_rain']), mm=num(day['rain_mm'], 0))}]")
    if risks := _risks(day):
        st.markdown("  \n".join(risks))


def _compact(bulletin: pd.DataFrame, today: pd.Timestamp) -> None:
    """On a phone the cards would stack into a long column: one line per day instead."""
    lines = []
    for _, day in bulletin.iterrows():
        icon, words = _rain_words(day["p_rain"])
        lines.append(f"**{fmt_date(day['date'], 'weekday')}** :gray[{_when(day, today)}]  \n"
                     f"**{num(day['tx'], 0)}°** / {num(day['tn'], 0)}° · {icon} {words}, {pct(day['p_rain'])}"
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
    for var, colour in (("tx", MAX), ("tn", MIN)):
        name = t(f"pred.{var}")
        fig.add_scatter(x=proj["date"], y=proj[f"{var}_high"], mode="lines", line=dict(width=0), hoverinfo="skip",
                        showlegend=False, legendgroup=var, meta={"csv": t("pred.csv_high", name=name)})
        fig.add_scatter(x=proj["date"], y=proj[f"{var}_low"], mode="lines", line=dict(width=0), fill="tonexty",
                        fillcolor=rgba(colour, 0.16), hoverinfo="skip", showlegend=False, legendgroup=var,
                        meta={"csv": t("pred.csv_low", name=name)})
        fig.add_scatter(x=rec["date"], y=rec[var], mode="lines+markers", name=name, legendgroup=var,
                        line=dict(color=colour, width=2), marker=dict(size=6, color=colour),
                        customdata=rec[f"{var}_normal"], meta={"csv": t("pred.csv_measured", name=name)},
                        hovertemplate=t("pred.hover_measured", name=name))
        # The dashed line joins the last measurement to the first projected day, as the band does (no hover: the
        # two points have their own; unnamed, so the CSV leaves it out)
        fig.add_scatter(x=proj["date"].iloc[:2], y=proj[var].iloc[:2], mode="lines", showlegend=False,
                        legendgroup=var, line=dict(color=colour, width=2, dash="dash"), hoverinfo="skip")
        fig.add_scatter(x=days["date"], y=days[var], mode="lines", showlegend=False, legendgroup=var,
                        line=dict(color=colour, width=2, dash="dash"), meta={"csv": t("pred.csv_projected", name=name)},
                        customdata=np.stack([days[f"{var}_low"], days[f"{var}_high"], days[f"{var}_normal"]], axis=1),
                        hovertemplate=t("pred.hover_projected", name=name))
        fig.add_scatter(x=pd.concat([rec["date"], days["date"]]), y=pd.concat([rec[f"{var}_normal"], days[f"{var}_normal"]]),
                        mode="lines", line=dict(color=MUTED, width=1.2, dash="dot"), hoverinfo="skip",
                        name=t("pred.normal"), legendgroup="normal", showlegend=var == "tx",
                        meta={"csv": t("pred.csv_normal", name=name)})
        fig.add_annotation(x=days["date"].iloc[-1], y=days[var].iloc[-1], text=name, showarrow=False, xanchor="left",
                           xshift=6, font=dict(color=INK2, size=12))
    fig.add_scatter(x=[None], y=[None], mode="lines", line=dict(color=MUTED, width=2, dash="dash"),
                    name=t("pred.projected"), hoverinfo="skip")
    _mark(fig, p.anchor, t("pred.mark_last"), left=True)
    _mark(fig, today, t("pred.mark_today"))
    half_day = pd.Timedelta(hours=12)
    fig.update_layout(yaxis_title="°C", hovermode="x unified", margin=dict(r=72), xaxis=dict(
        range=[rec["date"].iloc[0] - half_day, days["date"].iloc[-1] + half_day], showspikes=True, spikecolor=MUTED,
        spikethickness=1, spikedash="solid", spikemode="across"))
    chart(fig, 440, t("pred.temp_title", city=city), t("pred.temp_sub", years=YEARS), source=t("source.projection_csv"))


def _rain_chart(p: Projection, today: pd.Timestamp) -> None:
    days = p.days
    fig = go.Figure()
    fig.add_bar(x=days["date"], y=days["p_rain"] * 100, name=t("pred.rain_chance"), marker=dict(color=SERIES[0],
                cornerradius=4, line=dict(color=SURFACE, width=2)), customdata=days["rain_mm"],
                hovertemplate=t("pred.rain_hover"))
    fig.add_scatter(x=days["date"], y=days["p_rain_normal"] * 100, mode="markers", name=t("pred.rain_usual"),
                    marker=dict(symbol="line-ew-open", size=22, color=INK2, line=dict(width=2)),
                    hovertemplate=t("pred.rain_usual_hover"))
    _mark(fig, today - pd.Timedelta(hours=12), t("pred.mark_today"))
    fig.update_layout(yaxis=dict(title=t("pred.rain_axis"), range=[0, 100], ticksuffix=t("pred.percent_suffix")),
                      hovermode="x unified", bargap=0.35)
    rain_csv = pd.DataFrame({t("chart.csv_date"): days["date"].dt.strftime("%Y-%m-%d"),
                             t("pred.csv_rain_chance"): (days["p_rain"] * 100).round(1),
                             t("pred.csv_rain_usual"): (days["p_rain_normal"] * 100).round(1),
                             t("pred.csv_rain_mm"): days["rain_mm"].round(1)})
    chart(fig, 300, t("pred.rain_title", rain_day=num(TH["rain_day"], 0)), t("pred.rain_sub"), data=rain_csv,
          source=t("source.projection_csv"))


def _skill_chart(s: pd.DataFrame, lead_today: int) -> None:
    fig = go.Figure()
    fig.add_scatter(x=s["lead"], y=s["normal"], mode="lines+markers", name=t("pred.skill_normal"),
                    line=dict(color=MUTED, width=2), marker=dict(size=8, color=MUTED, line=dict(color=SURFACE, width=2)),
                    hovertemplate=t("pred.skill_hover", which=t("pred.skill_normal_short")))
    fig.add_scatter(x=s["lead"], y=s["projection"], mode="lines+markers", name=t("pred.skill_projection"),
                    line=dict(color=SERIES[0], width=2),
                    marker=dict(size=8, color=SERIES[0], line=dict(color=SURFACE, width=2)),
                    hovertemplate=t("pred.skill_hover", which=t("pred.skill_projection_short")))
    if lead_today <= s["lead"].max():
        _mark(fig, lead_today, t("pred.mark_today"))
    fig.update_layout(xaxis=dict(title=t("pred.skill_x"), dtick=1),
                      yaxis=dict(title=t("pred.skill_y"), rangemode="tozero"), hovermode="closest")
    useful = s[s["projection"] < 0.95 * s["normal"]]["lead"]
    horizon = int(useful.max()) if len(useful) else 0
    beats = t("pred.skill_beats", day=horizon) if horizon else t("pred.skill_never")
    now = t("pred.skill_now_beyond" if lead_today > horizon else "pred.skill_now_within", day=lead_today)
    chart(fig, 320, t("pred.skill_title"), t("pred.skill_sub", years=YEARS, beats=beats, now=now),
          source=t("source.projection_csv"))


def _kept(p: Projection, lead: int) -> str:
    row = p.days[p.days["lead"] == lead]
    return pct(max(float(row["tx_kept"].iloc[0]), 0)) if len(row) else "–"


def _table(p: Projection) -> None:
    days = p.days.copy()
    for c in ("p_rain", "p_rain_normal", "p_frost", "p_hot"):
        days[c] *= 100
    cols = ["date", "tx", "tx_low", "tx_high", "tn", "tn_low", "tn_high", "p_rain", "p_rain_normal", "rain_mm",
            "p_frost", "p_hot"]
    share = number(0, "%" if t("fmt.decimal") == "." else " %")
    formats = {"date": date_text("weekday"), "rain_mm": number(0, "mm"), "p_rain": share, "p_rain_normal": share,
               "p_frost": share, "p_hot": share, **{c: number(1, "°C") for c in cols[1:7]}}
    st.dataframe(styled(days[cols], formats), hide_index=True, column_config={
        "date": t("pred.col_day"), "tx": t("pred.col_max"), "tx_low": t("pred.col_max_low"),
        "tx_high": t("pred.col_max_high"), "tn": t("pred.col_min"), "tn_low": t("pred.col_min_low"),
        "tn_high": t("pred.col_min_high"), "p_rain": t("pred.col_rain"), "p_rain_normal": t("pred.col_rain_usual"),
        "rain_mm": t("pred.col_rain_mm"), "p_frost": t("pred.col_frost"), "p_hot": f"≥ {TH['hot_day']} °C"})


def page_projection():
    page_header(t("pred.title"), tab=t("app.nav_prediction"), lead=t("pred.lead"))
    st.warning(t("pred.warning"), icon=":material/info:")
    p_ = theme()
    st.markdown(f"<style>.pj-temp {{ font-size: 1.7rem; color: {p_['ink']}; line-height: 1.2; margin-top: 0.2rem; }}"
                f".pj-temp span {{ color: {p_['ink2']}; font-size: 1.25rem; }}"
                f".pj-range {{ color: {p_['ink2']}; font-size: 0.82rem; margin-bottom: 0.6rem; }}</style>",
                unsafe_allow_html=True)

    restore("proj_city", "city", "Paris", valid=lambda v: v in CITIES)
    city = st.selectbox(t("common.city"), CITIES, key="proj_city")
    publish({"city": city})
    today = pd.Timestamp.today().normalize()
    p = _projection(city, today + pd.Timedelta(days=DAYS - 1))
    gap = (today - p.anchor).days
    past = t("pred.past_many", n=gap - 1) if gap > 2 else (t("pred.past_yesterday") if gap == 2 else "")
    st.caption(t("pred.last_measurement", date=fmt_date(p.anchor, "weekday_long"), past=past))
    if gap > 30:
        st.info(t("pred.old_data"))

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

    with st.expander(t("pred.values_title")):
        _table(p)
        st.caption(t("pred.values_caption"))
    with st.expander(t("pred.how_title")):
        st.markdown(t("pred.how", years=YEARS, city=city, kept1=_kept(p, 1), kept3=_kept(p, 3), kept7=_kept(p, 7)))
    source_note()
