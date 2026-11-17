"""The last weeks against the normal: the "Lately" panel of the Overview and city pages."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from meteo.climatology import doy_climatology, latest, mean_series, month_so_far, ordinal, records_between
from ui import COOL, INK, MUTED, REF, START, SURFACE, WARM, chart, num, signed, table

DAYS = 60              # days shown on the chart
RECORD_DAYS = 30       # days searched for records


@st.cache_data(show_spinner=False)
def _series(cities: tuple[str, ...]) -> pd.DataFrame:
    """Daily mean temperature of these cities (their mean, corrected for the cities not measured yet)."""
    d = table("city_daily")
    d = d[d["city"].isin(cities)]
    if len(cities) == 1:
        return d[["date", "tmean"]].rename(columns={"tmean": "value"}).dropna().reset_index(drop=True)
    return mean_series(d, "tmean", REF)


@st.cache_data(show_spinner=False)
def _clim(cities: tuple[str, ...]) -> pd.DataFrame:
    return doy_climatology(_series(cities), "value", REF)


@st.cache_data(show_spinner=False)
def _records(cities: tuple[str, ...], end: pd.Timestamp) -> pd.DataFrame:
    d = table("city_daily")
    return records_between(d[d["city"].isin(cities)], end - pd.Timedelta(days=RECORD_DAYS - 1), end, START)


def last_measurement() -> pd.Timestamp | None:
    """The latest day with a temperature measured in at least one city."""
    d = table("city_daily").dropna(subset=["tmean"])
    return d["date"].max() if not d.empty else None


def lately_panel(cities: list[str], label: str) -> None:
    """Chart of the last 60 days against the normal, the month so far and its rank, and the records of the last
    30 days, for one city or the mean of several."""
    key = tuple(sorted(cities))
    series = _series(key)
    recent = latest(series, _clim(key), DAYS)
    if recent.empty:
        st.info("No recent measurement.")
        return
    last = pd.Timestamp(recent["date"].max())
    month = month_so_far(series, REF, START)

    k = st.columns(3)
    k[0].metric("Latest measurement", last.strftime("%d %b %Y"), border=True,
                help="Météo-France publishes each day's values a few days later; this site reloads them every day.")
    last_30 = recent[recent["date"] > last - pd.Timedelta(days=30)]
    k[1].metric("Last 30 days vs normal", f"{signed(last_30['anomaly'].mean(), 1)} °C", border=True,
                help=f"Mean daily temperature over the last 30 days measured, against the {REF[0]}-{REF[1]} normal "
                     "of the same days.")
    if month:
        # Ranked from the end it is closest to: a month below the 1991-2020 normal can still be warmer than most years
        warm = month["rank_warm"] <= month["rank_cold"]
        rank = month["rank_warm"] if warm else month["rank_cold"]
        so_far = "" if month["last_day"] >= pd.Timestamp(last).days_in_month else " so far"
        k[2].metric(f"{month['month_name']} {month['year']}{so_far}", f"{signed(month['anomaly'], 1)} °C", border=True,
                    help=f"1-{month['last_day']} {month['month_name']}: mean {num(month['value'], 1)} °C, against "
                         f"{num(month['normal'], 1)} °C on the same days of {REF[0]}-{REF[1]}. Ranked among the same "
                         f"days of the {month['n_years']} years since {month['first_year']}.")
        st.markdown(f"<div class='kpi-note'>{month['month_name']} {month['year']}{so_far}: "
                    f"<b>{ordinal(rank)} {'warmest' if warm else 'coldest'}</b> since {month['first_year']} "
                    f"(days 1 to {month['last_day']}, {month['n_years']} years compared)</div>",
                    unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_scatter(x=recent["date"], y=recent["p90"], mode="lines", line=dict(width=0), hoverinfo="skip",
                    showlegend=False)
    fig.add_scatter(x=recent["date"], y=recent["p10"], mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(195,194,183,0.35)", name=f"80% of days in {REF[0]}-{REF[1]}", hoverinfo="skip")
    fig.add_scatter(x=recent["date"], y=recent["mean"], mode="lines", line=dict(color=MUTED, width=1.6),
                    name="Normal", hoverinfo="skip")
    fig.add_scatter(x=recent["date"], y=recent["value"], mode="lines", line=dict(color=INK, width=2), name=label,
                    customdata=recent[["mean", "anomaly"]].to_numpy(),
                    hovertemplate="<b>%{x|%a %d %b}</b>: %{y:.1f} °C<br>normal %{customdata[0]:.1f} °C · "
                                  "%{customdata[1]:+.1f} °C<extra></extra>")
    hot, cold = recent[recent["value"] > recent["p90"]], recent[recent["value"] < recent["p10"]]
    fig.add_scatter(x=hot["date"], y=hot["value"], mode="markers", name="Above the 90th percentile",
                    marker=dict(color=WARM, size=8, line=dict(color=SURFACE, width=1.5)), hoverinfo="skip")
    fig.add_scatter(x=cold["date"], y=cold["value"], mode="markers", name="Below the 10th percentile",
                    marker=dict(color=COOL, size=8, line=dict(color=SURFACE, width=1.5)), hoverinfo="skip")
    fig.update_layout(yaxis_title="°C", hovermode="closest")
    chart(fig, 340, f"Mean daily temperature, last {DAYS} days measured",
          f"{label} against the {REF[0]}-{REF[1]} normal, smoothed over ±15 days")

    rec = _records(key, last)
    if rec.empty:
        st.caption(f"No daily record in the last {RECORD_DAYS} days measured.")
        return
    lines = []
    for r in rec.sort_values("date", ascending=False).head(8).itertuples():
        what = "highest maximum" if r.kind == "hot" else "lowest minimum"
        lines.append(f"{r.date:%d %b}: {r.city}, {what} for that day, {num(r.value, 1)} °C "
                     f"(record before: {num(r.previous, 1)} °C in {int(r.previous_year)})")
    n_hot, n_cold = int((rec["kind"] == "hot").sum()), int((rec["kind"] == "cold").sum())
    st.markdown(f"**Daily records in the last {RECORD_DAYS} days**: {n_hot} hot, {n_cold} cold"
                + ("" if len(rec) <= 8 else " (latest 8 below)"))
    for line in lines:
        st.markdown(f"<div class='finding'>{line}</div>", unsafe_allow_html=True)
