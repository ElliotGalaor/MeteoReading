"""The last weeks against the normal: the "Lately" panel of the Overview and city pages."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from meteo.climatology import doy_climatology, latest, mean_series, month_so_far, ordinal, records_between
from ui import COOL, INK, MUTED, REF, START, SURFACE, WARM, chart, fmt_date, num, signed, t, table

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
        st.info(t("lately.none"))
        return
    last = pd.Timestamp(recent["date"].max())
    month = month_so_far(series, REF, START)

    k = st.columns(3)
    k[0].metric(t("lately.latest"), fmt_date(last, "short"), border=True, help=t("lately.latest_help"))
    last_30 = recent[recent["date"] > last - pd.Timedelta(days=30)]
    k[1].metric(t("lately.last30"), f"{signed(last_30['anomaly'].mean(), 1)} °C", border=True,
                help=t("lately.last30_help", first=REF[0], last=REF[1]))
    if month:
        # Ranked from the end it is closest to: a month below the 1991-2020 normal can still be warmer than most years
        warm = month["rank_warm"] <= month["rank_cold"]
        rank = month["rank_warm"] if warm else month["rank_cold"]
        whole = month["last_day"] >= pd.Timestamp(last).days_in_month
        title = t("lately.month" if whole else "lately.month_so_far", month=month["month_name"], year=month["year"])
        title = title[:1].upper() + title[1:]
        k[2].metric(title, f"{signed(month['anomaly'], 1)} °C", border=True,
                    help=t("lately.month_help", last_day=month["last_day"], month=month["month_name"],
                           value=num(month["value"], 1), normal=num(month["normal"], 1), first=REF[0], last=REF[1],
                           n=month["n_years"], since=month["first_year"]))
        ranked = t("lately.rank_warm" if warm else "lately.rank_cold", rank=ordinal(rank))
        note = t("lately.rank_note", title=title, rank=ranked, since=month["first_year"], last_day=month["last_day"],
                 n=month["n_years"])
        st.markdown(f"<div class='kpi-note'>{note}</div>", unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_scatter(x=recent["date"], y=recent["p90"], mode="lines", line=dict(width=0), hoverinfo="skip",
                    showlegend=False)
    fig.add_scatter(x=recent["date"], y=recent["p10"], mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(195,194,183,0.35)", name=t("common.band_80_days", first=REF[0], last=REF[1]),
                    hoverinfo="skip")
    fig.add_scatter(x=recent["date"], y=recent["mean"], mode="lines", line=dict(color=MUTED, width=1.6),
                    name=t("common.normal"), hoverinfo="skip")
    fig.add_scatter(x=recent["date"], y=recent["value"], mode="lines", line=dict(color=INK, width=2), name=label,
                    customdata=recent[["mean", "anomaly"]].to_numpy(),
                    hovertemplate=t("lately.hover"))
    hot, cold = recent[recent["value"] > recent["p90"]], recent[recent["value"] < recent["p10"]]
    fig.add_scatter(x=hot["date"], y=hot["value"], mode="markers", name=t("common.above_p90"),
                    marker=dict(color=WARM, size=8, line=dict(color=SURFACE, width=1.5)), hoverinfo="skip")
    fig.add_scatter(x=cold["date"], y=cold["value"], mode="markers", name=t("common.below_p10"),
                    marker=dict(color=COOL, size=8, line=dict(color=SURFACE, width=1.5)), hoverinfo="skip")
    fig.update_layout(yaxis_title="°C", hovermode="closest")
    chart(fig, 340, t("lately.chart_title", days=DAYS), t("lately.chart_sub", label=label, first=REF[0], last=REF[1]))

    rec = _records(key, last)
    if rec.empty:
        st.caption(t("lately.no_record", days=RECORD_DAYS))
        return
    lines = []
    for r in rec.sort_values("date", ascending=False).head(8).itertuples():
        lines.append(t("lately.record_hot" if r.kind == "hot" else "lately.record_cold",
                       date=fmt_date(r.date, "day_month"), city=r.city, value=num(r.value, 1),
                       previous=num(r.previous, 1), year=int(r.previous_year)))
    n_hot, n_cold = int((rec["kind"] == "hot").sum()), int((rec["kind"] == "cold").sum())
    st.markdown(t("lately.records", days=RECORD_DAYS, hot=n_hot, cold=n_cold)
                + ("" if len(rec) <= 8 else t("lately.records_latest")))
    for line in lines:
        st.markdown(f"<div class='finding'>{line}</div>", unsafe_allow_html=True)
