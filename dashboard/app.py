"""Météo Analytics interactive dashboard.

Run from the project root:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run` only puts the script's own folder (dashboard/) on the import path:
# add the project root so that the `meteo` package is found, before importing it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from date_page import page_date
from future_page import page_future
from gestures import chart_gestures
from idle import pause_when_idle
from meteo.climatology import day_of_year, record_envelope
from meteo.stats import SEASONS, trend
from pivot_page import page_pivot
from projection_page import page_projection
from recent_panel import last_measurement, lately_panel
from route_page import page_route
from seasons_panel import seasons_panel
from share import from_link, publish, restore
from ui import (
    CFG,
    COOL,
    DIVERGING,
    END,
    GRID,
    HEAT,
    INK,
    MUTED,
    OLD,
    RAIN,
    RECENT,
    REF,
    SERIES,
    START,
    SURFACE,
    TH,
    WARM,
    chart,
    city_colors,
    doy_climatology,
    era5,
    is_dark,
    is_phone,
    num,
    page_header,
    pvalue,
    results,
    rolling,
    signed,
    signed_color,
    source_note,
    table,
    theme,
    warehouse_ready,
)
from wind_panel import wind_panel

st.set_page_config(page_title="Météo Analytics", page_icon="🌡️", layout="wide")

CITIES = [c.name for c in CFG.cities]
INDICATORS = {  # label -> (column, unit, available by season)
    "Mean temperature": ("tmean", "°C", True),
    "Maximum temperature": ("tx", "°C", True),
    "Minimum temperature": ("tn", "°C", True),
    "Total rainfall": ("rr", "mm", True),
    f"Hot days (TX ≥ {TH['hot_day']} °C)": ("hot_days", "days", False),
    f"Very hot days (TX ≥ {TH['very_hot_day']} °C)": ("very_hot_days", "days", False),
    f"Tropical nights (TN ≥ {TH['tropical_night']} °C)": ("tropical_nights", "days", False),
    f"Frost days (TN < {TH['frost_day']} °C)": ("frost_days", "days", False),
    f"Heat-episode days (≥ {TH['heatwave_min_days']} days at {TH['hot_day']} °C)": ("heatwave_days", "days", False),
    f"Heavy-rain days (≥ {TH['heavy_rain']} mm)": ("heavy_rain_days", "days", False),
    "Longest dry spell": ("longest_dry_spell", "days", False),
    "Highest temperature of the year": ("tx_max", "°C", False),
    "Lowest temperature of the year": ("tn_min", "°C", False),
}
COUNT_INDICATORS = {k: v for k, v in INDICATORS.items() if v[1] == "days"}
INDICATOR_OF_CODE = {v[0]: k for k, v in INDICATORS.items()}   # links carry the short code ("tropical_nights")
SEASON_LABELS = ["Whole year", "Winter", "Spring", "Summer", "Autumn"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# =============================================================================
def page_overview():
    r = results()
    w, ex, pl = r["warming"], r["extremes"], r.get("pipeline", {})
    tr = w["france_trend_1951"]
    last = last_measurement()
    page_header(
        f"Climate of 20 French cities, {START}-{END}", tab="Overview", lead=
        f"{num(pl.get('n_rows', 0), 0)} daily Météo-France records from {num(pl.get('n_stations', 0), 0)} stations. "
        f"All values are anomalies vs the {REF[0]}-{REF[1]} normal unless stated otherwise."
        + (f" Latest measurement: {last:%d %B %Y}." if last is not None else ""),
    )

    hot = ex["france_old_vs_recent"]["hot_days"]
    k = st.columns(4)
    k[0].metric("Warming per decade", f"{signed(tr['slope'], 2)} °C", border=True,
                help=f"Slope of the {START}-{END} linear regression on the average of the 20 cities. "
                     f"95% confidence interval: {num(tr['ci_low'], 2)} to {num(tr['ci_high'], 2)} °C. "
                     f"Mann-Kendall test: {pvalue(tr['mk_p_value'])}.")
    k[1].metric(f"{RECENT[0]}-{str(RECENT[1])[2:]} vs {OLD[0]}-{str(OLD[1])[2:]}", f"{signed(w['delta_recent_vs_old'], 2)} °C",
                border=True, help="Difference in mean temperature between the two periods (average of the 20 cities).")
    k[2].metric(f"Days ≥ {TH['hot_day']} °C", f"× {num(hot['ratio'], 1)}", border=True,
                help=f"{num(hot['old'], 1)} days per year and per city in {OLD[0]}-{OLD[1]}, "
                     f"{num(hot['recent'], 1)} in {RECENT[0]}-{RECENT[1]}.")
    k[3].metric("Hot vs cold records", f"{num(r['records']['last_ratio'], 0)} : 1",
                border=True, help="Daily records broken since 2020: highest TX vs lowest TN ever measured on that date. "
                                  "In a stable climate the ratio would be close to 1.")

    fra = pd.read_csv(CFG.reports_dir / "tables" / "france_anomaly.csv")
    years, anom = fra["year"], fra["anomaly_c"]
    fig = go.Figure(go.Bar(
        x=years, y=[1] * len(years), marker=dict(color=anom, colorscale=DIVERGING, cmid=0, line_width=0),
        customdata=anom, hovertemplate="<b>%{x}</b><br>%{customdata:+.2f} °C vs the normal<extra></extra>"))
    fig.update_layout(bargap=0, yaxis=dict(visible=False), xaxis=dict(showline=False, showgrid=False, dtick=10))
    chart(fig, 170, "Each stripe is a year", f"Annual temperature anomaly vs the {REF[0]}-{REF[1]} normal, "
          "average of the 20 cities (hover over a stripe)", margin=dict(l=0, r=0, b=28))

    left, right = st.columns([3, 2], gap="large")
    with left:
        fig = go.Figure()
        fig.add_bar(x=years, y=anom, marker_color=signed_color(anom), name="Annual anomaly", showlegend=False,
                    hovertemplate="%{y:+.2f} °C<extra>%{x}</extra>")
        fig.add_scatter(x=years, y=rolling(anom), mode="lines", line=dict(color=INK, width=2.5),
                        name="10-year moving average", hovertemplate="%{y:+.2f} °C<extra>10-yr avg</extra>")
        fig.update_layout(yaxis_title="°C", hovermode="x unified")
        chart(fig, 420, f"{signed(tr['slope'], 2)} °C per decade since {START}",
              "Red: year warmer than normal, blue: colder · "
              + ("the 10 warmest years all came after 2010" if w["n_top10_since_2011"] == 10
                 else f"{w['n_top10_since_2011']} of the 10 warmest years came after 2010"))
    with right:
        cities = pd.DataFrame(r["heterogeneity"]["cities"])
        ref = table("ref_station").query("priority == 1")[["city", "lat", "lon", "station_name"]]
        m = cities.merge(ref, on="city")
        fig = go.Figure(go.Scattermap(
            lat=m["lat"], lon=m["lon"], mode="markers",
            marker=dict(size=17, color=m["slope"], colorscale=HEAT, cmin=0, opacity=0.95,
                        colorbar=dict(title=dict(text="°C per decade", side="top"), thickness=10, len=0.7,
                                      orientation="h", y=-0.02, yanchor="top")),
            customdata=np.stack([m["city"], m["slope"], m["delta"], m["station_name"]], axis=1),
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]:+.2f} °C per decade"
                          "<br>%{customdata[2]:+.2f} °C between 1961-1990 and 2011-2025"
                          "<br><span style='color:#52514e'>station: %{customdata[3]}</span><extra></extra>"))
        # Centred on the cities' bounds and zoomed to fit them all, Ajaccio included (narrower on phones)
        lat0, lat1, lon0, lon1 = m["lat"].min(), m["lat"].max(), m["lon"].min(), m["lon"].max()
        width = 300 if is_phone() else 420
        zoom = min(np.log2(width * 360 / (512 * (lon1 - lon0 + 1))),
                   np.log2(330 * 360 / (512 * (lat1 - lat0 + 1) / np.cos(np.radians((lat0 + lat1) / 2)))))
        fig.update_layout(map=dict(style="carto-positron", center=dict(lat=(lat0 + lat1) / 2, lon=(lon0 + lon1) / 2),
                                   zoom=float(zoom)))
        chart(fig, 440, "Warming rate by city", "Hover over a dot for details", margin=dict(l=0, r=0, b=60))

    st.subheader("Lately")
    lately_panel(CITIES, "Mean of the 20 cities")

    st.subheader("Key takeaways")
    pr, h = r["precipitation"], r["heterogeneity"]
    findings = [
        f"<b>Clear, significant warming</b>: {signed(tr['slope'], 2)} °C per decade since {START}, "
        f"and {signed(w['france_trend_1981']['slope'], 2)} °C since 1981: it is speeding up.",
        f"<b>{h['fastest_season']}</b> is the season warming the fastest "
        f"({signed(h['season_trends'][h['fastest_season']]['slope'], 2)} °C per decade). "
        f"{h['fastest_city']} is warming the most, {h['slowest_city']} the least.",
        f"<b>Hot extremes are soaring</b>: days ≥ {TH['very_hot_day']} °C × "
        f"{num(ex['france_old_vs_recent']['very_hot_days']['ratio'], 1)}, tropical nights × "
        f"{num(ex['france_old_vs_recent']['tropical_nights']['ratio'], 1)}, frost days "
        f"{signed((ex['france_old_vs_recent']['frost_days']['ratio'] - 1) * 100, 0)}%.",
        f"<b>Rainfall: no significant trend</b> in annual totals "
        f"({signed(pr['annual_trend_pct']['slope'], 1)}% per decade, Mann-Kendall {pvalue(pr['annual_trend_pct']['mk_p_value'])}).",
        f"<b>All-time record</b>: {num(ex['hottest_day']['tx'], 1)} °C in {ex['hottest_day']['city']} on "
        f"{pd.Timestamp(ex['hottest_day']['date']).strftime('%d %B %Y')}.",
    ]
    for f_ in findings:
        st.markdown(f"<div class='finding'>{f_}</div>", unsafe_allow_html=True)
    source_note()


# =============================================================================
def _city_series(col: str, season: str) -> pd.DataFrame:
    """Annual (or seasonal) series per city for one indicator."""
    valid = "valid_rain" if col == "rr" else "valid_temp"
    if season == "Whole year":
        y = table("city_yearly")
        return y[y[valid] & y["year"].between(START, END)][["city", "year", col]]
    m = table("city_monthly")
    m = m[m[valid]].copy()
    m["season"] = m["month"].map(SEASONS)
    m["year"] = m["year"] + (m["month"] == 12).astype(int)
    g = (m[m["season"] == season].groupby(["city", "year"])[col]
         .agg(["sum" if col == "rr" else "mean", "count"]).reset_index())
    g = g[(g["count"] == 3) & g["year"].between(START, END)]
    return g.rename(columns={g.columns[2]: col})[["city", "year", col]]


def page_city():
    page_header("Explore a city", "Compare up to three cities on the indicator of your choice.")
    restore("city_sel", "cities", ["Paris", "Marseille"], multi=True,
            valid=lambda v: 0 < len(v) <= 3 and set(v) <= set(CITIES))
    restore("city_ind", "indicator", list(INDICATORS)[0], parse=INDICATOR_OF_CODE.__getitem__)
    restore("city_season", "season", "Whole year", parse=str.capitalize, valid=lambda v: v in SEASON_LABELS)
    restore("city_mode", "display", "Values", parse=str.capitalize, valid=lambda v: v in ("Values", "Anomaly"))
    first, last = from_link("from"), from_link("to")
    if first or last:
        try:
            lo, hi = sorted((int(first or START), int(last or END)))
            if START <= lo < hi <= END:
                st.session_state["city_period"] = (lo, hi)
        except ValueError:
            pass
    st.session_state.setdefault("city_period", (START, END))
    f = st.columns([3, 2, 1.2, 1.5])
    sel = f[0].multiselect("Cities", CITIES, max_selections=3, key="city_sel",
                           help="Three cities at most, so that each colour stays clearly distinct.")
    ind = f[1].selectbox("Indicator", list(INDICATORS), key="city_ind")
    col, unit, seasonal_ok = INDICATORS[ind]
    if not seasonal_ok:
        st.session_state["city_season"] = "Whole year"
    season = f[2].selectbox("Season", SEASON_LABELS if seasonal_ok else ["Whole year"], disabled=not seasonal_ok,
                            key="city_season", help="Winter = December of the previous year, January and February.")
    mode = f[3].segmented_control("Display", ["Values", "Anomaly"], required=True, key="city_mode") or "Values"
    period = st.slider("Period", START, END, key="city_period")
    publish({"cities": sel, "indicator": col, "season": None if season == "Whole year" else season.lower(),
             "display": None if mode == "Values" else mode.lower(),
             "from": None if period[0] == START else period[0], "to": None if period[1] == END else period[1]})
    if period[1] - period[0] < 1:
        st.info("Pick a period of at least two years.")
        return
    if not sel:
        st.info("Pick at least one city.")
        return

    colors = city_colors(sel, "city")
    df = _city_series(col, season)
    df = df[df["city"].isin(sel)].copy()
    relative = col == "rr"
    anomaly = mode == "Anomaly"
    if anomaly:
        normals = df[df["year"].between(*REF)].groupby("city")[col].mean()
        base = df["city"].map(normals)
        df[col] = (df[col] / base - 1) * 100 if relative else df[col] - base
    # Moving average on the whole series, then cut to the period: narrowing the period must not change it
    df = df.sort_values(["city", "year"])
    df["avg"] = df.groupby("city")[col].transform(lambda s: rolling(s.reset_index(drop=True)).to_numpy())
    df = df[df["year"].between(*period)]
    y_unit = "%" if (anomaly and relative) else unit

    fig = go.Figure()
    for city in sel:
        d = df[df["city"] == city].sort_values("year")
        c = colors[city]
        fig.add_scatter(x=d["year"], y=d[col], mode="lines", line=dict(color=c, width=1.2), opacity=0.45,
                        legendgroup=city, showlegend=False, name=city,
                        hovertemplate=f"{city}: %{{y:.1f}} {y_unit}<extra></extra>")
        fig.add_scatter(x=d["year"], y=d["avg"], mode="lines",
                        line=dict(color=c, width=2.8), legendgroup=city, name=city,
                        hovertemplate=f"{city} (10-yr avg): %{{y:.1f}} {y_unit}<extra></extra>")
    if anomaly:
        fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.update_layout(hovermode="x unified", yaxis_title=y_unit, xaxis=dict(tickformat="d"))
    label = ind if season == "Whole year" else f"{ind} · {season.lower()}"
    chart(fig, 460, label + (f": anomaly vs the {REF[0]}-{REF[1]} normal" if anomaly else ""),
          f"Thin line: annual value · thick line: 10-year moving average · {period[0]}-{period[1]}")

    cols = st.columns(len(sel))
    for c_, city in zip(cols, sel, strict=True):
        d = df[df["city"] == city]
        try:
            t = trend(d["year"], d[col])
            verdict = "significant" if t.significant else "not significant"
            c_.metric(f"Trend in {city}", f"{signed(t.slope, 2)} {y_unit} / decade", border=True,
                      help=f"Linear regression, 95% CI [{num(t.ci_low, 2)}, {num(t.ci_high, 2)}]. "
                           f"Theil-Sen slope: {signed(t.sen_slope, 2)}.")
            # Both tests decide (significant = both p < 0.05), so both p-values are shown
            c_.markdown(f"<div class='kpi-note'>Regression {pvalue(t.p_value)} · Mann-Kendall {pvalue(t.mk_p_value)}"
                        f" · trend {verdict}</div>", unsafe_allow_html=True)
        except ValueError:
            c_.metric(f"Trend in {city}", "period too short", border=True)

    st.subheader("Seasonal cycle: then and now")
    # The monthly variable the indicator is about: rainfall for rain indicators, TX for hot days, TN for nights
    mcol = {"hot_days": "tx", "very_hot_days": "tx", "heatwave_days": "tx", "tx_max": "tx", "tropical_nights": "tn",
            "frost_days": "tn", "tn_min": "tn", "heavy_rain_days": "rr", "longest_dry_spell": "rr"}.get(col, col)
    mcol = mcol if mcol in ("tmean", "tx", "tn", "rr") else "tmean"
    m = table("city_monthly")
    m = m[m["city"].isin(sel) & m["valid_rain" if mcol == "rr" else "valid_temp"]]
    fig = go.Figure()
    for city in sel:
        for per, width, opacity in ((OLD, 1.6, 0.5), (REF, 2.8, 1.0)):
            d = m[(m["city"] == city) & m["year"].between(*per)].groupby("month")[mcol].mean()
            fig.add_scatter(x=MONTHS, y=d.values, mode="lines+markers", name=f"{city} {per[0]}-{per[1]}",
                            line=dict(color=colors[city], width=width), opacity=opacity, marker=dict(size=8),
                            hovertemplate=f"{city} {per[0]}-{per[1]}: %{{y:.1f}} {'mm' if mcol == 'rr' else '°C'}<extra></extra>")
    fig.update_layout(hovermode="x unified", yaxis_title="mm" if mcol == "rr" else "°C")
    name = {"tmean": "Mean temperature", "tx": "Maximum temperature", "tn": "Minimum temperature", "rr": "Rainfall"}[mcol]
    chart(fig, 400, f"{name}, month by month", f"{OLD[0]}-{OLD[1]} normal (light line) and {REF[0]}-{REF[1]} (dark line)")

    seasons_panel(sel)

    st.subheader("Lately")
    who = sel[0] if len(sel) == 1 else st.selectbox("City", sel, key="lately_city",
                                                    help="The last weeks of one of the cities picked above.")
    lately_panel([who], who)

    with st.expander("Weather stations used"):
        rs = table("ref_station")
        rs = rs[rs["city"].isin(sel)].copy()
        rs["role"] = np.where(rs["priority"] == 1, "main", "backup (spliced series)")
        rs["completeness_temp"] = rs["completeness_temp"] * 100   # shown as "98%" whatever the browser's language
        st.dataframe(rs[["city", "role", "station_id", "station_name", "altitude", "distance_km",
                         "first_date", "last_date", "completeness_temp"]],
                     hide_index=True, column_config={
                         "city": "City", "role": "Role", "station_id": "Météo-France ID", "station_name": "Station",
                         "altitude": st.column_config.NumberColumn("Altitude", format="%d m"),
                         "distance_km": st.column_config.NumberColumn("Distance", format="%.1f km"),
                         "first_date": st.column_config.DateColumn("Start", format="D MMM YYYY"),
                         "last_date": st.column_config.DateColumn("End", format="D MMM YYYY"),
                         "completeness_temp": st.column_config.ProgressColumn(
                             f"Completeness {START}-{END}", min_value=0, max_value=100, format="%.0f%%")})
    source_note()


# =============================================================================
def page_year():
    page_header("A year in detail", "Every day of the year compared with the city's normal and its records.")
    d_all = table("city_daily")
    last_year = int(d_all.dropna(subset=["tx"])["date"].dt.year.max())
    years = list(range(last_year, START - 1, -1))
    variables = ["Maximum", "Minimum", "Mean", "Rainfall"]
    restore("year_city", "city", "Paris", valid=lambda v: v in CITIES)
    restore("year_year", "year", 2022 if 2022 in years else years[0], parse=int, valid=lambda v: v in years)
    restore("year_variable", "variable", "Maximum", parse=str.capitalize, valid=lambda v: v in variables)
    f = st.columns([2, 1.4, 3])
    city = f[0].selectbox("City", CITIES, key="year_city")
    year = f[1].selectbox("Year", years, key="year_year",
                          format_func=lambda y: f"{y} (in progress)" if y > END else str(y))
    var_label = f[2].segmented_control("Variable", variables, required=True, key="year_variable") or "Maximum"
    publish({"city": city, "year": year, "variable": var_label.lower()})
    d = d_all[d_all["city"] == city]
    if var_label == "Rainfall":
        _year_rain(city, year, d)
        source_note()
        return
    col = {"Maximum": "tx", "Minimum": "tn", "Mean": "tmean"}[var_label]

    clim = doy_clim(city, col)
    yr = d[d["date"].dt.year == year][["date", col]].dropna().copy()
    yr = yr[~((yr["date"].dt.month == 2) & (yr["date"].dt.day == 29))]
    yr["doy"] = day_of_year(yr["date"]).astype(int)
    yr = yr.merge(clim, on="doy")
    yr["anom"] = yr[col] - yr["mean"]
    ref_dates = _doy_dates(year, clim["doy"])

    # Records "at the time": the highest and lowest value of each calendar day in the years before this one
    env = record_envelope(d, col, year)
    env = env[~((env["month"] == 2) & (env["day"] == 29))].copy()
    env["date"] = pd.to_datetime(dict(year=year, month=env["month"], day=env["day"]))
    env = env.sort_values("date")
    yr = yr.assign(month=yr["date"].dt.month, day=yr["date"].dt.day).merge(env.drop(columns="date"),
                                                                              on=["month", "day"], how="left")
    rec_hi, rec_lo = yr[yr[col] >= yr["high"]], yr[yr[col] <= yr["low"]]

    k = st.columns(4)
    k[0].metric("Anomaly vs normal", f"{signed(yr['anom'].mean(), 1)} °C", border=True,
                help="Average gap over the year between each day's value and that day's normal.")
    k[1].metric("Days above normal", f"{(yr['anom'] > 0).sum()}", border=True,
                help=f"Out of {len(yr)} days with data.")
    k[2].metric("Daily records", f"{len(rec_hi)} high · {len(rec_lo)} low", border=True,
                help=f"Days whose {var_label.lower()} temperature equalled or beat the highest (or lowest) value of "
                     f"that calendar day in every year from {START} to {year - 1}.")
    # The day that stands out for the variable: hottest day, coldest night, warmest day on average
    kpi, i_k = {"Maximum": ("Hottest day", yr[col].idxmax()), "Minimum": ("Coldest night", yr[col].idxmin()),
                "Mean": ("Warmest day", yr[col].idxmax())}[var_label]
    k[3].metric(kpi, f"{num(yr.loc[i_k, col], 1)} °C", border=True,
                help=f"{var_label} temperature, on {yr.loc[i_k, 'date'].strftime('%d %B %Y')}")

    fig = go.Figure()
    if not env.empty:
        fig.add_scatter(x=env["date"], y=env["high"], mode="lines", line=dict(color=WARM, width=1, dash="dot"),
                        name=f"Record high before {year}", customdata=env["high_year"],
                        hovertemplate="%{x|%d %b}: record %{y:.1f} °C in %{customdata}<extra></extra>")
        fig.add_scatter(x=env["date"], y=env["low"], mode="lines", line=dict(color=COOL, width=1, dash="dot"),
                        name=f"Record low before {year}", customdata=env["low_year"],
                        hovertemplate="%{x|%d %b}: record %{y:.1f} °C in %{customdata}<extra></extra>")
    fig.add_scatter(x=ref_dates, y=clim["p90"], mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False)
    fig.add_scatter(x=ref_dates, y=clim["p10"], mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(195,194,183,0.35)", name=f"80% of days in {REF[0]}-{REF[1]}", hoverinfo="skip")
    fig.add_scatter(x=ref_dates, y=clim["mean"], mode="lines", line=dict(color=MUTED, width=1.6),
                    name=f"{REF[0]}-{REF[1]} normal", hoverinfo="skip")
    fig.add_scatter(x=yr["date"], y=yr[col], mode="lines", line=dict(color=INK, width=1.6), name=str(year),
                    customdata=np.stack([yr["mean"], yr["anom"]], axis=1),
                    hovertemplate="<b>%{x|%d %b}</b>: %{y:.1f} °C<br>normal %{customdata[0]:.1f} °C · "
                                  "anomaly %{customdata[1]:+.1f} °C<extra></extra>")
    hot, cold = yr[yr[col] > yr["p90"]], yr[yr[col] < yr["p10"]]
    fig.add_scatter(x=hot["date"], y=hot[col], mode="markers", name="Above the 90th percentile",
                    marker=dict(color=WARM, size=8, line=dict(color=SURFACE, width=1.5)), hoverinfo="skip")
    fig.add_scatter(x=cold["date"], y=cold[col], mode="markers", name="Below the 10th percentile",
                    marker=dict(color=COOL, size=8, line=dict(color=SURFACE, width=1.5)), hoverinfo="skip")
    for recs, colour, what, prev, prev_year in ((rec_hi, WARM, "record high", "high", "high_year"),
                                               (rec_lo, COOL, "record low", "low", "low_year")):
        if recs.empty:
            continue
        tie = np.where(recs[col] == recs[prev], "equals", "beats")
        fig.add_scatter(x=recs["date"], y=recs[col], mode="markers", name=f"{what.capitalize()} of the day",
                        marker=dict(symbol="star", color=colour, size=13, line=dict(color=SURFACE, width=1.5)),
                        customdata=np.stack([recs[prev], recs[prev_year].astype(int), tie], axis=1),
                        hovertemplate=f"<b>%{{x|%d %b}}: %{{y:.1f}} °C</b>, {what}<br>%{{customdata[2]}} the previous "
                                      "record: %{customdata[0]:.1f} °C in %{customdata[1]}<extra></extra>")
    month_starts = [pd.Timestamp(year=year, month=m, day=1) for m in range(1, 13)]
    fig.update_layout(yaxis_title="°C", xaxis=dict(tickvals=month_starts, ticktext=MONTHS), hovermode="closest")
    chart(fig, 520, f"{var_label} temperature in {city} in {year}",
          f"Compared with the {REF[0]}-{REF[1]} normal (smoothed over ±15 days) and with the records of each "
          f"calendar day since {START} · stars: records set in {year}")

    yr["month"] = yr["date"].dt.month
    mon = yr.groupby("month")["anom"].mean()
    fig = go.Figure(go.Bar(x=[MONTHS[i - 1] for i in mon.index], y=mon.values, marker_color=signed_color(mon.values),
                           hovertemplate="%{x}: %{y:+.1f} °C<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.update_layout(yaxis_title="°C", showlegend=False)
    chart(fig, 320, "Anomaly vs normal, month by month")
    source_note()


def _doy_dates(year: int, doys) -> pd.Series:
    """Dates of the calendar days 1-365 in `year`: in a leap year, from 1 March they fall one day later."""
    leap = pd.Timestamp(year=year, month=1, day=1).is_leap_year
    doys = pd.Series(np.asarray(doys, dtype=int))
    return pd.Timestamp(year=year, month=1, day=1) + pd.to_timedelta(doys - 1 + (leap & (doys >= 60)), unit="D")


def _daily_rain(d: pd.DataFrame, year: int) -> tuple[np.ndarray, np.ndarray]:
    """Rainfall of each calendar day 1-365 of `year` (29 February added to the 28th) and which days were measured."""
    x = d[d["date"].dt.year == year][["date", "rr"]]
    doy = day_of_year(x["date"]).fillna(59).astype(int).to_numpy() - 1
    rain, measured = np.zeros(365), np.zeros(365, dtype=bool)
    np.add.at(rain, doy, x["rr"].fillna(0).to_numpy())
    measured[doy[x["rr"].notna().to_numpy()]] = True
    return rain, measured


@st.cache_data(show_spinner=False)
def rain_clim(city: str) -> pd.DataFrame:
    """Normal cumulative rainfall of each calendar day: mean, 10th and 90th percentiles over the complete years of
    the reference period."""
    d = table("city_daily")
    d = d[d["city"] == city]
    y = table("city_yearly")
    valid = y[(y["city"] == city) & y["valid_rain"] & y["year"].between(*REF)]["year"]
    cum = np.array([np.cumsum(_daily_rain(d, int(v))[0]) for v in valid])
    if not len(cum):
        return pd.DataFrame(columns=["doy", "mean", "p10", "p90"])
    return pd.DataFrame({"doy": np.arange(1, 366), "mean": cum.mean(axis=0), "p10": np.percentile(cum, 10, axis=0),
                         "p90": np.percentile(cum, 90, axis=0)})


def _year_rain(city: str, year: int, d: pd.DataFrame) -> None:
    """Rainfall view of the year page: cumulative rainfall against the normal, daily rainfall below."""
    rain, measured = _daily_rain(d, year)
    clim = rain_clim(city)
    if not measured.any() or clim.empty:
        st.info("No rainfall data for this city and year.")
        return
    last = int(np.flatnonzero(measured).max())          # last calendar day measured (0-based)
    cum = np.cumsum(rain)[: last + 1]
    dates = _doy_dates(year, np.arange(1, last + 2))
    total, normal_to_date = float(cum[-1]), float(clim["mean"].iloc[last])
    wet = rain[: last + 1] >= TH["rain_day"]
    rainy = int(wet.sum())
    dry_runs = np.diff(np.flatnonzero(np.concatenate([[True], wet, [True]]))) - 1
    so_far = last < 364
    k = st.columns(4)
    k[0].metric("Rainfall" + (" so far" if so_far else ""), f"{num(total, 0)} mm", border=True,
                delta=f"{signed((total / normal_to_date - 1) * 100, 0)}% vs normal" if normal_to_date else None,
                delta_color="off",
                help=f"Normal for the same days: {num(normal_to_date, 0)} mm ({REF[0]}-{REF[1]}). "
                     f"{int(measured[: last + 1].sum())} of {last + 1} days measured.")
    k[1].metric(f"Rainy days (≥ {TH['rain_day']} mm)", f"{rainy}", border=True)
    i_wet = int(np.argmax(rain[: last + 1]))
    k[2].metric("Wettest day", f"{num(rain[i_wet], 1)} mm", border=True, help=f"On {dates.iloc[i_wet]:%d %B %Y}.")
    k[3].metric("Longest dry spell", f"{int(dry_runs.max()) if len(dry_runs) else 0} days", border=True,
                help=f"Consecutive days with less than {TH['rain_day']} mm.")

    ref_dates = _doy_dates(year, clim["doy"])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.68, 0.32])
    fig.add_scatter(x=ref_dates, y=clim["p90"], mode="lines", line=dict(width=0), hoverinfo="skip",
                    showlegend=False, row=1, col=1)
    fig.add_scatter(x=ref_dates, y=clim["p10"], mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(195,194,183,0.35)", name=f"80% of years in {REF[0]}-{REF[1]}", hoverinfo="skip",
                    row=1, col=1)
    fig.add_scatter(x=ref_dates, y=clim["mean"], mode="lines", line=dict(color=MUTED, width=1.6),
                    name="Normal", hovertemplate="%{x|%d %b}: normal %{y:.0f} mm<extra></extra>", row=1, col=1)
    fig.add_scatter(x=dates, y=cum, mode="lines", line=dict(color=SERIES[0], width=2.4), name=str(year),
                    customdata=clim["mean"].iloc[: last + 1].to_numpy(),
                    hovertemplate="<b>%{x|%d %b}</b>: %{y:.0f} mm since 1 January<br>normal %{customdata:.0f} mm"
                                  "<extra></extra>", row=1, col=1)
    fig.add_bar(x=dates, y=rain[: last + 1], marker_color=SERIES[0], name="Daily rainfall", showlegend=False,
                hovertemplate="%{x|%d %b}: %{y:.1f} mm<extra></extra>", row=2, col=1)
    month_starts = [pd.Timestamp(year=year, month=m, day=1) for m in range(1, 13)]
    fig.update_xaxes(tickvals=month_starts, ticktext=MONTHS,
                     range=[month_starts[0] - pd.Timedelta(days=3), pd.Timestamp(year=year, month=12, day=31)])
    fig.update_yaxes(title_text="mm since 1 Jan", rangemode="tozero", row=1, col=1)
    fig.update_yaxes(title_text="mm / day", rangemode="tozero", row=2, col=1)
    fig.update_layout(hovermode="closest", bargap=0.1)
    chart(fig, 560, f"Rainfall in {city} in {year}" + (" so far" if so_far else ""),
          f"Cumulative rainfall since 1 January against the {REF[0]}-{REF[1]} normal · daily rainfall below")


@st.cache_data(show_spinner=False)
def doy_clim(city: str, col: str) -> pd.DataFrame:
    d = table("city_daily")
    return doy_climatology(d[d["city"] == city], col, REF)


# =============================================================================
def page_extremes():
    page_header("Extremes and records", "How the frequency of extreme days has changed, city by city.")
    r = results()
    restore("ext_ind", "indicator", list(COUNT_INDICATORS)[0], parse=INDICATOR_OF_CODE.__getitem__,
            valid=lambda v: v in COUNT_INDICATORS)
    ind = st.selectbox("Indicator", list(COUNT_INDICATORS), key="ext_ind")
    col = COUNT_INDICATORS[ind][0]
    publish({"indicator": col})
    scale = RAIN if col in ("frost_days", "heavy_rain_days") else HEAT

    rain_ind = col in ("heavy_rain_days", "longest_dry_spell")
    # A count of days per year, except the longest dry spell: the length of the year's longest run without rain
    per_year = col != "longest_dry_spell"
    unit_short, unit_long = ("days/yr", "days per year") if per_year else ("days", "days")
    y = table("city_yearly")
    y = y[y["valid_rain" if rain_ind else "valid_temp"] & y["year"].between(START, END)].copy()
    y["decade"] = y["year"] // 10 * 10
    pv = y.pivot_table(index="city", columns="decade", values=col, aggfunc="mean")
    pv.columns = [f"{max(c, START)}-{str(min(c + 9, END))[2:]}" for c in pv.columns]
    pv = pv.loc[pv.iloc[:, -1].sort_values().index]
    fig = go.Figure(go.Heatmap(
        z=pv.values, x=pv.columns, y=pv.index, colorscale=scale, xgap=2, ygap=2,
        colorbar=dict(title=dict(text=unit_short, side="top"), thickness=12),
        hovertemplate="<b>%{y}</b>, %{x}: %{z:.1f} " + unit_long + "<extra></extra>"))
    # Values written in each cell, in a colour that stands out from it: the cells darken with the value in light
    # mode and brighten in dark mode, so the text switches between dark and light at mid-scale
    vals = pv.to_numpy(dtype=float)
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    frac = (vals - lo) / (hi - lo) if hi > lo else np.zeros_like(vals)
    light_text = (frac < 0.55) if is_dark() else (frac >= 0.55)
    xs, ys = np.meshgrid(list(pv.columns), list(pv.index))
    shown = ~np.isnan(vals)
    fig.add_scatter(x=xs[shown], y=ys[shown], mode="text", text=[num(v, 0) for v in vals[shown]], hoverinfo="skip",
                    textfont=dict(size=11, color=np.where(light_text[shown], "#f8f7f4", "#141413").tolist()),
                    showlegend=False)
    fig.update_layout(xaxis=dict(side="top", showline=False, type="category", showgrid=False),
                      yaxis=dict(showline=False, type="category", showgrid=False))
    chart(fig, 640, ind, margin=dict(l=120, t=40),
          subtitle=("Average number of days per year" if per_year else "Average length of the year's longest dry "
                    "spell, in days") + ", by decade (cities sorted by the latest decade)")

    old = y[y["year"].between(*OLD)].groupby("city")[col].mean()
    new = y[y["year"].between(*RECENT)].groupby("city")[col].mean()
    dd = pd.DataFrame({"old": old, "new": new}).sort_values("new")
    fig = go.Figure()
    xs, ys = [], []
    for c, row in dd.iterrows():
        xs += [row["old"], row["new"], None]
        ys += [c, c, None]
    fig.add_scatter(x=xs, y=ys, mode="lines", line=dict(color=GRID, width=3), hoverinfo="skip", showlegend=False)
    fig.add_scatter(x=dd["old"], y=dd.index, mode="markers", name=f"{OLD[0]}-{OLD[1]}",
                    marker=dict(color=SERIES[0], size=11, line=dict(color=SURFACE, width=2)),
                    hovertemplate="%{y}: %{x:.1f} days<extra>" + f"{OLD[0]}-{OLD[1]}</extra>")
    fig.add_scatter(x=dd["new"], y=dd.index, mode="markers", name=f"{RECENT[0]}-{RECENT[1]}",
                    marker=dict(color=SERIES[1], size=11, line=dict(color=SURFACE, width=2)),
                    hovertemplate="%{y}: %{x:.1f} days<extra>" + f"{RECENT[0]}-{RECENT[1]}</extra>")
    fig.update_layout(xaxis_title=unit_long, yaxis=dict(showgrid=False))
    chart(fig, 600, "Then and now", f"{OLD[0]}-{OLD[1]} average vs {RECENT[0]}-{RECENT[1]}")

    left, right = st.columns([1, 1], gap="large")
    with left:
        rec = pd.DataFrame(r["records"]["by_decade"])
        fig = go.Figure(go.Bar(
            x=rec["decade"].astype(str), y=rec["ratio"], marker_color=[WARM if v >= 1 else COOL for v in rec["ratio"]],
            customdata=np.stack([rec["hot"], rec["cold"]], axis=1),
            text=[f"{num(v, 1)} : 1" for v in rec["ratio"]], textposition="outside",
            hovertemplate="<b>%{x}</b><br>%{customdata[0]} hot records<br>%{customdata[1]} cold records<extra></extra>"))
        fig.add_hline(y=1, line=dict(color=MUTED, width=1))
        top = max(32, rec["ratio"].max() * 2)
        ticks = [v for v in (0.5, 1, 2, 4, 8, 16, 32, 64) if v <= top]
        fig.update_layout(yaxis=dict(type="log", title="hot / cold records", range=[np.log10(0.7), np.log10(top)],
                                     tickvals=ticks, ticktext=[f"{num(v, 0)} : 1" if v >= 1 else "1 : 2" for v in ticks]),
                          xaxis=dict(type="category"), showlegend=False)
        chart(fig, 420, "Heat records vs cold records", "Daily records broken, 20 cities (log scale)")
    with right:
        st.markdown(f"**Heat episodes** (at least {TH['heatwave_min_days']} consecutive days at {TH['hot_day']} °C or more)")
        hw = table("city_heatwaves")
        hw = hw[hw["start_date"].dt.year <= END]
        c1, c2 = st.columns([2, 1])
        cities = c1.multiselect("Cities", CITIES, default=[], placeholder="All cities", key="hw_cities")
        min_days = c2.number_input("Minimum length", 3, 60, 7)
        hw = hw[hw["n_days"] >= min_days]
        if cities:
            hw = hw[hw["city"].isin(cities)]
        st.dataframe(hw.sort_values(["n_days", "tx_peak"], ascending=False), hide_index=True, height=330,
                     column_order=["city", "start_date", "end_date", "n_days", "tx_peak", "tx_mean"],
                     column_config={
                         "city": "City",
                         "start_date": st.column_config.DateColumn("Start", format="D MMM YYYY"),
                         "end_date": st.column_config.DateColumn("End", format="D MMM YYYY"),
                         "n_days": st.column_config.NumberColumn("Length", format="%d days"),
                         "tx_peak": st.column_config.NumberColumn("Peak", format="%.1f °C"),
                         "tx_mean": st.column_config.NumberColumn("Mean TX", format="%.1f °C")})
        st.caption(f"{num(len(hw), 0)} episodes match the filters, longest first (scroll for more). Heatwaves, "
                   "cold waves and dry spells ranked by intensity: Explore a date.")
    wind_panel()
    source_note()


# =============================================================================
def page_quality():
    page_header("Data, method and quality",
                "Where the numbers come from, how they are cleaned, and how far they can be trusted.")
    r = results()
    pl = r.get("pipeline", {})
    build_s = sum(v for k, v in pl.items() if k.startswith("duration_"))
    p = theme()
    st.graphviz_chart(f"""
    digraph {{
      rankdir={"TB" if is_phone() else "LR"}; bgcolor="transparent"; nodesep=0.35; ranksep=0.45;
      node [shape=box style="rounded,filled" fillcolor="{p['surface']}" color="{p['grid']}" fontname="Segoe UI" fontsize=11 fontcolor="{p['ink']}" margin="0.18,0.1"];
      edge [color="{p['muted']}" arrowsize=0.7];
      api  [label="data.gouv.fr API\\nfile discovery"];
      raw  [label="{pl.get('n_files', 0)} CSV.gz files\\n{num(pl.get('raw_bytes', 0) / 1e6, 0)} MB · bronze"];
      stg  [label="DuckDB staging\\n{num(pl.get('n_rows', 0) / 1e6, 1)} M typed rows"];
      dq   [label="Quality control\\ncodes, bounds, consistency"];
      st   [label="Reference stations\\n+ series splicing"];
      gold [label="Gold tables\\nday · month · year"];
      out  [label="Analyses · report\\ndashboard · SQL" fillcolor="{p['series']}" fontcolor="#ffffff" color="{p['series']}"];
      api -> raw -> stg -> dq -> st -> gold -> out;
    }}""", width="stretch")
    st.caption(f"Full pipeline: parallel download (~30 s), warehouse build in {num(build_s, 1)} s, "
               f"analyses in ~6 s. One command: `python -m meteo all`.")

    k = st.columns(4)
    k[0].metric("Raw rows", num(pl.get("n_rows", 0), 0), border=True)
    k[1].metric("Stations", num(pl.get("n_stations", 0), 0), border=True)
    k[2].metric("Rejected values", num(pl.get("n_measures", 0) - pl.get("n_valid_measures", 0), 0), border=True,
                help=f"Out of {num(pl.get('n_measures', 0) / 1e6, 1)} million measurements present: "
                     "'suspect' quality codes, impossible values or TN > TX.")
    k[3].metric("Station-day duplicates", num(pl.get("n_duplicates", 0), 0), border=True)

    st.subheader("Quality summary by variable")
    dq = pd.DataFrame(r["quality"]["by_variable"])
    labels = {"TN": "Min temperature", "TX": "Max temperature", "TM": "Mean hourly temperature",
              "RR": "Rainfall", "FFM": "Mean wind", "FXI": "Max gust"}
    dq["variable"] = dq["variable"].map(labels)
    for c in ("suspect", "out_of_range", "inconsistent"):
        if c not in dq:
            dq[c] = 0
    # Text, so that "11,326,531" reads the same as the rest of the page whatever the browser's language
    for c in ("valid", "missing", "suspect", "out_of_range", "inconsistent"):
        dq[c] = [num(v, 0) for v in dq[c]]
    st.dataframe(dq[["variable", "valid", "missing", "suspect", "out_of_range", "inconsistent", "pct_missing"]],
                 hide_index=True, column_config={
                     "variable": "Variable",
                     "valid": st.column_config.TextColumn("Valid"),
                     "missing": st.column_config.TextColumn("Missing"),
                     "suspect": st.column_config.TextColumn("Suspect (code 2)"),
                     "out_of_range": st.column_config.TextColumn("Out of range"),
                     "inconsistent": st.column_config.TextColumn("TN > TX"),
                     "pct_missing": st.column_config.ProgressColumn("% missing", min_value=0, max_value=100, format="%.0f%%")})
    st.markdown(
        "Météo-France already validates its data: very few values are rejected. **The real issue is completeness**: "
        "about half of the stations ever opened measured rainfall only, and the network keeps changing (openings, "
        "closures, relocations). "
        "Hence the choice of long, complete stations, and series splicing when a station has moved.")

    nw = table("network_yearly")
    nw = nw[nw["year"] <= END]
    fig = go.Figure()
    fig.add_scatter(x=nw["year"], y=nw["n_stations_rain"], name="measuring rainfall", line=dict(color=SERIES[0], width=2.2),
                    hovertemplate="%{y} stations<extra>rainfall</extra>")
    fig.add_scatter(x=nw["year"], y=nw["n_stations_temp"], name="measuring temperature", line=dict(color=SERIES[1], width=2.2),
                    hovertemplate="%{y} stations<extra>temperature</extra>")
    fig.update_layout(hovermode="x unified", yaxis=dict(rangemode="tozero", title="active stations"))
    chart(fig, 380, "The measurement network changes over time", "Active stations per year in the 20 downloaded départements")

    st.subheader("Reference stations and series splicing")
    off = pd.DataFrame(r["quality"]["offsets"])
    rs = table("ref_station")
    rs = rs.merge(off[["station_id", "offset_tn", "offset_tx", "n_overlap_days"]], on="station_id", how="left")
    for c in ("offset_tn", "offset_tx"):  # text: an empty cell rather than "None"
        rs[c] = [f"{signed(v, 2)} °C" if pd.notna(v) else "" for v in rs[c]]
    rs["n_overlap_days"] = [num(v, 0) if pd.notna(v) else "" for v in rs["n_overlap_days"]]
    rs["completeness_temp"] = rs["completeness_temp"] * 100
    st.dataframe(rs[["city", "priority", "station_name", "altitude", "distance_km", "first_date", "last_date",
                     "completeness_temp", "offset_tn", "offset_tx", "n_overlap_days"]],
                 hide_index=True, height=420, column_config={
                     "city": "City", "priority": st.column_config.NumberColumn("Rank", help="1 = main station"),
                     "station_name": "Station",
                     "altitude": st.column_config.NumberColumn("Alt.", format="%d m"),
                     "distance_km": st.column_config.NumberColumn("Distance", format="%.1f km"),
                     "first_date": st.column_config.DateColumn("Start", format="YYYY"),
                     "last_date": st.column_config.DateColumn("End", format="YYYY"),
                     "completeness_temp": st.column_config.ProgressColumn(f"Completeness {START}-{END}", min_value=0, max_value=100, format="%.0f%%"),
                     "offset_tn": st.column_config.TextColumn("TN correction", help="Mean bias corrected, computed month by month"),
                     "offset_tx": st.column_config.TextColumn("TX correction"),
                     "n_overlap_days": st.column_config.TextColumn("Shared days", help="Days measured by both stations")})

    e = era5()
    if e is not None:
        st.subheader("Cross-validation against the ERA5 reanalysis")
        e = e.sort_values("bias_tn")
        fig = go.Figure()
        fig.add_bar(y=e["city"], x=e["bias_tx"], orientation="h", name="Maximum temperature", marker_color=SERIES[1],
                    hovertemplate="%{y}: %{x:+.2f} °C<extra>TX</extra>")
        fig.add_bar(y=e["city"], x=e["bias_tn"], orientation="h", name="Minimum temperature", marker_color=SERIES[0],
                    hovertemplate="%{y}: %{x:+.2f} °C<extra>TN</extra>")
        fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08, xaxis_title="mean bias ERA5 − station (°C)",
                          yaxis=dict(showgrid=False))
        chart(fig, 620, r.get("era5", {}).get("title", "ERA5 vs stations"),
              f"{CFG.era5['start'][:4]}-{CFG.era5['end'][:4]} · ERA5 (~25 km grid) via the Open-Meteo API")
        x = r.get("era5", {})
        e["hot_ratio"] = e["hot_days_era5"] / e["hot_days_obs"].clip(lower=1)
        miss = e[e["hot_days_obs"] >= 50].sort_values("hot_ratio").iloc[0]
        st.markdown(
            f"Mean correlation of daily TX: **{num(x.get('mean_corr_tx', 0), 3)}**: the reanalysis tracks day-to-day "
            f"weather very well. But a 25 km grid cell mixes sea, city and countryside: **{x.get('worst_city')}** shows a "
            f"{signed(x.get('worst_bias_tn', 0), 1)} °C bias on minimum temperatures, and in **{miss['city']}** ERA5 counts "
            f"only {num(int(miss['hot_days_era5']), 0)} days ≥ {TH['hot_day']} °C against {num(int(miss['hot_days_obs']), 0)} observed. "
            "To study local extremes, stations remain essential.")

    with st.expander("Limitations of the analysis"):
        st.markdown(f"""
- **Non-homogenised series**: changes of instruments, urbanisation around stations or relocations can create
  breaks. Météo-France publishes homogenised series for research; here only the identified relocations are
  corrected (splicing with a monthly bias).
- **One station per city**: it represents the city, not its whole area (urban heat island, terrain).
- **Linear trend**: useful as a summary, but warming has been speeding up since the 1980s.
- **Autocorrelation**: the tests assume independent years, so the p-values are slightly optimistic.
- **Current year**: {END + 1} is incomplete and excluded from annual statistics.
""")
    source_note("; ERA5: Copernicus Climate Change Service, via Open-Meteo")


# =============================================================================
# =============================================================================
route = st.Page(page_route, title="Route weather", icon=":material/route:", url_path="route")
if warehouse_ready():
    nav = st.navigation({
        "Climate analysis": [
            st.Page(page_overview, title="Overview", icon=":material/thermostat:", default=True),
            st.Page(page_city, title="Explore a city", icon=":material/location_city:", url_path="city"),
            st.Page(page_year, title="A year in detail", icon=":material/calendar_month:", url_path="year"),
            st.Page(page_date, title="Explore a date", icon=":material/event:", url_path="date"),
            st.Page(page_extremes, title="Extremes and records", icon=":material/local_fire_department:", url_path="extremes"),
            st.Page(page_future, title="Climate ahead", icon=":material/trending_up:", url_path="ahead"),
            st.Page(page_quality, title="Data and quality", icon=":material/fact_check:", url_path="quality"),
            st.Page(page_pivot, title="Pivot table", icon=":material/pivot_table_chart:", url_path="pivot"),
        ],
        "Tools": [st.Page(page_projection, title="Prediction", icon=":material/online_prediction:",
                          url_path="prediction"), route],
    })
else:
    # Without the warehouse, only the route tool (which uses the weather API) is available
    nav = st.navigation([route])
    st.sidebar.info("Climate pages appear once the data is built: run `python -m meteo all`.")
with st.sidebar:
    st.markdown("### Météo Analytics")
    st.caption(f"Climate of 20 French cities, {START}-{END}, from Météo-France open data, "
               "and weather along any GPS track.")
pause_when_idle()
chart_gestures()
nav.run()
