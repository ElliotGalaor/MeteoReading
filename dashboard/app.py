"""Météo Analytics interactive dashboard, in English and French (text in meteo/locales/).

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

import language
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
    date_text,
    doy_climatology,
    era5,
    fmt_date,
    is_dark,
    is_phone,
    num,
    number,
    page_header,
    percent,
    plural,
    pvalue,
    results,
    rolling,
    signed,
    signed_color,
    source_note,
    styled,
    t,
    table,
    theme,
    warehouse_ready,
    words,
)
from wind_panel import wind_panel

st.set_page_config(page_title="Météo Analytics", page_icon="🌡️", layout="wide")
language.setup()

CITIES = [c.name for c in CFG.cities]
INDICATORS = {  # code (also the one links carry) -> (unit, available by season)
    "tmean": ("°C", True), "tx": ("°C", True), "tn": ("°C", True), "rr": ("mm", True),
    "hot_days": ("days", False), "very_hot_days": ("days", False), "tropical_nights": ("days", False),
    "frost_days": ("days", False), "heatwave_days": ("days", False), "heavy_rain_days": ("days", False),
    "longest_dry_spell": ("days", False), "tx_max": ("°C", False), "tn_min": ("°C", False),
}
COUNT_INDICATORS = [c for c, (unit, _) in INDICATORS.items() if unit == "days"]
SEASON_LABELS = ["Whole year", "Winter", "Spring", "Summer", "Autumn"]   # codes; shown with t("season.…")


def ind_label(code: str) -> str:
    return t(f"ind.{code}", **TH)


def unit_label(unit: str) -> str:
    return t("common.days_unit") if unit == "days" else unit


def season_label(code: str) -> str:
    return t(f"season.{code}")


def period(p: tuple[int, int], short: bool = False) -> str:
    return f"{p[0]}-{str(p[1])[2:]}" if short else f"{p[0]}-{p[1]}"


# =============================================================================
def page_overview():
    r = results()
    w, ex, pl = r["warming"], r["extremes"], r.get("pipeline", {})
    tr = w["france_trend_1951"]
    last = last_measurement()
    page_header(
        t("overview.title", start=START, end=END), tab=t("app.nav_overview"), lead=
        t("overview.lead", rows=num(pl.get("n_rows", 0), 0), stations=num(pl.get("n_stations", 0), 0),
          first=REF[0], last=REF[1])
        + (t("overview.lead_latest", date=fmt_date(last)) if last is not None else ""),
    )

    hot = ex["france_old_vs_recent"]["hot_days"]
    k = st.columns(4)
    k[0].metric(t("overview.kpi_warming"), f"{signed(tr['slope'], 2)} °C", border=True,
                help=t("overview.kpi_warming_help", start=START, end=END, low=num(tr["ci_low"], 2),
                       high=num(tr["ci_high"], 2), p=pvalue(tr["mk_p_value"])))
    k[1].metric(t("overview.kpi_delta", recent=period(RECENT, True), old=period(OLD, True)),
                f"{signed(w['delta_recent_vs_old'], 2)} °C", border=True, help=t("overview.kpi_delta_help"))
    k[2].metric(t("overview.kpi_hot", hot_day=TH["hot_day"]), f"× {num(hot['ratio'], 1)}", border=True,
                help=t("overview.kpi_hot_help", old_n=num(hot["old"], 1), old=period(OLD),
                       recent_n=num(hot["recent"], 1), recent=period(RECENT)))
    k[3].metric(t("overview.kpi_records"), f"{num(r['records']['last_ratio'], 0)} : 1",
                border=True, help=t("overview.kpi_records_help"))

    fra = pd.read_csv(CFG.reports_dir / "tables" / "france_anomaly.csv")
    years, anom = fra["year"], fra["anomaly_c"]
    fig = go.Figure(go.Bar(
        x=years, y=[1] * len(years), marker=dict(color=anom, colorscale=DIVERGING, cmid=0, line_width=0),
        customdata=anom, hovertemplate=t("overview.stripes_hover")))
    fig.update_layout(bargap=0, yaxis=dict(visible=False), xaxis=dict(showline=False, showgrid=False, dtick=10))
    anomaly_csv = pd.DataFrame({t("chart.csv_year"): years, t("overview.csv_anomaly"): anom.round(3)})
    chart(fig, 170, t("overview.stripes_title"), t("overview.stripes_sub", first=REF[0], last=REF[1]),
          margin=dict(l=0, r=0, b=28), data=anomaly_csv)

    left, right = st.columns([3, 2], gap="large")
    with left:
        fig = go.Figure()
        fig.add_bar(x=years, y=anom, marker_color=signed_color(anom), name=t("overview.anomaly_name"),
                    showlegend=False, hovertemplate="%{y:+.2f} °C<extra>%{x}</extra>")
        fig.add_scatter(x=years, y=rolling(anom), mode="lines", line=dict(color=INK, width=2.5),
                        name=t("overview.avg_name"), hovertemplate=t("overview.avg_hover"))
        fig.update_layout(yaxis_title="°C", hovermode="x unified")
        chart(fig, 420, t("overview.trend_title", slope=signed(tr["slope"], 2), start=START),
              t("overview.trend_sub") + (t("overview.top10_all") if w["n_top10_since_2011"] == 10
                                         else t("overview.top10_some", n=w["n_top10_since_2011"])))
    with right:
        cities = pd.DataFrame(r["heterogeneity"]["cities"])
        ref = table("ref_station").query("priority == 1")[["city", "lat", "lon", "station_name"]]
        m = cities.merge(ref, on="city")
        fig = go.Figure(go.Scattermap(
            lat=m["lat"], lon=m["lon"], mode="markers",
            marker=dict(size=17, color=m["slope"], colorscale=HEAT, cmin=0, opacity=0.95,
                        colorbar=dict(title=dict(text=t("overview.map_colorbar"), side="top"), thickness=10, len=0.7,
                                      orientation="h", y=-0.02, yanchor="top")),
            customdata=np.stack([m["city"], m["slope"], m["delta"], m["station_name"]], axis=1),
            hovertemplate=t("overview.map_hover", old=period(OLD), recent=period(RECENT))))
        # Centred on the cities' bounds and zoomed to fit them all, Ajaccio included (narrower on phones)
        lat0, lat1, lon0, lon1 = m["lat"].min(), m["lat"].max(), m["lon"].min(), m["lon"].max()
        width = 300 if is_phone() else 420
        zoom = min(np.log2(width * 360 / (512 * (lon1 - lon0 + 1))),
                   np.log2(330 * 360 / (512 * (lat1 - lat0 + 1) / np.cos(np.radians((lat0 + lat1) / 2)))))
        fig.update_layout(map=dict(style="carto-positron", center=dict(lat=(lat0 + lat1) / 2, lon=(lon0 + lon1) / 2),
                                   zoom=float(zoom)))
        map_csv = pd.DataFrame({t("common.city"): m["city"], t("common.station"): m["station_name"],
                                t("overview.csv_lat"): m["lat"], t("overview.csv_lon"): m["lon"],
                                t("overview.csv_slope"): m["slope"].round(3),
                                t("overview.csv_delta", old=period(OLD), recent=period(RECENT)): m["delta"].round(3)})
        chart(fig, 440, t("overview.map_title"), t("overview.map_sub"), margin=dict(l=0, r=0, b=60), data=map_csv)

    st.subheader(t("overview.lately"))
    lately_panel(CITIES, t("overview.lately_mean"))

    st.subheader(t("overview.takeaways"))
    pr, h = r["precipitation"], r["heterogeneity"]
    old_new = ex["france_old_vs_recent"]
    findings = [
        t("overview.f_warming", slope=signed(tr["slope"], 2), start=START,
          slope81=signed(w["france_trend_1981"]["slope"], 2)),
        t("overview.f_season", season=season_label(h["fastest_season"]),
          slope=signed(h["season_trends"][h["fastest_season"]]["slope"], 2), fastest=h["fastest_city"],
          slowest=h["slowest_city"]),
        t("overview.f_extremes", very_hot_day=TH["very_hot_day"], very_hot=num(old_new["very_hot_days"]["ratio"], 1),
          nights=num(old_new["tropical_nights"]["ratio"], 1),
          frost=percent((old_new["frost_days"]["ratio"] - 1) * 100, 0, sign=True)),
        t("overview.f_rain", slope=percent(pr["annual_trend_pct"]["slope"], 1, sign=True),
          p=pvalue(pr["annual_trend_pct"]["mk_p_value"])),
        t("overview.f_record", value=num(ex["hottest_day"]["tx"], 1), city=ex["hottest_day"]["city"],
          date=fmt_date(ex["hottest_day"]["date"])),
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
    page_header(t("city.title"), t("city.lead"))
    restore("city_sel", "cities", ["Paris", "Marseille"], multi=True,
            valid=lambda v: 0 < len(v) <= 3 and set(v) <= set(CITIES))
    restore("city_ind", "indicator", "tmean", valid=lambda v: v in INDICATORS)
    restore("city_season", "season", "Whole year", parse=str.capitalize, valid=lambda v: v in SEASON_LABELS)
    restore("city_mode", "display", "Values", parse=str.capitalize, valid=lambda v: v in ("Values", "Anomaly"))
    if st.session_state.get("city_ind") not in INDICATORS:   # a label kept from before the language switch
        st.session_state["city_ind"] = "tmean"
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
    sel = f[0].multiselect(t("common.cities"), CITIES, max_selections=3, key="city_sel", help=t("city.cities_help"))
    col = f[1].selectbox(t("common.indicator"), list(INDICATORS), key="city_ind", format_func=ind_label)
    unit, seasonal_ok = INDICATORS[col]
    if not seasonal_ok:
        st.session_state["city_season"] = "Whole year"
    season = f[2].selectbox(t("common.season"), SEASON_LABELS if seasonal_ok else ["Whole year"],
                            disabled=not seasonal_ok, key="city_season", format_func=season_label,
                            help=t("season.winter_help"))
    mode = f[3].segmented_control(t("city.display"), ["Values", "Anomaly"], required=True, key="city_mode",
                                  format_func=lambda v: t(f"city.{v}")) or "Values"
    period_ = st.slider(t("common.period"), START, END, key="city_period")
    publish({"cities": sel, "indicator": col, "season": None if season == "Whole year" else season.lower(),
             "display": None if mode == "Values" else mode.lower(),
             "from": None if period_[0] == START else period_[0], "to": None if period_[1] == END else period_[1]})
    if period_[1] - period_[0] < 1:
        st.info(t("city.pick_period"))
        return
    if not sel:
        st.info(t("city.pick_city"))
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
    df = df[df["year"].between(*period_)]
    y_unit = "%" if (anomaly and relative) else unit_label(unit)

    fig = go.Figure()
    for city in sel:
        d = df[df["city"] == city].sort_values("year")
        c = colors[city]
        fig.add_scatter(x=d["year"], y=d[col], mode="lines", line=dict(color=c, width=1.2), opacity=0.45,
                        legendgroup=city, showlegend=False, name=city, meta={"csv": t("city.csv_annual", city=city)},
                        hovertemplate=t("city.hover_value", city=city, unit=y_unit))
        fig.add_scatter(x=d["year"], y=d["avg"], mode="lines", line=dict(color=c, width=2.8), legendgroup=city,
                        name=city, meta={"csv": t("city.csv_avg", city=city)},
                        hovertemplate=t("city.hover_avg", city=city, unit=y_unit))
    if anomaly:
        fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.update_layout(hovermode="x unified", yaxis_title=y_unit, xaxis=dict(tickformat="d"))
    label = ind_label(col) if season == "Whole year" else t("city.title_season", indicator=ind_label(col),
                                                             season=season_label(season).lower())
    chart(fig, 460, t("city.title_anomaly", label=label, first=REF[0], last=REF[1]) if anomaly else label,
          t("city.sub", first=period_[0], last=period_[1]))

    cols = st.columns(len(sel))
    for c_, city in zip(cols, sel, strict=True):
        d = df[df["city"] == city]
        try:
            tr_ = trend(d["year"], d[col])
            verdict = t("common.significant") if tr_.significant else t("common.not_significant")
            c_.metric(t("city.trend_in", city=city), t("city.trend_value", slope=signed(tr_.slope, 2), unit=y_unit),
                      border=True, help=t("city.trend_help", low=num(tr_.ci_low, 2), high=num(tr_.ci_high, 2),
                                          sen=signed(tr_.sen_slope, 2)))
            # Both tests decide (significant = both p < 0.05), so both p-values are shown
            c_.markdown(f"<div class='kpi-note'>{t('city.trend_note', p=pvalue(tr_.p_value), mk=pvalue(tr_.mk_p_value), verdict=verdict)}</div>",
                        unsafe_allow_html=True)
        except ValueError:
            c_.metric(t("city.trend_in", city=city), t("city.too_short"), border=True)

    st.subheader(t("city.cycle"))
    # The monthly variable the indicator is about: rainfall for rain indicators, TX for hot days, TN for nights
    mcol = {"hot_days": "tx", "very_hot_days": "tx", "heatwave_days": "tx", "tx_max": "tx", "tropical_nights": "tn",
            "frost_days": "tn", "tn_min": "tn", "heavy_rain_days": "rr", "longest_dry_spell": "rr"}.get(col, col)
    mcol = mcol if mcol in ("tmean", "tx", "tn", "rr") else "tmean"
    m = table("city_monthly")
    m = m[m["city"].isin(sel) & m["valid_rain" if mcol == "rr" else "valid_temp"]]
    months = words("fmt.months_short")
    m_unit = "mm" if mcol == "rr" else "°C"
    fig = go.Figure()
    for city in sel:
        for per, width, opacity in ((OLD, 1.6, 0.5), (REF, 2.8, 1.0)):
            d = m[(m["city"] == city) & m["year"].between(*per)].groupby("month")[mcol].mean()
            fig.add_scatter(x=months, y=d.values, mode="lines+markers", name=f"{city} {period(per)}",
                            line=dict(color=colors[city], width=width), opacity=opacity, marker=dict(size=8),
                            hovertemplate=t("city.cycle_hover", city=city, period=period(per), unit=m_unit))
    fig.update_layout(hovermode="x unified", yaxis_title=m_unit)
    name = t("city.var_rr") if mcol == "rr" else ind_label(mcol)
    chart(fig, 400, t("city.cycle_title", name=name), t("city.cycle_sub", old=period(OLD), ref=period(REF)))

    seasons_panel(sel)

    st.subheader(t("overview.lately"))
    who = sel[0] if len(sel) == 1 else st.selectbox(t("common.city"), sel, key="lately_city",
                                                    help=t("city.lately_city_help"))
    lately_panel([who], who)

    with st.expander(t("city.stations")):
        rs = table("ref_station")
        rs = rs[rs["city"].isin(sel)].copy()
        rs["role"] = np.where(rs["priority"] == 1, t("city.role_main"), t("city.role_backup"))
        rs["completeness_temp"] = rs["completeness_temp"] * 100
        view = rs[["city", "role", "station_id", "station_name", "altitude", "distance_km", "first_date", "last_date",
                   "completeness_temp"]]
        st.dataframe(styled(view, {"altitude": number(0, "m"), "distance_km": number(1, "km"),
                                   "first_date": date_text(), "last_date": date_text()}),
                     hide_index=True, column_config={
                         "city": t("common.city"), "role": t("city.col_role"), "station_id": t("city.col_id"),
                         "station_name": t("common.station"), "altitude": t("city.col_altitude"),
                         "distance_km": t("city.col_distance"), "first_date": t("common.start"),
                         "last_date": t("common.end"),
                         "completeness_temp": st.column_config.ProgressColumn(
                             t("city.col_completeness", start=START, end=END), min_value=0, max_value=100,
                             format=t("fmt.progress_pct"))})
    source_note()


# =============================================================================
YEAR_VARIABLES = ["Maximum", "Minimum", "Mean", "Rainfall"]   # codes; shown with t("year.…")


def page_year():
    page_header(t("year.title"), t("year.lead"))
    d_all = table("city_daily")
    last_year = int(d_all.dropna(subset=["tx"])["date"].dt.year.max())
    years = list(range(last_year, START - 1, -1))
    restore("year_city", "city", "Paris", valid=lambda v: v in CITIES)
    restore("year_year", "year", 2022 if 2022 in years else years[0], parse=int, valid=lambda v: v in years)
    restore("year_variable", "variable", "Maximum", parse=str.capitalize, valid=lambda v: v in YEAR_VARIABLES)
    f = st.columns([2, 1.4, 3])
    city = f[0].selectbox(t("common.city"), CITIES, key="year_city")
    year = f[1].selectbox(t("common.year"), years, key="year_year",
                          format_func=lambda y: t("common.in_progress", year=y) if y > END else str(y))
    var = f[2].segmented_control(t("year.variable"), YEAR_VARIABLES, required=True, key="year_variable",
                                 format_func=lambda v: t(f"year.{v}")) or "Maximum"
    publish({"city": city, "year": year, "variable": var.lower()})
    d = d_all[d_all["city"] == city]
    if var == "Rainfall":
        _year_rain(city, year, d)
        source_note()
        return
    col = {"Maximum": "tx", "Minimum": "tn", "Mean": "tmean"}[var]
    variable = t(f"year.temp_{var}")          # "maximum temperature"
    months = words("fmt.months_short")

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
    k[0].metric(t("year.kpi_anomaly"), f"{signed(yr['anom'].mean(), 1)} °C", border=True, help=t("year.kpi_anomaly_help"))
    k[1].metric(t("year.kpi_above"), f"{(yr['anom'] > 0).sum()}", border=True, help=t("year.kpi_above_help", n=len(yr)))
    # Records against the years before this one: none in the first year of the city's data, few just after it
    first = int(d["date"].dt.year.min())
    if year <= first:
        records, records_help = "–", t("year.kpi_records_first", year=year)
    else:
        records = t("year.kpi_records_value", high=len(rec_hi), low=len(rec_lo))
        records_help = t("year.kpi_records_help", variable=variable, start=first, last=year - 1) + (
            plural("year.kpi_records_few", year - first) if year - first < 10 else "")
    k[2].metric(t("year.kpi_records"), records, border=True, help=records_help)
    # The day that stands out for the variable: hottest day, coldest night, warmest day on average
    kpi, i_k = {"Maximum": ("kpi_hottest", yr[col].idxmax()), "Minimum": ("kpi_coldest", yr[col].idxmin()),
                "Mean": ("kpi_warmest", yr[col].idxmax())}[var]
    k[3].metric(t(f"year.{kpi}"), f"{num(yr.loc[i_k, col], 1)} °C", border=True,
                help=t("year.kpi_day_help", variable=variable.capitalize(), date=fmt_date(yr.loc[i_k, "date"])))

    fig = go.Figure()
    if not env.empty:
        fig.add_scatter(x=env["date"], y=env["high"], mode="lines", line=dict(color=WARM, width=1, dash="dot"),
                        name=t("year.record_high_name", year=year), customdata=env["high_year"],
                        hovertemplate=t("year.record_hover"))
        fig.add_scatter(x=env["date"], y=env["low"], mode="lines", line=dict(color=COOL, width=1, dash="dot"),
                        name=t("year.record_low_name", year=year), customdata=env["low_year"],
                        hovertemplate=t("year.record_hover"))
    fig.add_scatter(x=ref_dates, y=clim["p90"], mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False)
    fig.add_scatter(x=ref_dates, y=clim["p10"], mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(195,194,183,0.35)", name=t("common.band_80_days", first=REF[0], last=REF[1]),
                    hoverinfo="skip")
    fig.add_scatter(x=ref_dates, y=clim["mean"], mode="lines", line=dict(color=MUTED, width=1.6),
                    name=t("common.normal_of", first=REF[0], last=REF[1]), hoverinfo="skip")
    fig.add_scatter(x=yr["date"], y=yr[col], mode="lines", line=dict(color=INK, width=1.6), name=str(year),
                    customdata=np.stack([yr["mean"], yr["anom"]], axis=1), hovertemplate=t("year.year_hover"))
    hot, cold = yr[yr[col] > yr["p90"]], yr[yr[col] < yr["p10"]]
    fig.add_scatter(x=hot["date"], y=hot[col], mode="markers", name=t("common.above_p90"),
                    marker=dict(color=WARM, size=8, line=dict(color=SURFACE, width=1.5)), hoverinfo="skip")
    fig.add_scatter(x=cold["date"], y=cold[col], mode="markers", name=t("common.below_p10"),
                    marker=dict(color=COOL, size=8, line=dict(color=SURFACE, width=1.5)), hoverinfo="skip")
    for recs, colour, which, prev, prev_year in ((rec_hi, WARM, "high", "high", "high_year"),
                                                (rec_lo, COOL, "low", "low", "low_year")):
        if recs.empty:
            continue
        tie = np.where(recs[col] == recs[prev], t("year.rec_equals"), t("year.rec_beats"))
        fig.add_scatter(x=recs["date"], y=recs[col], mode="markers", name=t(f"year.rec_{which}_day"),
                        marker=dict(symbol="star", color=colour, size=13, line=dict(color=SURFACE, width=1.5)),
                        customdata=np.stack([recs[prev], recs[prev_year].astype(int), tie], axis=1),
                        hovertemplate=t("year.rec_hover", what=t(f"year.rec_what_{which}")))
    month_starts = [pd.Timestamp(year=year, month=m, day=1) for m in range(1, 13)]
    fig.update_layout(yaxis_title="°C", xaxis=dict(tickvals=month_starts, ticktext=months), hovermode="closest")
    chart(fig, 520, t("year.chart_title", variable=variable.capitalize(), city=city, year=year),
          t("year.chart_sub", first=REF[0], last=REF[1], start=START, year=year))

    yr["month"] = yr["date"].dt.month
    mon = yr.groupby("month")["anom"].mean()
    fig = go.Figure(go.Bar(x=[months[i - 1] for i in mon.index], y=mon.values, marker_color=signed_color(mon.values),
                           name=t("year.month_name"), hovertemplate="%{x}: %{y:+.1f} °C<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.update_layout(yaxis_title="°C", showlegend=False)
    chart(fig, 320, t("year.month_title"))
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
        st.info(t("year.no_rain"))
        return
    last = int(np.flatnonzero(measured).max())          # last calendar day measured (0-based)
    cum = np.cumsum(rain)[: last + 1]
    dates = _doy_dates(year, np.arange(1, last + 2))
    total, normal_to_date = float(cum[-1]), float(clim["mean"].iloc[last])
    wet = rain[: last + 1] >= TH["rain_day"]
    rainy = int(wet.sum())
    dry_runs = np.diff(np.flatnonzero(np.concatenate([[True], wet, [True]]))) - 1
    so_far = last < 364
    gap = round((total / normal_to_date - 1) * 100) if normal_to_date else 0
    k = st.columns(4)
    # The arrow comes from the number: Streamlit reads it from the text, where the typographic minus "−" looks
    # positive (an up arrow on a deficit, #74)
    k[0].metric(t("year.rain_kpi_so_far" if so_far else "year.rain_kpi"), f"{num(total, 0)} mm", border=True,
                delta=t("year.rain_delta", pct=percent(gap, 0, sign=True)) if normal_to_date else None,
                delta_color="off", delta_arrow="up" if gap > 0 else "down" if gap < 0 else "off",
                help=t("year.rain_help", normal=num(normal_to_date, 0), first=REF[0], last=REF[1],
                       measured=int(measured[: last + 1].sum()), days=last + 1))
    k[1].metric(t("year.rainy_days", rain_day=num(TH["rain_day"], 0)), f"{rainy}", border=True)
    i_wet = int(np.argmax(rain[: last + 1]))
    k[2].metric(t("year.wettest"), f"{num(rain[i_wet], 1)} mm", border=True,
                help=t("year.wettest_help", date=fmt_date(dates.iloc[i_wet])))
    k[3].metric(t("year.dry"), plural("common.n_days", int(dry_runs.max()) if len(dry_runs) else 0), border=True,
                help=t("year.dry_help", rain_day=num(TH["rain_day"], 0)))

    ref_dates = _doy_dates(year, clim["doy"])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.68, 0.32])
    fig.add_scatter(x=ref_dates, y=clim["p90"], mode="lines", line=dict(width=0), hoverinfo="skip",
                    showlegend=False, row=1, col=1)
    fig.add_scatter(x=ref_dates, y=clim["p10"], mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(195,194,183,0.35)", name=t("common.band_80_years", first=REF[0], last=REF[1]),
                    hoverinfo="skip", row=1, col=1)
    fig.add_scatter(x=ref_dates, y=clim["mean"], mode="lines", line=dict(color=MUTED, width=1.6),
                    name=t("common.normal"), hovertemplate=t("year.rain_normal_hover"), row=1, col=1)
    fig.add_scatter(x=dates, y=cum, mode="lines", line=dict(color=SERIES[0], width=2.4), name=str(year),
                    customdata=clim["mean"].iloc[: last + 1].to_numpy(), hovertemplate=t("year.rain_cum_hover"),
                    row=1, col=1)
    fig.add_bar(x=dates, y=rain[: last + 1], marker_color=SERIES[0], name=t("year.daily_rain"), showlegend=False,
                hovertemplate=t("year.daily_rain_hover"), row=2, col=1)
    month_starts = [pd.Timestamp(year=year, month=m, day=1) for m in range(1, 13)]
    fig.update_xaxes(tickvals=month_starts, ticktext=words("fmt.months_short"),
                     range=[month_starts[0] - pd.Timedelta(days=3), pd.Timestamp(year=year, month=12, day=31)])
    fig.update_yaxes(title_text=t("year.axis_cum"), rangemode="tozero", row=1, col=1)
    fig.update_yaxes(title_text=t("year.axis_day"), rangemode="tozero", row=2, col=1)
    fig.update_layout(hovermode="closest", bargap=0.1)
    chart(fig, 560, t("year.rain_title_so_far" if so_far else "year.rain_title", city=city, year=year),
          t("year.rain_sub", first=REF[0], last=REF[1]))


@st.cache_data(show_spinner=False)
def doy_clim(city: str, col: str) -> pd.DataFrame:
    d = table("city_daily")
    return doy_climatology(d[d["city"] == city], col, REF)


# =============================================================================
def page_extremes():
    page_header(t("extremes.title"), t("extremes.lead"))
    r = results()
    restore("ext_ind", "indicator", COUNT_INDICATORS[0], valid=lambda v: v in COUNT_INDICATORS)
    if st.session_state.get("ext_ind") not in COUNT_INDICATORS:
        st.session_state["ext_ind"] = COUNT_INDICATORS[0]
    col = st.selectbox(t("common.indicator"), COUNT_INDICATORS, key="ext_ind", format_func=ind_label)
    publish({"indicator": col})
    scale = RAIN if col in ("frost_days", "heavy_rain_days") else HEAT

    rain_ind = col in ("heavy_rain_days", "longest_dry_spell")
    # A count of days per year, except the longest dry spell: the length of the year's longest run without rain
    per_year = col != "longest_dry_spell"
    unit_short, unit_long = ((t("extremes.unit_short"), t("extremes.unit_long")) if per_year
                             else (t("common.days_unit"), t("common.days_unit")))
    y = table("city_yearly")
    y = y[y["valid_rain" if rain_ind else "valid_temp"] & y["year"].between(START, END)].copy()
    y["decade"] = y["year"] // 10 * 10
    pv = y.pivot_table(index="city", columns="decade", values=col, aggfunc="mean")
    pv.columns = [f"{max(c, START)}-{str(min(c + 9, END))[2:]}" for c in pv.columns]
    pv = pv.loc[pv.iloc[:, -1].sort_values().index]
    fig = go.Figure(go.Heatmap(
        z=pv.values, x=pv.columns, y=pv.index, colorscale=scale, xgap=2, ygap=2,
        colorbar=dict(title=dict(text=unit_short, side="top"), thickness=12), meta={"row": t("common.city")},
        hovertemplate=t("extremes.heat_hover", unit=unit_long)))
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
    chart(fig, 640, ind_label(col), margin=dict(l=120, t=40),
          subtitle=(t("extremes.heat_sub_year") if per_year else t("extremes.heat_sub_spell"))
          + t("extremes.heat_sub_tail"))

    old = y[y["year"].between(*OLD)].groupby("city")[col].mean()
    new = y[y["year"].between(*RECENT)].groupby("city")[col].mean()
    dd = pd.DataFrame({"old": old, "new": new}).sort_values("new")
    fig = go.Figure()
    xs, ys = [], []
    for c, row in dd.iterrows():
        xs += [row["old"], row["new"], None]
        ys += [c, c, None]
    fig.add_scatter(x=xs, y=ys, mode="lines", line=dict(color=GRID, width=3), hoverinfo="skip", showlegend=False)
    fig.add_scatter(x=dd["old"], y=dd.index, mode="markers", name=period(OLD),
                    marker=dict(color=SERIES[0], size=11, line=dict(color=SURFACE, width=2)),
                    hovertemplate=t("extremes.then_hover", period=period(OLD)))
    fig.add_scatter(x=dd["new"], y=dd.index, mode="markers", name=period(RECENT),
                    marker=dict(color=SERIES[1], size=11, line=dict(color=SURFACE, width=2)),
                    hovertemplate=t("extremes.then_hover", period=period(RECENT)))
    fig.update_layout(xaxis_title=unit_long, yaxis=dict(showgrid=False))
    then_csv = pd.DataFrame({t("common.city"): dd.index, f"{period(OLD)} ({unit_long})": dd["old"].round(2).to_numpy(),
                             f"{period(RECENT)} ({unit_long})": dd["new"].round(2).to_numpy()})
    chart(fig, 600, t("extremes.then_title"), t("extremes.then_sub", old=period(OLD), recent=period(RECENT)),
          data=then_csv)

    left, right = st.columns([1, 1], gap="large")
    with left:
        rec = pd.DataFrame(r["records"]["by_decade"])
        fig = go.Figure(go.Bar(
            x=rec["decade"].astype(str), y=rec["ratio"], marker_color=[WARM if v >= 1 else COOL for v in rec["ratio"]],
            customdata=np.stack([rec["hot"], rec["cold"]], axis=1),
            text=[f"{num(v, 1)} : 1" for v in rec["ratio"]], textposition="outside",
            hovertemplate=t("extremes.rec_hover")))
        fig.add_hline(y=1, line=dict(color=MUTED, width=1))
        top = max(32, rec["ratio"].max() * 2)
        ticks = [v for v in (0.5, 1, 2, 4, 8, 16, 32, 64) if v <= top]
        fig.update_layout(yaxis=dict(type="log", title=t("extremes.rec_axis"), range=[np.log10(0.7), np.log10(top)],
                                     tickvals=ticks, ticktext=[f"{num(v, 0)} : 1" if v >= 1 else "1 : 2" for v in ticks]),
                          xaxis=dict(type="category"), showlegend=False)
        rec_csv = pd.DataFrame({t("extremes.csv_decade"): rec["decade"], t("extremes.csv_hot"): rec["hot"],
                                t("extremes.csv_cold"): rec["cold"], t("extremes.rec_axis"): rec["ratio"].round(3)})
        chart(fig, 420, t("extremes.rec_title"), t("extremes.rec_sub"), data=rec_csv)
    with right:
        st.markdown(t("extremes.episodes_title", days=TH["heatwave_min_days"], hot_day=TH["hot_day"]))
        hw = table("city_heatwaves")
        hw = hw[hw["start_date"].dt.year <= END]
        c1, c2 = st.columns([2, 1])
        cities = c1.multiselect(t("common.cities"), CITIES, default=[], placeholder=t("common.all_cities"),
                                key="hw_cities")
        min_days = c2.number_input(t("extremes.min_length"), 3, 60, 7)
        hw = hw[hw["n_days"] >= min_days]
        if cities:
            hw = hw[hw["city"].isin(cities)]
        view = hw.sort_values(["n_days", "tx_peak"], ascending=False)[
            ["city", "start_date", "end_date", "n_days", "tx_peak", "tx_mean"]]
        st.dataframe(styled(view, {"start_date": date_text(), "end_date": date_text(),
                                   "n_days": number(0, t("common.days_unit")), "tx_peak": number(1, "°C"),
                                   "tx_mean": number(1, "°C")}),
                     hide_index=True, height=330, column_config={
                         "city": t("common.city"), "start_date": t("common.start"), "end_date": t("common.end"),
                         "n_days": t("common.length"), "tx_peak": t("extremes.col_peak"),
                         "tx_mean": t("extremes.col_mean_tx")})
        st.caption(plural("extremes.episodes_caption", len(hw)))
    wind_panel()
    source_note()


# =============================================================================
def page_quality():
    page_header(t("quality.title"), t("quality.lead"), tab=t("app.nav_quality"))
    r = results()
    pl = r.get("pipeline", {})
    build_s = sum(v for k, v in pl.items() if k.startswith("duration_"))
    p = theme()
    st.graphviz_chart(f"""
    digraph {{
      rankdir={"TB" if is_phone() else "LR"}; bgcolor="transparent"; nodesep=0.35; ranksep=0.45;
      node [shape=box style="rounded,filled" fillcolor="{p['surface']}" color="{p['grid']}" fontname="Segoe UI" fontsize=11 fontcolor="{p['ink']}" margin="0.18,0.1"];
      edge [color="{p['muted']}" arrowsize=0.7];
      api  [label="{t('quality.g_api')}"];
      raw  [label="{t('quality.g_raw', files=pl.get('n_files', 0), mb=num(pl.get('raw_bytes', 0) / 1e6, 0))}"];
      stg  [label="{t('quality.g_stg', rows=num(pl.get('n_rows', 0) / 1e6, 1))}"];
      dq   [label="{t('quality.g_dq')}"];
      st   [label="{t('quality.g_st')}"];
      gold [label="{t('quality.g_gold')}"];
      out  [label="{t('quality.g_out')}" fillcolor="{p['series']}" fontcolor="#ffffff" color="{p['series']}"];
      api -> raw -> stg -> dq -> st -> gold -> out;
    }}""", width="stretch")
    st.caption(t("quality.pipeline_caption", seconds=num(build_s, 1)))

    k = st.columns(4)
    k[0].metric(t("quality.kpi_rows"), num(pl.get("n_rows", 0), 0), border=True)
    k[1].metric(t("quality.kpi_stations"), num(pl.get("n_stations", 0), 0), border=True)
    k[2].metric(t("quality.kpi_rejected"), num(pl.get("n_measures", 0) - pl.get("n_valid_measures", 0), 0), border=True,
                help=t("quality.kpi_rejected_help", millions=num(pl.get("n_measures", 0) / 1e6, 1)))
    k[3].metric(t("quality.kpi_dupes"), num(pl.get("n_duplicates", 0), 0), border=True)

    st.subheader(t("quality.by_variable"))
    dq = pd.DataFrame(r["quality"]["by_variable"])
    dq["variable"] = [t(f"quality.var_{v}") for v in dq["variable"]]
    for c in ("suspect", "out_of_range", "inconsistent"):
        if c not in dq:
            dq[c] = 0
    view = dq[["variable", "valid", "missing", "suspect", "out_of_range", "inconsistent", "pct_missing"]]
    st.dataframe(styled(view, {c: number(0) for c in ("valid", "missing", "suspect", "out_of_range", "inconsistent")}),
                 hide_index=True, column_config={
                     "variable": t("quality.col_variable"), "valid": t("quality.col_valid"),
                     "missing": t("quality.col_missing"), "suspect": t("quality.col_suspect"),
                     "out_of_range": t("quality.col_out"), "inconsistent": "TN > TX",
                     "pct_missing": st.column_config.ProgressColumn(t("quality.col_pct_missing"), min_value=0,
                                                                    max_value=100, format=t("fmt.progress_pct"))})
    st.markdown(t("quality.completeness_text"))

    nw = table("network_yearly")
    nw = nw[nw["year"] <= END]
    fig = go.Figure()
    fig.add_scatter(x=nw["year"], y=nw["n_stations_rain"], name=t("quality.net_rain"),
                    line=dict(color=SERIES[0], width=2.2), hovertemplate=t("quality.net_hover_rain"))
    fig.add_scatter(x=nw["year"], y=nw["n_stations_temp"], name=t("quality.net_temp"),
                    line=dict(color=SERIES[1], width=2.2), hovertemplate=t("quality.net_hover_temp"))
    fig.update_layout(hovermode="x unified", yaxis=dict(rangemode="tozero", title=t("quality.net_axis")))
    chart(fig, 380, t("quality.net_title"), t("quality.net_sub"))

    st.subheader(t("quality.ref_title"))
    off = pd.DataFrame(r["quality"]["offsets"])
    rs = table("ref_station")
    rs = rs.merge(off[["station_id", "offset_tn", "offset_tx", "n_overlap_days"]], on="station_id", how="left")
    rs["completeness_temp"] = rs["completeness_temp"] * 100
    view = rs[["city", "priority", "station_name", "altitude", "distance_km", "first_date", "last_date",
               "completeness_temp", "offset_tn", "offset_tx", "n_overlap_days"]]
    st.dataframe(styled(view, {"altitude": number(0, "m"), "distance_km": number(1, "km"),
                               "first_date": lambda v: f"{pd.Timestamp(v).year}" if pd.notna(v) else "–",
                               "last_date": lambda v: f"{pd.Timestamp(v).year}" if pd.notna(v) else "–",
                               "offset_tn": number(2, "°C", sign=True), "offset_tx": number(2, "°C", sign=True),
                               "n_overlap_days": number(0)}),
                 hide_index=True, height=420, column_config={
                     "city": t("common.city"),
                     "priority": st.column_config.Column(t("quality.col_rank"), help=t("quality.col_rank_help")),
                     "station_name": t("common.station"), "altitude": t("quality.col_alt"),
                     "distance_km": t("city.col_distance"), "first_date": t("common.start"),
                     "last_date": t("common.end"),
                     "completeness_temp": st.column_config.ProgressColumn(
                         t("city.col_completeness", start=START, end=END), min_value=0, max_value=100,
                         format=t("fmt.progress_pct")),
                     "offset_tn": st.column_config.Column(t("quality.col_tn_corr"), help=t("quality.col_tn_corr_help")),
                     "offset_tx": t("quality.col_tx_corr"),
                     "n_overlap_days": st.column_config.Column(t("quality.col_shared"),
                                                               help=t("quality.col_shared_help"))})

    e = era5()
    if e is not None:
        st.subheader(t("quality.era5_title"))
        e = e.sort_values("bias_tn")
        fig = go.Figure()
        fig.add_bar(y=e["city"], x=e["bias_tx"], orientation="h", name=t("ind.tx"), marker_color=SERIES[1],
                    hovertemplate="%{y}: %{x:+.2f} °C<extra>TX</extra>")
        fig.add_bar(y=e["city"], x=e["bias_tn"], orientation="h", name=t("ind.tn"), marker_color=SERIES[0],
                    hovertemplate="%{y}: %{x:+.2f} °C<extra>TN</extra>")
        fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08, xaxis_title=t("quality.era5_axis"),
                          yaxis=dict(showgrid=False))
        chart(fig, 620, t("quality.era5_chart_title"),
              t("quality.era5_sub", start=CFG.era5["start"][:4], end=CFG.era5["end"][:4]), source=t("source.era5_csv"))
        x = r.get("era5", {})
        e["hot_ratio"] = e["hot_days_era5"] / e["hot_days_obs"].clip(lower=1)
        miss = e[e["hot_days_obs"] >= 50].sort_values("hot_ratio").iloc[0]
        st.markdown(t("quality.era5_text", corr=num(x.get("mean_corr_tx", 0), 3), worst=x.get("worst_city"),
                      bias=signed(x.get("worst_bias_tn", 0), 1), miss=miss["city"],
                      era5=num(int(miss["hot_days_era5"]), 0), hot_day=TH["hot_day"],
                      obs=num(int(miss["hot_days_obs"]), 0)))

    with st.expander(t("quality.limits_title")):
        st.markdown(t("quality.limits", next=END + 1))
    source_note(t("source.era5"))


# =============================================================================
# =============================================================================
route = st.Page(page_route, title=t("app.nav_route"), icon=":material/route:", url_path="route")
if warehouse_ready():
    nav = st.navigation({
        t("app.section_climate"): [
            st.Page(page_overview, title=t("app.nav_overview"), icon=":material/thermostat:", default=True),
            st.Page(page_city, title=t("app.nav_city"), icon=":material/location_city:", url_path="city"),
            st.Page(page_year, title=t("app.nav_year"), icon=":material/calendar_month:", url_path="year"),
            st.Page(page_date, title=t("app.nav_date"), icon=":material/event:", url_path="date"),
            st.Page(page_extremes, title=t("app.nav_extremes"), icon=":material/local_fire_department:",
                    url_path="extremes"),
            st.Page(page_future, title=t("app.nav_ahead"), icon=":material/trending_up:", url_path="ahead"),
            st.Page(page_quality, title=t("app.nav_quality"), icon=":material/fact_check:", url_path="quality"),
            st.Page(page_pivot, title=t("app.nav_pivot"), icon=":material/pivot_table_chart:", url_path="pivot"),
        ],
        t("app.section_tools"): [st.Page(page_projection, title=t("app.nav_prediction"),
                                         icon=":material/online_prediction:", url_path="prediction"), route],
    })
else:
    # Without the warehouse, only the route tool (which uses the weather API) is available
    nav = st.navigation([route])
    st.sidebar.info(t("app.no_warehouse"))
with st.sidebar:
    language.picker()
    st.markdown("### Météo Analytics")
    st.caption(t("app.tagline", start=START, end=END))
pause_when_idle()
chart_gestures(is_phone())
nav.run()
