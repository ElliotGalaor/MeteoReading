"""Extremes page: storms (days with gusts of 100 km/h or more) and the strongest gusts recorded."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from meteo.wind import MIN_COVERAGE, MIN_YEARS, STORM_KMH, gust_records, storm_trends, storm_years
from ui import END, HEAT, START, chart, date_text, is_dark, listing, num, number, pct, pvalue, signed, styled, t, table


@st.cache_data(show_spinner=False)
def _years():
    return storm_years(table("city_daily"), END)


@st.cache_data(show_spinner=False)
def _records():
    return gust_records(table("city_daily"))


def wind_panel() -> None:
    st.subheader(t("wind.title"))
    y = _years()
    valid = y[y["valid"] & y["year"].between(START, END)].copy()
    if valid.empty:
        st.info(t("wind.none"))
        return
    valid["decade"] = valid["year"] // 10 * 10
    # A decade is shown when at least 8 of its years have gusts measured on most days
    pv = valid.groupby(["city", "decade"]).agg(days=("storm_days", "mean"), n=("year", "count")).reset_index()
    pv = pv[pv["n"] >= 8].pivot(index="city", columns="decade", values="days")
    pv.columns = [f"{max(c, START)}-{str(min(c + 9, END))[2:]}" for c in pv.columns]
    pv = pv.loc[pv.mean(axis=1).sort_values().index]
    fig = go.Figure(go.Heatmap(z=pv.values, x=pv.columns, y=pv.index, colorscale=HEAT, xgap=2, ygap=2,
                               colorbar=dict(title=dict(text=t("extremes.unit_short"), side="top"), thickness=12),
                               meta={"row": t("common.city")}, hovertemplate=t("wind.heat_hover")))
    vals = pv.to_numpy(dtype=float)
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    frac = (vals - lo) / (hi - lo) if hi > lo else np.zeros_like(vals)
    light = (frac < 0.55) if is_dark() else (frac >= 0.55)
    xs, ys = np.meshgrid(list(pv.columns), list(pv.index))
    shown = ~np.isnan(vals)
    fig.add_scatter(x=xs[shown], y=ys[shown], mode="text", text=[num(v, 1) for v in vals[shown]], hoverinfo="skip",
                    textfont=dict(size=11, color=np.where(light[shown], "#f8f7f4", "#141413").tolist()),
                    showlegend=False)
    fig.update_layout(xaxis=dict(side="top", showline=False, type="category", showgrid=False),
                      yaxis=dict(showline=False, type="category", showgrid=False))
    missing = sorted(set(table("ref_station")["city"]) - set(pv.index))
    chart(fig, 620, t("wind.heat_title", kmh=num(STORM_KMH, 0)),
          t("wind.heat_sub") + (t("wind.heat_missing", cities=listing(missing)) if missing else ""),
          margin=dict(l=120, t=40))

    tr = storm_trends(y[y["year"].between(START, END)])
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.markdown(t("wind.trend_title"))
        tr = tr.sort_values("mean_days", ascending=False)
        tr["trend"] = [t("wind.trend_value", slope=signed(s, 2)) if np.isfinite(s) else t("common.too_few_years")
                       for s in tr["slope"]]
        tr["test"] = [f"{pvalue(p)} · {t('common.significant') if sig else t('common.not_significant')}"
                      if np.isfinite(p) else "" for p, sig in zip(tr["mk_p_value"], tr["significant"], strict=True)]
        view = tr[["city", "mean_days", "trend", "test", "first_year", "n_years"]]
        st.dataframe(styled(view, {"mean_days": number(1)}), hide_index=True, height=400,
                     column_config={"city": t("common.city"), "mean_days": t("wind.col_days"),
                                    "trend": t("wind.col_trend"), "test": st.column_config.TextColumn(
                                        "Mann-Kendall", help=t("wind.col_test_help")),
                                    "first_year": st.column_config.NumberColumn(t("wind.col_from"), format="%d"),
                                    "n_years": st.column_config.NumberColumn(t("wind.col_years"), format="%d")})
        n_up = int(((tr["slope"] > 0) & tr["significant"]).sum())
        n_down = int(((tr["slope"] < 0) & tr["significant"]).sum())
        st.caption(t("wind.trend_caption", up=n_up, down=n_down, coverage=pct(MIN_COVERAGE), years=MIN_YEARS))
    with right:
        st.markdown(t("wind.records_title"))
        rec = _records()
        st.dataframe(styled(rec, {"date": date_text(), "gust_kmh": number(0, "km/h")}), hide_index=True, height=400,
                     column_config={"city": t("common.city"), "date": t("common.date"), "gust_kmh": t("wind.col_gust")})
