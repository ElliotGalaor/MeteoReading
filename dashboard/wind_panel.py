"""Extremes page: storms (days with gusts of 100 km/h or more) and the strongest gusts recorded."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from meteo.wind import MIN_COVERAGE, MIN_YEARS, STORM_KMH, gust_records, storm_trends, storm_years
from ui import END, HEAT, START, chart, is_dark, num, pvalue, signed, table


@st.cache_data(show_spinner=False)
def _years():
    return storm_years(table("city_daily"), END)


@st.cache_data(show_spinner=False)
def _records():
    return gust_records(table("city_daily"))


def wind_panel() -> None:
    st.subheader("Storms and gusts")
    y = _years()
    valid = y[y["valid"] & y["year"].between(START, END)].copy()
    if valid.empty:
        st.info("No gust measurements.")
        return
    valid["decade"] = valid["year"] // 10 * 10
    # A decade is shown when at least 8 of its years have gusts measured on most days
    pv = valid.groupby(["city", "decade"]).agg(days=("storm_days", "mean"), n=("year", "count")).reset_index()
    pv = pv[pv["n"] >= 8].pivot(index="city", columns="decade", values="days")
    pv.columns = [f"{max(c, START)}-{str(min(c + 9, END))[2:]}" for c in pv.columns]
    pv = pv.loc[pv.mean(axis=1).sort_values().index]
    fig = go.Figure(go.Heatmap(z=pv.values, x=pv.columns, y=pv.index, colorscale=HEAT, xgap=2, ygap=2,
                               colorbar=dict(title=dict(text="days/yr", side="top"), thickness=12),
                               hovertemplate="<b>%{y}</b>, %{x}: %{z:.1f} storm days per year<extra></extra>"))
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
    chart(fig, 620, f"Storm days: strongest gust ≥ {num(STORM_KMH, 0)} km/h",
          "Average number of days per year, by decade · a blank cell: gusts missing on too many days"
          + (f" · no gust series: {', '.join(missing)}" if missing else ""), margin=dict(l=120, t=40))

    tr = storm_trends(y[y["year"].between(START, END)])
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.markdown("**Are storms more frequent?** Trend of storm days per city")
        tr = tr.sort_values("mean_days", ascending=False)
        tr["trend"] = [f"{signed(s, 2)} days/yr per decade" if np.isfinite(s) else "too few years" for s in tr["slope"]]
        tr["test"] = [f"{pvalue(p)} · {'significant' if sig else 'not significant'}" if np.isfinite(p) else ""
                      for p, sig in zip(tr["mk_p_value"], tr["significant"], strict=True)]
        st.dataframe(tr[["city", "mean_days", "trend", "test", "first_year", "n_years"]], hide_index=True, height=400,
                     column_config={"city": "City",
                                    "mean_days": st.column_config.NumberColumn("Storm days/yr", format="%.1f"),
                                    "trend": "Trend", "test": st.column_config.TextColumn(
                                        "Mann-Kendall", help="Significant = regression and Mann-Kendall both p < 0.05"),
                                    "first_year": st.column_config.NumberColumn("From", format="%d"),
                                    "n_years": st.column_config.NumberColumn("Years", format="%d")})
        n_up = int(((tr["slope"] > 0) & tr["significant"]).sum())
        n_down = int(((tr["slope"] < 0) & tr["significant"]).sum())
        st.caption(f"Significant rise in {n_up} cities, significant fall in {n_down}. Years count when gusts were "
                   f"measured on at least {num(MIN_COVERAGE * 100, 0)}% of their days; a trend needs {MIN_YEARS} "
                   "such years. Anemometers and their sites changed over the decades: a trend can come from a "
                   "change of instrument as much as from the climate.")
    with right:
        st.markdown("**Strongest gust ever recorded**, by city")
        rec = _records()
        st.dataframe(rec, hide_index=True, height=400, column_config={
            "city": "City", "date": st.column_config.DateColumn("Date", format="D MMM YYYY"),
            "gust_kmh": st.column_config.NumberColumn("Gust", format="%d km/h")})
