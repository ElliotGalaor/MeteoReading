"""Dashboard page: the weather of one day in context, and the notable heatwaves, cold waves and dry spells."""
from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st
from plotly.subplots import make_subplots

from meteo.climatology import MONTH_NAMES
from meteo.episodes import (
    COLD_PERCENTILES,
    HEAT_PERCENTILES,
    MAX_BREAK_DAYS,
    MIN_DRY_DAYS,
    day_in_context,
    days_around,
    dry_spells,
    episodes,
    rank_text,
)
from share import publish, restore
from ui import CFG, MUTED, REF, SERIES, START, TH, chart, num, page_header, source_note, table

CITIES = [c.name for c in CFG.cities]
KINDS = ["Heatwaves", "Cold waves", "Dry spells"]


@st.cache_data(show_spinner=False)
def _episodes(kind: str) -> pd.DataFrame:
    if kind == "Dry spells":
        return dry_spells(table("city_dry_spells"), table("city_yearly"), REF)
    return episodes(table("city_daily"), REF, heat=kind == "Heatwaves")


def _last_day() -> date:
    d = table("city_daily").dropna(subset=["tx"])
    return d["date"].max().date()


def _open_day(key: str, view: pd.DataFrame) -> None:
    """Opens the day lookup on the peak day of the episode selected in the table."""
    rows = st.session_state[key].selection.rows
    if rows:
        r = view.iloc[rows[0]]
        st.session_state["date_city"] = r["city"]
        st.session_state["date_day"] = pd.Timestamp(r["peak_date"]).date()
        st.session_state["date_scroll_top"] = True


def page_date():
    page_header("Explore a date", "The weather of any day since 1951 in one of the 20 cities, ranked against the "
                "same day of every year, and the notable heatwaves, cold waves and dry spells.")
    if st.session_state.pop("date_scroll_top", False):
        st.html("<script>(document.querySelector('[data-testid=\"stMain\"]') || document.scrollingElement)"
                ".scrollTo({top: 0, behavior: 'smooth'}); window.scrollTo(0, 0);</script>", unsafe_allow_javascript=True)
    last = _last_day()
    restore("date_city", "city", "Lyon", valid=lambda v: v in CITIES)
    restore("date_day", "date", date(2003, 8, 12), parse=date.fromisoformat,
            valid=lambda v: date(START, 1, 1) <= v <= last)
    if st.session_state["date_day"] > last:
        st.session_state["date_day"] = last
    f = st.columns([2, 2, 3])
    city = f[0].selectbox("City", CITIES, key="date_city")
    day = f[1].date_input("Date", min_value=date(START, 1, 1), max_value=last, key="date_day", format="DD/MM/YYYY")
    publish({"city": city, "date": day.isoformat()})
    day = pd.Timestamp(day)
    d = table("city_daily")
    d = d[d["city"] == city]
    ctx = day_in_context(d, day)
    label = f"{day.day} {MONTH_NAMES[day.month - 1]}"

    k = st.columns(5)
    for col_, (key, info) in zip(k, ctx.items(), strict=True):
        v = info["value"]
        value = "–" if v is None else (f"{num(v, 1)} {info['unit']}" if key in ("tx", "tn", "rr") else f"{num(v, 0)} {info['unit']}")
        record_word = "record low" if key == "tn" else "record"
        help_ = (f"{record_word.capitalize()} for {label}: {num(info['record'], 1)} {info['unit']} in {info['record_year']}."
                 if "record" in info else None)
        col_.metric(info["label"], value, border=True, help=help_)
        col_.markdown(f"<div class='kpi-note'>{rank_text(info, day)}</div>", unsafe_allow_html=True)

    names = {"tx": "highest maximum", "tn": "lowest minimum", "rr": "most rain", "fxi": "strongest gust"}
    rec = [f"{names[key]} {num(info['record'], 1)} {info['unit']} ({info['record_year']})"
           for key, info in ctx.items() if key in names and "record" in info]
    st.markdown(f"<div class='finding'><b>Records for {label} in {city}</b> since {START}: {' · '.join(rec)}</div>",
                unsafe_allow_html=True)

    win = days_around(d, day, REF)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
    for col, colour, name in (("tx", SERIES[1], "Maximum"), ("tn", SERIES[0], "Minimum")):
        fig.add_scatter(x=win["date"], y=win[f"{col}_normal"], mode="lines", line=dict(color=colour, width=1, dash="dot"),
                        name=f"{name}, normal", hovertemplate=f"%{{x|%d %b}}: normal {name.lower()} %{{y:.1f}} °C<extra></extra>",
                        row=1, col=1)
        fig.add_scatter(x=win["date"], y=win[col], mode="lines+markers", line=dict(color=colour, width=2.2),
                        marker=dict(size=7), name=name,
                        hovertemplate=f"%{{x|%a %d %b %Y}}: {name.lower()} %{{y:.1f}} °C<extra></extra>", row=1, col=1)
    fig.add_bar(x=win["date"], y=win["rr"], marker_color=SERIES[0], name="Rainfall", showlegend=False,
                hovertemplate="%{x|%d %b}: %{y:.1f} mm<extra></extra>", row=2, col=1)
    fig.add_vline(x=day, line=dict(color=MUTED, width=1, dash="dash"))
    fig.update_yaxes(title_text="°C", row=1, col=1)
    fig.update_yaxes(title_text="mm", rangemode="tozero", row=2, col=1)
    fig.update_layout(hovermode="x unified", bargap=0.2)
    chart(fig, 460, f"{city}, ten days before and after {day:%d %B %Y}",
          f"Daily maximum and minimum against their {REF[0]}-{REF[1]} normals (dotted), rainfall below")

    st.subheader("Notable episodes")
    c1, c2 = st.columns([2, 3])
    kind = c1.segmented_control("Episodes", KINDS, default="Heatwaves", required=True, key="episodes_kind") or "Heatwaves"
    cities = c2.multiselect("Cities", CITIES, default=[], placeholder="All cities", key="episodes_cities")
    e = _episodes(kind)
    if cities:
        e = e[e["city"].isin(cities)]
    view = e.head(200).reset_index(drop=True)
    key = f"episodes_{kind}_{abs(hash(tuple(cities)))}"
    if kind == "Dry spells":
        cols = ["city", "start_date", "end_date", "n_days", "usual", "intensity"]
        config = {"start_date": st.column_config.DateColumn("Start", format="D MMM YYYY"),
                  "end_date": st.column_config.DateColumn("End", format="D MMM YYYY"),
                  "n_days": st.column_config.NumberColumn("Length", format="%d days"),
                  "usual": st.column_config.NumberColumn("Usual longest", format="%d days",
                                                         help="Median of the longest spell of each year, "
                                                              f"{REF[0]}-{REF[1]}"),
                  "intensity": st.column_config.NumberColumn("× usual", format="%.1f")}
    else:
        cols = ["city", "start", "end", "n_days", "peak", "peak_date", "intensity"]
        config = {"start": st.column_config.DateColumn("Start", format="D MMM YYYY"),
                  "end": st.column_config.DateColumn("End", format="D MMM YYYY"),
                  "n_days": st.column_config.NumberColumn("Length", format="%d days"),
                  "peak": st.column_config.NumberColumn("Peak daily mean", format="%.1f °C"),
                  "peak_date": st.column_config.DateColumn("Peak day", format="D MMM YYYY"),
                  "intensity": st.column_config.NumberColumn("Intensity", format="%.0f °C·days",
                                                             help="Sum of the daily gaps beyond the start threshold")}
    st.dataframe(view[cols], hide_index=True, height=380, key=key, on_select=lambda: _open_day(key, view),
                 selection_mode="single-row", column_config={"city": "City", **config})
    st.caption(f"{num(len(e), 0)} episodes, most intense first (200 at most listed). Select a row to look up its "
               f"{'last' if kind == 'Dry spells' else 'peak'} day above.")
    with st.expander("How are episodes detected?"):
        hp, cp = HEAT_PERCENTILES, COLD_PERCENTILES
        st.markdown(f"""
- **Heatwaves and cold waves** follow Météo-France's method for its national indicators (Ouzeau et al., 2016), on
  each city's daily mean temperature: three thresholds from the {REF[0]}-{REF[1]} daily means, the
  {num(hp[0], 1)}th, {num(hp[1], 1)}th and {num(hp[2], 0)}th percentiles for heat (the {num(cp[0], 1)}th,
  {num(cp[1], 1)}th and {num(cp[2], 0)}th for cold). An episode is a run of days beyond the second threshold that
  reaches the first at least once; it carries on through up to {MAX_BREAK_DAYS} days that stay beyond the third.
- **Intensity**: the sum, over the days of the episode, of the gap between the daily mean and the second threshold.
- **Dry spells**: runs of at least {MIN_DRY_DAYS} days with less than {TH['rain_day']} mm of rain, ranked by their
  length divided by the city's usual longest spell of a year: Mediterranean summers are always dry.
""")
    source_note()
