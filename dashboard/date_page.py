"""Dashboard page: the weather of one day in context, and the notable heatwaves, cold waves and dry spells."""
from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st
from plotly.subplots import make_subplots

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
from ui import (
    CFG,
    MUTED,
    REF,
    SERIES,
    START,
    TH,
    chart,
    date_text,
    fmt_date,
    num,
    number,
    page_header,
    source_note,
    styled,
    t,
    table,
)

CITIES = [c.name for c in CFG.cities]
KINDS = ["Heatwaves", "Cold waves", "Dry spells"]   # codes; shown with t("date.kind_…")


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
    page_header(t("date.title"), t("date.lead", start=START))
    if st.session_state.pop("date_scroll_top", False):
        st.html("<script>(document.querySelector('[data-testid=\"stMain\"]') || document.scrollingElement)"
                ".scrollTo({top: 0, behavior: 'smooth'}); window.scrollTo(0, 0);</script>", unsafe_allow_javascript=True)
    last = _last_day()
    restore("date_city", "city", "Lyon", valid=lambda v: v in CITIES)
    first = date(START, 1, 1)

    def nearest(text: str) -> date:
        """A link's date, moved to the nearest measured day when it is outside the data (and said, #82)."""
        asked = date.fromisoformat(text)
        shown = min(max(asked, first), last)
        if shown != asked:
            st.session_state["date_outside"] = asked
        return shown
    restore("date_day", "date", date(2003, 8, 12), parse=nearest)
    if st.session_state["date_day"] > last:
        st.session_state["date_day"] = last
    if (asked := st.session_state.pop("date_outside", None)) is not None:
        st.info(t("date.outside", asked=fmt_date(asked), first=fmt_date(first), last=fmt_date(last),
                  shown=fmt_date(st.session_state["date_day"])), icon=":material/info:")
    f = st.columns([2, 2, 3])
    city = f[0].selectbox(t("common.city"), CITIES, key="date_city")
    day = f[1].date_input(t("common.date"), min_value=first, max_value=last, key="date_day", format="DD/MM/YYYY",
                          help=t("date.range_help", first=fmt_date(first), last=fmt_date(last)))
    publish({"city": city, "date": day.isoformat()})
    day = pd.Timestamp(day)
    d = table("city_daily")
    d = d[d["city"] == city]
    ctx = day_in_context(d, day)
    label = fmt_date(day, "day_month_long")

    k = st.columns(5)
    for col_, (key, info) in zip(k, ctx.items(), strict=True):
        v = info["value"]
        value = "–" if v is None else (f"{num(v, 1)} {info['unit']}" if key in ("tx", "tn", "rr") else f"{num(v, 0)} {info['unit']}")
        help_ = (t("date.record_low_help" if key == "tn" else "date.record_help", day=label,
                   value=f"{num(info['record'], 1)} {info['unit']}", year=info["record_year"])
                 if "record" in info else None)
        col_.metric(info["label"], value, border=True, help=help_)
        col_.markdown(f"<div class='kpi-note'>{rank_text(info, day)}</div>", unsafe_allow_html=True)

    rec = [t(f"date.rec_{key}", value=f"{num(info['record'], 1)} {info['unit']}", year=info["record_year"])
           for key, info in ctx.items() if key in ("tx", "tn", "rr", "fxi") and "record" in info]
    st.markdown(f"<div class='finding'>{t('date.records', day=label, city=city, start=START, records=' · '.join(rec))}"
                "</div>", unsafe_allow_html=True)

    win = days_around(d, day, REF)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
    for col, colour in (("tx", SERIES[1]), ("tn", SERIES[0])):
        name, word = t(f"date.{col}"), t(f"date.{col}_word")
        fig.add_scatter(x=win["date"], y=win[f"{col}_normal"], mode="lines", line=dict(color=colour, width=1, dash="dot"),
                        name=t("date.normal_name", name=name), hovertemplate=t("date.normal_hover", word=word),
                        row=1, col=1)
        fig.add_scatter(x=win["date"], y=win[col], mode="lines+markers", line=dict(color=colour, width=2.2),
                        marker=dict(size=7), name=name, hovertemplate=t("date.value_hover", word=word), row=1, col=1)
    fig.add_bar(x=win["date"], y=win["rr"], marker_color=SERIES[0], name=t("date.rain"), showlegend=False,
                hovertemplate=t("date.rain_hover"), row=2, col=1)
    fig.add_vline(x=day, line=dict(color=MUTED, width=1, dash="dash"))
    fig.update_yaxes(title_text="°C", row=1, col=1)
    fig.update_yaxes(title_text="mm", rangemode="tozero", row=2, col=1)
    fig.update_layout(hovermode="x unified", bargap=0.2)
    chart(fig, 460, t("date.chart_title", city=city, date=fmt_date(day)),
          t("date.chart_sub", first=REF[0], last=REF[1]))

    st.subheader(t("date.episodes"))
    c1, c2 = st.columns([2, 3])
    kind = c1.segmented_control(t("date.episodes_kind"), KINDS, default="Heatwaves", required=True, key="episodes_kind",
                                format_func=lambda k: t(f"date.kind_{k.replace(' ', '_')}")) or "Heatwaves"
    cities = c2.multiselect(t("common.cities"), CITIES, default=[], placeholder=t("common.all_cities"),
                            key="episodes_cities")
    e = _episodes(kind)
    if cities:
        e = e[e["city"].isin(cities)]
    view = e.head(200).reset_index(drop=True)
    key = f"episodes_{kind}_{abs(hash(tuple(cities)))}"
    days = t("common.days_unit")
    if kind == "Dry spells":
        cols = ["city", "start_date", "end_date", "n_days", "usual", "intensity"]
        formats = {"start_date": date_text(), "end_date": date_text(), "n_days": number(0, days),
                   "usual": number(0, days), "intensity": number(1)}
        config = {"start_date": t("common.start"), "end_date": t("common.end"), "n_days": t("common.length"),
                  "usual": st.column_config.Column(t("date.col_usual"), help=t("date.col_usual_help", first=REF[0],
                                                                                 last=REF[1])),
                  "intensity": t("date.col_times_usual")}
    else:
        cols = ["city", "start", "end", "n_days", "peak", "peak_date", "intensity"]
        formats = {"start": date_text(), "end": date_text(), "n_days": number(0, days), "peak": number(1, "°C"),
                   "peak_date": date_text(), "intensity": number(0, t("date.degree_days"))}
        config = {"start": t("common.start"), "end": t("common.end"), "n_days": t("common.length"),
                  "peak": t("date.col_peak"), "peak_date": t("date.col_peak_day"),
                  "intensity": st.column_config.Column(t("date.col_intensity"), help=t("date.col_intensity_help"))}
    st.dataframe(styled(view[cols], formats), hide_index=True, height=380, key=key,
                 on_select=lambda: _open_day(key, view), selection_mode="single-row",
                 column_config={"city": t("common.city"), **config})
    st.caption(t("date.episodes_caption_dry" if kind == "Dry spells" else "date.episodes_caption", n=num(len(e), 0)))
    with st.expander(t("date.how_title")):
        hp, cp = HEAT_PERCENTILES, COLD_PERCENTILES
        st.markdown(t("date.how", first=REF[0], last=REF[1], h0=num(hp[0], 1), h1=num(hp[1], 1), h2=num(hp[2], 0),
                      c0=num(cp[0], 1), c1=num(cp[1], 1), c2=num(cp[2], 0), max_break=MAX_BREAK_DAYS,
                      min_dry=MIN_DRY_DAYS, rain=num(TH["rain_day"], 0)))
    source_note()
