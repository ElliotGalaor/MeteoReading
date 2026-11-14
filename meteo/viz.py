"""Static charts (matplotlib) for the report and the README.

Readability rules applied everywhere:
- one colour per series, in a fixed order; at most 3 colours on a chart;
- blue <-> red diverging scale with a neutral grey midpoint for anomalies;
- single-hue sequential scales for magnitudes (heat, rain);
- thin marks, discreet grid, never two Y axes on the same chart;
- selective direct labels rather than a value on every point.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

from .palette import (  # noqa: E402
    AXIS, BLUE, COOL, DIVERGING_STOPS, GRID, HEAT_STOPS, INK, INK2, MUTED, ORANGE, RAIN_STOPS,
    SERIES, SOURCE, SURFACE, WARM,
)

DIVERGING = LinearSegmentedColormap.from_list("anomalies", DIVERGING_STOPS)
HEAT = LinearSegmentedColormap.from_list("heat", HEAT_STOPS)
RAIN = LinearSegmentedColormap.from_list("rain", RAIN_STOPS)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 10,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 12.5, "axes.titleweight": "semibold",
    "axes.titlelocation": "left", "axes.titlepad": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.axisbelow": True, "axes.unicode_minus": True,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "legend.frameon": False, "legend.fontsize": 9.5, "lines.linewidth": 2,
    "figure.dpi": 100, "savefig.dpi": 150, "savefig.bbox": "tight",
})


def _finish(fig, path: Path, source: str = SOURCE) -> Path:
    fig.text(0.01, -0.01, source, color=MUTED, fontsize=7.5, ha="left", va="top")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _title(ax, title: str, subtitle: str | None = None) -> None:
    ax.set_title(title, loc="left")
    if subtitle:
        ax.text(0, 1.012, subtitle, transform=ax.transAxes, color=INK2, fontsize=9.5, va="bottom")
        ax.set_title(title, loc="left", pad=24)


def _sym_norm(values) -> TwoSlopeNorm:
    v = float(np.nanmax(np.abs(values))) or 1.0
    return TwoSlopeNorm(vcenter=0, vmin=-v, vmax=v)


def _num(v: float, digits: int = 1) -> str:
    return f"{v:.{digits}f}".replace("-", "−")


def _signed(v: float, unit: str = "", digits: int = 1) -> str:
    return f"{v:+.{digits}f}".replace("-", "−") + unit


# ---------------------------------------------------------------------------
def warming_stripes(series: pd.Series, path: Path, ref: tuple[int, int]) -> Path:
    """Warming stripes: one stripe per year."""
    fig, ax = plt.subplots(figsize=(11, 2.9))
    ax.bar(series.index, 1, width=1.0, color=DIVERGING(_sym_norm(series)(series.values)), linewidth=0)
    ax.set_xlim(series.index.min() - 0.5, series.index.max() + 0.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.grid(False)
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.tick_params(axis="x", length=0)
    _title(ax, "Each stripe is a year: France is turning red",
           f"Annual temperature anomaly vs the {ref[0]}-{ref[1]} normal, average of 20 cities")
    return _finish(fig, path)


def anomaly_bars(series: pd.Series, path: Path, tr, ref: tuple[int, int]) -> Path:
    """Annual anomaly (bars) + 10-year moving average."""
    fig, ax = plt.subplots(figsize=(11, 4.6))
    colors = np.where(series.values >= 0, WARM, COOL)
    ax.bar(series.index, series.values, width=0.72, color=colors, linewidth=0)
    rolling = series.rolling(10, center=True, min_periods=6).mean()
    ax.plot(rolling.index, rolling.values, color=INK, lw=2)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_ylabel("Anomaly vs normal (°C)")
    ax.set_xlim(series.index.min() - 1, series.index.max() + 1)
    ax.legend(handles=[Patch(color=WARM, label="year warmer than normal"),
                       Patch(color=COOL, label="year colder than normal"),
                       Line2D([], [], color=INK, lw=2, label="10-year moving average")],
              loc="upper left")
    hottest = series.idxmax()
    ax.annotate(f"{hottest}: {_signed(series.max(), ' °C')}", (hottest, series.max()),
                xytext=(0, 6), textcoords="offset points", ha="center", color=INK, fontsize=9)
    _title(ax, f"Warming of {_signed(tr.slope, ' °C', 2)} per decade since {series.index.min()}",
           f"Annual anomaly vs the {ref[0]}-{ref[1]} normal (average of 20 cities) · "
           f"95% CI [{_num(tr.ci_low, 2)}, {_num(tr.ci_high, 2)}] °C/decade · "
           + ("Mann-Kendall p < 0.001" if tr.mk_p_value < 0.001 else f"Mann-Kendall p = {tr.mk_p_value:.3f}"))
    return _finish(fig, path)


def city_trends(df: pd.DataFrame, path: Path) -> Path:
    """Cities ranked by warming rate, with confidence intervals."""
    d = df.sort_values("slope")
    fig, ax = plt.subplots(figsize=(8.5, 7))
    y = np.arange(len(d))
    ax.hlines(y, d["ci_low"], d["ci_high"], color=BLUE, alpha=0.35, lw=3)
    ax.scatter(d["slope"], y, color=BLUE, s=46, zorder=3, edgecolor=SURFACE, linewidth=1.5)
    ax.set_yticks(y, d["city"])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Trend in annual mean temperature (°C per decade)")
    for yi, (_, r) in zip(y, d.iterrows()):
        ax.text(r["ci_high"] + 0.01, yi, _num(r["slope"], 2), va="center", color=INK2, fontsize=8.5)
    ax.set_xlim(0, d["ci_high"].max() + 0.08)
    all_up = bool((d["ci_low"] > 0).all())
    _title(ax, "Every city is warming, at different rates" if all_up else "Warming by city: different rates",
           "1951-2025 slope and 95% confidence interval (linear regression)")
    return _finish(fig, path)


def season_heatmap(pivot: pd.DataFrame, path: Path, title: str) -> Path:
    """Trend by city and by season (°C per decade)."""
    fig, ax = plt.subplots(figsize=(7.5, 7.6))
    norm = _sym_norm(pivot.values)
    im = ax.imshow(pivot.values, cmap=DIVERGING, norm=norm, aspect="auto")
    ax.set_xticks(range(pivot.shape[1]), pivot.columns)
    ax.set_yticks(range(pivot.shape[0]), pivot.index)
    ax.tick_params(length=0)
    ax.grid(False)
    ax.spines[:].set_visible(False)
    ax.xaxis.tick_top()
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, _num(v, 2), ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if abs(norm(v) - 0.5) > 0.32 else INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.03)
    cb.outline.set_visible(False)
    cb.set_label("°C per decade", color=INK2)
    ax.set_title(title, loc="left", pad=46)
    ax.text(0, 1.055, "1951-2025 trend in seasonal mean temperature (°C per decade)",
            transform=ax.transAxes, color=INK2, fontsize=9.5)
    return _finish(fig, path)


def extremes_by_decade(df: pd.DataFrame, path: Path, labels: dict[str, str], title: str) -> Path:
    """Small multiples: average number of extreme days per year, by decade."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.6), sharex=True)
    x = np.arange(len(df))
    for ax, (col, label) in zip(axes.flat, labels.items()):
        vals = df[col].values
        ax.bar(x, vals, width=0.7, color=BLUE, linewidth=0)
        ax.set_title(label, fontsize=11)
        ax.grid(axis="x", visible=False)
        for i in (0, len(vals) - 1):
            ax.text(x[i], vals[i], _num(vals[i], 1), ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_ylim(0, vals.max() * 1.18)
    for ax in axes[1]:
        ax.set_xticks(x, df.index, rotation=0)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13.5, fontweight="semibold", color=INK)
    fig.text(0.01, 0.935, "Average number of days per year and per city, by decade (20 cities)",
             color=INK2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _finish(fig, path)


def records_ratio(df: pd.DataFrame, path: Path) -> Path:
    """Ratio of hot records to cold records, by decade (log scale)."""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ratio = df["ratio"]
    colors = np.where(ratio >= 1, WARM, COOL)
    x = np.arange(len(ratio))
    ax.bar(x, np.log2(ratio), color=colors, width=0.65, linewidth=0)
    ax.axhline(0, color=AXIS, lw=0.8)
    ticks = [-2, -1, 0, 1, 2, 3, 4]
    ax.set_yticks(ticks, ["1:4", "1:2", "1:1", "2:1", "4:1", "8:1", "16:1"])
    ax.set_xticks(x, df.index)
    ax.grid(axis="x", visible=False)
    for xi, r in zip(x, ratio):
        txt = f"{r:.1f}:1" if r >= 1 else f"1:{1 / r:.1f}"
        ax.text(xi, np.log2(r) + (0.12 if r >= 1 else -0.12), txt, ha="center",
                va="bottom" if r >= 1 else "top", fontsize=9, color=INK)
    ax.set_ylabel("Hot records : cold records")
    lo = min(-1.2, np.log2(ratio.min()) - 0.5)
    ax.set_ylim(lo, np.log2(ratio.max()) + 0.7)
    _title(ax, "Heat records now far outnumber cold records" if ratio.iloc[-1] >= 2
           else "Hot and cold records by decade",
           "Daily records broken (highest TX / lowest TN ever measured on that date), 20 cities")
    return _finish(fig, path)


def climatology_heatmap(pivot: pd.DataFrame, path: Path, ref: tuple[int, int]) -> Path:
    fig, ax = plt.subplots(figsize=(10, 7.5))
    im = ax.imshow(pivot.values, cmap=HEAT, aspect="auto")
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.set_xticks(range(12), months)
    ax.set_yticks(range(len(pivot)), pivot.index)
    ax.tick_params(length=0)
    ax.grid(False)
    ax.spines[:].set_visible(False)
    vmin, vmax = np.nanmin(pivot.values), np.nanmax(pivot.values)
    for i in range(pivot.shape[0]):
        for j in range(12):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if (v - vmin) / (vmax - vmin) > 0.55 else INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label("°C", color=INK2)
    _title(ax, "Climatology: from the oceanic north to the Mediterranean south",
           f"Monthly mean temperature, {ref[0]}-{ref[1]} normal (cities sorted from warmest to coldest)")
    return _finish(fig, path)


def precip_anomaly(series: pd.Series, path: Path, tr, ref: tuple[int, int]) -> Path:
    fig, ax = plt.subplots(figsize=(11, 4.4))
    colors = np.where(series.values >= 0, BLUE, ORANGE)
    ax.bar(series.index, series.values, width=0.72, color=colors, linewidth=0)
    rolling = series.rolling(10, center=True, min_periods=6).mean()
    ax.plot(rolling.index, rolling.values, color=INK, lw=2)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_ylabel("Anomaly vs normal total (%)")
    ax.set_xlim(series.index.min() - 1, series.index.max() + 1)
    ax.legend(handles=[Patch(color=BLUE, label="wetter year"),
                       Patch(color=ORANGE, label="drier year"),
                       Line2D([], [], color=INK, lw=2, label="10-year moving average")],
              loc="upper right", ncols=3)
    ax.set_ylim(top=series.max() * 1.25)
    _title(ax, "Annual rainfall: no clear signal (trend not significant)" if tr.mk_p_value >= 0.05
           else "Annual rainfall: significant trend",
           f"Annual total anomaly vs the {ref[0]}-{ref[1]} normal, average of 20 cities · "
           f"slope {_signed(tr.slope, '%')} per decade (Mann-Kendall p = {_num(tr.mk_p_value, 2)})")
    return _finish(fig, path)


def clusters_scatter(coords: pd.DataFrame, path: Path, names: dict[int, str], var_ratio, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    markers = ["o", "s", "^"]
    for k in sorted(coords["cluster"].unique()):
        d = coords[coords["cluster"] == k]
        ax.scatter(d["pc1"], d["pc2"], s=80, color=SERIES[k], marker=markers[k], label=names[k],
                   edgecolor=SURFACE, linewidth=2, zorder=3)
        for _, r in d.iterrows():
            ax.annotate(r["city"], (r["pc1"], r["pc2"]), xytext=(6, 4), textcoords="offset points",
                        fontsize=8.5, color=INK2)
    ax.set_xlabel(f"Component 1 ({var_ratio[0] * 100:.0f}% of variance) · warmer →")
    ax.set_ylabel(f"Component 2 ({var_ratio[1] * 100:.0f}% of variance)")
    ax.legend(loc="best", title=None)
    _title(ax, title,
           "PCA + k-means on the 1991-2020 monthly normals (temperature and rainfall), standardised variables")
    return _finish(fig, path)


def network(df: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(df["year"], df["n_stations_rain"], color=SERIES[0], label="measuring rainfall")
    ax.plot(df["year"], df["n_stations_temp"], color=SERIES[1], label="measuring temperature")
    ax.set_ylabel("Active stations")
    ax.legend(loc="upper left")
    ax.set_ylim(0)
    _title(ax, "The measurement network changes over time",
           "Number of active stations per year in the 20 downloaded départements")
    return _finish(fig, path)


def anomaly_small_multiples(df: pd.DataFrame, path: Path, ref: tuple[int, int]) -> Path:
    cities = sorted(df["city"].unique())
    fig, axes = plt.subplots(4, 5, figsize=(14, 9), sharex=True, sharey=True)
    for ax, city in zip(axes.flat, cities):
        d = df[df["city"] == city]
        ax.bar(d["year"], d["tmean_anom"], width=0.9,
               color=np.where(d["tmean_anom"] >= 0, WARM, COOL), linewidth=0)
        coef = np.polyfit(d["year"], d["tmean_anom"], 1)
        ax.plot(d["year"], np.polyval(coef, d["year"]), color=INK, lw=1.5)
        ax.set_title(f"{city}  {_signed(coef[0] * 10, ' °C/dec.', 2)}", fontsize=10, pad=4)
        ax.axhline(0, color=AXIS, lw=0.6)
        ax.grid(axis="x", visible=False)
    fig.suptitle("City by city, the same trend", x=0.01, ha="left", fontsize=13.5,
                 fontweight="semibold", color=INK)
    fig.text(0.01, 0.947, f"Annual anomaly vs the {ref[0]}-{ref[1]} normal (°C) and trend line",
             color=INK2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _finish(fig, path)


def dumbbell(df: pd.DataFrame, path: Path, col_old: str, col_new: str, labels: tuple[str, str],
             title: str, subtitle: str, xlabel: str) -> Path:
    """Two periods per city: one dot per period, joined by a line."""
    d = df.sort_values(col_new)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    y = np.arange(len(d))
    ax.hlines(y, d[col_old], d[col_new], color=AXIS, lw=2)
    ax.scatter(d[col_old], y, color=SERIES[0], s=50, zorder=3, label=labels[0], edgecolor=SURFACE, linewidth=1.5)
    ax.scatter(d[col_new], y, color=SERIES[1], s=50, zorder=3, label=labels[1], edgecolor=SURFACE, linewidth=1.5)
    for yi, (_, r) in zip(y, d.iterrows()):
        ax.text(r[col_new] + 0.6, yi, f"{r[col_new]:.0f}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, d["city"])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    ax.legend(loc="lower right")
    _title(ax, title, subtitle)
    return _finish(fig, path)


def era5_bias(df: pd.DataFrame, path: Path, title: str, period: str) -> Path:
    d = df.sort_values("bias_tx")
    fig, ax = plt.subplots(figsize=(8.5, 7))
    y = np.arange(len(d))
    h = 0.38
    ax.barh(y + h / 2, d["bias_tx"], height=h - 0.04, color=SERIES[1], label="Maximum temperature")
    ax.barh(y - h / 2, d["bias_tn"], height=h - 0.04, color=SERIES[0], label="Minimum temperature")
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.set_yticks(y, d["city"])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean bias ERA5 − station (°C)")
    ax.legend(loc="upper right")
    _title(ax, title, f"Mean difference between ERA5 (~25 km grid) and the Météo-France station, {period}")
    return _finish(fig, path, SOURCE.replace(" · analysis", " and ERA5 (Copernicus) via the Open-Meteo API · analysis"))


def route_profile(points: pd.DataFrame, track_points: pd.DataFrame, path: Path, title: str, subtitle: str) -> Path:
    """Small multiples along a route: elevation, temperature, rain and head/tail wind (shared distance axis)."""
    fig, axes = plt.subplots(4, 1, figsize=(10, 8.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.1, 0.8, 1]})
    km = points["dist_km"]
    ele_x = track_points["dist_km"]
    ele_y = track_points["ele"] if track_points["ele"].notna().mean() > 0.9 else np.interp(ele_x, km, points["ele_profile"])
    ax = axes[0]
    ax.fill_between(ele_x, ele_y, np.nanmin(ele_y) - 20, color=AXIS, alpha=0.45, linewidth=0)
    ax.plot(ele_x, ele_y, color=MUTED, lw=1.5)
    ax.set_ylabel("Elevation (m)")
    ax = axes[1]
    ax.plot(km, points["temperature_2m"], color=SERIES[1], marker="o", ms=4, label="temperature")
    ax.plot(km, points["apparent_temperature"], color=SERIES[0], marker="o", ms=4, label="feels like")
    ax.set_ylabel("°C")
    ax.legend(loc="upper right", ncols=2)
    ax = axes[2]
    ax.bar(km, points["precipitation"].fillna(0), width=max(km.max() / len(km) * 0.6, 0.2), color=BLUE, linewidth=0)
    ax.set_ylim(0, max(points["precipitation"].max() * 1.3, 1))
    ax.set_ylabel("Rain (mm/h)")
    ax = axes[3]
    head = points["headwind"]
    ax.bar(km, head, width=max(km.max() / len(km) * 0.6, 0.2), color=np.where(head >= 0, WARM, COOL), linewidth=0)
    ax.plot(km, points["wind_gusts_10m"], color=INK, lw=1.5, label="gusts")
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_ylabel("Wind (km/h)")
    ax.set_xlabel("Distance (km)")
    ax.legend(handles=[Patch(color=WARM, label="headwind"), Patch(color=COOL, label="tailwind"),
                       Line2D([], [], color=INK, lw=1.5, label="gusts")], loc="upper left", ncols=3)
    for a in axes:
        a.grid(axis="x", visible=False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13.5, fontweight="semibold", color=INK)
    fig.text(0.01, 0.948, subtitle, color=INK2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _finish(fig, path, "Weather: Open-Meteo (forecast models / ERA5 archive) · analysis: Météo Analytics")
