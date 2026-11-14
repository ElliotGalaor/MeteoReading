"""Danger and comfort scores along a route, and generic advice for the weather met on the way.

Both scores go from 0 to 100 and are meant to be explained in one sentence each, not to look scientific.

Danger (0 = nothing to worry about, 100 = do not go)
    At each point, every hazard (thunderstorm, ice, snow, heavy rain, gusts, heat, cold, poor visibility,
    darkness) gets a score that rises linearly from a threshold where it starts to matter to one where it is
    severe. The thresholds depend on the activity: a 60 km/h gust is a danger on a bike, not in a car. The
    hazards of a point add up like independent risks, 1 - (1 - a)(1 - b)..., so two moderate hazards at once
    weigh more than either of them. The score of the route mixes its worst moment (70%) and the average over
    the time spent on the way (30%): a trip is as dangerous as its worst part, and a long exposure counts
    more than a brief one.

Comfort (100 = ideal conditions for the activity)
    At each point, penalties are taken off 100: temperature outside the comfortable range of the activity,
    muggy air, rain, wind (the headwind on a bike), snow, strong sun, darkness, fog. The score of the route is
    the average over the time spent on the way.

The advice is generic: it comes from simple rules on the conditions met (where and when), worded for the
activity. It does not replace the official warnings.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

THUNDER, HAIL = {95}, {96, 99}
FREEZING = {56, 57, 66, 67}
SNOW = {71, 73, 75, 77, 85, 86}
FOG = {45, 48}

DANGER_LEVELS = [(75, "Severe"), (50, "High"), (25, "Moderate"), (0, "Low")]
COMFORT_LEVELS = [(80, "Very pleasant"), (60, "Pleasant"), (40, "Mixed"), (20, "Unpleasant"), (0, "Very unpleasant")]
PEAK_WEIGHT = 0.7  # share of the worst moment in the route's danger score
SIGNIFICANT = 10   # a hazard below this score is neither listed nor advised on
WARN_FROM = 25     # from this hazard score (moderate), the advice points to the official warnings


@dataclass(frozen=True)
class Profile:
    """How an activity reacts to the weather (danger thresholds: where a hazard starts, where it is severe)."""

    gusts: tuple[float, float]       # km/h
    heat: tuple[float, float]        # feels-like °C
    cold: tuple[float, float]        # feels-like °C (going down)
    storm: float                     # danger of a thunderstorm (with hail: +20)
    wet_frost: float                 # danger of rain or drizzle at 1 °C or below (ice forming)
    snow: float                      # danger of falling snow, +50 at 3 cm/h
    visibility: float                # weight of poor visibility (fog, 100 m visibility = this x 100)
    darkness: float                  # danger of travelling at night
    comfort_range: tuple[float, float]  # feels-like °C
    thermal_weight: float            # 1 outdoors; a car has heating and air conditioning
    rain_weight: float
    exposed: bool                    # outdoors (sun, muggy air, wind are felt)


PROFILES = {
    "Walking": Profile(gusts=(60, 110), heat=(30, 42), cold=(-10, -30), storm=70, wet_frost=40, snow=20,
                       visibility=0.4, darkness=15, comfort_range=(10, 22), thermal_weight=1.0, rain_weight=1.0,
                       exposed=True),
    "Cycling": Profile(gusts=(40, 80), heat=(30, 42), cold=(-5, -25), storm=70, wet_frost=70, snow=50,
                       visibility=0.7, darkness=30, comfort_range=(12, 22), thermal_weight=1.0, rain_weight=1.0,
                       exposed=True),
    "Driving": Profile(gusts=(70, 130), heat=(40, 50), cold=(-15, -35), storm=40, wet_frost=60, snow=40,
                       visibility=0.9, darkness=10, comfort_range=(5, 28), thermal_weight=0.3, rain_weight=0.4,
                       exposed=False),
}


@dataclass
class Factor:
    """What drives a score: a hazard (danger, peak value) or a penalty (comfort, average value)."""

    name: str
    value: float
    km: float
    time: pd.Timestamp


@dataclass
class Score:
    value: int
    level: str
    factors: list[Factor] = field(default_factory=list)


@dataclass
class Advice:
    icon: str
    title: str
    text: str
    kind: str  # "danger", "comfort" or "info"


@dataclass
class Assessment:
    activity: str
    danger: Score
    comfort: Score
    advice: list[Advice]
    points: pd.DataFrame  # dist_km, time, danger, comfort, then one column per hazard (h_...) and penalty (p_...)


# --- Helpers --------------------------------------------------------------------------------
def ramp(x, start: float, severe: float) -> np.ndarray:
    """0 before `start`, 100 from `severe` on, linear in between; works both ways (cold: start > severe)."""
    x = np.asarray(x, dtype=float)
    return np.nan_to_num(np.clip((x - start) / (severe - start), 0, 1) * 100, nan=0.0)


def column(df: pd.DataFrame, name: str, default: float = np.nan) -> np.ndarray:
    if name not in df:
        return np.full(len(df), default, dtype=float)
    return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)


def codes_in(df: pd.DataFrame, codes: set[int]) -> np.ndarray:
    if "weather_code" not in df:
        return np.zeros(len(df), dtype=bool)
    return np.array([c is not None and pd.notna(c) and int(c) in codes for c in df["weather_code"]])


def weights(df: pd.DataFrame) -> np.ndarray:
    """Time spent around each point: averages are over time, not over points."""
    w = column(df, "hours_here", 0.0)
    w = np.nan_to_num(w, nan=0.0)
    return w if w.sum() > 0 else np.ones(len(df))


def level(value: float, levels: list[tuple[int, str]]) -> str:
    return next(name for floor, name in levels if value >= floor)


def is_night(df: pd.DataFrame) -> np.ndarray:
    return column(df, "is_day", 1.0) < 0.5


# --- Danger ---------------------------------------------------------------------------------
def hazards(df: pd.DataFrame, p: Profile) -> pd.DataFrame:
    """Danger of each hazard at each point, from 0 to 100."""
    temp, feels = column(df, "temperature_2m"), column(df, "apparent_temperature")
    rain, snowfall = np.nan_to_num(column(df, "precipitation"), nan=0.0), np.nan_to_num(column(df, "snowfall"), nan=0.0)
    freezing, snowing = codes_in(df, FREEZING), codes_in(df, SNOW) | (snowfall > 0)
    ice = np.where(freezing, 90.0, np.where((temp <= 1) & (rain >= 0.1) & ~snowing, p.wet_frost, 0.0))
    snow = np.where(snowing, np.minimum(p.snow + ramp(snowfall, 0, 3) / 2, 100), 0.0)
    storm = np.where(codes_in(df, HAIL), min(p.storm + 20, 100), np.where(codes_in(df, THUNDER), p.storm, 0.0))
    fog_code = np.where(codes_in(df, FOG), 50 * p.visibility, 0.0)
    return pd.DataFrame({
        "Thunderstorm": storm,
        "Ice": ice,
        "Snow": snow,
        "Heavy rain": ramp(rain, 4, 20),
        "Gusts": ramp(column(df, "wind_gusts_10m"), *p.gusts),
        "Heat": ramp(feels, *p.heat),
        "Cold": ramp(feels, *p.cold),
        "Poor visibility": np.maximum(fog_code, ramp(column(df, "visibility"), 1000, 100) * p.visibility),
        "Darkness": np.where(is_night(df), p.darkness, 0.0),
    }, index=df.index)


def combine(h: pd.DataFrame) -> np.ndarray:
    """Hazards add up like independent risks: 1 - (1 - a)(1 - b)..."""
    return 100 * (1 - np.prod(1 - h.to_numpy() / 100, axis=1))


def danger_score(df: pd.DataFrame, h: pd.DataFrame) -> Score:
    point = combine(h)
    value = PEAK_WEIGHT * point.max() + (1 - PEAK_WEIGHT) * np.average(point, weights=weights(df))
    factors = []
    for name in h.columns:
        i = int(np.argmax(h[name].to_numpy()))
        if h[name].iloc[i] >= SIGNIFICANT:
            factors.append(Factor(name, float(h[name].iloc[i]), float(df["dist_km"].iloc[i]), df["time"].iloc[i]))
    factors.sort(key=lambda f: -f.value)
    return Score(int(round(value)), level(value, DANGER_LEVELS), factors)


# --- Comfort ---------------------------------------------------------------------------------
def penalties(df: pd.DataFrame, p: Profile, activity: str) -> pd.DataFrame:
    """Comfort points lost at each point, by cause."""
    feels, rain = column(df, "apparent_temperature"), np.nan_to_num(column(df, "precipitation"), nan=0.0)
    low, high = p.comfort_range
    outside = np.nan_to_num(np.maximum(low - feels, 0) + np.maximum(feels - high, 0), nan=0.0)
    gusts = np.nan_to_num(column(df, "wind_gusts_10m"), nan=0.0)
    if activity == "Cycling":
        head = np.nan_to_num(column(df, "headwind"), nan=0.0)
        wind = np.minimum(np.maximum(head - 5, 0) * 1.5, 35) + np.minimum(np.maximum(gusts - 35, 0) * 0.5, 15)
    elif activity == "Walking":
        wind = np.minimum(np.maximum(np.nan_to_num(column(df, "wind_speed_10m"), nan=0.0) - 20, 0), 25)
    else:
        wind = np.minimum(np.maximum(gusts - 60, 0) * 0.5, 15)
    uv = np.nan_to_num(column(df, "uv_index"), nan=0.0)
    dew = np.nan_to_num(column(df, "dew_point_2m"), nan=0.0)
    snowing = codes_in(df, SNOW) | (np.nan_to_num(column(df, "snowfall"), nan=0.0) > 0)
    return pd.DataFrame({
        "Temperature": np.minimum(outside * 3, 60) * p.thermal_weight,
        "Muggy air": np.minimum(np.maximum(dew - 16, 0) * 3, 25) * p.exposed,
        "Rain": np.where(rain >= 0.2, np.minimum(15 + 10 * rain, 55), 0.0) * p.rain_weight,
        "Wind": wind,
        "Snow": np.where(snowing, 40.0 if activity == "Cycling" else 25.0, 0.0),
        "Strong sun": np.minimum(np.maximum(uv - 5, 0) * 4, 20) * p.exposed,
        "Darkness": np.where(is_night(df), 5.0 if activity == "Driving" else 10.0, 0.0),
        "Fog": np.where(codes_in(df, FOG), 15.0 if activity == "Driving" else 10.0, 0.0),
    }, index=df.index)


def comfort_score(df: pd.DataFrame, pen: pd.DataFrame) -> tuple[Score, np.ndarray]:
    point = np.clip(100 - pen.sum(axis=1).to_numpy(), 0, 100)
    w = weights(df)
    value = float(np.average(point, weights=w))
    factors = []
    for name in pen.columns:
        mean = float(np.average(pen[name].to_numpy(), weights=w))
        if mean >= 3:
            i = int(np.argmax(pen[name].to_numpy()))
            factors.append(Factor(name, mean, float(df["dist_km"].iloc[i]), df["time"].iloc[i]))
    factors.sort(key=lambda f: -f.value)
    return Score(int(round(value)), level(value, COMFORT_LEVELS), factors), point


# --- Advice -----------------------------------------------------------------------------------
def advise(df: pd.DataFrame, activity: str, h: pd.DataFrame, tz: str) -> list[Advice]:
    local = df["time"].dt.tz_convert(tz)
    clock = "%a %H:%M" if local.dt.date.nunique() > 1 else "%H:%M"

    def where(mask) -> str:
        """'around km 12 (14:30)' for the first point where `mask` is true."""
        i = int(np.argmax(np.asarray(mask)))
        return f"around km {df['dist_km'].iloc[i]:.0f} ({local.iloc[i].strftime(clock)})"

    bike, car = activity == "Cycling", activity == "Driving"
    feels, rain = column(df, "apparent_temperature"), np.nan_to_num(column(df, "precipitation"), nan=0.0)
    gusts, uv = column(df, "wind_gusts_10m"), np.nan_to_num(column(df, "uv_index"), nan=0.0)
    out: list[Advice] = []

    if (h["Thunderstorm"] >= SIGNIFICANT).any():
        text = ("Heavy downpours and hail possible: slow down, never drive through a flooded road." if car else
                "Stay away from summits, ridges and open ground, do not shelter under an isolated tree, and "
                "consider postponing.")
        out.append(Advice("⛈️", f"Thunderstorms {where(h['Thunderstorm'] >= SIGNIFICANT)}", text, "danger"))
    if (h["Ice"] >= SIGNIFICANT).any():
        text = {"Walking": "Shoes with a good grip, and poles if you have them; ice is hard to see.",
                "Cycling": "Ice is invisible on the road: consider not riding, or lower the tyre pressure and avoid "
                           "braking in bends.",
                "Driving": "Slow down, keep a long distance and brake gently; bridges freeze first."}[activity]
        out.append(Advice("🧊", f"Risk of ice {where(h['Ice'] >= SIGNIFICANT)}", text, "danger"))
    if (h["Snow"] >= SIGNIFICANT).any():
        text = {"Walking": "Waterproof boots and warm layers; in the mountains, check the avalanche bulletin.",
                "Cycling": "Grip and visibility drop sharply: shorten the ride or postpone it.",
                "Driving": "Winter tyres or chains, more time for the journey, and a blanket in the car."}[activity]
        out.append(Advice("❄️", f"Snow {where(h['Snow'] >= SIGNIFICANT)}", text, "danger"))
    if (h["Gusts"] >= SIGNIFICANT).any():
        text = {"Walking": "Avoid ridges and forests (falling branches), and keep children close.",
                "Cycling": "Hold the handlebars firmly and watch for crosswinds on open sections, on bridges and "
                           "when lorries pass.",
                "Driving": "Slow down, especially with a trailer, a van or a roof box."}[activity]
        out.append(Advice("💨", f"Gusts up to {np.nanmax(gusts):.0f} km/h", text, "danger"))
    if (h["Heavy rain"] >= SIGNIFICANT).any():
        text = ("Slow down (aquaplaning), switch on dipped headlights and leave more distance." if car else
                "Waterproof jacket and trousers; paths and roads get slippery, and streams can rise quickly.")
        out.append(Advice("🌧️", f"Heavy rain {where(h['Heavy rain'] >= SIGNIFICANT)}", text, "danger"))
    elif (rain >= 0.2).any():
        text = ("Wet roads: leave more distance." if car else
                "Take a waterproof layer" + (", mudguards and a rear light." if bike else "."))
        out.append(Advice("☔", f"Rain {where(rain >= 0.2)}", text, "comfort"))
    if (h["Poor visibility"] >= SIGNIFICANT).any():
        text = ("Dipped headlights or fog lights, and slow down." if car else
                "Wear bright clothing" + (" and switch on your lights, even by day." if bike else "."))
        out.append(Advice("🌫️", f"Fog or poor visibility {where(h['Poor visibility'] >= SIGNIFICANT)}", text, "danger"))

    hot = feels >= 28
    if hot.any():
        text = ("Keep water in the car and never leave anyone in a parked car." if car else
                "Drink regularly (about half a litre per hour of effort), set off earlier, cover your head and "
                "slow down in the hottest hours.")
        kind = "danger" if (h["Heat"] >= SIGNIFICANT).any() else "comfort"
        out.append(Advice("🥵", f"Heat: feels like up to {np.nanmax(feels):.0f} °C", text, kind))
    cold = feels <= 5
    if cold.any() and not car:
        text = ("Several thin layers, gloves and a hat" +
                (", a windproof jacket and overshoes: riding adds wind chill." if bike else "."))
        kind = "danger" if (h["Cold"] >= SIGNIFICANT).any() else "comfort"
        out.append(Advice("🧤", f"Cold: feels like down to {np.nanmin(feels):.0f} °C", text, kind))
    temps = column(df, "temperature_2m")
    if not car and np.nanmax(temps) - np.nanmin(temps) >= 8:
        out.append(Advice("🧥", f"From {np.nanmin(temps):.0f} to {np.nanmax(temps):.0f} °C along the way",
                          "Altitude or time of day: bring a layer you can put on and take off.", "comfort"))
    if (uv >= 6).any() and not car:
        out.append(Advice("🕶️", f"Strong sun (UV index up to {uv.max():.0f})",
                          "Sunscreen, sunglasses and a cap; refill water more often.", "comfort"))
    night = is_night(df)
    if night.any():
        text = ("Headlights on, and take breaks: night driving is more tiring." if car else
                "Front and rear lights, reflective clothing" + ("." if bike else ", and a head torch."))
        out.append(Advice("🌙", f"In the dark {where(night)}", text, "danger" if bike else "comfort"))
    if not car:
        head = column(df, "headwind")
        w = weights(df)
        head_share = float(w[head > 5].sum() / w.sum())
        tail_share = float(w[head < -5].sum() / w.sum())
        if head_share >= 0.4 and tail_share < 0.15:
            out.append(Advice("🧭", f"Headwind on {head_share:.0%} of the way",
                              "Plan more time and energy. Done the other way round, most of it would be with the "
                              "wind behind you, if the wind holds.", "comfort"))

    if (h.drop(columns="Darkness").max() >= WARN_FROM).any():   # night is not in the weather warnings
        out.append(Advice("📢", "Check the official warnings before you go",
                          "In France: Météo-France vigilance (vigilance.meteofrance.fr). Forecasts change: look "
                          "again on the day.", "info"))
    elif not out:
        out.append(Advice("👍", "Good conditions", "Nothing in the forecast calls for special care: enjoy the trip. "
                          "Forecasts change, so look again on the day.", "info"))
    order = {"danger": 0, "comfort": 1, "info": 2}
    return sorted(out, key=lambda a: order[a.kind])


# --- Entry point ------------------------------------------------------------------------------
def assess(df: pd.DataFrame, activity: str, tz: str) -> Assessment:
    """Danger and comfort scores and advice for a weather reading (the table built by route_weather)."""
    if activity not in PROFILES:
        raise ValueError(f"Unknown activity {activity!r}: choose one of {', '.join(PROFILES)}")
    p = PROFILES[activity]
    h = hazards(df, p)
    pen = penalties(df, p, activity)
    comfort, comfort_points = comfort_score(df, pen)
    points = pd.concat([
        df[["dist_km", "time"]].reset_index(drop=True),
        pd.DataFrame({"danger": combine(h), "comfort": comfort_points}),
        h.add_prefix("h_").reset_index(drop=True),
        pen.add_prefix("p_").reset_index(drop=True),
    ], axis=1)
    return Assessment(activity, danger_score(df, h), comfort, advise(df, activity, h, tz), points)
