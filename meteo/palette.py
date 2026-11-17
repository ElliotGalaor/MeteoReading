"""Palette shared by the static charts (matplotlib) and the dashboard (Plotly).

The first 3 categorical colours stay distinguishable pairwise, including for colour-blind
readers: no chart shows more than 3 coloured series.
"""

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # blue, orange, aqua
BLUE, ORANGE = SERIES[0], SERIES[1]
WARM, COOL, NEUTRAL = "#d63c3b", "#256abf", "#f0efec"  # diverging poles + neutral midpoint

# Diverging (anomalies): blue <-> neutral grey <-> red, same number of steps on each side
DIVERGING_STOPS = ["#0d366b", "#256abf", "#86b6ef", NEUTRAL, "#f0a09e", "#d63c3b", "#8a1c1b"]
# Single-hue sequential scales
HEAT_STOPS = ["#fdf1ea", "#f7c2a6", "#eb6834", "#a8400f", "#5c2208"]
RAIN_STOPS = ["#eef5fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"]

# Status (danger and comfort levels): always shown with a label, never by colour alone
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

# Dark mode (dashboard, when the device is in dark mode): the same hues stepped for the dark surface, from the
# dataviz reference palette. Checked with its validator against #1a1a19: lightness band, chroma, colour-blind
# separation and contrast all pass for the series and the anomaly poles.
DARK_SURFACE, DARK_INK, DARK_INK2, DARK_MUTED = "#1a1a19", "#f0efec", "#c3c2b7", "#8f8d87"
DARK_GRID, DARK_AXIS, DARK_NEUTRAL = "#2f2e2b", "#4a4944", "#383835"
DARK_SERIES = ["#3987e5", "#d95926", "#199e70"]
DARK_WARM, DARK_COOL = "#e66767", "#3987e5"
# Sequential and diverging scales on a dark surface: the low end (or the neutral midpoint) recedes into the surface,
# the strong values are the bright ones
DARK_DIVERGING_STOPS = ["#b7d3f6", "#3987e5", "#184f95", DARK_NEUTRAL, "#8a2a29", "#e66767", "#f6b9b8"]
DARK_HEAT_STOPS = ["#3a1b0d", "#7a3414", "#d95926", "#f29a72", "#fcd9c6"]
DARK_RAIN_STOPS = ["#132a4a", "#184f95", "#3987e5", "#86b6ef", "#cde2fb"]

SOURCE = "Source: Météo-France daily climatological data (data.gouv.fr) · analysis: Météo Analytics"
