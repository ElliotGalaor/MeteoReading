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

SOURCE = "Source: Météo-France daily climatological data (data.gouv.fr) · analysis: Météo Analytics"
