"""Palette partagée par les graphiques statiques (matplotlib) et le dashboard (Plotly).

Les 3 premières couleurs catégorielles restent distinguables deux à deux, y compris
pour les personnes daltoniennes : aucun graphique n'affiche plus de 3 séries colorées.
"""

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # bleu, orange, vert d'eau
BLUE, ORANGE = SERIES[0], SERIES[1]
WARM, COOL, NEUTRAL = "#d63c3b", "#256abf", "#f0efec"  # pôles du divergent + centre neutre

# Divergent (anomalies) : bleu <-> gris neutre <-> rouge, autant de pas de chaque côté
DIVERGING_STOPS = ["#0d366b", "#256abf", "#86b6ef", NEUTRAL, "#f0a09e", "#d63c3b", "#8a1c1b"]
# Séquentiels à une seule teinte
HEAT_STOPS = ["#fdf1ea", "#f7c2a6", "#eb6834", "#a8400f", "#5c2208"]
RAIN_STOPS = ["#eef5fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"]

SOURCE = "Source : Météo-France, données climatologiques quotidiennes (data.gouv.fr) · traitement : Météo Analytics"
