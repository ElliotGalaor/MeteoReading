"""Dark mode of the dashboard (dashboard/ui.py): figures built with the light palette get its dark steps."""
from __future__ import annotations

import plotly.graph_objects as go
import pytest

from meteo import palette


@pytest.fixture
def ui(monkeypatch):
    monkeypatch.syspath_prepend("dashboard")
    import ui

    return ui


def figure(ui) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=[1, 2], y=[3, 4], line=dict(color=palette.SERIES[0]), marker=dict(line=dict(color=palette.SURFACE)))
    fig.add_heatmap(z=[[1, 2]], colorscale=ui.HEAT)
    fig.add_scattermap(lat=[45], lon=[6], marker=dict(color=[1], colorscale=ui.DIVERGING))
    fig.update_layout(map=dict(style="carto-positron"), font=dict(color=palette.INK2),
                      annotations=[dict(text="x", font=dict(color=palette.MUTED))])
    return fig


def test_light_mode_leaves_the_figure_as_built(ui, monkeypatch):
    monkeypatch.setattr(ui, "is_dark", lambda: False)
    fig = figure(ui)
    assert ui.themed(fig) is fig


def test_dark_mode_steps_every_colour(ui, monkeypatch):
    monkeypatch.setattr(ui, "is_dark", lambda: True)
    d = ui.themed(figure(ui))
    assert d.data[0].line.color == palette.DARK_SERIES[0]
    assert d.data[0].marker.line.color == palette.DARK_SURFACE
    assert [c for _, c in d.data[1].colorscale] == palette.DARK_HEAT_STOPS
    assert [c for _, c in d.data[2].marker.colorscale] == palette.DARK_DIVERGING_STOPS
    assert d.layout.map.style == "carto-darkmatter"
    assert d.layout.font.color == palette.DARK_INK2 and d.layout.annotations[0].font.color == palette.DARK_MUTED
    # the template of the dashboard (backgrounds, grid, axes) is recoloured too
    assert d.layout.template.layout.paper_bgcolor == palette.DARK_SURFACE
    assert d.layout.template.layout.xaxis.gridcolor == palette.DARK_GRID


def test_status_and_unknown_colours_are_kept(ui, monkeypatch):
    monkeypatch.setattr(ui, "is_dark", lambda: True)
    fig = go.Figure(go.Bar(x=[1], y=[1], marker_color=palette.STATUS["critical"]))
    fig.add_scatter(x=[1], y=[1], line=dict(color="#123456"))
    d = ui.themed(fig)
    assert d.data[0].marker.color == palette.STATUS["critical"] and d.data[1].line.color == "#123456"


def test_transparent_fills_take_the_dark_step(ui, monkeypatch):
    monkeypatch.setattr(ui, "is_dark", lambda: True)
    fig = go.Figure(go.Scatter(x=[1], y=[1], fill="tonexty", fillcolor=ui.rgba(palette.SERIES[1], 0.16)))
    fig.add_scatter(x=[1], y=[1], fillcolor="rgba(18,52,86,0.5)")
    d = ui.themed(fig)
    assert d.data[0].fillcolor == ui.rgba(palette.DARK_SERIES[1], 0.16)
    assert d.data[1].fillcolor == "rgba(18,52,86,0.5)"     # not a palette colour: kept


def test_css_follows_the_theme(ui):
    assert palette.DARK_SURFACE in ui.css(ui.DARK) and palette.SURFACE not in ui.css(ui.DARK)
    assert palette.SURFACE in ui.css(ui.LIGHT)


@pytest.mark.parametrize("dark", [False, True])
def test_the_drawing_map_needs_no_api_key(ui, dark):
    from route_page import _draw_map

    html = _draw_map((43.53, 5.45), 12, dark=dark, phone=False).get_root().render()
    assert "tile.openstreetmap.org" in html
    assert "cartocdn" not in html      # CARTO's raster tiles now need a key: "API KEY REQUIRED" on every tile
    assert ("invert(1)" in html) == dark
