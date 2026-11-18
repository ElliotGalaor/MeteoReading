"""Data behind the charts (#69): the CSV holds the values drawn, with clear columns, the source and the licence."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))

from meteo import i18n  # noqa: E402
from ui import csv_bytes, figure_frame, slug  # noqa: E402


def read(payload: bytes, sep: str = ",", decimal: str = ".") -> tuple[list[str], pd.DataFrame]:
    text = payload.decode("utf-8-sig")
    header = [line for line in text.splitlines() if line.startswith("#")]
    return header, pd.read_csv(io.StringIO(text), comment="#", sep=sep, decimal=decimal)


def test_lines_on_a_shared_x_become_columns_with_their_unit():
    years = list(range(2000, 2005))
    fig = go.Figure()
    fig.add_scatter(x=years, y=[1.0, 2, 3, 4, 5], name="Paris")
    fig.add_scatter(x=years, y=[2.0, 3, 4, 5, 6], name="Lyon")
    fig.update_layout(yaxis_title="°C")
    frame = figure_frame(fig)
    assert list(frame.columns) == ["year", "Paris (°C)", "Lyon (°C)"]
    assert frame["Lyon (°C)"].tolist() == [2, 3, 4, 5, 6]


def test_bands_highlights_and_helpers():
    dates = pd.date_range("2022-07-01", periods=4)
    fig = go.Figure()
    fig.add_scatter(x=dates, y=[30, 31, 32, 33], line=dict(width=0), hoverinfo="skip", showlegend=False)  # band top
    fig.add_scatter(x=dates, y=[20, 21, 22, 23], fill="tonexty", name="80% of days", hoverinfo="skip")
    fig.add_scatter(x=dates, y=[25, 36, 27, 28], name="2022")
    fig.add_scatter(x=dates[[1]], y=[36], mode="markers", name="Above the 90th percentile")   # a highlight
    fig.add_scatter(x=[None], y=[None], name="legend only")
    frame = figure_frame(fig)
    assert list(frame.columns) == ["date", "80% of days, upper bound", "80% of days, lower bound", "2022"]
    assert frame["2022"].tolist() == [25, 36, 27, 28]


def test_error_bars_give_the_lowest_and_highest():
    fig = go.Figure(go.Bar(x=["2050", "2100"], y=[20.0, 40.0], name="Scenario",
                           error_y=dict(type="data", array=[5, 10], arrayminus=[2, 4], symmetric=False)))
    frame = figure_frame(fig)
    assert frame["Scenario, lowest"].tolist() == [18, 36] and frame["Scenario, highest"].tolist() == [25, 50]


def test_a_heatmap_gives_its_grid_and_subplots_share_their_x():
    fig = go.Figure(go.Heatmap(z=[[1, 2], [3, 4]], x=["1951-59", "1960-69"], y=["Brest", "Nice"],
                               meta={"row": "City"}))
    assert figure_frame(fig).to_dict("list") == {"City": ["Brest", "Nice"], "1951-59": [1.0, 3.0], "1960-69": [2.0, 4.0]}
    sub = make_subplots(rows=2, cols=1, shared_xaxes=True)
    sub.add_scatter(x=[0, 1, 2], y=[10, 11, 12], name="Temperature", row=1, col=1)
    sub.add_bar(x=[0, 1, 2], y=[0.0, 0.5, 0.0], name="Rain", row=2, col=1)
    sub.update_yaxes(title_text="°C", row=1, col=1)
    sub.update_yaxes(title_text="mm/h", row=2, col=1)
    assert list(figure_frame(sub).columns)[1:] == ["Temperature (°C)", "Rain (mm/h)"]


def test_the_file_says_where_the_data_comes_from():
    frame = pd.DataFrame({"year": [2000, 2001], "Paris (°C)": [12.5, 13.25]})
    header, back = read(csv_bytes(frame, "Mean temperature", "A subtitle"))
    assert header[0] == "# Météo Analytics: Mean temperature" and header[1] == "# A subtitle"
    assert any("Etalab 2.0" in line for line in header)
    assert back["Paris (°C)"].tolist() == [12.5, 13.25]


@pytest.fixture
def french():
    previous = i18n._provider
    i18n.set_provider(lambda: "fr")
    yield
    i18n.set_provider(previous)


def test_french_files_use_semicolons_and_decimal_commas(french):
    frame = pd.DataFrame({"année": [2000], "Paris (°C)": [12.5]})
    payload = csv_bytes(frame, "Température moyenne")
    assert payload.startswith("﻿".encode())                      # spreadsheets read the accents
    assert "2000;12,5" in payload.decode("utf-8-sig")
    header, back = read(payload, sep=";", decimal=",")
    assert header[0] == "# Météo Analytics : Température moyenne"
    assert any("licence ouverte Etalab 2.0" in line for line in header)
    assert back["Paris (°C)"].tolist() == [12.5]


def test_file_names_come_from_the_title():
    assert slug("Hot days (TX ≥ 30 °C)") == "hot_days_tx_30_c"
    assert slug("Jours très chauds à Nîmes") == "jours_tres_chauds_a_nimes"
    assert slug(None) == "chart"
