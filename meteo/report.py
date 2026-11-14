"""Reporting: self-contained HTML report (a single file with embedded images, printable)."""
from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .config import PROJECT_ROOT, Config
from .fmt import num, pvalue, signed, verdict


def _img(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def build_report(cfg: Config) -> Path:
    res_path = cfg.reports_dir / "results.json"
    if not res_path.exists():
        raise FileNotFoundError("Results missing: run `python -m meteo analyze` first")
    res = json.loads(res_path.read_text(encoding="utf-8"))
    figs = {p.stem: _img(p) for p in sorted(cfg.figures_dir.glob("*.png"))}

    env = Environment(loader=FileSystemLoader(PROJECT_ROOT / "templates"),
                      autoescape=select_autoescape(["html"]), trim_blocks=True, lstrip_blocks=True)
    env.filters.update(num=num, signed=signed, pvalue=pvalue, verdict=verdict)
    html = env.get_template("report.html.j2").render(
        r=res, figs=figs, generated=datetime.now().strftime("%d %B %Y"),
        cities=[c.name for c in cfg.cities],
    )
    out = cfg.reports_dir / "climate_report.html"
    out.write_text(html, encoding="utf-8")
    return out
