"""Restitution : rapport HTML autonome (un seul fichier, images intégrées, imprimable)."""
from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .config import PROJECT_ROOT, Config
from .fmt import fr, pvalue, sfr, verdict


def _img(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def build_report(cfg: Config) -> Path:
    res_path = cfg.reports_dir / "resultats.json"
    if not res_path.exists():
        raise FileNotFoundError("Résultats absents : lancer d'abord `python -m meteo analyze`")
    res = json.loads(res_path.read_text(encoding="utf-8"))
    figs = {p.stem: _img(p) for p in sorted(cfg.figures_dir.glob("*.png"))}

    env = Environment(loader=FileSystemLoader(PROJECT_ROOT / "templates"),
                      autoescape=select_autoescape(["html"]), trim_blocks=True, lstrip_blocks=True)
    env.filters.update(fr=fr, sfr=sfr, pvalue=pvalue, verdict=verdict)
    html = env.get_template("rapport.html.j2").render(
        r=res, figs=figs, generated=datetime.now().strftime("%d/%m/%Y"),
        cities=[c.name for c in cfg.cities],
    )
    out = cfg.reports_dir / "rapport_climat.html"
    out.write_text(html, encoding="utf-8")
    return out
