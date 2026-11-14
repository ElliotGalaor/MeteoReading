"""Chargement et validation de la configuration (config.yaml)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class City:
    name: str
    dep: str
    lat: float
    lon: float
    alt: float | None = None
    stations: tuple[str, ...] = ()  # station(s) imposée(s), par ordre de priorité


@dataclass(frozen=True)
class Config:
    dataset_api: str
    periods: list[str]
    extra_departments: list[str] | str
    workers: int
    raw_dir: Path
    warehouse: Path
    reports_dir: Path
    analysis: dict
    thresholds: dict
    era5: dict
    cities: list[City] = field(default_factory=list)

    @property
    def departments(self) -> list[str] | str:
        """Départements à télécharger : ceux des villes + les départements en plus."""
        if self.extra_departments == "all":
            return "all"
        deps = {c.dep for c in self.cities} | set(self.extra_departments or [])
        return sorted(deps)

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    root = Path(path).resolve().parent
    cities = [
        City(**{**c, "dep": str(c["dep"]).zfill(2), "stations": tuple(str(s) for s in c.get("stations", []))})
        for c in raw["cities"]
    ]
    if len({c.name for c in cities}) != len(cities):
        raise ValueError("Deux villes portent le même nom dans config.yaml")

    a = raw["analysis"]
    if not a["start_year"] < a["end_year"]:
        raise ValueError("analysis.start_year doit être inférieur à analysis.end_year")

    src, paths = raw["source"], raw["paths"]
    return Config(
        dataset_api=src["dataset_api"],
        periods=list(src["periods"]),
        extra_departments=src.get("extra_departments") or [],
        workers=int(src.get("workers", 4)),
        raw_dir=root / paths["raw"],
        warehouse=root / paths["warehouse"],
        reports_dir=root / paths["reports"],
        analysis=a,
        thresholds=raw["thresholds"],
        era5=raw.get("era5", {}),
        cities=cities,
    )
