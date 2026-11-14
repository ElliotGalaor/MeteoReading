"""Mise en forme des nombres à la française (rapport et dashboard)."""
from __future__ import annotations

MINUS, NNBSP = "−", " "


def fr(value, digits: int = 1) -> str:
    """0.3315 -> '0,3' ; 25026059 -> '25 026 059' (espace fine insécable)."""
    if value is None:
        return "–"
    if isinstance(value, int) or digits == 0:
        return f"{round(value):,}".replace(",", NNBSP).replace("-", MINUS)
    return f"{value:,.{digits}f}".replace(",", NNBSP).replace(".", ",").replace("-", MINUS)


def sfr(value, digits: int = 1) -> str:
    """Nombre signé : +0,3 / −1,2."""
    if value is None:
        return "–"
    return ("+" if value >= 0 else "") + fr(value, digits)


def pvalue(p) -> str:
    return "p < 0,001" if p < 0.001 else f"p = {fr(p, 3)}"


def verdict(trend: dict) -> str:
    """Formulation prudente d'une tendance selon les deux tests (OLS et Mann-Kendall)."""
    if trend["significant"]:
        return "significative"
    if trend["mk_p_value"] < 0.1:
        return "à la limite de la significativité"
    return "non significative"
