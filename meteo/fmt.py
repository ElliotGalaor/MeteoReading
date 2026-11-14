"""Number formatting shared by the report and the dashboard."""
from __future__ import annotations

MINUS = "−"


def num(value, digits: int = 1) -> str:
    """0.3315 -> '0.3' ; 25026059 -> '25,026,059' (typographic minus sign)."""
    if value is None:
        return "–"
    if isinstance(value, int) or digits == 0:
        return f"{round(value):,}".replace("-", MINUS)
    return f"{value:,.{digits}f}".replace("-", MINUS)


def signed(value, digits: int = 1) -> str:
    """Signed number: +0.3 / −1.2."""
    if value is None:
        return "–"
    return ("+" if value >= 0 else "") + num(value, digits)


def pvalue(p) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {num(p, 3)}"


def verdict(trend: dict) -> str:
    """Cautious wording of a trend based on both tests (OLS and Mann-Kendall)."""
    if trend["significant"]:
        return "significant"
    if trend["mk_p_value"] < 0.1:
        return "borderline significant"
    return "not significant"
