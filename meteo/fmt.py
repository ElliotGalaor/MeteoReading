"""Number formatting shared by the report and the dashboard (decimal comma in French, see meteo/i18n.py)."""
from __future__ import annotations

from .i18n import NNBSP, lang

MINUS = "−"


def num(value, digits: int = 1) -> str:
    """0.3315 -> '0.3' ; 25026059 -> '25,026,059' (typographic minus sign); in French '0,3' and '25 026 059'."""
    if value is None:
        return "–"
    text = f"{round(value):,}" if isinstance(value, int) or digits == 0 else f"{value:,.{digits}f}"
    if lang() == "fr":
        text = text.replace(",", NNBSP).replace(".", ",")
    return text.replace("-", MINUS)


def signed(value, digits: int = 1) -> str:
    """Signed number: +0.3 / −1.2."""
    if value is None:
        return "–"
    return ("+" if value >= 0 else "") + num(value, digits)


def percent(value, digits: int = 0, sign: bool = False) -> str:
    """A value already in percent: 42 -> '42%' ('42 %' in French), with its sign if `sign`."""
    if value is None:
        return "–"
    return (signed if sign else num)(value, digits) + (NNBSP + "%" if lang() == "fr" else "%")


def pct(share, digits: int = 0) -> str:
    """A share: 0.42 -> '42%' ('42 %' in French)."""
    return "–" if share is None else percent(share * 100, digits)


def pvalue(p) -> str:
    return "p < " + num(0.001, 3) if p < 0.001 else f"p = {num(p, 3)}"


def verdict(trend: dict) -> str:
    """Cautious wording of a trend based on both tests (OLS and Mann-Kendall)."""
    if trend["significant"]:
        return "significant"
    if trend["mk_p_value"] < 0.1:
        return "borderline significant"
    return "not significant"
