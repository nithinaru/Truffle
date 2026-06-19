"""Numeral-grounding for LLM-generated explanations.

The single hard rule for explain (BLUEPRINT §6):

    *Every number in the explanation must appear in the SolutionReport.*

This module enforces that rule with a regex-extract + multi-rendering match.
We extract every numeral from the candidate explanation and check that each
can be reconciled with at least one renderable form of at least one report
value.

Allowed renderings for a single report value ``v``:

* Raw: ``v`` rounded to 1–6 significant figures.
* Percent: ``v * 100`` rounded similarly.
* Basis points: ``v * 10000`` rounded to integer or one decimal.

Plus a *small-integer allowlist* for counts that are demonstrably in the
report (``n_assets``, ``nonzero_names``), counts of binding constraints,
and the integers 0..12 (which appear constantly in English like "five
constraints" but are bounded and unlikely to be mistaken for material
quantities).

If verification fails, the chat loop calls :func:`explain` once more with
the offending numerals named, then falls back to a deterministic template
summary (built in :mod:`agent.explain`) so the user never sees an
ungrounded narration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from core.report import SolutionReport

# Numeric token: optional sign, digits with optional thousands commas and
# optional decimal, optional %/bps suffix. Trailing $ prefix is also caught.
_NUM_RE = re.compile(
    r"""
    (?P<lead>[-+]?)
    (?:\$\s*)?                       # $ prefix
    (?P<int>\d{1,3}(?:,\d{3})+|\d+)  # integer (with optional thousands commas)
    (?:\.(?P<frac>\d+))?             # optional fractional part
    (?:\s*(?P<suffix>%|bps))?        # optional suffix
    """,
    re.VERBOSE,
)

SMALL_INT_ALLOWLIST: set[int] = set(range(0, 13))


@dataclass(frozen=True)
class GroundingResult:
    """Outcome of one verification pass."""

    ok: bool
    unmatched: list[str] = field(default_factory=list)


def _extract_numerals(text: str) -> list[tuple[float, str | None]]:
    """Return ``(value, suffix)`` pairs found in ``text``.

    ``value`` is the raw numeric value with any thousands commas removed.
    ``suffix`` is one of ``'%'``, ``'bps'``, or ``None``. The suffix matters
    because "0.04" and "4%" should both resolve to the report value 0.04
    via *different* rendering candidates.
    """
    out: list[tuple[float, str | None]] = []
    for m in _NUM_RE.finditer(text):
        int_part = m.group("int").replace(",", "")
        frac = m.group("frac")
        raw = f"{m.group('lead') or ''}{int_part}" + (f".{frac}" if frac else "")
        try:
            value = float(raw)
        except ValueError:
            continue
        out.append((value, m.group("suffix")))
    return out


def _candidate_values(report: SolutionReport) -> list[float]:
    """All numeric quantities the explanation is allowed to reference."""
    vals: list[float] = [report.objective_value, report.solve_time_ms]
    if report.var is not None:
        vals.append(report.var)
    if report.optimality_gap is not None:
        vals.append(report.optimality_gap)
    vals.extend(report.weights.values())
    for b in report.binding:
        vals.append(b.shadow_price)
    return vals


def _renderings(value: float) -> set[str]:
    """Generate the set of string renderings that count as 'matches' for ``value``.

    We produce both rounded and percent/bps forms at multiple precisions so
    typical LLM phrasings (``"3.94%"``, ``"0.04"``, ``"4 bps"``) all resolve
    to the same underlying number.
    """
    candidates: set[float] = set()
    for v in (value, value * 100.0, value * 10000.0):
        for digits in range(0, 5):
            candidates.add(round(v, digits))
        for sig in range(1, 5):
            if v != 0.0:
                from math import floor, log10  # noqa: PLC0415

                dec = sig - int(floor(log10(abs(v)))) - 1
                candidates.add(round(v, dec))
            else:
                candidates.add(0.0)
    return {f"{c:g}" for c in candidates}


def _matches_any(
    numeral: float,
    report_values: list[float],
    suffix: str | None,
    tol_rel: float = 5e-3,
    tol_abs: float = 5e-4,
) -> bool:
    """Is ``numeral`` (with optional %/bps suffix) within tolerance of any
    raw/percent/bps rendering of any report value?
    """
    scaled_candidates: list[float] = []
    for v in report_values:
        scaled_candidates.extend([v, v * 100.0, v * 10000.0])

    # The suffix narrows what the numeral encodes:
    #   "%"   means the writer meant value/100; compare against (raw * 100).
    #   "bps" means the writer meant value/10000; compare against (raw * 10000).
    # Either way the raw token is `numeral`; we just compare to scaled values.
    for sv in scaled_candidates:
        if abs(sv - numeral) <= max(tol_abs, tol_rel * max(abs(sv), abs(numeral))):
            return True
    return suffix is None and abs(numeral) <= 12 and numeral == int(numeral)


def verify(explanation: str, report: SolutionReport) -> GroundingResult:
    """Check every numeral in ``explanation`` against ``report``.

    Returns:
        ``GroundingResult(ok=True)`` if every numeral matches something in
        the report (or is an allowed small integer). Otherwise ``ok=False``
        with ``unmatched`` populated; the chat loop uses that list to ask
        the model to regenerate.
    """
    numerals = _extract_numerals(explanation)
    if not numerals:
        return GroundingResult(ok=True)

    report_values = _candidate_values(report)
    allowed_counts: set[int] = {report.n_assets, report.nonzero_names, len(report.binding)}
    # The selected-name count ("13 selected names") is a real, in-report count
    # for MIP solves; it equals nonzero_names but we add it explicitly so the
    # conditional narration's count phrasing always grounds.
    if report.selected_names is not None:
        allowed_counts.add(len(report.selected_names))
    allowed_ints: set[int] = SMALL_INT_ALLOWLIST | allowed_counts

    unmatched: list[str] = []
    for value, suffix in numerals:
        # Small integer allowlist for counts/cardinal phrasings.
        if suffix is None and value == int(value) and int(value) in allowed_ints:
            continue
        if _matches_any(value, report_values, suffix):
            continue
        unmatched.append(f"{value:g}{suffix or ''}")
    return GroundingResult(ok=not unmatched, unmatched=unmatched)
