"""Numeral-grounding for LLM-generated explanations.

The single hard rule for explain (BLUEPRINT §6):

    *Every number in the explanation must appear in the SolutionReport.*

This module enforces that rule with a regex-extract + field-aware rendering
match. We extract every numeral from the candidate explanation, infer an
explicit nearby field label when one is present, and check that the numeral can
be reconciled with that report field. A variance claim therefore cannot borrow
an expected-return value, and a sensitivity claim cannot borrow a portfolio
weight merely because their rendered numerals happen to coincide.

Renderings are unit-aware. Every value permits its raw representation;
portfolio/return fractions also permit percent and basis-point forms, while
dimensionless ratios permit percent only. Variance, objective scores,
coefficients, runtimes, and sensitivity derivatives remain raw. Counts must be
present in the report; there is no global small-integer allowlist.

If verification fails, the chat loop calls :func:`explain` once more with
the offending numerals named, then falls back to a deterministic template
summary (built in :mod:`agent.explain`) so the user never sees an
ungrounded narration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from core.report import ConflictReport, GroundValue, SolutionReport

# Numeric token: optional sign/currency prefix, a conventional or leading-dot
# decimal (with optional thousands separators), optional scientific notation,
# and an optional %/bps suffix.  The boundaries are intentionally strict: a
# malformed token such as ``1,2`` must not be accepted as two independently
# grounded numbers, and ``.25`` must not silently become ``25``.
_NUM_RE = re.compile(
    r"""
    (?<![\w.,])
    (?P<lead>[-+]?)
    (?:\$\s*)?
    (?P<number>
        (?:
            \d{1,3}(?:,\d{3})+(?:\.\d*)?  # thousands separators
            | \d+(?:\.\d*)?                 # integer / conventional decimal
            | \.\d+                          # leading-dot decimal
        )
        (?:[eE][-+]?\d+)?                    # scientific notation
    )
    (?:\s*(?P<suffix>%|bps))?
    (?!\w)
    (?!,\d)
    """,
    re.IGNORECASE | re.VERBOSE,
)

@dataclass(frozen=True)
class GroundingResult:
    """Outcome of one verification pass."""

    ok: bool
    unmatched: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _SolutionValue:
    """One solution fact plus the renderings its unit permits."""

    value: float
    rendering: Literal["raw", "fraction", "ratio"] = "raw"
    fields: frozenset[str] = frozenset()


@dataclass(frozen=True)
class _NumeralToken:
    """One parsed numeral together with its source span."""

    value: float
    suffix: str | None
    start: int
    end: int


@dataclass(frozen=True)
class _FieldMention:
    """One explicit semantic field phrase found near a numeral."""

    field: str
    start: int
    end: int


# These phrases are deliberately about report fields, not general finance
# vocabulary. Unknown prose keeps the legacy global numeric check, while an
# explicit known label narrows matching to that field and fails closed when the
# report does not contain it.
_FIELD_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("cvar", re.compile(r"\b(?:cvar|conditional value at risk|expected shortfall)\b", re.I)),
    ("var", re.compile(r"\b(?:var|value at risk)\b", re.I)),
    ("tracking_error", re.compile(r"\btracking(?:[-\s]+error)(?:[-\s]+variance)?\b", re.I)),
    ("excess_return", re.compile(r"\bexcess(?:[-\s]+expected)?[-\s]+return\b", re.I)),
    ("expected_return", re.compile(r"\b(?:annualized[-\s]+)?expected[-\s]+return\b", re.I)),
    ("risk_contribution", re.compile(r"\brisk[-\s]+contribution(?:[-\s]+share)?\b", re.I)),
    ("transaction_cost", re.compile(r"\b(?:modeled[-\s]+)?transaction[-\s]+costs?\b", re.I)),
    ("variance", re.compile(r"\b(?:portfolio[-\s]+)?variance\b", re.I)),
    ("volatility", re.compile(r"\b(?:portfolio[-\s]+)?volatility\b", re.I)),
    ("sharpe", re.compile(r"\bsharpe(?:[-\s]+ratio)?\b", re.I)),
    ("turnover", re.compile(r"\b(?:l1[-\s]+)?turnover\b", re.I)),
    ("drawdown", re.compile(r"\b(?:maximum[-\s]+|max[-\s]+)?drawdown\b", re.I)),
    ("gap", re.compile(r"\b(?:relative[-\s]+|optimality[-\s]+)?gap\b", re.I)),
    (
        "sensitivity",
        re.compile(
            r"\b(?:sensitiv(?:ity|ities)|shadow[-\s]+prices?|"
            r"(?:objective[-\s]+)?derivatives?|dual(?:[-\s]+(?:value|multiplier))?s?)\b",
            re.I,
        ),
    ),
    ("parameter_scale", re.compile(r"\bparameter[-\s]+scale\b", re.I)),
    ("coefficient", re.compile(r"\bcoefficients?\b", re.I)),
    ("reconstruction_error", re.compile(r"\breconstruction[-\s]+error\b", re.I)),
    ("confidence", re.compile(r"\bconfidence(?:[-\s]+level)?\b", re.I)),
    (
        "annualization",
        re.compile(r"\b(?:annualization|periods?[-\s]*/[-\s]*year)\b", re.I),
    ),
    ("slack", re.compile(r"\bslacks?\b", re.I)),
    ("primal", re.compile(r"\bprimal(?:[-\s]+value)?\b", re.I)),
    ("bound", re.compile(r"\b(?:bounds?|caps?|floors?|limits?)\b", re.I)),
    (
        "objective",
        re.compile(r"\b(?:solver[-\s]+)?objective(?:[-\s]+(?:value|score))?\b", re.I),
    ),
    ("time", re.compile(r"\b(?:solve[-\s]+time|runtime|milliseconds?|ms)\b", re.I)),
    ("count", re.compile(r"\b(?:assets?|names?|nonzero[-\s]+positions?|constraints?|rows?)\b", re.I)),
    ("weight", re.compile(r"\b(?:portfolio[-\s]+)?weights?\b|\ballocation\b", re.I)),
)

_SENTENCE_BREAK_RE = re.compile(r"[!?;\n]+|\.(?=\s|$)")


def _extract_numeral_tokens(text: str) -> list[_NumeralToken]:
    """Return parsed numeral tokens without discarding their prose spans."""

    out: list[_NumeralToken] = []
    for match in _NUM_RE.finditer(text):
        number = match.group("number").replace(",", "")
        raw = f"{match.group('lead') or ''}{number}"
        try:
            value = float(raw)
        except ValueError:
            continue
        suffix = match.group("suffix")
        out.append(
            _NumeralToken(
                value=value,
                suffix=suffix.lower() if suffix is not None else None,
                start=match.start(),
                end=match.end(),
            )
        )
    return out


def _extract_numerals(text: str) -> list[tuple[float, str | None]]:
    """Return ``(value, suffix)`` pairs found in ``text``.

    ``value`` is the raw numeric value with any thousands commas removed.
    ``suffix`` is one of ``'%'``, ``'bps'``, or ``None``. The suffix matters
    because "0.04" and "4%" should both resolve to the report value 0.04
    via *different* rendering candidates.
    """
    return [(token.value, token.suffix) for token in _extract_numeral_tokens(text)]


def _unit_rendering(unit: str) -> Literal["raw", "fraction", "ratio"]:
    """Map an explicit report unit to permitted prose conversions."""

    if unit in {
        "fraction_per_year",
        "fraction_per_sqrt_year",
        "fraction_per_scenario_period",
        "portfolio_weight_fraction",
        "l1_weight_fraction",
        "portfolio_value_fraction",
        "scenario_loss_fraction",
        "annualized_volatility_fraction",
    }:
        return "fraction"
    if unit in {"dimensionless_ratio", "fraction_of_total_variance"}:
        return "ratio"
    return "raw"


def _semantic_field(key: str, label: str = "") -> str:
    """Return the canonical prose field for a typed report key.

    Stable report keys remain the source of truth. Labels are considered only
    to support custom/adapted report producers that retain the conventional
    human-readable names.
    """

    key_text = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    label_text = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    text = f"{key_text} {label_text}"

    # Order matters: the more specific fields must win over their substrings.
    if "cvar_var_threshold" in key_text or "var_threshold" in text:
        return "var"
    if "conditional_value_at_risk" in text or re.search(r"(?:^|_)cvar(?:_|$)", text):
        return "cvar"
    if "value_at_risk" in text:
        return "var"
    if "tracking_error" in text:
        return "tracking_error"
    if "excess_expected_return" in text or "excess_return" in text:
        return "excess_return"
    if "expected_return" in text:
        return "expected_return"
    if "risk_contribution" in text:
        return "risk_contribution"
    if "transaction_cost" in text:
        return "transaction_cost"
    if "mean_variance_score" in text:
        return "objective"
    if "variance" in text:
        return "variance"
    if "volatility" in text:
        return "volatility"
    if "sharpe" in text:
        return "sharpe"
    if "turnover" in text:
        return "turnover"
    if "drawdown" in text:
        return "drawdown"
    # Dynamic aliases use this same stable fallback, so unknown typed metrics
    # are still field-aware instead of reverting to a global value pool.
    fallback = key_text or label_text
    return f"reported:{fallback}"


def _solution_value(
    value: float,
    rendering: Literal["raw", "fraction", "ratio"] = "raw",
    *fields: str,
) -> _SolutionValue:
    return _SolutionValue(float(value), rendering, frozenset(fields))


def _legacy_objective_field(objective_kind: str) -> str | None:
    """Semantic field for legacy reports whose objective is itself a metric."""

    return {
        "min_variance": "variance",
        "min_cvar": "cvar",
        "min_tracking_error": "tracking_error",
    }.get(objective_kind)


def _candidate_values(report: SolutionReport) -> list[_SolutionValue]:
    """All typed numeric quantities the explanation may reference."""

    vals: list[_SolutionValue] = [
        _solution_value(report.objective_value, "raw", "objective"),
        _solution_value(report.solve_time_ms, "raw", "time"),
    ]
    if report.objective_decomposition is None:
        legacy_field = _legacy_objective_field(report.objective_kind)
        if legacy_field is not None:
            vals[0] = _solution_value(
                report.objective_value,
                "raw",
                "objective",
                legacy_field,
            )
    if report.var is not None:
        vals.append(_solution_value(report.var, "fraction", "var"))
    if report.optimality_gap is not None:
        vals.append(_solution_value(report.optimality_gap, "ratio", "gap"))
    vals.extend(
        _solution_value(value, "fraction", "weight") for value in report.weights.values()
    )

    if report.objective_decomposition is not None:
        decomposition = report.objective_decomposition
        vals.extend(
            [
                _solution_value(decomposition.solver_value, "raw", "objective"),
                _solution_value(
                    decomposition.reconstruction_error,
                    "raw",
                    "reconstruction_error",
                ),
            ]
        )
        for term in decomposition.terms:
            semantic_field = _semantic_field(term.key, term.label)
            vals.extend(
                [
                    _solution_value(
                        term.natural_value,
                        _unit_rendering(term.natural_unit),
                        semantic_field,
                    ),
                    _solution_value(term.coefficient, "raw", "coefficient"),
                    _solution_value(
                        term.objective_contribution,
                        "raw",
                        "objective",
                        semantic_field,
                    ),
                ]
            )
    for metric in report.metrics:
        semantic_field = _semantic_field(metric.key, metric.label)
        vals.append(
            _solution_value(
                metric.value,
                _unit_rendering(metric.unit),
                semantic_field,
            )
        )
        if metric.annualization_periods is not None:
            vals.append(
                _solution_value(
                    float(metric.annualization_periods),
                    "raw",
                    "annualization",
                )
            )
        if metric.confidence_level is not None:
            vals.append(
                _solution_value(metric.confidence_level, "ratio", "confidence")
            )
    for sensitivity in report.sensitivities:
        bound_rendering = _unit_rendering(sensitivity.bound_unit)
        vals.extend(
            [
                _solution_value(sensitivity.bound_value, bound_rendering, "bound"),
                _solution_value(sensitivity.primal_value, bound_rendering, "primal"),
                _solution_value(sensitivity.slack, bound_rendering, "slack"),
                _solution_value(sensitivity.raw_solver_dual, "raw", "sensitivity"),
                _solution_value(
                    sensitivity.parameter_scale,
                    "raw",
                    "parameter_scale",
                ),
                _solution_value(
                    sensitivity.objective_derivative_per_bound_unit,
                    "raw",
                    "sensitivity",
                ),
            ]
        )
    for b in report.binding:
        vals.append(_solution_value(b.shadow_price, "raw", "sensitivity"))
    return vals


def _alias_pattern(alias: str) -> re.Pattern[str] | None:
    """Compile an exact label/key alias with flexible separators."""

    words = re.findall(r"[a-z0-9]+", alias.lower())
    if not words:
        return None
    body = r"[-\s_:]+".join(re.escape(word) for word in words)
    return re.compile(rf"(?<!\w){body}(?!\w)", re.I)


def _field_mentions(text: str, report: SolutionReport) -> list[_FieldMention]:
    """Find static and report-provided semantic field labels in ``text``."""

    aliases: list[tuple[str, re.Pattern[str]]] = list(_FIELD_PATTERNS)
    for metric in report.metrics:
        semantic_field = _semantic_field(metric.key, metric.label)
        for alias in (metric.key, metric.label):
            pattern = _alias_pattern(alias)
            if pattern is not None:
                aliases.append((semantic_field, pattern))
    if report.objective_decomposition is not None:
        for term in report.objective_decomposition.terms:
            semantic_field = _semantic_field(term.key, term.label)
            for alias in (term.key, term.label):
                pattern = _alias_pattern(alias)
                if pattern is not None:
                    aliases.append((semantic_field, pattern))

    mentions: dict[tuple[str, int, int], _FieldMention] = {}
    for semantic_field, pattern in aliases:
        for match in pattern.finditer(text):
            mention = _FieldMention(semantic_field, match.start(), match.end())
            mentions[(semantic_field, mention.start, mention.end)] = mention
    return list(mentions.values())


def _sentence_bounds(text: str, token: _NumeralToken) -> tuple[int, int]:
    """Return the prose sentence containing ``token`` without splitting decimals."""

    left = 0
    right = len(text)
    for match in _SENTENCE_BREAK_RE.finditer(text):
        if match.end() <= token.start:
            left = match.end()
        elif match.start() >= token.end:
            right = match.start()
            break
    return left, right


def _field_for_token(
    text: str,
    token: _NumeralToken,
    mentions: list[_FieldMention],
) -> str | None:
    """Choose the nearest explicit field label in the numeral's sentence."""

    left, right = _sentence_bounds(text, token)
    local_mentions = [
        mention
        for mention in mentions
        if mention.start >= left and mention.end <= right
    ]
    if not local_mentions:
        return None

    def _rank(mention: _FieldMention) -> tuple[int, int, int]:
        if mention.end <= token.start:
            distance = token.start - mention.end
            intervening = text[mention.end : token.start]
            follows_numeral = 0
        elif mention.start >= token.end:
            distance = mention.start - token.end
            intervening = text[token.end : mention.start]
            follows_numeral = 1
        else:
            distance = 0
            intervening = ""
            follows_numeral = 0
        # A parenthetical field normally annotates a different value. Keep it
        # eligible (for "variance (0.04)"), but prefer a label on the same
        # side of the parenthesis (for "CVaR 0.02 (VaR 0.01)").
        distance += 1000 * sum(intervening.count(mark) for mark in "()")
        # Prefer a fuller phrase on a tie ("excess expected return" over
        # "expected return") and then the conventional label-before-value form.
        return distance, -(mention.end - mention.start), follows_numeral

    nearest = min(local_mentions, key=_rank)
    return nearest.field


def _conflict_values(report: ConflictReport) -> list[GroundValue]:
    """Typed values exposed to free-form infeasibility narration.

    Repair targets and elastic-solver internals are deliberately absent.  The
    model receives the same narrow surface in :mod:`agent.diagnose`; verified
    repairs are rendered separately from server-owned ``Repair`` objects.
    """

    values: list[GroundValue] = [
        GroundValue(
            key="n_assets", value=float(report.n_assets), unit="count", source="spec"
        )
    ]
    for evidence in report.evidence:
        values.extend(evidence.values)
    for member in report.conflict_set:
        values.extend(member.parameters)
    return values


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
    report_values: list[_SolutionValue],
    suffix: str | None,
    field: str | None = None,
    tol_rel: float = 5e-3,
    tol_abs: float = 5e-4,
) -> bool:
    """Is ``numeral`` (with optional %/bps suffix) within tolerance of any
    raw/percent/bps rendering of the applicable report field?
    """
    def _close(expected: float) -> bool:
        return abs(expected - numeral) <= max(
            tol_abs,
            tol_rel * max(abs(expected), abs(numeral)),
        )

    candidates = [
        candidate
        for candidate in report_values
        if field is None or field in candidate.fields
    ]
    matched = False
    for candidate in candidates:
        if suffix is None:
            expected = candidate.value
        elif suffix == "%" and candidate.rendering in {"fraction", "ratio"}:
            expected = candidate.value * 100.0
        elif suffix == "bps" and candidate.rendering == "fraction":
            expected = candidate.value * 10000.0
        else:
            continue
        if _close(expected):
            matched = True
            break
    if not matched:
        return False

    # When prose contains no recognized field label, preserve the legacy
    # fail-closed behavior for a suffixed value shared by incompatible units.
    # Explicit field labels already narrow ``candidates`` and remove that
    # ambiguity (e.g. expected return and variance can both equal 0.08).
    if suffix is not None and field is None:
        raw_value = numeral / (100.0 if suffix == "%" else 10000.0)
        allowed_renderings = {"fraction", "ratio"} if suffix == "%" else {"fraction"}
        for candidate in report_values:
            same_value = abs(candidate.value - raw_value) <= max(
                tol_abs,
                tol_rel * max(abs(candidate.value), abs(raw_value)),
            )
            if same_value and candidate.rendering not in allowed_renderings:
                return False
    return True


def _matches_ground_value(
    numeral: float,
    suffix: str | None,
    candidate: GroundValue,
    *,
    tol_rel: float = 5e-3,
    tol_abs: float = 5e-4,
) -> bool:
    """Unit-aware match for a conflict-report value.

    Unlike legacy solution grounding, a count cannot authorize a percentage
    and a fraction cannot authorize an arbitrary raw/bps rendering.  This is
    important for repair targets: ``34 names`` and ``34%`` are different facts.
    """

    expected: float
    if candidate.unit == "fraction":
        if suffix == "%":
            expected = candidate.value * 100.0
        elif suffix == "bps":
            expected = candidate.value * 10000.0
        else:
            expected = candidate.value
    elif candidate.unit == "bps":
        if suffix not in {None, "bps"}:
            return False
        expected = candidate.value
    elif candidate.unit == "count":
        if suffix is not None or numeral != int(numeral):
            return False
        expected = candidate.value
    else:  # raw / milliseconds
        if suffix is not None:
            return False
        expected = candidate.value
    return abs(expected - numeral) <= max(
        tol_abs, tol_rel * max(abs(expected), abs(numeral))
    )


def _verify_conflict(explanation: str, report: ConflictReport) -> GroundingResult:
    numerals = _extract_numerals(explanation)
    if not numerals:
        return GroundingResult(ok=True)
    candidates = _conflict_values(report)
    unmatched: list[str] = []
    for value, suffix in numerals:
        if any(_matches_ground_value(value, suffix, candidate) for candidate in candidates):
            continue
        unmatched.append(f"{value:g}{suffix or ''}")
    return GroundingResult(ok=not unmatched, unmatched=unmatched)


def verify(explanation: str, report: SolutionReport | ConflictReport) -> GroundingResult:
    """Check every numeral in ``explanation`` against ``report``.

    Returns:
        ``GroundingResult(ok=True)`` if every numeral matches something in
        the report. Otherwise ``ok=False`` with ``unmatched`` populated; the
        chat loop uses that list to ask the model to regenerate.
    """
    if not explanation.strip():
        return GroundingResult(ok=False, unmatched=["<blank>"])

    if isinstance(report, ConflictReport):
        return _verify_conflict(explanation, report)

    numerals = _extract_numeral_tokens(explanation)
    if not numerals:
        return GroundingResult(ok=True)

    report_values = _candidate_values(report)
    mentions = _field_mentions(explanation, report)
    allowed_counts: set[int] = {report.n_assets, report.nonzero_names, len(report.binding)}
    # The selected-name count ("13 selected names") is a real, in-report count
    # for MIP solves; it equals nonzero_names but we add it explicitly so the
    # conditional narration's count phrasing always grounds.
    if report.selected_names is not None:
        allowed_counts.add(len(report.selected_names))
    allowed_ints = allowed_counts

    unmatched: list[str] = []
    for token in numerals:
        value = token.value
        suffix = token.suffix
        semantic_field = _field_for_token(explanation, token, mentions)
        # Reported integers authorize only an unlabeled numeral or an explicit
        # count phrase, never a claim about another semantic field.
        if (
            semantic_field in {None, "count"}
            and suffix is None
            and value == int(value)
            and int(value) in allowed_ints
        ):
            continue
        if _matches_any(value, report_values, suffix, semantic_field):
            continue
        unmatched.append(f"{value:g}{suffix or ''}")
    return GroundingResult(ok=not unmatched, unmatched=unmatched)
