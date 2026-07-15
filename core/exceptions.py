"""Typed exceptions for the Truffle solver pipeline.

These are designed to be surfaced to the user (CLI / future agent tool output)
verbatim. Messages should be structured: state *what* failed, *which* spec
element caused it, and *what to try*. Avoid stack-trace-only diagnostics.
"""

from __future__ import annotations


class TruffleError(Exception):
    """Base class for all Truffle errors."""


class SpecValidationError(TruffleError):
    """Raised when a PortfolioSpec is structurally or semantically invalid."""


class CompilationError(TruffleError):
    """Raised when the IR cannot be turned into a well-formed CVXPY problem."""


class SolverError(TruffleError):
    """Raised when the solver fails for a reason other than infeasibility/unboundedness."""


class InfeasibleError(SolverError):
    """Raised when the solver reports the problem is infeasible.

    Callers may opt into diagnosis, in which case ``conflict_report`` carries
    the deterministic elastic/IIS result and verified repair candidates.
    """

    def __init__(self, message: str, *, conflict_report: object | None = None) -> None:
        super().__init__(message)
        self.conflict_report = conflict_report


class UnboundedError(SolverError):
    """Raised when the solver reports the problem is unbounded."""


class DualsUnavailableError(TruffleError):
    """Raised when dual values are requested but the problem has not been solved
    (or was solved with a method that does not produce duals, e.g. MIP)."""


class ParseFailedError(TruffleError):
    """Raised when the LLM cannot produce a valid ParseResult even after
    structured-output repair attempts. The chat loop catches this and asks
    the user to rephrase rather than crashing the session."""


class GroundingFailedError(TruffleError):
    """Raised when an LLM-generated explanation contains numerals that are
    not present in the SolutionReport, even after one repair attempt. The
    chat loop falls back to a deterministic template summary."""


class DiagnosisError(TruffleError):
    """Raised when diagnosis is inapplicable or cannot prove a safe result."""
