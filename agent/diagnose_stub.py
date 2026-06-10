"""Placeholder for Sprint 3's elastic-relaxation infeasibility diagnoser.

When the solver reports infeasible, BLUEPRINT §7 specifies an elastic
relaxation pass that identifies the minimal conflicting set of constraints
and proposes plain-English repair options. That is Sprint 3 scope.

Until then, this stub raises NotImplementedError, which the chat loop
catches and degrades to the typed InfeasibleError message — clean for the
user, clearly labeled in code as a TODO.
"""

from __future__ import annotations

from core.compiler import CompiledProblem


def diagnose_infeasibility(compiled: CompiledProblem) -> str:
    """TODO(Sprint 3): elastic relaxation per BLUEPRINT §7.

    Will return a plain-language conflict report identifying the minimal
    set of conflicting constraints and proposing repair options. For now,
    raises NotImplementedError so the chat loop falls back to the typed
    error message.
    """
    raise NotImplementedError(
        "Elastic-relaxation infeasibility diagnosis is Sprint 3 (BLUEPRINT §7)."
    )
