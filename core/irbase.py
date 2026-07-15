"""Shared base for IR nodes.

This is a deliberately dependency-light *leaf* module: it imports nothing from
the rest of ``core``. The per-node constraint/objective modules under
``core/constraints`` and ``core/objectives`` subclass :class:`_IRModel` from
here rather than from :mod:`core.ir`, so importing a single node module never
drags in (and never cycles through) the big ``core.ir`` module that assembles
the discriminated unions.
"""

from __future__ import annotations

import math
import uuid
from typing import ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

ProblemClassImpact = Literal["convex", "mip"]


def _new_id(prefix: str) -> str:
    """Short stable id, prefixed by constraint kind for readability in logs."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class _IRModel(BaseModel):
    # Numeric intent must be finite before it can become part of a confirmed
    # specification. This applies recursively to typed containers such as
    # PortfolioSpec.current_weights as well as to scalar objective/constraint
    # fields.
    model_config = ConfigDict(
        extra="forbid",
        frozen=False,
        allow_inf_nan=False,
    )


class _ConstraintIRModel(_IRModel):
    """Shared metadata for constraint-like IR nodes.

    Objectives deliberately inherit :class:`_IRModel` directly: elasticity is
    a feasibility-repair concern and must never become part of an objective's
    public schema. ``TransactionCost`` also uses this base for a uniform
    constraint union, but keeps ``elastic_default = False`` because it is an
    objective penalty and cannot make a problem infeasible.

    ``slack_scale`` is the node's natural (possibly dynamic) unit. Diagnosis
    minimizes *relative* slack, rather than comparing unlike raw units such as
    portfolio weight and CVaR. Consumers should use
    :meth:`natural_slack_scale`, which guarantees a finite positive divisor:
    degenerate zero-width/zero-bound metadata falls back to ``1.0``.
    """

    elastic: bool | None = Field(
        default=None,
        description=(
            "Optional one-way elasticity override. False makes a normally relaxable "
            "constraint hard; True cannot soften a structurally hard constraint."
        ),
    )

    elastic_default: ClassVar[bool] = False
    slack_scale: ClassVar[float | None] = None
    big_m: ClassVar[float | None] = None

    @model_validator(mode="after")
    def _reject_softening_hard_constraints(self) -> Self:
        if self.elastic is True and not type(self).elastic_default:
            raise ValueError(
                f"{type(self).__name__} is never elastic; elastic=True cannot make "
                "a structural constraint or objective penalty relaxable."
            )
        return self

    @property
    def elasticity_supported(self) -> bool:
        """Whether this particular node instance has a complete relaxation.

        Most node types support all of their rows whenever ``elastic_default``
        is true.  A compound node may override this property when only a
        subset of its shapes has a sound elastic formulation and repair.
        """

        return type(self).elastic_default

    @property
    def effective_elastic(self) -> bool:
        """Whether diagnosis may relax this instance.

        The override is intentionally one-way: ``False`` can lock a normally
        soft-able node, while ``True`` never promotes a hard node.
        """
        return self.elasticity_supported and self.elastic is not False

    @property
    def is_elastic(self) -> bool:
        """Effective elasticity alias used by the diagnosis compiler."""
        return self.effective_elastic

    def natural_slack_scale(self) -> float:
        """Return a finite positive scale suitable for slack normalization."""
        raw_scale = self.slack_scale
        if raw_scale is None or not math.isfinite(raw_scale) or raw_scale <= 0.0:
            return 1.0
        return float(raw_scale)

    def diagnostic_big_m(self, universe_size: int | None = None) -> float | None:
        """Return the bound used to make this node vacuous during diagnosis.

        Most nodes have a static bound or require problem data and therefore
        expose ``None`` here. ``Cardinality`` overrides this helper because its
        vacuous count cap is the universe size.
        """
        del universe_size
        return self.big_m
