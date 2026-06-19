"""Shared base for IR nodes.

This is a deliberately dependency-light *leaf* module: it imports nothing from
the rest of ``core``. The per-node constraint/objective modules under
``core/constraints`` and ``core/objectives`` subclass :class:`_IRModel` from
here rather than from :mod:`core.ir`, so importing a single node module never
drags in (and never cycles through) the big ``core.ir`` module that assembles
the discriminated unions.
"""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict

ProblemClassImpact = Literal["convex", "mip"]


def _new_id(prefix: str) -> str:
    """Short stable id, prefixed by constraint kind for readability in logs."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class _IRModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)
