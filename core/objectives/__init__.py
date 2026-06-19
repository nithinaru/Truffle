"""Sprint 3 convex objective nodes.

* ``MinTrackingError`` — a plain QP in ``w``; contributes a base objective
  expression like min-variance does.
* ``MaxSharpe`` — non-convex as stated, made convex by the Charnes–Cooper
  change of variables; builds its own transformed problem.
* ``RiskParity`` — equal risk contribution via the convex log-barrier
  surrogate; builds its own problem and normalizes the result.

The transformed objectives (``MaxSharpe``, ``RiskParity``) return a fully built
``CompiledProblem`` with a ``weight_recovery`` hook, because they optimize over
a transformed variable rather than ``w`` directly.
"""
