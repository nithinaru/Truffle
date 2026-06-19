"""Sprint 3 convex constraint nodes.

Each module here follows the Box template: a typed IR model (unique ``id``,
``problem_class_impact = "convex"``, field validation) plus a ``build(node, ctx)``
function the compiler dispatches to. All nodes preserve convexity — no
mixed-integer work (cardinality is Sprint 4).
"""
