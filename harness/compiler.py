"""
Compiler — the clean public step from a validated spec to a playable world.

Thin by design: the heavy lifting lives in engine/world.py (hybrid runtime) and
engine/gridlogic.py (semantics). This module is the named seam in the pipeline
  text -> generator -> [compiler] -> world
and the place to attach the executable predicate program.
"""

from __future__ import annotations

from harness.dsl.schema import EnvSpec
from harness.engine.world import World


def compile_spec(spec: EnvSpec) -> World:
    """spec -> runtime World (grid logic + pymunk props + predicate tracking)."""
    return World(spec)


def predicate_program(spec: EnvSpec) -> str:
    """Human-readable form of the code-level objective (for READMEs / logs)."""
    parts = []
    for p in spec.objective:
        if p.kind == "holding":
            parts.append(f"holding({p.item})")
        elif p.kind == "reached_exit":
            parts.append("agent.at(exit)")
        elif p.kind == "at_start":
            parts.append("agent.at(start)")
        elif p.kind == "item_at":
            parts.append(f"{p.item}.at({tuple(p.cell)})")
        elif p.kind == "collected_all_coins":
            parts.append("all_coins_collected()")
    return " AND ".join(parts)
