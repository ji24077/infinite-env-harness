"""
Rollout-legality checker — a proof-of-concept / DIRECTION, not a proven capability.

The idea: the same gridlogic that proves solvability can also check whether a rollout obeys the
environment's rules. Given a sequence of engine STATES, it asks of every transition — *is there a
single legal action, under the real grid rules, that produces it?* — and flags the ones with no
legal action (teleports, wall phasing, ungrounded pickups, no-push crate moves).

Motivation: GI's founders authored DIAMOND, a diffusion world model; such models hallucinate
dynamics that a per-frame VLM / pixel reward-model reads as plausible. Code-truth can catch them.

HONEST SCOPE (read before believing the pitch):
  * The illustration (_demo / forge_hallucination) *injects* the very corruptions critique()
    detects — it demonstrates the mechanism, it does not discover anything on its own.
  * It operates on discrete engine STATE, not pixels. Catching a real DIAMOND-class model's
    pixel-space hallucinations would require decoding predicted frames back to state first.
So: a direction worth building, illustrated concretely — not a load-bearing result.

Run the illustration:  uv run python -m harness.critic
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from harness.engine import gridlogic as G


@dataclass
class Violation:
    step: int
    frm: G.GridState
    to: G.GridState
    reason: str


def rollout_from_plan(level: G.Level, plan: List[str]) -> List[G.GridState]:
    """Replay a legal action sequence into the state sequence a faithful world model should predict."""
    st = G.initial_state(level)
    states = [st]
    for a in plan:
        nxt = G.step(level, st, a)
        if nxt is None:
            break
        st = nxt
        states.append(st)
    return states


def critique(level: G.Level, states: List[G.GridState]) -> List[Violation]:
    """Flag every transition not reachable by a single legal action (a no-op counts as legal).
    Sound: it never flags a physically legal transition; it catches teleports, wall phasing,
    ungrounded pickups, key-less door passages, and illegal crate moves."""
    out: List[Violation] = []
    for i in range(len(states) - 1):
        s, n = states[i], states[i + 1]
        if n == s:
            continue  # world model predicted "no change" — always legal
        if not any(G.step(level, s, a) == n for a in G.ACTIONS):
            out.append(Violation(i, s, n, _diagnose(level, s, n)))
    return out


def _diagnose(level: G.Level, s: G.GridState, n: G.GridState) -> str:
    ax, ay = s.agent; bx, by = n.agent
    if abs(ax - bx) + abs(ay - by) > 1:
        return f"agent teleported {ax,ay}->{bx,by} (>1 cell in one step)"
    if (bx, by) != (ax, ay) and level.blocked_tile(bx, by):
        return f"agent moved into a wall/hazard at {bx,by}"
    gained = n.held - s.held
    if gained:
        return f"held {sorted(gained)} appeared without a grounded pickup"
    if n.crates != s.crates:
        return "a crate moved without a valid push"
    return "no single legal action produces this transition"


def score(level: G.Level, states: List[G.GridState]) -> float:
    """Fraction of transitions that are physically legal (1.0 = fully consistent dream)."""
    trans = max(1, len(states) - 1)
    return round(1.0 - len(critique(level, states)) / trans, 3)


def forge_hallucination(level: G.Level, real: List[G.GridState]) -> List[G.GridState]:
    """Corrupt a faithful rollout the way a neural world model might dream — a DIVERSE set of
    physically-impossible transitions (teleport, wall phasing, ungrounded pickup, illegal crate
    move). Every one is invisible to a per-frame VLM but caught exactly by the critic."""
    d = list(real)
    if len(d) > 8:
        ax, ay = d[3].agent
        d[4] = G.GridState(agent=(ax + 4, ay), held=d[3].held, crates=d[3].crates)      # teleport
        # ungrounded pickup: the can is suddenly "held" with no approach
        d[7] = G.GridState(agent=d[6].agent, held=d[6].held | {"can1"}, crates=d[6].crates)
        # illegal crate move: a crate jumps with no push
        if d[8].crates:
            cid, cx, cy = d[8].crates[0]
            moved = tuple(sorted([(cid, cx, cy + 3)] + list(d[8].crates[1:])))
            d[9] = G.GridState(agent=d[8].agent, held=d[8].held, crates=moved)
    return d


# ── illustration ──────────────────────────────────────────────────────────────

def _demo():
    from harness import fixtures as F
    from harness.dsl.schema import EnvSpec
    from harness.verifier import solve

    spec = EnvSpec(**F.key_crate_return())
    level = G.build_level(spec)
    plan, _ = solve(level, spec.objective)
    real = rollout_from_plan(level, plan)

    print("rollout-legality checker (PoC; violations injected) — scene:", spec.name)
    print(f"  FAITHFUL rollout ({len(real)} states): consistency = {score(level, real):.0%}, "
          f"violations = {len(critique(level, real))}")

    dreamed = forge_hallucination(level, real)
    viols = critique(level, dreamed)
    print(f"  HALLUCINATED rollout: consistency = {score(level, dreamed):.0%}, "
          f"violations = {len(viols)} (a VLM / pixel reward-model would likely miss these):")
    for v in viols:
        print(f"    - step {v.step}: {v.reason}")


if __name__ == "__main__":
    _demo()
