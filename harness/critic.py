"""
World-model critic — code-truth as an automated critic for neural world-model "dreams".

GI's founders authored DIAMOND, a diffusion world model that predicts future frames. A world
model can hallucinate: an object teleports, a wall is walked through, an item is "held" that was
never reached, a door opens with no key. A VLM or a learned reward model judging pixels cannot
reliably catch these — but the code-defined environment CAN, exactly and for free.

Given a rollout (a sequence of engine states — e.g. decoded from a world model's predicted
frames), this critic asks of every transition: *is there a single legal action, under the real
grid rules, that produces it?* If not, the transition is flagged as hallucinated, with the rule
it broke. This turns the same gridlogic that verifies solvability into a dynamics critic — a
direct, on-thesis use of code-truth for world-model training/eval.

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


# ── illustration ──────────────────────────────────────────────────────────────

def _demo():
    from harness import fixtures as F
    from harness.dsl.schema import EnvSpec
    from harness.verifier import solve

    spec = EnvSpec(**F.key_crate_return())
    level = G.build_level(spec)
    plan, _ = solve(level, spec.objective)
    real = rollout_from_plan(level, plan)

    print("code-truth as a world-model critic — scene:", spec.name)
    print(f"  FAITHFUL rollout ({len(real)} states): consistency = {score(level, real):.0%}, "
          f"violations = {len(critique(level, real))}")

    # forge a 'hallucinated' rollout a world model might dream: two illegal transitions
    dreamed = list(real)
    # (a) teleport the agent 4 cells through a wall
    tp = G.GridState(agent=(real[3].agent[0] + 4, real[3].agent[1]), held=real[3].held, crates=real[3].crates)
    dreamed[4] = tp
    # (b) an ungrounded pickup: the can is suddenly held with no approach
    hall = G.GridState(agent=dreamed[6].agent, held=dreamed[6].held | {"can1"}, crates=dreamed[6].crates)
    dreamed[7] = hall

    viols = critique(level, dreamed)
    print(f"  HALLUCINATED rollout: consistency = {score(level, dreamed):.0%}, "
          f"violations = {len(viols)} (a VLM/reward-model on pixels would likely miss these):")
    for v in viols:
        print(f"    - step {v.step}: {v.reason}")


if __name__ == "__main__":
    _demo()
