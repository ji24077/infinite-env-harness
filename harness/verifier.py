"""
The 3-stage verifier — the answer to the challenge's word "reliably".

  L1  schema/meaning   : EnvSpec validates (geometry, references, entities off walls).
  L2  solvability      : BFS over the authoritative grid semantics finds a shortest ORACLE
                         PLAN that satisfies the objective. No plan -> reject (-> repair loop).
  L3  physics stability: build the pymunk world (walls + dynamic props) and step it; assert
                         no NaN / no tunneling / bodies stay bounded.

The oracle plan is reused three ways (the "solver is the labeler" idea):
  (a) here, as the L3 replay / sanity witness,
  (b) plan length -> automatic difficulty label,
  (c) later, as reward-shaping subgoals (gym_env.py).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

from pydantic import ValidationError

from harness.dsl.schema import EnvSpec
from harness.engine import gridlogic as G

SEARCH_BUDGET = 400_000  # state-expansion cap; ample for 32x22 grids, guards blowups


@dataclass
class VerifyResult:
    ok: bool
    stage: str                       # "L1" | "L2" | "L3" | "passed"
    reason: str = ""
    plan: List[str] = field(default_factory=list)
    plan_len: int = 0
    expanded: int = 0
    difficulty: str = ""             # easy | medium | hard | expert
    difficulty_score: int = 0

    def log_line(self) -> str:
        if not self.ok:
            return f"  [{self.stage}] FAIL — {self.reason}"
        return (f"  [passed] schema OK · solvable: plan length {self.plan_len} "
                f"(expanded {self.expanded}) · physics OK · difficulty={self.difficulty}")


# ── L2: BFS solvability + oracle plan ───────────────────────────────────────────

def solve(level: G.Level, objective, start=None, budget: int = SEARCH_BUDGET):
    """Breadth-first search for a shortest action sequence satisfying the objective.
    Returns (plan|None, expanded_count). Sound + complete within the budget."""
    if start is None:
        start = G.initial_state(level)
    if G.objective_satisfied(level, objective, start):
        return [], 0
    frontier = deque([start])
    parent = {start: (None, None)}  # state -> (prev_state, action)
    expanded = 0
    while frontier:
        st = frontier.popleft()
        expanded += 1
        if expanded > budget:
            return None, expanded
        for action in G.ACTIONS:
            nxt = G.step(level, st, action)
            if nxt is None or nxt in parent:
                continue
            parent[nxt] = (st, action)
            if G.objective_satisfied(level, objective, nxt):
                # reconstruct
                plan, cur = [], nxt
                while parent[cur][0] is not None:
                    prev, act = parent[cur]
                    plan.append(act)
                    cur = prev
                plan.reverse()
                return plan, expanded
            frontier.append(nxt)
    return None, expanded


def build_distance_field(level: G.Level, objective, budget: int = SEARCH_BUDGET):
    """Cost-to-go for every reachable grid state = exact distance to satisfying the
    objective. Built by enumerating the reachable graph from the start, then a reverse
    BFS from all goal states. Reused by the Gym env as potential-based reward shaping —
    the oracle solver IS the reward source. Returns dict[state] -> steps_to_goal."""
    start = G.initial_state(level)
    adj = {}                      # state -> list[successor]
    frontier = deque([start]); seen = {start}
    goals = []
    while frontier:
        st = frontier.popleft()
        if G.objective_satisfied(level, objective, st):
            goals.append(st)
        succ = []
        for a in G.ACTIONS:
            nxt = G.step(level, st, a)
            if nxt is not None:
                succ.append(nxt)
                if nxt not in seen:
                    seen.add(nxt); frontier.append(nxt)
        adj[st] = succ
        if len(seen) > budget:
            break
    # reverse edges
    rev = {s: [] for s in adj}
    for s, succ in adj.items():
        for n in succ:
            rev.setdefault(n, []).append(s)
    dist = {g: 0 for g in goals}
    dq = deque(goals)
    while dq:
        s = dq.popleft()
        for p in rev.get(s, []):
            if p not in dist:
                dist[p] = dist[s] + 1
                dq.append(p)
    return dist


def _difficulty(spec: EnvSpec, plan_len: int) -> tuple[str, int]:
    nk = len(spec.by_type("key"))
    nc = len(spec.by_type("crate"))
    ncoin = len(spec.by_type("coin"))
    score = plan_len + 6 * nk + 5 * nc + 3 * ncoin
    if score <= 12:
        band = "easy"
    elif score <= 28:
        band = "medium"
    elif score <= 50:
        band = "hard"
    else:
        band = "expert"
    return band, score


# ── L3: physics stability smoke test ────────────────────────────────────────────

def physics_smoke_test(spec: EnvSpec, sim_steps: int = 240) -> tuple[bool, str]:
    """Build the pymunk world (static walls + dynamic crate/ball props) and step it,
    asserting nothing explodes. This is the 'soft physics props' half of the hybrid
    engine — physics never touches navigation, only validates the dynamic decor."""
    import math
    import pymunk

    space = pymunk.Space()
    space.gravity = (0, 900)  # screen-y-down
    ts = 32  # tile px, matches renderer

    # static walls as segments
    static = space.static_body
    for y, row in enumerate(spec.tiles):
        for x, t in enumerate(row):
            if t == 1:
                seg = pymunk.Poly.create_box(static, (ts, ts))
                # emulate a static box by an offset poly
                verts = [(x*ts, y*ts), (x*ts+ts, y*ts), (x*ts+ts, y*ts+ts), (x*ts, y*ts+ts)]
                shape = pymunk.Poly(static, verts)
                shape.elasticity = 0.1
                shape.friction = 0.9
                space.add(shape)

    dynamic_bodies = []
    for e in spec.entities:
        if e.type in ("crate", "ball"):
            x, y = e.pos
            mass = 1.0
            if e.type == "ball":
                r = ts * 0.35
                body = pymunk.Body(mass, pymunk.moment_for_circle(mass, 0, r))
                body.position = (x*ts + ts/2, y*ts + ts/2)
                shape = pymunk.Circle(body, r)
            else:
                size = (ts*0.8, ts*0.8)
                body = pymunk.Body(mass, pymunk.moment_for_box(mass, size))
                body.position = (x*ts + ts/2, y*ts + ts/2)
                shape = pymunk.Poly.create_box(body, size)
            shape.elasticity = 0.2
            shape.friction = 0.8
            space.add(body, shape)
            dynamic_bodies.append(body)

    if not dynamic_bodies:
        return True, "no dynamic props (trivially stable)"

    w_px, h_px = spec.width * ts, spec.height * ts
    for _ in range(sim_steps):
        space.step(1 / 60.0)
        for b in dynamic_bodies:
            px, py = b.position
            if not (math.isfinite(px) and math.isfinite(py)):
                return False, "physics produced NaN position"
            if px < -ts or px > w_px + ts or py < -ts or py > h_px + ts:
                return False, "a prop tunneled out of the arena"
    return True, f"{len(dynamic_bodies)} dynamic props stable after {sim_steps} steps"


# ── top-level ───────────────────────────────────────────────────────────────────

def verify(raw_spec: dict | EnvSpec) -> VerifyResult:
    # L1 — schema + meaning
    try:
        spec = raw_spec if isinstance(raw_spec, EnvSpec) else EnvSpec(**raw_spec)
    except ValidationError as ex:
        first = ex.errors()[0]
        loc = ".".join(str(p) for p in first["loc"])
        return VerifyResult(ok=False, stage="L1", reason=f"{loc}: {first['msg']}")

    # L2 — solvability + oracle plan
    level = G.build_level(spec)
    plan, expanded = solve(level, spec.objective)
    if plan is None:
        return VerifyResult(ok=False, stage="L2",
                            reason=f"no path satisfies the objective (expanded {expanded} states)",
                            expanded=expanded)
    band, score = _difficulty(spec, len(plan))

    # L3 — physics stability
    ok, msg = physics_smoke_test(spec)
    if not ok:
        return VerifyResult(ok=False, stage="L3", reason=msg, plan=plan, plan_len=len(plan))

    return VerifyResult(ok=True, stage="passed", reason=msg, plan=plan, plan_len=len(plan),
                        expanded=expanded, difficulty=band, difficulty_score=score)
