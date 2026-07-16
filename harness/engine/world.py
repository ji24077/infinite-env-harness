"""
Runtime world — the hybrid engine.

Navigation, keys, doors, crates and pickups are GRID-authoritative (deterministic, and
identical to the verifier's semantics, so oracle plans replay frame-exact). On top of that,
pymunk drives the *soft physics props*: a `ball` rolls/settles under gravity, and crate
bodies are synced to their grid cells so they can nudge the ball. Physics is decorative and
never affects game logic — the "physics-engine credential" without the physics risk.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import pymunk

from harness.dsl.schema import EnvSpec
from harness.engine import gridlogic as G

TILE = 32


class World:
    def __init__(self, spec: EnvSpec):
        self.spec = spec
        self.level = G.build_level(spec)
        self.reset()

    def reset(self) -> None:
        self.state = G.initial_state(self.level)
        self.step_count = 0
        self.done = False
        self.won = False
        self.message = ""
        # continuous agent pose (for the controller-contract / transfer story)
        ax, ay = self.state.agent
        self.pose = [ax * TILE + TILE / 2, ay * TILE + TILE / 2, 0.0]  # x_px, y_px, heading
        self._build_physics()

    # ── physics (props only) ─────────────────────────────────────────────────────

    def _build_physics(self) -> None:
        self.space = pymunk.Space()
        self.space.gravity = (0, 900)
        static = self.space.static_body
        for y, row in enumerate(self.spec.tiles):
            for x, t in enumerate(row):
                if t == 1:
                    verts = [(x*TILE, y*TILE), (x*TILE+TILE, y*TILE),
                             (x*TILE+TILE, y*TILE+TILE), (x*TILE, y*TILE+TILE)]
                    shp = pymunk.Poly(static, verts)
                    shp.friction = 0.9
                    self.space.add(shp)
        self.balls: Dict[str, pymunk.Body] = {}
        self.crate_bodies: Dict[str, pymunk.Body] = {}
        for e in self.spec.entities:
            x, y = e.pos
            cx, cy = x*TILE + TILE/2, y*TILE + TILE/2
            if e.type == "ball":
                r = TILE * 0.32
                b = pymunk.Body(1.0, pymunk.moment_for_circle(1.0, 0, r))
                b.position = (cx, cy)
                s = pymunk.Circle(b, r); s.friction = 0.7; s.elasticity = 0.4
                self.space.add(b, s); self.balls[e.id] = b
            elif e.type == "crate":
                size = (TILE*0.82, TILE*0.82)
                b = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
                b.position = (cx, cy)
                s = pymunk.Poly.create_box(b, size); s.friction = 0.9
                self.space.add(b, s); self.crate_bodies[e.id] = b

    def _sync_physics(self) -> None:
        # move crate kinematic bodies toward their grid cells; step the sim a few substeps
        cp = self.state.crate_pos()
        pos_of = {cid: cell for cell, cid in cp.items()}
        for cid, body in self.crate_bodies.items():
            tx, ty = pos_of.get(cid, (0, 0))
            body.position = (tx*TILE + TILE/2, ty*TILE + TILE/2)
        for _ in range(3):
            self.space.step(1 / 60.0)
        for b in self.balls.values():
            if not (math.isfinite(b.position.x) and math.isfinite(b.position.y)):
                b.position = (0, 0); b.velocity = (0, 0)

    # ── stepping ─────────────────────────────────────────────────────────────────

    def step(self, action: str) -> None:
        """Discrete action: up/down/left/right/wait/interact. Grid-authoritative, with
        deterministic patrolling enemies (contact = death), matching the verifier exactly."""
        if self.done:
            return
        self.step_count += 1
        nxt, event = G.resolve(self.level, self.state, action)
        if event == "dead":
            self.done, self.won = True, False
            self.message = "caught by an enemy!"
            self._sync_physics()
            return
        if event == "ok" and nxt is not None:
            if nxt.held != self.state.held:
                self.message = "picked up " + ",".join(sorted(nxt.held - self.state.held))
            self.state = nxt
        # update continuous pose to grid center + facing
        ax, ay = self.state.agent
        if action in G.DELTAS:
            dx, dy = G.DELTAS[action]
            self.pose[2] = math.atan2(dy, dx)
        self.pose[0], self.pose[1] = ax*TILE + TILE/2, ay*TILE + TILE/2
        self._sync_physics()

        if G.objective_satisfied(self.level, self.spec.objective, self.state):
            self.done, self.won = True, True
            self.message = "objective satisfied"
        elif self.step_count >= self.spec.time_limit:
            self.done, self.won = True, False
            self.message = "time limit reached"

    # ── introspection (code-truth ground signals) ───────────────────────────────

    def predicate_states(self) -> Dict[str, bool]:
        from harness.engine.gridlogic import predicate_true
        out = {}
        for i, p in enumerate(self.spec.objective):
            label = p.kind + (f"({p.item})" if p.item else "")
            out[f"{i}:{label}"] = predicate_true(self.level, p, self.state)
        return out

    def code_state(self) -> dict:
        return {
            "agent": list(self.state.agent),
            "held": sorted(self.state.held),
            "crates": {cid: [x, y] for cid, x, y in self.state.crates},
            "predicates": self.predicate_states(),
            "step": self.step_count,
            "done": self.done,
            "won": self.won,
        }
