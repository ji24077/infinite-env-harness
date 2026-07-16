"""
Scripted oracle controller — the API-free solvability oracle.

It asks the verifier for the shortest plan from the current state and executes it. This is
the honest role of "the agent" in this project: PROOF that a generated environment is
beatable, not a product. An optional epsilon makes it a noisy oracle so evaluation batches
show a spread of efficiencies rather than all-perfect runs.

(This also stands in, in --offline mode, for the 3D navigation that GI's vision policy would
perform: same env, same discrete/controller interface, a policy mounted on the game object.)
"""

from __future__ import annotations

import numpy as np

from harness import verifier
from harness.gym_env import DISCRETE_ACTIONS

_AIDX = {a: i for i, a in enumerate(DISCRETE_ACTIONS)}


class ScriptedOracle:
    def __init__(self, epsilon: float = 0.0, seed: int = 0):
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.plan = []
        self.i = 0

    def reset(self, env):
        plan, _ = verifier.solve(env.world.level, env.spec.objective)
        self.plan = plan or []
        self.i = 0

    def __call__(self, env, obs, info) -> int:
        if self.epsilon and self.rng.random() < self.epsilon:
            self.plan, self.i = [], 0      # invalidate cached plan; re-solve next call
            return int(self.rng.integers(0, 4))
        # if the plan is exhausted or was invalidated, re-solve from the current state
        if self.i >= len(self.plan):
            plan, _ = verifier.solve(env.world.level, env.spec.objective, start=env.world.state)
            self.plan = plan or []
            self.i = 0
            if not self.plan:
                return _AIDX["interact"]
        a = self.plan[self.i]
        self.i += 1
        return _AIDX[a]
