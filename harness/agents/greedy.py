"""
Greedy heuristic policy — a deliberately myopic agent used as an API-free "regret" probe for
ACCEL-inspired mutation curation (mutate.py). It steps toward the nearest objective-relevant
cell with no multi-step planning: it will pick up keys/coins/cans on the way (and doors then
auto-open), but it cannot plan a maneuver like pushing a crate from the correct side or routing
around a hazard wall. So it clears simple navigation/pickup layouts and fails ones that require
planning — and oracle_success - greedy_success is a cheap learnability/regret proxy.
"""

from __future__ import annotations

import numpy as np

from harness.gym_env import DISCRETE_ACTIONS
from harness.engine import gridlogic as G

_AIDX = {a: i for i, a in enumerate(DISCRETE_ACTIONS)}


class GreedyHeuristic:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self, env):
        pass

    def _targets(self, env):
        lvl = env.world.level
        held = env.world.state.held
        t = []
        for cell, kid in lvl.keys.items():
            if kid not in held:
                t.append(cell)
        for cid, cell in lvl.cans.items():
            if cid not in held:
                t.append(cell)
        for cell, coin in lvl.coins.items():
            if coin not in held:
                t.append(cell)
        if not t and lvl.exit:
            t.append(lvl.exit)
        return t

    def __call__(self, env, obs, info) -> int:
        ax, ay = env.world.state.agent
        targets = self._targets(env)
        if not targets:
            return _AIDX["interact"]
        tx, ty = min(targets, key=lambda c: abs(c[0]-ax) + abs(c[1]-ay))
        # greedily reduce the larger axis gap; try a step, fall back if blocked
        options = []
        if abs(tx - ax) >= abs(ty - ay):
            options = ["right" if tx > ax else "left", "down" if ty > ay else "up"]
        else:
            options = ["down" if ty > ay else "up", "right" if tx > ax else "left"]
        for a in options:
            if G.step(env.world.level, env.world.state, a) is not None:
                return _AIDX[a]
        return int(self.rng.integers(0, 4))  # stuck: wiggle
