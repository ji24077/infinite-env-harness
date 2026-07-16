"""
VStarGreedy — the zero-learning baseline that makes the shaped-reward circularity explicit.

The "shaped" reward's potential is phi = -cost_to_go/max_d, i.e. the EXACT optimal value
function V* of the env (verifier.build_distance_field). This policy does no learning at all: at
every step it greedily takes the action that most reduces that same distance field. If it already
solves an env at oracle-optimal length, then a PPO curve trained under the shaped reward is
demonstrating plumbing, not a learning challenge — it is following a hand-provided optimal compass.

The honest RL test is therefore reward_mode="sparse" + leak_goal_vectors=False, under which this
compass is unavailable in both the reward and the observation. See baselines.py.
"""

from __future__ import annotations

from harness.gym_env import DISCRETE_ACTIONS
from harness.engine import gridlogic as G

_AIDX = {a: i for i, a in enumerate(DISCRETE_ACTIONS)}
_BIG = 1e9


class VStarGreedy:
    """Greedy on the oracle cost-to-go field env._dist (the exact V*). No learning."""

    def reset(self, env):
        pass

    def __call__(self, env, obs, info) -> int:
        st = env.world.state
        best_a, best_d = "wait", env._dist.get(st, _BIG)
        for a in DISCRETE_ACTIONS:
            nxt = G.step(env.world.level, st, a)
            if nxt is None:
                continue
            d = env._dist.get(nxt, _BIG)
            if d < best_d:
                best_d, best_a = d, a
        return _AIDX[best_a]
