"""
baselines.py — de-circularize the "these environments feed RL" claim.

An honest reviewer objection to the PPO capstone: the shaped reward's potential IS the oracle's
exact optimal value function V*, so "PPO climbs the curve" might only mean "a hand-provided
optimal compass makes an already-solved grid greedily solvable." This script makes that explicit
and separates plumbing from a real learning challenge, using only offline policies (no API key,
no training):

  * random            — sanity floor.
  * V*-greedy         — ZERO learning; greedily follows the oracle cost-to-go field env._dist,
                        which is exactly the potential the "shaped" reward is built from.

The result to read: V*-greedy solves the shaped envs at oracle-optimal length with no learning at
all. That is the circularity — shaped reward == handing the policy V*. The honest RL test is
therefore reward_mode="sparse" + leak_goal_vectors=False (compass removed from BOTH reward and
observation); see learnability.py --reward-mode sparse --no-leak.

  uv run python baselines.py
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np

from harness import fixtures as F
from harness.gym_env import make_from_spec
from harness.rollout import run_episode
from harness.agents.reward_greedy import VStarGreedy
from harness.agents.scripted import ScriptedOracle

ENVS = ["open_can", "push_delivery", "coins_hazard", "three_rooms"]


class Random:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self, env):
        pass

    def __call__(self, env, obs, info) -> int:
        return int(self.rng.integers(0, 5))


def _run(spec, factory, episodes=1):
    won, steps = 0, []
    for i in range(episodes):
        out = run_episode(make_from_spec(spec, obs_mode="state"), factory(), seed=i)
        won += int(out["won"])
        if out["won"]:
            steps.append(out["steps"])
    return won / episodes, (float(np.mean(steps)) if steps else None)


def main():
    print("De-circularizing 'these environments feed RL' — offline policies, no learning, no key.\n")
    print(f"{'env':<15}{'oracle':>7}{'random':>10}{'V*-greedy (0 learning)':>26}")
    print("-" * 58)
    for n in ENVS:
        spec = F.ALL[n]()
        orc = run_episode(make_from_spec(spec), ScriptedOracle())["oracle_len"]
        rw, _ = _run(spec, lambda: Random(), episodes=5)
        gw, gs = _run(spec, lambda: VStarGreedy(), episodes=1)
        g = "fail" if gs is None else f"{gw*100:.0f}% @ {gs:.0f} steps"
        opt = " (=oracle)" if gs is not None and abs(gs - orc) < 1e-6 else ""
        print(f"{n:<15}{orc:>7}{rw*100:>9.0f}%{g:>22}{opt}")
    print("\nV*-greedy solves at oracle-optimal with ZERO learning because the shaped reward IS V*.")
    print("=> 'PPO climbs the shaped curve' proves plumbing, not difficulty.")
    print("Honest RL test (compass removed from reward AND obs):")
    print("   uv run --extra rl python learnability.py --env open_can --reward-mode sparse --no-leak")


if __name__ == "__main__":
    main()
