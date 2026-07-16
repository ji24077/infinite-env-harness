"""
Episode runner shared by the demo, the agents, and the evaluator.

A `policy` is any callable (env, obs, info) -> action-index, optionally with a `.reset(env)`.
run_episode returns the outcome plus, on request, the rendered frames + a per-step trace
(frame, action, code-truth state, reward) — i.e. exactly the (pixels, code-reward) pairs a
reward model would train on.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from harness.engine import renderer as R
from harness.gym_env import HarnessEnv, DISCRETE_ACTIONS


def run_episode(env: HarnessEnv, policy: Callable, max_steps: Optional[int] = None,
                collect_frames: bool = False, seed: Optional[int] = None) -> dict:
    obs, info = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset(env)
    steps = max_steps or env.max_steps
    frames: List = []
    trace: List[dict] = []
    total = 0.0
    tick = 0

    def snap():
        if collect_frames:
            frames.append(R.to_pil(R.render_surface(env.world, tick=tick)))

    snap()
    for t in range(steps):
        a = int(policy(env, obs, info))
        obs, r, term, trunc, info = env.step(a)
        total += r
        tick += 1
        trace.append({
            "step": env.world.step_count,
            "action": DISCRETE_ACTIONS[a] if a < len(DISCRETE_ACTIONS) else str(a),
            "reward": round(r, 4),
            "code_state": info["code_state"],
        })
        snap()
        if term or trunc:
            break

    return {
        "won": bool(env.world.won),
        "steps": env.world.step_count,
        "total_reward": round(total, 3),
        "oracle_len": _oracle_len(env),
        "frames": frames,
        "trace": trace,
    }


def _oracle_len(env: HarnessEnv):
    from harness import verifier
    plan, _ = verifier.solve(env.world.level, env.spec.objective)
    return len(plan) if plan is not None else None
