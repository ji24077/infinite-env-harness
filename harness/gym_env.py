"""
Gymnasium environment — the harness's top-level interface, and the evidence that this is
an RL *environment factory*, not a toy. Any Gymnasium-compatible learner (PPO here; GI's
vision policy in principle) mounts on it unchanged:

    env = make_from_spec(spec)          # or make("a room with a locked door...")
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)

Two design points that matter for GI:
  * obs_mode="state" | "pixels" — the SAME env yields a compact vector OR a rendered frame,
    so you can train/eval on code-truth features or on pixels (their policy's input).
  * reward is potential-based shaping from the oracle cost-to-go (verifier.build_distance_field)
    plus the sparse code-truth terminal — the solver that proves solvability also feeds RL.
  * info["code_state"]["predicates"] is the frame-exact, code-defined reward signal.

The 6-input action SHAPE [fwd,back,left,right,mouseDX,mouseDY] that GI's policy emits is exposed
via action_mode="controller" (a grid adapter — mouseDX drives facing, mouseDY reserved) as the
2D->3D transfer *interface shape*, not full continuous kinematics; the working demo uses Discrete(5).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from harness.dsl.schema import EnvSpec
from harness.compiler import compile_spec
from harness.engine import gridlogic as G
from harness.engine import renderer as R
from harness import verifier

DISCRETE_ACTIONS = ["up", "down", "left", "right", "wait"]  # wait = pass one tick (dodge patrols)
MAX_PREDS = 4

GAMMA = 0.99      # shaping discount; keep the learner's gamma equal to this (see learnability.py)
STEP_COST = 0.01  # per-step cost; shaping is zeroed on a no-op so a wall-bump nets exactly -STEP_COST


class HarnessEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, spec: EnvSpec, obs_mode: str = "state",
                 action_mode: str = "discrete", max_steps: Optional[int] = None):
        super().__init__()
        self.spec = spec
        self.obs_mode = obs_mode
        self.action_mode = action_mode
        self.world = compile_spec(spec)
        self.max_steps = max_steps or spec.time_limit
        self._dist = verifier.build_distance_field(self.world.level, spec.objective)
        self._max_d = max(self._dist.values()) if self._dist else 1

        if action_mode == "controller":
            # the 6-input action SHAPE GI's policy emits: [fwd, back, left, right, mouseDX, mouseDY]
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(len(DISCRETE_ACTIONS))

        if obs_mode == "pixels":
            surf = R.render_surface(self.world)
            w, h = surf.get_size()
            self.observation_space = spaces.Box(0, 255, shape=(h, w, 3), dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(-1.0, 1.0, shape=(self._state_dim(),), dtype=np.float32)

    # ── obs ──────────────────────────────────────────────────────────────────────

    def _state_dim(self) -> int:
        return 4 + 1 + MAX_PREDS + 6  # agent, exit, held_frac, preds, nearest key/can/coin

    def _state_vec(self) -> np.ndarray:
        w, h = self.spec.width, self.spec.height
        ax, ay = self.world.state.agent
        v = [ax / w * 2 - 1, ay / h * 2 - 1]
        if self.world.level.exit:
            ex, ey = self.world.level.exit
            v += [ex / w * 2 - 1, ey / h * 2 - 1]
        else:
            v += [0.0, 0.0]
        total_items = max(1, len(self.world.level.keys) + len(self.world.level.cans)
                          + len(self.world.level.all_coin_ids))
        v.append(len(self.world.state.held) / total_items)
        preds = list(self.world.predicate_states().values())
        for i in range(MAX_PREDS):
            v.append(1.0 if (i < len(preds) and preds[i]) else 0.0)
        v += self._nearest_vec(self.world.level.keys.keys())
        v += self._nearest_vec(self.world.level.cans.values())
        v += self._nearest_vec(self.world.level.coins.keys())
        return np.array(v, dtype=np.float32)

    def _nearest_vec(self, cells):
        ax, ay = self.world.state.agent
        w, h = self.spec.width, self.spec.height
        best, bd = None, 1e9
        for (cx, cy) in cells:
            d = abs(ax - cx) + abs(ay - cy)
            if d < bd:
                bd, best = d, (cx, cy)
        if best is None:
            return [0.0, 0.0]
        return [(best[0] - ax) / w, (best[1] - ay) / h]

    def _obs(self):
        if self.obs_mode == "pixels":
            surf = R.render_surface(self.world, tick=self.world.step_count)
            arr = np.array(R.to_pil(surf))
            return arr
        return self._state_vec()

    # ── gym API ───────────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.world.reset()
        return self._obs(), {"code_state": self.world.code_state()}

    def _phi(self):
        d = self._dist.get(self.world.state, self._max_d + 1)
        return -d / max(1, self._max_d)

    def step(self, action):
        if self.action_mode == "controller":
            act = self._decode_controller(action)
        else:
            act = DISCRETE_ACTIONS[int(action)]

        phi_before = self._phi()
        won_before = self.world.won
        state_before = self.world.state
        self.world.step(act)
        phi_after = self._phi()

        # Potential-based shaping F = GAMMA*phi(s') - phi(s) (Ng et al.; policy-invariant when the
        # learner's discount == GAMMA). On a genuine no-op (wall-bump / idle) the state is
        # unchanged, so we zero the shaping — otherwise PBRS's (GAMMA-1)*phi term would refund
        # part of the step cost and faintly reward bumping. Net for a no-op: exactly -STEP_COST.
        shaping = 0.0 if self.world.state == state_before else (GAMMA * phi_after - phi_before)
        reward = -STEP_COST + shaping
        terminated = self.world.done and self.world.won
        truncated = (self.world.done and not self.world.won) or (self.world.step_count >= self.max_steps)
        if self.world.won and not won_before:
            reward += 10.0                                # sparse code-truth terminal

        info = {"code_state": self.world.code_state(),
                "oracle_remaining": self._dist.get(self.world.state, None),
                "won": self.world.won}
        return self._obs(), float(reward), terminated, truncated, info

    def _decode_controller(self, a) -> str:
        """Adapter from the 6-input action shape [fwd,back,left,right,mouseDX,mouseDY] onto the
        discrete grid. mouseDX rotates the rendered facing; mouseDY is reserved (unused in 2D).
        This is the action-space SHAPE GI's policy emits, not full continuous kinematics — the
        engine is grid-authoritative by design. Below the dead-zone, no move is taken."""
        fwd, back, left, right, mdx, mdy = [float(x) for x in a]
        self.world.pose[2] += mdx * 0.3                  # facing follows mouseDX (cosmetic)
        moves = [fwd, back, left, right]
        if max(moves) < 0.1:                             # dead-zone: no clear intent -> wait
            return "wait"
        return ["up", "down", "left", "right"][int(np.argmax(moves))]

    def render(self):
        return np.array(R.to_pil(R.render_surface(self.world, tick=self.world.step_count)))


# ── convenience constructors ─────────────────────────────────────────────────────

def make_from_spec(spec, **kwargs) -> HarnessEnv:
    if not isinstance(spec, EnvSpec):
        spec = EnvSpec(**spec)
    return HarnessEnv(spec, **kwargs)


def make(command: str, model: str = "claude-sonnet-4-5", **kwargs) -> HarnessEnv:
    """text command -> generated + verified env -> Gymnasium Env. Requires ANTHROPIC_API_KEY."""
    from harness.generator import generate
    spec, _vr, _log = generate(command, model=model)
    return HarnessEnv(spec, **kwargs)
