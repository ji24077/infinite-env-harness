"""
PPO learnability capstone — empirical proof that these auto-generated, code-rewarded
environments actually FEED reinforcement learning.

We do NOT train GI's navigation policy (that's theirs, and vision-based). We simply mount a
small off-the-shelf PPO (stable-baselines3) on one of our Gymnasium envs and show the reward
curve climb: the environment emits a learnable signal. The reward it climbs is the potential
shaping sourced from the oracle cost-to-go + the sparse code-truth terminal — i.e. the same
solver that proves solvability also supplies the training signal.

Optional dependency group:  uv run --extra rl learnability.py
Outputs: assets/learnability.png (reward curve), assets/learnability_solved.gif (trained agent).
"""

from __future__ import annotations

import argparse
import os
import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from harness import fixtures as F
from harness.gym_env import make_from_spec
from harness.rollout import run_episode
from harness.engine import renderer as R


def main():
    # defaults reproduce the committed assets/learnability.png exactly
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="coins_hazard", choices=list(F.ALL))
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--out", default="assets/learnability.png")
    ap.add_argument("--reward-mode", default="shaped", choices=["shaped", "sparse", "curiosity"],
                    help="'shaped' = PBRS from V* (easy by construction); 'sparse' = honest RL test")
    ap.add_argument("--no-leak", action="store_true",
                    help="drop goal-relative obs vectors — removes the observation compass too")
    args = ap.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec = F.ALL[args.env]()
    print(f"[learnability] env='{spec['name']}' reward_mode={args.reward_mode} "
          f"leak_goal_vectors={not args.no_leak} — training PPO for {args.steps} steps")

    def make():
        return Monitor(make_from_spec(spec, obs_mode="state",
                                      reward_mode=args.reward_mode,
                                      leak_goal_vectors=not args.no_leak))

    env = make()

    class Curve(BaseCallback):
        def __init__(self):
            super().__init__()
            self.x, self.y = [], []

        def _on_step(self) -> bool:
            if self.num_timesteps % 2000 < 1 and self.model.ep_info_buffer:
                rews = [e["r"] for e in self.model.ep_info_buffer]
                self.x.append(self.num_timesteps)
                self.y.append(float(np.mean(rews)))
            return True

    from harness.gym_env import GAMMA
    cb = Curve()
    model = PPO("MlpPolicy", env, verbose=0, n_steps=1024, batch_size=256,
                gae_lambda=0.95, gamma=GAMMA, ent_coef=0.01, seed=0)  # gamma == shaping GAMMA
    model.learn(total_timesteps=args.steps, callback=cb)

    # plot reward curve
    plt.figure(figsize=(6.4, 3.4), dpi=120)
    plt.plot(cb.x, cb.y, color="#56b4ff", lw=2)
    plt.axhline(0, color="#555", lw=0.8, ls="--")
    plt.title(f"PPO learns '{spec['name']}' ({args.reward_mode} reward"
              f"{', no obs compass' if args.no_leak else ''}) — the env feeds RL", fontsize=11)
    plt.xlabel("environment steps"); plt.ylabel("mean episode reward")
    plt.grid(alpha=0.2); plt.tight_layout()
    plt.savefig(args.out)
    print(f"[learnability] saved reward curve -> {args.out}")
    if cb.y:
        print(f"[learnability] mean episode reward: {cb.y[0]:.2f} (start) -> {cb.y[-1]:.2f} (end)")

    # roll out the trained policy and save a GIF
    class SB3Policy:
        def reset(self, env): pass
        def __call__(self, env, obs, info):
            a, _ = model.predict(obs, deterministic=True)
            return int(a)

    ev = make_from_spec(spec, obs_mode="state",
                        reward_mode=args.reward_mode, leak_goal_vectors=not args.no_leak)
    out = run_episode(ev, SB3Policy(), collect_frames=True)
    R.save_gif(out["frames"], "assets/learnability_solved.gif", fps=8)
    print(f"[learnability] trained agent: won={out['won']} steps={out['steps']} "
          f"(oracle {out['oracle_len']}) -> assets/learnability_solved.gif")


if __name__ == "__main__":
    main()
