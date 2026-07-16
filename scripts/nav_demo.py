"""
Claude navigates a generated environment — and surfaces the pixel-nav gap.

The challenge asks that an agent maneuver through the generated environments. Here Claude (not the
scripted oracle) does exactly that, two ways on the SAME level:
  * STATE  — Claude reasons over the coordinate-tagged engine state.
  * PIXELS — Claude sees only the rendered frame (GI's vision-policy modality).

Requires ANTHROPIC_API_KEY. Regenerates assets/agent_nav.gif + assets/nav_result.json.
  uv run --env-file .env python scripts/nav_demo.py [--env open_can]
"""
import argparse
import json
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from harness import fixtures as F
from harness.gym_env import make_from_spec
from harness.rollout import run_episode
from harness.agents.state_agent import StateAgent
from harness.agents.pixel_agent import PixelAgent
from harness.engine import renderer as R

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="open_can", choices=list(F.ALL))
    ap.add_argument("--cap", type=int, default=28)
    args = ap.parse_args()

    env = make_from_spec(F.ALL[args.env]())
    s = run_episode(env, StateAgent(), max_steps=args.cap, collect_frames=True)
    for _ in range(6):
        s["frames"].append(R.to_pil(R.render_surface(env.world, tick=999)))
    R.save_gif(s["frames"], os.path.join(ASSETS, "agent_nav.gif"), fps=4)
    print(f"STATE  (coords)      : won={s['won']} steps={s['steps']} (oracle {s['oracle_len']})")

    env2 = make_from_spec(F.ALL[args.env]())
    p = run_episode(env2, PixelAgent(), max_steps=args.cap)
    print(f"PIXELS (frames-only) : won={p['won']} steps={p['steps']} (cap {args.cap})")
    print("-> Claude solves from code-state; the pixel gap is exactly what a vision policy fills.")

    json.dump({"env": args.env, "oracle_len": s["oracle_len"],
               "state_agent": {"won": s["won"], "steps": s["steps"]},
               "pixel_agent": {"won": p["won"], "steps": p["steps"], "cap": args.cap}},
              open(os.path.join(ASSETS, "nav_result.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
