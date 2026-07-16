"""
nav_probe.py — the missing measurement: does Claude actually NAVIGATE the generated
environments from STATE (coordinates) vs from PIXELS (rendered frame)?

Drives harness/agents/state_agent.py and pixel_agent.py (which were never wired into any
runner) through verified specs, comparing each to the BFS oracle plan length. Honest, small-n.

  uv run python nav_probe.py --envs open_can
  uv run python nav_probe.py --envs open_can push_delivery occlusion_can
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# --- load ANTHROPIC_API_KEY from .env (repo has no dotenv) ---
envfile = Path(__file__).parent / ".env"
if envfile.exists():
    for line in envfile.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from harness.verifier import verify
from harness.gym_env import make_from_spec
from harness.agents.state_agent import StateAgent
from harness.agents.pixel_agent import PixelAgent


def load(name):
    return json.load(open(f"specs/{name}.json"))["spec"]


def run(spec, agent, cap):
    env = make_from_spec(spec)          # obs_mode=state; pixel agent self-renders env.world
    obs, info = env.reset()
    agent.reset(env)
    won, steps, err = False, 0, None
    for _ in range(cap):
        try:
            a = agent(env, obs, info)
        except Exception as ex:
            err = f"{type(ex).__name__}: {str(ex)[:80]}"
            break
        obs, r, term, trunc, info = env.step(a)
        steps += 1
        if term or trunc:
            won = bool(info.get("won"))
            break
    return {"won": won, "steps": steps, "err": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", nargs="+", default=["open_can"])
    ap.add_argument("--cap-mult", type=int, default=3, help="max steps = min(cap_mult*oracle, cap_abs)")
    ap.add_argument("--cap-abs", type=int, default=30)
    args = ap.parse_args()

    print(f"model: {os.environ.get('HARNESS_MODEL', 'claude-sonnet-4-5')}  "
          f"key: {'set' if os.environ.get('ANTHROPIC_API_KEY') else 'MISSING'}")
    print(f"{'env':<16}{'oracle':>7}{'mode':>8}{'won':>6}{'steps':>7}{'eff':>7}  note")
    print("-" * 68)

    rows = []
    for name in args.envs:
        spec = load(name)
        vr = verify(spec)
        oracle = vr.plan_len
        cap = min(args.cap_mult * oracle, args.cap_abs)
        for label, Agent in [("state", StateAgent), ("pixel", PixelAgent)]:
            t0 = time.time()
            res = run(spec, Agent(), cap)
            dt = time.time() - t0
            eff = f"{oracle/res['steps']:.2f}" if (res["won"] and res["steps"]) else "-"
            note = res["err"] or (f"solved@oracle" if res["won"] and res["steps"] == oracle
                                  else "solved" if res["won"] else f"FAIL(cap={cap})")
            print(f"{name:<16}{oracle:>7}{label:>8}{str(res['won']):>6}{res['steps']:>7}{eff:>7}  {note}  [{dt:.0f}s]")
            rows.append({"env": name, "oracle": oracle, "mode": label, **res})

    Path("runs").mkdir(exist_ok=True)
    with open("runs/nav_probe.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("\nwrote runs/nav_probe.json")


if __name__ == "__main__":
    main()
