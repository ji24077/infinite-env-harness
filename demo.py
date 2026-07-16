"""
demo.py — the whole harness in one command (~2 min).

  uv run demo.py              # auto: live generation if ANTHROPIC_API_KEY is set, else --offline
  uv run demo.py --offline    # no API key needed: uses the verified specs/ cache
  uv run demo.py --online     # force live text->environment generation

It walks the full factory pipeline:
  1. text command  -> generated + L1/L2/L3-verified environment (streamed logs)
  2. oracle plan   -> replay GIF (proof the environment is beatable)
  3. rollout       -> trace.jsonl + frames/ (pixel frame + code-truth reward = a training shard)
  4. mutation      -> 10 new verified environments, ACCEL-inspired, auto difficulty labels
  5. scorecard     -> success / efficiency, difficulty-stratified (the eval use case)
  6. legality critic -> a direction: code-truth flags injected illegal transitions in a rollout
  7. reward model   -> a pixel reward model trained ONLY on code-truth labels (GI use-case #3)
  8. RL capstone   -> the headline: PPO climbs the reward curve (these envs feed RL)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from harness import fixtures as F
from harness.dsl.schema import EnvSpec
from harness.compiler import predicate_program
from harness.verifier import verify
from harness.gym_env import make_from_spec
from harness.rollout import run_episode
from harness.agents.scripted import ScriptedOracle
from harness.mutate import mutate
from harness import eval as E
from harness.engine import renderer as R

RUNS = "runs"
DEMO_KEYS = ["open_can", "key_crate_return", "three_rooms"]


def hr(title): print("\n" + "=" * 72 + f"\n  {title}\n" + "=" * 72)


def load_cached(name):
    with open(os.path.join("specs", f"{name}.json")) as f:
        return json.load(f)["spec"]


def get_specs(online: bool):
    """Return list of (label, spec_dict). Online -> generate; offline -> cached."""
    specs = []
    if online:
        from harness.generator import generate
        for name in DEMO_KEYS:
            cmd = F.DEMO_COMMANDS[name]
            print(f"\n>>> generating from command: {cmd!r}")
            spec, vr, _ = generate(cmd)
            specs.append((name, spec.model_dump()))
    else:
        from harness.generator import generate_offline
        for name in DEMO_KEYS:
            print(f"\n>>> loading cached spec (no API): {name}")
            spec_dict = load_cached(name)
            _spec, vr, _ = generate_offline(spec_dict)   # re-verifies; raises if not vr.ok
            print("   ", vr.log_line().strip())
            print("    objective (code):", predicate_program(_spec))
            specs.append((name, spec_dict))
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="use cached specs, no API key")
    ap.add_argument("--online", action="store_true", help="force live LLM generation")
    ap.add_argument("--variants", type=int, default=10)
    args = ap.parse_args()

    online = args.online or (bool(os.environ.get("ANTHROPIC_API_KEY")) and not args.offline)
    os.makedirs(RUNS, exist_ok=True)

    print("INFINITE ENVIRONMENT HARNESS — text -> verified RL environments")
    print("mode:", "ONLINE (live generation)" if online else "OFFLINE (cached specs, no API key)")

    # 1. GENERATE + VERIFY
    hr("1. TEXT COMMAND  ->  VERIFIED ENVIRONMENT  (L1 schema / L2 solvable / L3 physics)")
    try:
        specs = get_specs(online)
    except Exception as ex:
        if not online:
            raise
        print(f"\n[!] live generation unavailable ({type(ex).__name__}: {str(ex)[:140]})")
        print("[!] falling back to the pre-verified cached specs (check ANTHROPIC_API_KEY / billing).")
        online = False
        specs = get_specs(False)

    # 2. ORACLE REPLAY GIF (proof the env is beatable)
    hr("2. ORACLE PLAN REPLAY  ->  GIF  (verification that the env is solvable)")
    hero_name, hero_spec = specs[1]                 # the medium one is the nicest hero
    env = make_from_spec(hero_spec)
    out = run_episode(env, ScriptedOracle(), collect_frames=True)
    for _ in range(8):
        out["frames"].append(R.to_pil(R.render_surface(env.world, tick=999)))
    gif = os.path.join(RUNS, f"{hero_name}_oracle.gif")
    R.save_gif(out["frames"], gif, fps=7)
    print(f"    '{hero_spec['name']}': oracle solved in {out['steps']} steps -> {gif}")

    # 3. TRAJECTORY DATASET SHARD (each step: a saved frame + its code-defined reward)
    hr("3. ROLLOUT  ->  trace.jsonl + frames/  (pixel frame + code reward = a training shard)")
    frame_dir = os.path.join(RUNS, f"{hero_name}_frames")
    os.makedirs(frame_dir, exist_ok=True)
    trace_path = os.path.join(RUNS, f"{hero_name}_trace.jsonl")
    with open(trace_path, "w") as fh:
        for i, row in enumerate(out["trace"]):
            fpath = os.path.join(frame_dir, f"step_{i+1:04d}.png")   # frame AFTER this action
            out["frames"][i + 1].save(fpath)
            row = {**row, "frame": os.path.relpath(fpath, RUNS)}      # pair pixels with reward
            fh.write(json.dumps(row) + "\n")
    print(f"    wrote {len(out['trace'])} (frame, action, reward, code_state) rows -> {trace_path}")
    print(f"    + {len(out['trace'])} index-aligned frames -> {frame_dir}/")
    print(f"    e.g. final row: {json.dumps({k: v for k, v in {**out['trace'][-1]}.items() if k != 'code_state'})[:120]}...")

    # 4. MUTATION -> infinite verified variants
    hr(f"4. MUTATION  ->  {args.variants} NEW verified environments (ACCEL-curated, auto difficulty)")
    base_name = "key_crate_return"
    base = load_cached(base_name) if not online else hero_spec
    variants = mutate(base, n=args.variants, seed=7, accel=True)
    print(f"    base: '{base['name']}'   (regret = optimal solves but greedy fails)")
    print(f"    {'variant':<40} {'difficulty':<9} {'plan':<5} regret")
    print("    " + "-" * 64)
    for v in variants:
        desc = v["name"].split("·")[-1].strip()[:38]
        print(f"    {desc:<40} {v['difficulty']:<9} {v['plan_len']:<5} {v['regret']:+.0f}")

    # 5. EVAL SCORECARD
    hr("5. EVAL SCORECARD  (noisy oracle across all verified envs, difficulty-stratified)")
    all_specs = {n: load_cached(n) for n in F.ALL}
    sc = E.scorecard(all_specs, epsilon=0.12, seed=1)
    print(E.format_scorecard(sc))

    # 6. ROLLOUT-LEGALITY CRITIC (a direction, honestly scoped)
    hr("6. ROLLOUT-LEGALITY CRITIC  (a direction: code-truth checks rollout legality)")
    from harness import critic as CR
    from harness.verifier import solve
    from harness.engine import gridlogic as _G
    cspec = EnvSpec(**load_cached("key_crate_return"))
    clevel = _G.build_level(cspec)
    cplan, _ = solve(clevel, cspec.objective)
    real = CR.rollout_from_plan(clevel, cplan)
    dreamed = CR.forge_hallucination(clevel, real)   # we INJECT the illegal transitions here
    vio = CR.critique(clevel, dreamed)
    print(f"    faithful rollout    : consistency {CR.score(clevel, real):.0%}  (0 violations)")
    print(f"    hallucinated rollout: consistency {CR.score(clevel, dreamed):.0%}  ({len(vio)} injected illegal transitions caught)")
    for v in vio[:4]:
        print(f"      - step {v.step}: {v.reason}")
    print("    scope: proof-of-concept on discrete STATE (the demo plants the violations it catches);")
    print("    wiring to a real world model needs a frame->state decoder — a direction, not a claim.")

    # 7. CODE-TRUTH -> PIXEL REWARD MODEL (GI use-case #3)
    hr("7. CODE-TRUTH -> PIXEL REWARD MODEL  (train a pixel model on exact code labels)")
    rm_path = os.path.join("assets", "reward_model.json")
    if os.path.exists(rm_path):
        rm = json.load(open(rm_path))
        print(f"    code truth      : exact, label-free, ~microseconds ({rm['note']})")
        print(f"    pixel reward model: {rm['disagreement_rate']:.0%} held-out disagreement "
              f"(pickup recall {rm['pickup_recall']:.0%}, not-picked {rm['not_picked_recall']:.0%})")
        print("    -> a pixel model approximates the exact code label it was trained on. That code")
        print("       label is the supervision a vision reward model is trained toward (GI use-case #3).")
        print("    reproduce: uv run --extra rl python scripts/train_reward_model.py")
    else:
        print("    (run: uv run --extra rl python scripts/train_reward_model.py)")

    # 8. RL LEARNABILITY CAPSTONE (the headline result)
    hr("8. RL LEARNABILITY  (the headline: these verified envs feed reinforcement learning)")
    if os.path.exists("assets/learnability.png"):
        print("    assets/learnability.png — PPO reward on 'Coins & Hazard' climbs from failing")
        print("    (~ -1.3) to solving (~10.4), at oracle-optimal length.")
        print("    reproduce exactly: uv run --extra rl python learnability.py")
    else:
        print("    (run `uv run --extra rl python learnability.py` to train PPO and plot the curve)")

    print("\n" + "=" * 72)
    print("  DONE. Generated + verified environments, an oracle-solved GIF, a training-shard")
    print("  trace, 10 curated variants, an eval scorecard, a legality-checker direction, a")
    print("  code-trained pixel reward model, and a PPO learnability curve — the factory, end to end.")
    print("=" * 72)


if __name__ == "__main__":
    main()
