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
  6. world-model critic -> the headline: code-truth flags hallucinated dynamics a VLM would miss
  7. code vs pixel -> supporting illustrative micro-benchmark on a constructed occlusion scene
  8. RL capstone   -> pointer to the PPO reward curve (env feeds RL)
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
        for name in DEMO_KEYS:
            print(f"\n>>> loading cached spec (no API): {name}")
            spec = load_cached(name)
            vr = verify(spec)                      # real verification, streamed
            print("   ", vr.log_line().strip())
            print("    objective (code):", predicate_program(EnvSpec(**spec)))
            specs.append((name, spec))
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

    # 6. WORLD-MODEL CRITIC (headline)
    hr("6. WORLD-MODEL CRITIC  (headline: code-truth flags hallucinated dynamics)")
    from harness import critic as CR
    from harness.verifier import solve
    from harness.engine import gridlogic as _G
    cspec = EnvSpec(**load_cached("key_crate_return"))
    clevel = _G.build_level(cspec)
    cplan, _ = solve(clevel, cspec.objective)
    real = CR.rollout_from_plan(clevel, cplan)
    dreamed = CR.forge_hallucination(clevel, real)
    vio = CR.critique(clevel, dreamed)
    print(f"    faithful rollout    : consistency {CR.score(clevel, real):.0%}  (0 violations)")
    print(f"    hallucinated rollout: consistency {CR.score(clevel, dreamed):.0%}  ({len(vio)} illegal transitions flagged)")
    for v in vio[:4]:
        print(f"      - step {v.step}: {v.reason}")
    print("    -> code-truth catches teleports / ungrounded pickups a pixel judge would miss.  see assets/critic.png")

    # 7. CODE vs PIXEL CONTRAST (supporting illustration)
    hr("7. CODE-TRUTH vs PIXEL PERCEPTION  (supporting: illustrative micro-benchmark)")
    c = E.run_contrast(load_cached("occlusion_can"), use_vlm=False)
    strip = os.path.join("assets", "contrast.png")
    E.render_contrast_strip(c, strip)
    ratio = c['perc_time_us'] / max(c['code_time_us'], 1e-9)
    print(f"    scene (constructed): '{c['spec_name']}'  ({c['n_frames']} frames)")
    print(f"    code-truth  : pickup exact at frame {c['code_first_true']}; predicate check ~{c['code_time_us']} us (median)")
    print(f"    pixel stand-in: disagrees with code truth on {c['disagreements']}/{c['n_frames']} frames "
          f"(occlusion false positives); scan ~{c['perc_time_us']} us (median)")
    print(f"    -> code truth is exact and ~{ratio:.0f}x cheaper than the pixel scan.  strip -> {strip}")
    print("    (evaluate.py --vlm --live swaps the pixel stand-in for a real Claude VLM: ~1.7 s/frame + $)")

    # 8. RL LEARNABILITY CAPSTONE
    hr("8. RL LEARNABILITY  (these envs feed reinforcement learning)")
    if os.path.exists("assets/learnability.png"):
        print("    assets/learnability.png — PPO reward on 'Coins & Hazard' climbs from failing")
        print("    (~ -1.3) to solving (~10.4), at oracle-optimal length.")
        print("    reproduce exactly: uv run --extra rl python learnability.py")
    else:
        print("    (run `uv run --extra rl python learnability.py` to train PPO and plot the curve)")

    print("\n" + "=" * 72)
    print("  DONE. Generated + verified environments, an oracle-solved GIF, a training-shard")
    print("  trace, 10 curated variants, an eval scorecard, the world-model critic, the")
    print("  code-vs-pixel illustration, and a PPO learnability curve — the factory, end to end.")
    print("=" * 72)


if __name__ == "__main__":
    main()
