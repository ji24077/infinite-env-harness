"""
Evaluator — turns the environment factory into a benchmark suite (GI's "training AND
evaluating" use case), and hosts the headline: code-truth vs pixel-perception.

Two products:

1. scorecard(): run a policy across a batch of verified envs and report success rate,
   steps-vs-oracle efficiency, and difficulty-stratified breakdown — the kind of eval
   report you would run a candidate policy against.

2. code-vs-pixel contrast(): on the SAME saved frames, compare the frame-exact code-truth
   objective signal against a perception model reading pixels. The default perception model
   is a deterministic, offline pixel detector (no API); with --vlm --live it is Claude vision.
   An enemy patrol occludes the can, so the pixel model mis-reports the pickup while code
   truth stays exact — empirically demonstrating GI's founding rationale.
"""

from __future__ import annotations

import io
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from harness.dsl.schema import EnvSpec
from harness.verifier import verify
from harness.gym_env import make_from_spec
from harness.rollout import run_episode
from harness.agents.scripted import ScriptedOracle
from harness.engine import renderer as R
from harness.engine.world import TILE


# ── 1) scorecard ─────────────────────────────────────────────────────────────────

def scorecard(specs: Dict[str, dict], policy_factory: Callable[[], object] = None,
              epsilon: float = 0.12, seed: int = 0) -> dict:
    """specs: name -> spec dict. Returns per-env rows + aggregates (by difficulty)."""
    if policy_factory is None:
        policy_factory = lambda: ScriptedOracle(epsilon=epsilon, seed=seed)
    rows = []
    for name, spec in specs.items():
        vr = verify(spec)
        env = make_from_spec(spec)
        out = run_episode(env, policy_factory(), seed=seed)
        oracle = out["oracle_len"] or 1
        eff = round(oracle / max(1, out["steps"]), 3) if out["won"] else 0.0
        rows.append({
            "env": name, "difficulty": vr.difficulty, "won": out["won"],
            "steps": out["steps"], "oracle": oracle, "efficiency": eff,
            "reward": out["total_reward"],
        })
    by_diff = defaultdict(list)
    for r in rows:
        by_diff[r["difficulty"]].append(r)
    strata = {}
    for d, rs in by_diff.items():
        strata[d] = {
            "n": len(rs),
            "success_rate": round(sum(r["won"] for r in rs) / len(rs), 3),
            "mean_efficiency": round(np.mean([r["efficiency"] for r in rs]), 3),
        }
    agg = {
        "n": len(rows),
        "success_rate": round(sum(r["won"] for r in rows) / max(1, len(rows)), 3),
        "mean_efficiency": round(np.mean([r["efficiency"] for r in rows]), 3),
    }
    return {"rows": rows, "strata": strata, "aggregate": agg}


def format_scorecard(sc: dict) -> str:
    lines = ["  env                 difficulty  won  steps  oracle  eff",
             "  " + "-" * 58]
    for r in sc["rows"]:
        lines.append(f"  {r['env']:<19} {r['difficulty']:<10} "
                     f"{'Y' if r['won'] else 'N':^4} {r['steps']:>5} {r['oracle']:>6}  {r['efficiency']:.2f}")
    lines.append("  " + "-" * 58)
    for d in ("easy", "medium", "hard", "expert"):
        if d in sc["strata"]:
            s = sc["strata"][d]
            lines.append(f"  [{d:<6}] n={s['n']}  success={s['success_rate']:.0%}  mean_eff={s['mean_efficiency']:.2f}")
    a = sc["aggregate"]
    lines.append(f"  OVERALL   success={a['success_rate']:.0%}  mean_eff={a['mean_efficiency']:.2f}  (n={a['n']})")
    return "\n".join(lines)


# ── 2) code-vs-pixel contrast ────────────────────────────────────────────────────

def pixel_can_visible(frame: Image.Image, cell: Tuple[int, int], tile: int = TILE) -> bool:
    """A simple offline pixel-presence check: are 'can'-colored (light) pixels present at the
    can's location? It is a deterministic, no-API STAND-IN for a perception model (the real
    thing is a Claude VLM via --vlm --live). Like any pixel perceiver it is fooled by occlusion,
    which is the point. The count threshold sits between an occluded can (~24 px peeking) and a
    clear can (~156 px)."""
    x, y = cell
    box = (x * tile, y * tile, x * tile + tile, y * tile + tile)
    arr = np.asarray(frame.crop(box))
    light = (arr[:, :, 0] > 180) & (arr[:, :, 1] > 190) & (arr[:, :, 2] > 200)
    return int(light.sum()) > 80


def _bench_us(fn, iters: int = 2000) -> float:
    """Median microseconds per call, after warmup — a stable, defensible timing (not a single
    cold perf_counter sample)."""
    import statistics
    fn(); fn()
    samples = []
    for _ in range(iters):
        t = time.perf_counter(); fn(); samples.append(time.perf_counter() - t)
    return round(statistics.median(samples) * 1e6, 3)


def _vlm_judge_frame(frame: Image.Image, question: str, model: str) -> Tuple[bool, float]:
    """Live Claude-vision judge for one frame -> (yes, seconds). Requires ANTHROPIC_API_KEY."""
    import base64, anthropic
    client = anthropic.Anthropic()
    buf = io.BytesIO(); frame.save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode()
    t0 = time.time()
    resp = client.messages.create(
        model=model, max_tokens=10,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
            {"type": "text", "text": question + " Answer only 'yes' or 'no'."},
        ]}],
    )
    dt = time.time() - t0
    txt = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").lower()
    return ("yes" in txt), dt


def run_contrast(spec: dict, use_vlm: bool = False, model: str = "claude-sonnet-4-5") -> dict:
    """Replay a pickup episode; compare frame-exact code truth vs a pixel/VLM perception
    model detecting 'the can has been picked up'. Returns a timeline + summary stats."""
    if not isinstance(spec, EnvSpec):
        spec = EnvSpec(**spec)
    cans = spec.by_type("can")
    if not cans:
        raise ValueError("run_contrast needs a spec with a 'can' entity")
    can = cans[0]
    can_cell = tuple(can.pos)

    env = make_from_spec(spec)
    out = run_episode(env, ScriptedOracle(), collect_frames=True)
    frames = out["frames"]
    trace = out["trace"]

    # Per-frame detection. Code truth reads the (precomputed) engine state — a predicate
    # membership check; perception must read the pixels. Both answer "has the can been picked up".
    timeline = []
    for i, fr in enumerate(frames):
        cs = trace[i - 1]["code_state"] if i > 0 else {"held": []}
        code_picked = can.id in cs.get("held", [])
        if use_vlm:
            perc_picked, _ = _vlm_judge_frame(
                fr, "Has the small can been picked up (is it no longer on the table)?", model)
        else:
            perc_picked = not pixel_can_visible(fr, can_cell)
        timeline.append({"frame": i, "code": code_picked, "perc": perc_picked})

    code_first = next((t["frame"] for t in timeline if t["code"]), None)
    perc_first = next((t["frame"] for t in timeline if t["perc"]), None)
    disagreements = sum(1 for t in timeline if t["code"] != t["perc"])

    # Defensible timing: warmed-up median over many iterations on a representative frame.
    rep = frames[max(0, (code_first or 1) - 1)]
    rep_state = trace[max(0, (code_first or 1) - 2)]["code_state"] if len(trace) > 1 else {"held": []}
    code_us = _bench_us(lambda: can.id in rep_state.get("held", []))
    if use_vlm:
        # VLM timing comes from real calls; re-time a few for a median seconds/frame
        import statistics
        secs = [_vlm_judge_frame(rep, "Has the small can been picked up?", model)[1] for _ in range(3)]
        perc_us, perc_s = None, round(statistics.median(secs), 3)
    else:
        perc_us, perc_s = _bench_us(lambda: pixel_can_visible(rep, can_cell)), None

    return {
        "spec_name": spec.name,
        "can_cell": list(can_cell),
        "n_frames": len(frames),
        "timeline": timeline,
        "code_first_true": code_first,
        "perc_first_true": perc_first,
        "latency_frames": (None if (code_first is None or perc_first is None)
                           else perc_first - code_first),
        "disagreements": disagreements,
        "mode": "vlm" if use_vlm else "pixel",
        "code_time_us": code_us,
        "perc_time_us": perc_us,   # None in vlm mode
        "perc_time_s": perc_s,     # None in pixel mode
        "frames": frames,          # kept for strip rendering; drop before JSON dump
    }


def render_contrast_strip(contrast: dict, path: str) -> None:
    """Save a side-by-side timeline: top = code-truth, bottom = perception, mismatches red."""
    tl = contrast["timeline"]
    n = len(tl)
    cw, pad, top = 14, 40, 46
    W = pad + n * cw + 20
    H = top + 120
    img = Image.new("RGB", (W, H), (16, 16, 22))
    d = ImageDraw.Draw(img)
    d.text((8, 6), f"code-truth vs {contrast['mode']}-perception  |  "
                   f"'{contrast['spec_name']}'", fill=(210, 212, 228))
    row_code, row_perc = top + 16, top + 66
    d.text((4, row_code - 14), "code", fill=(150, 154, 174))
    d.text((4, row_perc - 14), contrast["mode"], fill=(150, 154, 174))
    for i, t in enumerate(tl):
        x = pad + i * cw
        code_c = (80, 220, 130) if t["code"] else (60, 62, 74)
        perc_c = (80, 220, 130) if t["perc"] else (60, 62, 74)
        d.rectangle([x, row_code, x + cw - 2, row_code + 24], fill=code_c)
        d.rectangle([x, row_perc, x + cw - 2, row_perc + 24], fill=perc_c)
        if t["code"] != t["perc"]:
            d.rectangle([x, row_perc, x + cw - 2, row_perc + 24], outline=(235, 70, 70), width=2)
    dis = contrast["disagreements"]
    msg = (f"{contrast['mode']} perception disagrees with code truth on {dis}/{len(tl)} frames "
           f"(occlusion false positives); code truth: exact.")
    d.text((8, H - 26), msg, fill=(235, 120, 90))
    img.save(path)
