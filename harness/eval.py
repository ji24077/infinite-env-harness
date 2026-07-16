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
    """Deterministic pixel detector: are 'can'-colored (light) pixels present in the can's
    tile region? Fooled by occlusion — exactly the failure GI cites for pixel perception."""
    x, y = cell
    box = (x * tile, y * tile, x * tile + tile, y * tile + tile)
    arr = np.asarray(frame.crop(box)).astype(int)
    light = (arr[:, :, 0] > 180) & (arr[:, :, 1] > 190) & (arr[:, :, 2] > 200)
    # threshold set between an occluded can (~24 px peeking) and a clear can (~156 px):
    # occlusion drops the count into "not visible", so the detector mis-reads a pickup.
    return int(light.sum()) > 80


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
    can = spec.by_type("can")[0]
    can_cell = tuple(can.pos)

    env = make_from_spec(spec)
    out = run_episode(env, ScriptedOracle(), collect_frames=True)
    frames = out["frames"]
    trace = out["trace"]

    timeline = []
    code_us, per_perc = [], []
    for i, fr in enumerate(frames):
        # code truth: O(1) dict lookup of the predicate — frame-exact
        t0 = time.perf_counter()
        cs = trace[i - 1]["code_state"] if i > 0 else {"held": []}
        code_picked = can.id in cs.get("held", [])
        code_us.append((time.perf_counter() - t0) * 1e6)

        # perception: pixel detector (offline) or VLM (live)
        if use_vlm:
            yes, dt = _vlm_judge_frame(fr, f"Has the small can been picked up (is it no longer on the table)?", model)
            perc_picked = yes
            per_perc.append(dt)
        else:
            t1 = time.perf_counter()
            visible = pixel_can_visible(fr, can_cell)
            per_perc.append((time.perf_counter() - t1) * 1e6)
            perc_picked = not visible
        timeline.append({"frame": i, "code": code_picked, "perc": perc_picked})

    code_first = next((t["frame"] for t in timeline if t["code"]), None)
    perc_first = next((t["frame"] for t in timeline if t["perc"]), None)
    disagreements = sum(1 for t in timeline if t["code"] != t["perc"])

    return {
        "spec_name": spec.name,
        "can_cell": can_cell,
        "n_frames": len(frames),
        "timeline": timeline,
        "code_first_true": code_first,
        "perc_first_true": perc_first,
        "latency_frames": (None if (code_first is None or perc_first is None)
                           else perc_first - code_first),
        "disagreements": disagreements,
        "mode": "vlm" if use_vlm else "pixel",
        "code_time_us": round(float(np.mean(code_us)), 3),
        "perc_time_us": round(float(np.mean(per_perc)), 3) if not use_vlm else None,
        "perc_time_s": round(float(np.mean(per_perc)), 3) if use_vlm else None,
        "frames": frames,   # kept for strip rendering; drop before JSON dump
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
    d.text((8, 6), f"code-truth vs {contrast['mode']}-perception  —  "
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
    lat = contrast["latency_frames"]
    if lat is None:
        msg = f"perception disagreed with code truth on {contrast['disagreements']} frames"
    elif lat < 0:
        msg = (f"{contrast['mode']} model mis-fired the pickup {abs(lat)} frames EARLY "
               f"(occlusion); wrong on {contrast['disagreements']} frames. code truth: exact.")
    else:
        msg = (f"{contrast['mode']} model detected the pickup {lat} frames late; "
               f"wrong on {contrast['disagreements']} frames. code truth: exact.")
    d.text((8, H - 26), msg, fill=(235, 120, 90))
    img.save(path)
