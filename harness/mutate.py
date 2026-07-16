"""
Mutation engine — turn one verified environment into a curated family of new ones.

This is the "infinite" in infinite generation without another LLM call: structural edits
(relocate entities, add/remove interior walls, drop a hazard, shift the exit) followed by
re-verification, so every survivor is still provably solvable and gets a fresh difficulty
label from its oracle-plan length.

ACCEL-style curation (Parker-Holder et al. 2022): instead of keeping random survivors, we
prefer high-REGRET variants — ones the optimal oracle solves but a myopic greedy agent fails.
Regret is approximated API-free as (oracle_success - greedy_success), a proxy for how much
learnable signal the level carries. Falls back to difficulty spread when regret is flat.
"""

from __future__ import annotations

import copy
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from harness.dsl.schema import EnvSpec
from harness.verifier import verify, VerifyResult
from harness.gym_env import make_from_spec
from harness.rollout import run_episode
from harness.agents.scripted import ScriptedOracle
from harness.agents.greedy import GreedyHeuristic


def _walkable(spec: dict, x: int, y: int) -> bool:
    if not (1 <= x < spec["width"] - 1 and 1 <= y < spec["height"] - 1):
        return False
    if spec["tiles"][y][x] in (1, 2):
        return False
    occupied = {tuple(e["pos"]) for e in spec["entities"]}
    return (x, y) not in occupied


def _free_cells(spec: dict) -> List[Tuple[int, int]]:
    return [(x, y) for y in range(1, spec["height"]-1) for x in range(1, spec["width"]-1)
            if _walkable(spec, x, y)]


def _mutations(spec: dict, rng) -> List[Tuple[str, dict]]:
    """Yield (description, mutated_spec_dict) candidates."""
    out = []
    free = _free_cells(spec)
    movable = [e for e in spec["entities"]
               if e["type"] in ("key", "coin", "ball", "exit", "table", "can", "crate")]

    # relocate an entity
    for e in movable:
        if not free:
            break
        nx, ny = free[rng.integers(0, len(free))]
        m = copy.deepcopy(spec)
        for me in m["entities"]:
            if me["id"] == e["id"]:
                me["pos"] = [int(nx), int(ny)]
                # keep can+table together
                if me["type"] == "can" and me.get("on"):
                    for t in m["entities"]:
                        if t["id"] == me["on"]:
                            t["pos"] = [int(nx), int(ny)]
        out.append((f"move {e['type']} {e['id']}", m))

    # add an interior wall block
    if free:
        nx, ny = free[rng.integers(0, len(free))]
        m = copy.deepcopy(spec)
        m["tiles"][ny][nx] = 1
        out.append((f"add wall at ({nx},{ny})", m))

    # add a hazard tile
    if free:
        nx, ny = free[rng.integers(0, len(free))]
        m = copy.deepcopy(spec)
        m["tiles"][ny][nx] = 2
        out.append((f"add hazard at ({nx},{ny})", m))

    return out


def _regret(spec: EnvSpec) -> Tuple[float, dict]:
    """oracle_success(=1 for verified) - greedy_success. Returns (regret, detail)."""
    env = make_from_spec(spec)
    g = run_episode(env, GreedyHeuristic(), collect_frames=False)
    env2 = make_from_spec(spec)
    o = run_episode(env2, ScriptedOracle(), collect_frames=False)
    regret = (1.0 if o["won"] else 0.0) - (1.0 if g["won"] else 0.0)
    return regret, {"greedy_won": g["won"], "oracle_won": o["won"],
                    "greedy_steps": g["steps"], "oracle_steps": o["steps"]}


def mutate(base_spec: dict, n: int = 10, seed: int = 0,
           accel: bool = True, log: Optional[Callable[[str], None]] = None) -> List[dict]:
    """Return up to n verified variants, each: {spec, name, difficulty, plan_len, regret}.
    With accel=True, survivors are ranked by regret (high first)."""
    emit = log or (lambda *_: None)
    rng = np.random.default_rng(seed)
    survivors: List[dict] = []
    seen_sigs = set()
    tries = 0
    max_tries = n * 12

    while len(survivors) < n and tries < max_tries:
        tries += 1
        cands = _mutations(base_spec, rng)
        if not cands:
            break
        desc, cand = cands[rng.integers(0, len(cands))]
        sig = (tuple(tuple(r) for r in cand["tiles"]),
               tuple((e["id"], tuple(e["pos"])) for e in cand["entities"]))
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)
        vr = verify(cand)
        if not vr.ok:
            continue
        spec = EnvSpec(**cand)
        regret, detail = _regret(spec) if accel else (0.0, {})
        survivors.append({
            "spec": cand, "name": f"{base_spec['name']} · {desc}",
            "difficulty": vr.difficulty, "plan_len": vr.plan_len,
            "regret": regret, "detail": detail,
        })
        emit(f"  variant: {desc:28s} difficulty={vr.difficulty:6s} plan={vr.plan_len:3d} regret={regret:+.0f}")

    if accel:
        survivors.sort(key=lambda s: (s["regret"], s["plan_len"]), reverse=True)
    return survivors
