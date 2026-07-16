"""
Mutation engine — turn one verified environment into a curated family of new ones.

This is the "infinite" in infinite generation without another LLM call: structural edits
(relocate entities, add/remove interior walls, drop a hazard, shift the exit) followed by
re-verification, so every survivor is still provably solvable and gets a fresh difficulty
label from its oracle-plan length.

ACCEL-INSPIRED curation (Parker-Holder et al. 2022): instead of keeping random survivors, we
prefer higher-regret variants — ones the optimal oracle solves but a myopic greedy agent fails.
This is a cheap, API-free BINARY regret PROXY: regret = oracle_success(=1 for any re-verified
variant) - greedy_success ∈ {0, 1}. It is NOT the full minimax-regret estimator of PAIRED/ACCEL
(no antagonist, no per-state regret) — just a learnability signal that correlates with "needs
planning". Ties break on oracle-plan length (difficulty).
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
        # keep a can and its table co-located regardless of which one moved
        if e["type"] == "can" and e.get("on"):
            for t in m["entities"]:
                if t["id"] == e["on"]:
                    t["pos"] = [int(nx), int(ny)]
        elif e["type"] == "table":
            for c in m["entities"]:
                if c["type"] == "can" and c.get("on") == e["id"]:
                    c["pos"] = [int(nx), int(ny)]
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


# ── compounding, generative evolution (MAP-Elites over objective x difficulty) ───────
# The single-edit mutate() above never changes the objective, entity roster, or topology, so its
# output is a jittered neighborhood of ONE template. evolve() fixes that: it keeps a MAP-Elites
# archive keyed by (objective-signature, difficulty-band), re-samples SURVIVORS as parents, and
# applies 1..k of the richer operators.py edits — so children compound edits across a lineage, and
# the objective / entities / rooms genuinely change. Every child still passes the unchanged verify().
#
# Honest scope (what this is NOT): the two archive descriptors are objective and difficulty; entity
# roster and room topology are changed by the operators but are NOT descriptor axes, so the archive
# does not explicitly select for topological spread. The ACCEL-inspired regret proxy is computed
# ONCE on the final archive to RANK it (accel=True) — it is a post-hoc ranking, not a live
# minimax-regret selection pressure during evolution. Both are deliberate simplicity/speed choices.

from harness.operators import ALL_OPERATORS, within_caps


def _obj_sig(spec: dict):
    return tuple(sorted((p["kind"], p.get("item"),
                         tuple(p["cell"]) if p.get("cell") else None) for p in spec["objective"]))


def _ext_sig(spec: dict):
    """Full structural signature: tiles + typed entity roster + objective (dedup key)."""
    return (tuple(tuple(r) for r in spec["tiles"]),
            tuple(sorted((e["type"], tuple(e["pos"]), e.get("requires")) for e in spec["entities"])),
            _obj_sig(spec))


def evolve(seeds: List[dict], generations: int = 80, max_ops: int = 3, seed: int = 0,
           accel: bool = True, log: Optional[Callable[[str], None]] = None) -> List[dict]:
    """Grow a diverse archive from `seeds` by compounding operators.py edits under verify().
    Returns records {spec, name, difficulty, plan_len, regret, lineage, ops}, best-regret-first."""
    emit = log or (lambda *_: None)
    rng = np.random.default_rng(seed)
    archive: Dict[tuple, dict] = {}          # (obj_sig, difficulty) -> best record in that cell
    seen = set()

    def _insert(spec: dict, vr: VerifyResult, lineage: int, ops: list):
        # quality within a MAP-Elites cell = harder (longer plan); regret is computed once at the
        # end for the final archive only (an oracle+greedy rollout per candidate is too slow).
        desc = (_obj_sig(spec), vr.difficulty)
        cur = archive.get(desc)
        if cur is None or vr.plan_len > cur["plan_len"]:
            archive[desc] = {"spec": spec, "name": spec.get("name", "evolved"),
                             "difficulty": vr.difficulty, "plan_len": vr.plan_len,
                             "regret": 0.0, "lineage": lineage, "ops": ops}

    for s in seeds:
        vr = verify(s)
        if vr.ok:
            seen.add(_ext_sig(s))
            _insert(copy.deepcopy(s), vr, 0, [])
    emit(f"  seeded archive: {len(archive)} cells from {len(seeds)} fixtures")

    for gen in range(generations):
        if not archive:
            break
        parent = list(archive.values())[int(rng.integers(0, len(archive)))]  # descriptor-uniform
        cand = copy.deepcopy(parent["spec"])
        ops: List[str] = []
        for _ in range(1 + int(rng.integers(0, max_ops))):
            op = ALL_OPERATORS[int(rng.integers(0, len(ALL_OPERATORS)))]
            out = op(cand, rng)
            if out is not None:
                cand = out
                ops.append(op.__name__)
        if not ops or not within_caps(cand):
            continue
        sig = _ext_sig(cand)
        if sig in seen:
            continue
        seen.add(sig)
        vr = verify(cand)
        if not vr.ok:
            continue
        cand["name"] = f"evolved ·gen{gen}· {'+'.join(ops)}"[:70]
        _insert(cand, vr, parent["lineage"] + 1, ops)
        emit(f"  gen{gen:3d} lineage={parent['lineage']+1} {vr.difficulty:6s} "
             f"plan={vr.plan_len:3d}  {'+'.join(ops)[:44]}")

    records = list(archive.values())
    if accel:                                 # ACCEL regret, computed once on the final archive
        for r in records:
            r["regret"] = _regret(EnvSpec(**r["spec"]))[0]
    return sorted(records, key=lambda r: (r["regret"], r["plan_len"]), reverse=True)
