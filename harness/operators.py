"""
Edit operators for the compounding mutation engine (harness/mutate.py: evolve()).

Each operator is a *dumb proposer*: op(spec_dict, rng) -> Optional[spec_dict]. It returns a
deep-copied, edited spec (or None if not applicable). It does NOT guarantee validity — the
unchanged verify() (L1 schema/reference + L2 solvability + L3 physics) is the single gate, so a
proposer can be reckless and let the verifier reject the ~unsolvable ones.

Crucially, unlike the original single-edit mutate() (which only relocated an entity or dropped one
wall/hazard, never touching the objective, entity roster, or topology), these operators DO change
all three — that is what turns "infinite = jittered clones of one template" into a genuinely
compositional generator. Entity/predicate vocabulary is still the fixed DSL (schema.py), so the
reachable space is combinatorially large, not literally infinite — the honest ceiling.
"""

from __future__ import annotations

import copy
from collections import Counter
from typing import List, Optional, Tuple

# ── caps: keep the accepted output inside the BFS solver's cheap regime ──────────────
# BFS state is (agent, held-subset, crate-config, enemy-phase); the exponential/combinatorial
# terms are collectibles, crates, and enemy period. These caps keep verify() well under its
# 400k-state budget so acceptance stays high (soundness is unaffected — verify still gates).
COLLECT_CAP = 6      # keys + coins + cans  -> 2^6 held-subsets
CRATE_CAP = 2
ENEMY_CAP = 3
ENTITY_CAP = 16


def within_caps(spec: dict) -> bool:
    t = Counter(e["type"] for e in spec["entities"])
    return (t["key"] + t["coin"] + t["can"] <= COLLECT_CAP
            and t["crate"] <= CRATE_CAP and t["enemy"] <= ENEMY_CAP
            and len(spec["entities"]) <= ENTITY_CAP)


# ── cell helpers ─────────────────────────────────────────────────────────────────
def _walkable(spec: dict, x: int, y: int) -> bool:
    if not (1 <= x < spec["width"] - 1 and 1 <= y < spec["height"] - 1):
        return False
    if spec["tiles"][y][x] in (1, 2):
        return False
    return (x, y) not in {tuple(e["pos"]) for e in spec["entities"]}


def _free_cells(spec: dict) -> List[Tuple[int, int]]:
    return [(x, y) for y in range(1, spec["height"] - 1) for x in range(1, spec["width"] - 1)
            if _walkable(spec, x, y)]


def _pick(rng, seq):
    return seq[int(rng.integers(0, len(seq)))]


def _fresh_id(spec: dict, prefix: str) -> str:
    existing = {e["id"] for e in spec["entities"]}
    n = 1
    while f"{prefix}{n}" in existing:
        n += 1
    return f"{prefix}{n}"


def _start(spec: dict):
    return next(tuple(e["pos"]) for e in spec["entities"] if e["type"] == "player_start")


# ── operators: geometry ─────────────────────────────────────────────────────────
def relocate(spec, rng):
    movable = [e for e in spec["entities"]
               if e["type"] in ("key", "coin", "ball", "exit", "table", "can", "crate")]
    free = _free_cells(spec)
    if not movable or not free:
        return None
    e = _pick(rng, movable)
    nx, ny = _pick(rng, free)
    m = copy.deepcopy(spec)
    for me in m["entities"]:
        if me["id"] == e["id"]:
            me["pos"] = [int(nx), int(ny)]
    # keep a can and its table co-located
    if e["type"] == "can" and e.get("on"):
        for t in m["entities"]:
            if t["id"] == e["on"]:
                t["pos"] = [int(nx), int(ny)]
    elif e["type"] == "table":
        for c in m["entities"]:
            if c["type"] == "can" and c.get("on") == e["id"]:
                c["pos"] = [int(nx), int(ny)]
    return m


def add_wall(spec, rng):
    free = _free_cells(spec)
    if not free:
        return None
    x, y = _pick(rng, free)
    m = copy.deepcopy(spec)
    m["tiles"][y][x] = 1
    return m


def add_hazard(spec, rng):
    free = _free_cells(spec)
    if not free:
        return None
    x, y = _pick(rng, free)
    m = copy.deepcopy(spec)
    m["tiles"][y][x] = 2
    return m


def _interior_tiles(spec, val):
    return [(x, y) for y in range(1, spec["height"] - 1) for x in range(1, spec["width"] - 1)
            if spec["tiles"][y][x] == val]


def remove_wall(spec, rng):
    walls = _interior_tiles(spec, 1)
    if not walls:
        return None
    x, y = _pick(rng, walls)
    m = copy.deepcopy(spec)
    m["tiles"][y][x] = 0
    return m


def remove_hazard(spec, rng):
    haz = _interior_tiles(spec, 2)
    if not haz:
        return None
    x, y = _pick(rng, haz)
    m = copy.deepcopy(spec)
    m["tiles"][y][x] = 0
    return m


def add_partition(spec, rng):
    """A full interior wall line with one gap — a real new room. verify() rejects it if the gap
    doesn't keep the objective reachable."""
    m = copy.deepcopy(spec)
    w, h = m["width"], m["height"]
    occupied = {tuple(e["pos"]) for e in m["entities"]}
    if rng.random() < 0.5 and w > 5:
        x = int(rng.integers(2, w - 2)); gap = int(rng.integers(1, h - 1))
        for y in range(1, h - 1):
            if y != gap and (x, y) not in occupied and m["tiles"][y][x] == 0:
                m["tiles"][y][x] = 1
    elif h > 5:
        y = int(rng.integers(2, h - 2)); gap = int(rng.integers(1, w - 1))
        for x in range(1, w - 1):
            if x != gap and (x, y) not in occupied and m["tiles"][y][x] == 0:
                m["tiles"][y][x] = 1
    return m


# ── operators: entity roster ─────────────────────────────────────────────────────
def add_coin(spec, rng):
    free = _free_cells(spec)
    if not free:
        return None
    x, y = _pick(rng, free)
    m = copy.deepcopy(spec)
    m["entities"].append({"type": "coin", "id": _fresh_id(m, "coin"), "pos": [int(x), int(y)]})
    return m


def add_crate(spec, rng):
    free = _free_cells(spec)
    if not free:
        return None
    x, y = _pick(rng, free)
    m = copy.deepcopy(spec)
    m["entities"].append({"type": "crate", "id": _fresh_id(m, "crate"), "pos": [int(x), int(y)]})
    return m


def add_enemy(spec, rng):
    free = set(_free_cells(spec))
    if len(free) < 2:
        return None
    start = _start(spec)
    cands = [c for c in free if c != start]
    if not cands:
        return None
    ax, ay = _pick(rng, cands)
    adj = [(ax + dx, ay + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)) if (ax + dx, ay + dy) in free]
    if not adj:
        return None
    bx, by = _pick(rng, adj)
    m = copy.deepcopy(spec)
    m["entities"].append({"type": "enemy", "id": _fresh_id(m, "enemy"), "pos": [ax, ay],
                          "patrol": [[ax, ay], [bx, by]]})
    return m


def add_key_door_pair(spec, rng):
    free = _free_cells(spec)
    if len(free) < 2:
        return None
    kx, ky = _pick(rng, free)
    rest = [c for c in free if c != (kx, ky)]
    dx, dy = _pick(rng, rest)
    m = copy.deepcopy(spec)
    kid, did = _fresh_id(m, "key"), _fresh_id(m, "door")
    m["entities"].append({"type": "key", "id": kid, "pos": [int(kx), int(ky)]})
    m["entities"].append({"type": "door", "id": did, "pos": [int(dx), int(dy)], "requires": kid})
    return m


def _remove_of_type(spec, rng, t):
    victims = [e["id"] for e in spec["entities"] if e["type"] == t]
    if not victims:
        return None
    vid = _pick(rng, victims)
    m = copy.deepcopy(spec)
    m["entities"] = [e for e in m["entities"] if e["id"] != vid and e.get("requires") != vid]
    return m


def remove_coin(spec, rng):
    return _remove_of_type(spec, rng, "coin")


def remove_enemy(spec, rng):
    return _remove_of_type(spec, rng, "enemy")


# ── operators: objective (the thing the old engine NEVER changed) ─────────────────
def mutate_objective(spec, rng):
    m = copy.deepcopy(spec)
    obj = m["objective"]
    kinds = {p["kind"] for p in obj}
    has_exit = any(e["type"] == "exit" for e in m["entities"])
    has_coin = any(e["type"] == "coin" for e in m["entities"])
    choices = []
    if not any(p["kind"] == "at_start" for p in obj):
        choices.append("add_at_start")             # "…then return home"
    if has_exit and "reached_exit" not in kinds:
        choices.append("add_reached_exit")
    if has_coin and "collected_all_coins" not in kinds:
        choices.append("add_all_coins")
    if len(obj) > 1:
        choices.append("drop_one")
    if "reached_exit" in kinds:
        choices.append("exit_to_start")
    if not choices:
        return None
    c = _pick(rng, choices)
    if c == "add_at_start":
        obj.append({"kind": "at_start"})
    elif c == "add_reached_exit":
        obj.append({"kind": "reached_exit"})
    elif c == "add_all_coins":
        obj.append({"kind": "collected_all_coins"})
    elif c == "drop_one":
        obj.pop(int(rng.integers(0, len(obj))))
    elif c == "exit_to_start":
        m["objective"] = [({"kind": "at_start"} if p["kind"] == "reached_exit" else p) for p in obj]
    return m


ALL_OPERATORS = [
    relocate, add_wall, add_hazard, remove_wall, remove_hazard, add_partition,
    add_coin, remove_coin, add_crate, add_enemy, remove_enemy, add_key_door_pair,
    mutate_objective,
]
