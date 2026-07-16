"""
Authoritative discrete grid semantics.

Both the runtime engine and the verifier's BFS import THIS module, so a plan the solver
finds is guaranteed to replay identically in the engine. That shared source of truth is
what upgrades "the LLM usually makes playable levels" into "every accepted level is
provably beatable" (see verifier.py L2).

Kept deliberately SOUND rather than maximally complete: when a rule is ambiguous we take
the conservative (more-restrictive) choice, so the solver never claims a level is solvable
when it is not. Over-strictness only costs an extra repair-loop regeneration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

Cell = Tuple[int, int]

ACTIONS = ("up", "down", "left", "right")  # interact is a runtime no-op, excluded from search
DELTAS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


@dataclass
class Level:
    """Static level data extracted once from an EnvSpec."""
    width: int
    height: int
    tiles: List[List[int]]
    start: Cell
    exit: Optional[Cell]
    doors: Dict[Cell, str]          # cell -> required key id
    keys: Dict[Cell, str]           # cell -> key id
    coins: Dict[Cell, str]          # cell -> coin id
    cans: Dict[str, Cell]           # can id -> cell (proximity pickup)
    crates0: Dict[str, Cell]        # can id -> initial cell
    tables: FrozenSet[Cell]         # static blockers
    all_coin_ids: FrozenSet[str]

    def blocked_tile(self, x: int, y: int) -> bool:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return True
        return self.tiles[y][x] in (1, 2)  # wall or hazard


@dataclass(frozen=True)
class GridState:
    agent: Cell
    held: FrozenSet[str]                       # picked key/can/coin ids
    crates: Tuple[Tuple[str, int, int], ...]   # sorted (id, x, y)

    def crate_pos(self) -> Dict[Cell, str]:
        return {(x, y): cid for cid, x, y in self.crates}


def build_level(spec) -> Level:
    doors, keys, coins, cans, crates0, tables = {}, {}, {}, {}, {}, set()
    exit_cell = None
    for e in spec.entities:
        p = tuple(e.pos)
        if e.type == "exit":
            exit_cell = p
        elif e.type == "door":
            doors[p] = e.requires
        elif e.type == "key":
            keys[p] = e.id
        elif e.type == "coin":
            coins[p] = e.id
        elif e.type == "can":
            cans[e.id] = p
        elif e.type == "crate":
            crates0[e.id] = p
        elif e.type == "table":
            tables.add(p)
    all_coins = frozenset(coins.values())
    return Level(
        width=spec.width, height=spec.height, tiles=spec.tiles,
        start=tuple(spec.start), exit=exit_cell, doors=doors, keys=keys, coins=coins,
        cans=cans, crates0=crates0, tables=frozenset(tables), all_coin_ids=all_coins,
    )


def initial_state(level: Level) -> GridState:
    crates = tuple(sorted((cid, x, y) for cid, (x, y) in level.crates0.items()))
    st = GridState(agent=level.start, held=frozenset(), crates=crates)
    return _auto_pickup(level, st)


def _auto_pickup(level: Level, st: GridState) -> GridState:
    """Pick up keys/coins on the agent cell and cans within Manhattan<=1 (grab from table)."""
    ax, ay = st.agent
    held = set(st.held)
    if (ax, ay) in level.keys:
        held.add(level.keys[(ax, ay)])
    if (ax, ay) in level.coins:
        held.add(level.coins[(ax, ay)])
    for cid, (cx, cy) in level.cans.items():
        if abs(ax - cx) + abs(ay - cy) <= 1:
            held.add(cid)
    if held != set(st.held):
        return GridState(agent=st.agent, held=frozenset(held), crates=st.crates)
    return st


def step(level: Level, st: GridState, action: str) -> Optional[GridState]:
    """Return the successor state, or None if the action is blocked. Deterministic."""
    if action not in DELTAS:
        return st  # interact / no-op
    dx, dy = DELTAS[action]
    ax, ay = st.agent
    nx, ny = ax + dx, ay + dy

    if level.blocked_tile(nx, ny):
        return None
    if (nx, ny) in level.tables:
        return None

    # closed door?
    if (nx, ny) in level.doors:
        if level.doors[(nx, ny)] not in st.held:
            return None  # need the key

    crate_at = st.crate_pos()
    if (nx, ny) in crate_at:
        # push the crate one cell further, if the destination is clear
        bx, by = nx + dx, ny + dy
        if level.blocked_tile(bx, by) or (bx, by) in level.tables:
            return None
        if (bx, by) in crate_at or (bx, by) in level.doors:
            return None
        cid = crate_at[(nx, ny)]
        new_crates = tuple(sorted(
            (c, (bx if c == cid else x), (by if c == cid else y))
            for c, x, y in st.crates
        ))
        moved = GridState(agent=(nx, ny), held=st.held, crates=new_crates)
        return _auto_pickup(level, moved)

    moved = GridState(agent=(nx, ny), held=st.held, crates=st.crates)
    return _auto_pickup(level, moved)


def predicate_true(level: Level, pred, st: GridState) -> bool:
    kind = pred.kind
    if kind == "reached_exit":
        return level.exit is not None and st.agent == level.exit
    if kind == "holding":
        return pred.item in st.held
    if kind == "at_start":
        return st.agent == level.start
    if kind == "item_at":
        target = tuple(pred.cell)
        cp = st.crate_pos()
        for cell, cid in cp.items():
            if cid == pred.item:
                return cell == target
        # non-crate item: satisfied if held-and-at? treat pickup items as location-agnostic
        return False
    if kind == "collected_all_coins":
        return level.all_coin_ids <= st.held
    return False


def objective_satisfied(level: Level, objective, st: GridState) -> bool:
    return all(predicate_true(level, p, st) for p in objective)
