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
from math import gcd
from typing import Dict, FrozenSet, List, Optional, Tuple

Cell = Tuple[int, int]

ACTIONS = ("up", "down", "left", "right")  # interact is a runtime no-op, excluded from search
WAIT = "wait"                              # stay one tick (lets a patrolling enemy pass)
DELTAS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


def _lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b) if a and b else max(a, b, 1)


def plan_actions(level: "Level") -> Tuple[str, ...]:
    """Search/agent action set. WAIT is only useful (and only added) when enemies exist."""
    return ACTIONS + ((WAIT,) if level.enemy_period > 1 else ())


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
    enemies: Tuple[Tuple[Cell, ...], ...] = ()   # each enemy = its per-tick patrol cycle of cells
    enemy_period: int = 1                         # lcm of cycle lengths (1 = no time dependence)

    def blocked_tile(self, x: int, y: int) -> bool:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return True
        return self.tiles[y][x] in (1, 2)  # wall or hazard

    def enemy_cells(self, phase: int) -> FrozenSet[Cell]:
        """Cells occupied by enemies at this time phase — deterministic, so avoidance is a
        solvable timing puzzle, not luck."""
        if not self.enemies:
            return frozenset()
        return frozenset(cyc[phase % len(cyc)] for cyc in self.enemies)


@dataclass(frozen=True)
class GridState:
    agent: Cell
    held: FrozenSet[str]                       # picked key/can/coin ids
    crates: Tuple[Tuple[str, int, int], ...]   # sorted (id, x, y)
    phase: int = 0                             # time mod enemy_period (always 0 with no enemies)

    def crate_pos(self) -> Dict[Cell, str]:
        return {(x, y): cid for cid, x, y in self.crates}


def build_level(spec) -> Level:
    doors, keys, coins, cans, crates0, tables = {}, {}, {}, {}, {}, set()
    exit_cell = None
    enemies = []
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
        elif e.type == "enemy":
            # patrol is the per-tick cycle of cells; default to a stationary sentinel at pos
            cyc = tuple(tuple(c) for c in (e.patrol or [e.pos]))
            enemies.append(cyc)
    period = 1
    for cyc in enemies:
        period = _lcm(period, len(cyc))
    all_coins = frozenset(coins.values())
    return Level(
        width=spec.width, height=spec.height, tiles=spec.tiles,
        start=tuple(spec.start), exit=exit_cell, doors=doors, keys=keys, coins=coins,
        cans=cans, crates0=crates0, tables=frozenset(tables), all_coin_ids=all_coins,
        enemies=tuple(enemies), enemy_period=period,
    )


def initial_state(level: Level) -> GridState:
    crates = tuple(sorted((cid, x, y) for cid, (x, y) in level.crates0.items()))
    return _auto_pickup(level, level.start, frozenset(), crates, 0)


def _auto_pickup(level: Level, agent: Cell, held: FrozenSet[str], crates, phase: int) -> GridState:
    """Pick up keys/coins on the agent cell and cans within Manhattan<=1 (grab from table)."""
    ax, ay = agent
    h = set(held)
    if (ax, ay) in level.keys:
        h.add(level.keys[(ax, ay)])
    if (ax, ay) in level.coins:
        h.add(level.coins[(ax, ay)])
    for cid, (cx, cy) in level.cans.items():
        if abs(ax - cx) + abs(ay - cy) <= 1:
            h.add(cid)
    return GridState(agent=agent, held=frozenset(h), crates=crates, phase=phase)


def resolve(level: Level, st: GridState, action: str):
    """Core transition. Returns (new_state | None, event) where event is:
        'ok'      — the action happened (a tick passed; enemies advanced)
        'blocked' — a move into a wall/hazard/closed door/immovable crate; nothing happens, no tick
        'dead'    — the action landed the agent on a patrolling enemy (a tick passed, then death)
    A tick passes on any move or WAIT (so enemies advance exactly once per player action).
    Death uses cell-occupation at the NEW phase — identical here and in the runtime engine, so a
    plan the solver certifies safe is provably safe when replayed."""
    if action not in DELTAS and action != WAIT:
        return st, "ok"                     # interact / unknown = pure no-op, no tick

    if action == WAIT:
        agent, crates, held = st.agent, st.crates, st.held
    else:
        dx, dy = DELTAS[action]
        ax, ay = st.agent
        nx, ny = ax + dx, ay + dy
        if level.blocked_tile(nx, ny) or (nx, ny) in level.tables:
            return None, "blocked"
        if (nx, ny) in level.doors and level.doors[(nx, ny)] not in st.held:
            return None, "blocked"
        crate_at = st.crate_pos()
        if (nx, ny) in crate_at:
            bx, by = nx + dx, ny + dy
            if (level.blocked_tile(bx, by) or (bx, by) in level.tables
                    or (bx, by) in crate_at or (bx, by) in level.doors):
                return None, "blocked"
            cid = crate_at[(nx, ny)]
            crates = tuple(sorted((c, (bx if c == cid else x), (by if c == cid else y))
                                  for c, x, y in st.crates))
        else:
            crates = st.crates
        agent, held = (nx, ny), st.held

    new_phase = (st.phase + 1) % level.enemy_period
    if agent in level.enemy_cells(new_phase):          # a patrolling enemy is (or steps) here
        return None, "dead"
    return _auto_pickup(level, agent, held, crates, new_phase), "ok"


def step(level: Level, st: GridState, action: str) -> Optional[GridState]:
    """Safe successor for search / planning: the new state, or None if the action is blocked
    OR would be lethal. Planners therefore avoid death automatically."""
    nxt, event = resolve(level, st, action)
    return nxt if event == "ok" else None


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
