"""
Hand-authored canonical environments.

They double as (a) the --offline cache so a reviewer with no API key still sees the full
pipeline, and (b) regression fixtures. Each is a plain dict matching the DSL; the verifier
proves them solvable when we dump them to specs/.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


def _room(w: int, h: int) -> List[List[int]]:
    """All-wall border, floor interior."""
    g = [[1] * w for _ in range(h)]
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            g[y][x] = 0
    return g


def _vwall(g, x, y0, y1, gap=None):
    for y in range(y0, y1 + 1):
        g[y][x] = 1
    if gap is not None:
        g[gap][x] = 0


def _hwall(g, y, x0, x1, gap=None):
    for x in range(x0, x1 + 1):
        g[y][x] = 1
    if gap is not None:
        g[y][gap] = 0


# ── 1) easy: open room, grab the can ─────────────────────────────────────────────
def open_can() -> dict:
    W, H = 16, 9
    g = _room(W, H)
    g[6][8] = 3; g[6][9] = 3  # a little grass for flavor
    return dict(
        name="Open Room - Grab the Can", width=W, height=H, tiles=g,
        entities=[
            {"type": "player_start", "id": "p", "pos": [2, 4]},
            {"type": "table", "id": "t1", "pos": [12, 4]},
            {"type": "can", "id": "can1", "pos": [12, 4], "on": "t1"},
            {"type": "ball", "id": "b1", "pos": [8, 2]},
            {"type": "exit", "id": "e", "pos": [13, 7]},
        ],
        objective=[{"kind": "holding", "item": "can1"}],
        objective_text="pick up the can from the table",
        time_limit=200,
    )


# ── 2) medium: push a crate off the key, unlock the door, return the can ─────────
def key_crate_return() -> dict:
    W, H = 19, 11
    g = _room(W, H)
    _vwall(g, 9, 1, H - 2, gap=5)              # wall dividing the room, door-gap at y=5
    return dict(
        name="Key, Crate & Return", width=W, height=H, tiles=g,
        entities=[
            {"type": "player_start", "id": "p", "pos": [2, 5]},
            {"type": "crate", "id": "c1", "pos": [4, 5]},     # blocks the corridor to the key
            {"type": "key", "id": "k1", "pos": [2, 8]},
            {"type": "door", "id": "d1", "pos": [9, 5], "requires": "k1"},
            {"type": "table", "id": "t1", "pos": [16, 5]},
            {"type": "can", "id": "can1", "pos": [16, 5], "on": "t1"},
            {"type": "ball", "id": "b1", "pos": [13, 2]},
        ],
        objective=[{"kind": "holding", "item": "can1"}, {"kind": "at_start"}],
        objective_text="grab the key, unlock the door, take the can, return to start",
        time_limit=300,
    )


# ── 3) hard: three rooms, chained keys ──────────────────────────────────────────
def three_rooms() -> dict:
    W, H = 25, 11
    g = _room(W, H)
    _vwall(g, 8, 1, H - 2, gap=5)              # room A | B door at (8,5)
    _vwall(g, 16, 1, H - 2, gap=5)             # room B | C door at (16,5)
    return dict(
        name="Three Rooms - Chained Keys", width=W, height=H, tiles=g,
        entities=[
            {"type": "player_start", "id": "p", "pos": [2, 5]},
            {"type": "key", "id": "kA", "pos": [3, 8]},       # room A key opens door A->B
            {"type": "door", "id": "dA", "pos": [8, 5], "requires": "kA"},
            {"type": "key", "id": "kB", "pos": [12, 2]},      # room B key opens door B->C
            {"type": "crate", "id": "c1", "pos": [12, 5]},    # a crate in room B
            {"type": "door", "id": "dB", "pos": [16, 5], "requires": "kB"},
            {"type": "table", "id": "t1", "pos": [22, 5]},
            {"type": "can", "id": "can1", "pos": [22, 5], "on": "t1"},
            {"type": "ball", "id": "b1", "pos": [20, 8]},
            {"type": "exit", "id": "e", "pos": [22, 8]},
        ],
        objective=[{"kind": "holding", "item": "can1"}, {"kind": "reached_exit"}],
        objective_text="unlock two chained doors, grab the can, reach the exit",
        time_limit=400,
    )


# ── 4) eval variety: coins around hazards ────────────────────────────────────────
def coins_hazard() -> dict:
    W, H = 17, 11
    g = _room(W, H)
    for x in range(4, 13):
        g[5][x] = 2                            # a lava strip
    g[5][8] = 0                                # one safe crossing
    return dict(
        name="Coins & Hazard", width=W, height=H, tiles=g,
        entities=[
            {"type": "player_start", "id": "p", "pos": [2, 2]},
            {"type": "coin", "id": "co1", "pos": [5, 2]},
            {"type": "coin", "id": "co2", "pos": [14, 2]},
            {"type": "coin", "id": "co3", "pos": [14, 8]},
            {"type": "exit", "id": "e", "pos": [2, 8]},
        ],
        objective=[{"kind": "collected_all_coins"}, {"kind": "reached_exit"}],
        objective_text="collect all coins across the lava, then reach the exit",
        time_limit=300,
    )


# ── 5) eval variety: push a crate onto a target pad ──────────────────────────────
def push_delivery() -> dict:
    W, H = 16, 10
    g = _room(W, H)
    return dict(
        name="Crate Delivery", width=W, height=H, tiles=g,
        entities=[
            {"type": "player_start", "id": "p", "pos": [2, 5]},
            {"type": "crate", "id": "c1", "pos": [5, 5]},
            {"type": "ball", "id": "b1", "pos": [10, 2]},
            {"type": "exit", "id": "e", "pos": [13, 5]},
        ],
        objective=[{"kind": "item_at", "item": "c1", "cell": [11, 5]},
                   {"kind": "reached_exit"}],
        objective_text="push the crate onto the pad at (11,5), then reach the exit",
        time_limit=250,
    )


# ── 6) contrast scene: an enemy patrol occludes the can early (at spawn) ─────────
def occlusion_can() -> dict:
    """Used by the code-vs-pixel contrast. The enemy sprite sits ON the can for the first few
    frames (then moves away), so a pixel detector loses sight of the can early and FALSELY
    reports 'picked up' while code truth stays exact until the agent actually grabs it."""
    W, H = 16, 9
    g = _room(W, H)
    return dict(
        name="Occlusion - Can + Patrol", width=W, height=H, tiles=g,
        entities=[
            {"type": "player_start", "id": "p", "pos": [2, 4]},
            {"type": "table", "id": "t1", "pos": [11, 4]},
            {"type": "can", "id": "can1", "pos": [11, 4], "on": "t1"},
            {"type": "enemy", "id": "en1", "pos": [11, 4],
             # sits ON the can early (occluding it), then moves away before the real pickup
             "patrol": [[11, 4], [11, 6], [11, 6], [11, 6]]},
            {"type": "exit", "id": "e", "pos": [13, 7]},
        ],
        objective=[{"kind": "holding", "item": "can1"}, {"kind": "reached_exit"}],
        objective_text="grab the can (an enemy keeps crossing it), then reach the exit",
        time_limit=200,
    )


ALL = {
    "open_can": open_can,
    "key_crate_return": key_crate_return,
    "three_rooms": three_rooms,
    "coins_hazard": coins_hazard,
    "push_delivery": push_delivery,
    "occlusion_can": occlusion_can,
}

# the three commands the live demo mirrors
DEMO_COMMANDS = {
    "open_can": "an open room with a can on a table; pick it up",
    "key_crate_return": "a locked door, the key behind a pushable crate, bring the can back to start",
    "three_rooms": "three rooms, each door's key is in the previous room, and a crate in the middle room",
}
