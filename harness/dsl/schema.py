"""
The DSL — a typed, constrained description of a 2D environment.

This single artifact is the center of the whole harness. It is, at once:
  * the LLM's generation target (produced via forced tool use),
  * the compiler's input (spec -> playable world),
  * the objective definition (a list of executable predicates = code-level ground truth),
  * the thing the verifier proves solvable.

Design split (see DESIGN.md): the LLM's tool-use schema guarantees SHAPE; this Pydantic
model + the L1 validators guarantee MEANING (reference integrity, geometry, reachability
preconditions). Structured outputs cannot enforce "the key is not inside a wall" — we do.

Tiles:  0 floor (walkable) · 1 wall (solid) · 2 hazard (impassable) · 3 grass (walkable, cosmetic)
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Literal
from pydantic import BaseModel, Field, field_validator, model_validator

Cell = Tuple[int, int]

# Entity kinds the engine understands. Kept small so the L2 solver stays sound.
ENTITY_KINDS = (
    "player_start",  # exactly one — agent spawn
    "exit",          # goal cell (for reached_exit objectives)
    "key",           # collectible; opens the door whose requires == this id
    "door",          # blocks passage until required key held; requires: <key id>
    "crate",         # pushable prop (kinematic grid logic + pymunk render)
    "can",           # pickup item (the "pick up the can" objective)
    "table",         # static prop a can rests on (cosmetic anchor)
    "ball",          # rollable prop (pure pymunk physics — the "physics engine" credential)
    "coin",          # collectible for collected_all_coins objectives
    "enemy",         # cosmetic patrol; can occlude items (used in the VLM-contrast demo)
)

PRED_KINDS = (
    "reached_exit",         # agent stands on the exit cell
    "holding",              # agent has picked up item <item>
    "at_start",             # agent is back on its spawn cell
    "item_at",              # item <item> occupies <cell> (e.g. crate delivered)
    "collected_all_coins",  # every coin picked up
)


class Entity(BaseModel):
    type: Literal[ENTITY_KINDS]  # type: ignore[valid-type]
    id: str
    pos: Cell
    requires: Optional[str] = None            # door -> key id
    patrol: Optional[List[Cell]] = None       # enemy -> waypoints
    on: Optional[str] = None                  # can -> table id (cosmetic)


class Predicate(BaseModel):
    """One conjunct of the objective. Win = every predicate true simultaneously."""
    kind: Literal[PRED_KINDS]  # type: ignore[valid-type]
    item: Optional[str] = None
    cell: Optional[Cell] = None


class EnvSpec(BaseModel):
    name: str
    description: str = ""
    width: int = Field(ge=12, le=32)
    height: int = Field(ge=8, le=22)
    tiles: List[List[int]]
    entities: List[Entity]
    objective: List[Predicate] = Field(min_length=1)
    objective_text: str = ""
    time_limit: int = Field(default=400, ge=20, le=2000)

    # ── L1 validators: geometry + reference integrity (the "meaning" the API can't enforce) ──

    @field_validator("tiles")
    @classmethod
    def _tiles_values(cls, v: List[List[int]]) -> List[List[int]]:
        for row in v:
            for t in row:
                if t not in (0, 1, 2, 3):
                    raise ValueError(f"tile value {t} out of range (0..3)")
        return v

    @model_validator(mode="after")
    def _check(self) -> "EnvSpec":
        # grid dimensions
        if len(self.tiles) != self.height:
            raise ValueError(f"tiles has {len(self.tiles)} rows, expected height={self.height}")
        for y, row in enumerate(self.tiles):
            if len(row) != self.width:
                raise ValueError(f"tiles row {y} has {len(row)} cols, expected width={self.width}")

        # perimeter must be walls (levels are enclosed)
        for x in range(self.width):
            if self.tiles[0][x] != 1 or self.tiles[self.height - 1][x] != 1:
                raise ValueError("top/bottom border must be walls (tile 1)")
        for y in range(self.height):
            if self.tiles[y][0] != 1 or self.tiles[y][self.width - 1] != 1:
                raise ValueError("left/right border must be walls (tile 1)")

        # unique ids
        ids = [e.id for e in self.entities]
        if len(ids) != len(set(ids)):
            raise ValueError("entity ids must be unique")
        id_to_type = {e.id: e.type for e in self.entities}

        # exactly one player_start
        starts = [e for e in self.entities if e.type == "player_start"]
        if len(starts) != 1:
            raise ValueError(f"need exactly one player_start, got {len(starts)}")

        # entities must sit on walkable tiles (not walls / hazards), and in bounds
        for e in self.entities:
            x, y = e.pos
            if not (0 <= x < self.width and 0 <= y < self.height):
                raise ValueError(f"entity {e.id} at {e.pos} out of bounds")
            if self.tiles[y][x] in (1, 2):
                raise ValueError(f"entity {e.id} at {e.pos} sits on a wall/hazard tile")

        # at most one exit (build_level keeps a single goal cell)
        if len([e for e in self.entities if e.type == "exit"]) > 1:
            raise ValueError("at most one exit entity is allowed")

        # door requires an existing key
        for e in self.entities:
            if e.type == "door":
                if e.requires is None or id_to_type.get(e.requires) != "key":
                    raise ValueError(f"door {e.id} requires a valid key id, got {e.requires!r}")

        # predicate items/cells must reference real entities of the right kind
        for p in self.objective:
            if p.kind == "holding" and id_to_type.get(p.item or "") not in ("can", "key", "coin"):
                raise ValueError(f"holding predicate references unknown item {p.item!r}")
            if p.kind == "item_at":
                # only crate positions are tracked, so item_at must target a crate
                if id_to_type.get(p.item or "") != "crate":
                    raise ValueError(f"item_at must reference a crate, got {p.item!r}")
                if p.cell is None:
                    raise ValueError("item_at predicate needs a cell")
            if p.kind == "reached_exit" and not any(e.type == "exit" for e in self.entities):
                raise ValueError("reached_exit objective but no exit entity")
            if p.kind == "collected_all_coins" and not any(e.type == "coin" for e in self.entities):
                # otherwise the empty-set subset check is vacuously satisfied at spawn
                raise ValueError("collected_all_coins objective but no coin entities")
        return self

    # ── convenience ──

    def entity(self, eid: str) -> Optional[Entity]:
        for e in self.entities:
            if e.id == eid:
                return e
        return None

    def by_type(self, t: str) -> List[Entity]:
        return [e for e in self.entities if e.type == t]

    @property
    def start(self) -> Cell:
        return self.by_type("player_start")[0].pos


# ── Tool-use schema for the generator (SHAPE guarantee) ─────────────────────────
# Deliberately hand-written and flattened so Claude's forced tool use produces clean
# JSON. Semantic constraints (bounds, references) are checked afterward by EnvSpec.

TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "width": {"type": "integer", "minimum": 12, "maximum": 32},
        "height": {"type": "integer", "minimum": 8, "maximum": 22},
        "tiles": {
            "type": "array",
            "description": "height rows x width cols. 0 floor, 1 wall, 2 hazard, 3 grass. Border all 1s.",
            "items": {"type": "array", "items": {"type": "integer"}},
        },
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": list(ENTITY_KINDS)},
                    "id": {"type": "string"},
                    "pos": {"type": "array", "items": {"type": "integer"}},
                    "requires": {"type": ["string", "null"]},
                    "patrol": {"type": ["array", "null"],
                               "items": {"type": "array", "items": {"type": "integer"}}},
                    "on": {"type": ["string", "null"]},
                },
                "required": ["type", "id", "pos"],
            },
        },
        "objective": {
            "type": "array",
            "description": "Conjunction (AND) of predicates that define winning.",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": list(PRED_KINDS)},
                    "item": {"type": ["string", "null"]},
                    "cell": {"type": ["array", "null"], "items": {"type": "integer"}},
                },
                "required": ["kind"],
            },
        },
        "objective_text": {"type": "string"},
        "time_limit": {"type": "integer", "minimum": 20, "maximum": 2000},
    },
    "required": ["name", "width", "height", "tiles", "entities", "objective", "objective_text"],
}
