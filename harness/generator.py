"""
Generator — text command -> verified EnvSpec, via Claude forced tool use + a repair loop.

Two-layer reliability:
  1. SHAPE is guaranteed by tool use: Claude must call `emit_environment` whose input_schema
     is the DSL (harness/dsl/schema.py:TOOL_SCHEMA). It cannot return prose or malformed JSON.
  2. MEANING is guaranteed by the verifier: the emitted spec is run through L1/L2/L3. On any
     failure we feed the STRUCTURED error back as a tool_result and ask for a corrected full
     environment — up to MAX_REPAIRS times (Voyager-style self-verification / EnvGen feedback).

If all repairs fail we raise, rather than emit an unverified environment. That is the whole
point: the factory never ships a level it cannot prove beatable.
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional, Tuple

from harness.dsl.schema import EnvSpec, TOOL_SCHEMA
from harness.verifier import verify, VerifyResult

MAX_REPAIRS = 3
DEFAULT_MODEL = os.environ.get("HARNESS_MODEL", "claude-sonnet-4-5")

SYSTEM = """You design small 2D grid environments for a reinforcement-learning environment factory.
You output ONE environment by calling the emit_environment tool. Rules:

GRID: width 12-32, height 8-22. Tiles: 0 floor, 1 wall, 2 hazard (impassable), 3 grass.
The entire border (row 0, last row, col 0, last col) MUST be wall (1).

ENTITIES (each needs a unique id and a pos [x,y] on a floor/grass tile, never on a wall/hazard):
- player_start (exactly one) : agent spawn.
- exit                        : goal cell for 'reached_exit' objectives.
- key / door                  : a door {requires:<key id>} blocks its cell until that key is picked up.
- crate                       : pushable one cell at a time if the cell beyond is clear.
- table + can                 : a 'can' is picked up when the agent is on OR next to its cell (grab from a table).
- ball                        : a physics prop (rolls); purely decorative.
- coin                        : collectible.
- enemy {patrol:[[x,y],...]}  : a DEADLY guard. patrol lists its per-tick cycle of cells (each
                                cell walkable, adjacent to the next); the agent dies on contact.
                                Leave a timed path through — never seal the only route. Don't start
                                a patrol on the player's spawn.

OBJECTIVE: a list of predicates ALL of which must hold to win. Kinds:
  reached_exit | holding(item) | at_start | item_at(item, cell) | collected_all_coins.

DESIGN GOALS: make it solvable but non-trivial; place the player far from the goal; if you use a
door, put its key somewhere reachable BEFORE the door; keep hazards from sealing off the goal.
Match the user's described theme. Always fill objective_text with a short natural-language goal."""

TOOL = {
    "name": "emit_environment",
    "description": "Emit one complete, playable 2D grid environment matching the user's description.",
    "input_schema": TOOL_SCHEMA,
}


def _first_tool_use(content):
    for block in content:
        if getattr(block, "type", None) == "tool_use":
            return block
    return None


def generate(command: str, model: str = DEFAULT_MODEL,
             log: Optional[Callable[[str], None]] = None) -> Tuple[EnvSpec, VerifyResult, List[str]]:
    """Return (spec, verify_result, transcript_log). Raises RuntimeError if unrepairable."""
    import anthropic
    client = anthropic.Anthropic()
    logs: List[str] = []

    def emit(msg: str):
        logs.append(msg)
        (log or print)(msg)

    emit(f"[generator] command: {command!r}")
    messages = [{"role": "user", "content":
                 f"Design an environment for this request:\n\n{command}"}]

    for attempt in range(1, MAX_REPAIRS + 2):
        resp = client.messages.create(
            model=model, max_tokens=4096, system=SYSTEM,
            tools=[TOOL], tool_choice={"type": "tool", "name": "emit_environment"},
            messages=messages,
        )
        tu = _first_tool_use(resp.content)
        if tu is None:
            # tool_choice forces a tool call, so this is a rare safety net; keep the message
            # sequence valid (assistant turn before the next user turn)
            emit(f"[generator] attempt {attempt}: model did not call the tool; retrying")
            messages.append({"role": "assistant", "content": resp.content})
            messages.append({"role": "user", "content": "Call emit_environment with a full environment."})
            continue

        raw = tu.input
        # L1: schema/meaning
        try:
            spec = EnvSpec(**raw)
        except Exception as ex:  # pydantic ValidationError et al.
            reason = f"schema/meaning error: {ex}"
            emit(f"[generator] attempt {attempt}: L1 FAIL — {str(ex)[:160]}")
        else:
            vr = verify(spec)
            if vr.ok:
                emit(f"[generator] attempt {attempt}: {vr.log_line().strip()}")
                emit(f"[generator] accepted '{spec.name}' (difficulty={vr.difficulty}, "
                     f"oracle plan {vr.plan_len} steps)")
                return spec, vr, logs
            reason = f"{vr.stage} failed: {vr.reason}"
            emit(f"[generator] attempt {attempt}: {vr.stage} FAIL — {vr.reason}")

        if attempt > MAX_REPAIRS:
            break
        # repair: hand the structured failure back and ask for a corrected FULL environment
        messages.append({"role": "assistant", "content": resp.content})
        messages.append({"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": tu.id, "is_error": True,
            "content": (f"The environment was rejected — {reason}. "
                        f"Fix ONLY that problem and call emit_environment again with the full corrected environment."),
        }]})

    raise RuntimeError(f"generation failed after {MAX_REPAIRS} repairs for command: {command!r}")


def generate_offline(spec_dict: dict) -> Tuple[EnvSpec, VerifyResult, List[str]]:
    """No-API path used by cached specs: just re-run the verifier so the demo still streams
    real L1/L2/L3 logs without a key."""
    spec = EnvSpec(**spec_dict)
    vr = verify(spec)
    return spec, vr, [vr.log_line()]
