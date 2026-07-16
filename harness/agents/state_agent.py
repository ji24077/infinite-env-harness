"""
State agent — Claude reasoning over a STRUCTURED, coordinate-tagged description of the world.
This is the "upper-bound" navigation baseline (explicit coordinates are the documented fix for
LLM spatial-reasoning weakness). Contrast with pixel_agent.py, which sees only the frame.

Role in this project: another solvability oracle / demonstration policy — NOT the product.
Requires ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import os
from typing import List

from harness.gym_env import DISCRETE_ACTIONS

_AIDX = {a: i for i, a in enumerate(DISCRETE_ACTIONS)}
_MODEL = os.environ.get("HARNESS_MODEL", "claude-sonnet-4-5")

_MOVE_TOOL = {
    "name": "move",
    "description": "Choose the next single action for the agent.",
    "input_schema": {
        "type": "object",
        "properties": {"action": {"type": "string", "enum": DISCRETE_ACTIONS},
                       "reason": {"type": "string"}},
        "required": ["action"],
    },
}


def _describe(env) -> str:
    w = env.world
    lvl = w.level
    ax, ay = w.state.agent
    lines = [f"Grid {env.spec.width}x{env.spec.height}. Agent at ({ax},{ay}). "
             f"Coordinates are (x=col from left, y=row from top). Moves: up=-y down=+y left=-x right=+x.",
             f"Objective: {env.spec.objective_text}"]
    if lvl.exit:
        lines.append(f"Exit at {tuple(lvl.exit)}.")
    for cell, kid in lvl.keys.items():
        if kid not in w.state.held:
            lines.append(f"Key '{kid}' at {cell} (pick up by stepping onto its cell; it auto-opens its door).")
    for cell, keyid in lvl.doors.items():
        state = "OPEN" if keyid in w.state.held else f"LOCKED(needs {keyid})"
        lines.append(f"Door at {cell} {state}.")
    for cid, cell in lvl.cans.items():
        if cid not in w.state.held:
            lines.append(f"Can '{cid}' at {cell} (pick up by standing on OR next to it).")
    for cell, coin in lvl.coins.items():
        if coin not in w.state.held:
            lines.append(f"Coin '{coin}' at {cell} (pick up by stepping onto its cell).")
    for cid, x, y in w.state.crates:
        lines.append(f"Crate '{cid}' at ({x},{y}) (push by walking into it).")
    if w.state.held:
        lines.append(f"Holding: {sorted(w.state.held)}.")
    lines.append(f"Walls are tile=1, hazards tile=2 (impassable). Step {w.step_count}.")
    return "\n".join(lines)


class StateAgent:
    def __init__(self, model: str = _MODEL):
        self.model = model
        self._client = None
        self.history: List[str] = []

    def reset(self, env):
        self.history = []

    def __call__(self, env, obs, info) -> int:
        import anthropic
        if self._client is None:
            self._client = anthropic.Anthropic()
        prompt = _describe(env)
        if self.history:
            prompt += "\nRecent actions: " + ", ".join(self.history[-6:])
        resp = self._client.messages.create(
            model=self.model, max_tokens=200,
            system="You navigate a 2D grid to satisfy the objective. Think briefly, then call move.",
            tools=[_MOVE_TOOL], tool_choice={"type": "tool", "name": "move"},
            messages=[{"role": "user", "content": prompt}],
        )
        for b in resp.content:
            if getattr(b, "type", None) == "tool_use":
                act = b.input.get("action", "wait")
                self.history.append(act)
                return _AIDX.get(act, _AIDX["wait"])
        return _AIDX["wait"]
