"""
Pixel agent — Claude sees ONLY the rendered frame (the observation modality GI's vision
policy uses) and picks an action. Paired with state_agent.py it exposes the vision gap:
identical objective, identical engine, but pixels-only vs coordinates. Requires ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import base64
import io
import os

from harness.gym_env import DISCRETE_ACTIONS
from harness.engine import renderer as R

_AIDX = {a: i for i, a in enumerate(DISCRETE_ACTIONS)}
_MODEL = os.environ.get("HARNESS_MODEL", "claude-sonnet-4-5")

_MOVE_TOOL = {
    "name": "move",
    "description": "Choose the next single action for the blue agent.",
    "input_schema": {
        "type": "object",
        "properties": {"action": {"type": "string", "enum": DISCRETE_ACTIONS}},
        "required": ["action"],
    },
}

_SYS = ("You control the BLUE circle in a top-down 2D grid game. Reach the objective. "
        "Green square = exit. Yellow = key. Brown = door/crate. Light cylinder = can. "
        "Red circle = enemy. Faint numbers on the edges are x (top) and y (left) coordinates. "
        "up decreases y, down increases y, left decreases x, right increases x. Call move.")


class PixelAgent:
    def __init__(self, model: str = _MODEL):
        self.model = model
        self._client = None

    def reset(self, env):
        pass

    def __call__(self, env, obs, info) -> int:
        import anthropic
        if self._client is None:
            self._client = anthropic.Anthropic()
        surf = R.render_surface(env.world, tick=env.world.step_count)
        buf = io.BytesIO(); R.to_pil(surf).save(buf, format="PNG")
        b64 = base64.standard_b64encode(buf.getvalue()).decode()
        resp = self._client.messages.create(
            model=self.model, max_tokens=100, system=_SYS,
            tools=[_MOVE_TOOL], tool_choice={"type": "tool", "name": "move"},
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                {"type": "text", "text": f"Objective: {env.spec.objective_text}. Next action?"},
            ]}],
        )
        for b in resp.content:
            if getattr(b, "type", None) == "tool_use":
                return _AIDX.get(b.input.get("action", "interact"), _AIDX["interact"])
        return _AIDX["interact"]
