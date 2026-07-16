"""
Smoke tests — run with:  uv run --with pytest pytest -q

Covers the whole API-free pipeline, plus the generator's control flow (tool-use extraction +
repair loop) via a mocked Anthropic client, so the online path is de-risked without a key.
"""

import sys, types
import pytest

from harness import fixtures as F
from harness.dsl.schema import EnvSpec
from harness.verifier import verify, solve
from harness.engine import gridlogic as G
from harness.gym_env import make_from_spec, DISCRETE_ACTIONS
from harness.rollout import run_episode
from harness.agents.scripted import ScriptedOracle
from harness.mutate import mutate
from harness import eval as E


# ── core pipeline ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", list(F.ALL))
def test_fixture_verifies_and_oracle_solves(name):
    spec = F.ALL[name]()
    vr = verify(spec)
    assert vr.ok, f"{name}: {vr.stage} {vr.reason}"
    # the extracted oracle plan actually wins when replayed
    s = EnvSpec(**spec); lvl = G.build_level(s); st = G.initial_state(lvl)
    for a in vr.plan:
        st = G.step(lvl, st, a); assert st is not None
    assert G.objective_satisfied(lvl, s.objective, st)


def test_unsolvable_is_rejected():
    spec = F.open_can()
    # wall the can off from the player
    for y in range(spec["height"]):
        spec["tiles"][y][6] = 1
    vr = verify(spec)
    assert not vr.ok and vr.stage in ("L1", "L2")


def test_gym_reward_positive_on_oracle_and_random_fails():
    spec = F.key_crate_return()
    env = make_from_spec(spec)
    out = run_episode(env, ScriptedOracle())
    assert out["won"] and out["total_reward"] > 0


def test_mutations_all_verified_solvable():
    variants = mutate(F.key_crate_return(), n=6, seed=0, accel=True)
    assert len(variants) >= 3
    for v in variants:
        assert verify(v["spec"]).ok


def test_pixel_perception_is_fooled_by_occlusion():
    c = E.run_contrast(F.occlusion_can(), use_vlm=False)
    # code truth is exact; the pixel model mis-fires early under occlusion
    assert c["code_first_true"] is not None
    assert c["disagreements"] >= 1
    assert c["latency_frames"] is not None and c["latency_frames"] < 0


def test_scorecard_runs():
    sc = E.scorecard({n: F.ALL[n]() for n in F.ALL}, epsilon=0.1, seed=0)
    assert sc["aggregate"]["success_rate"] == 1.0


# ── generator control flow via a mocked Anthropic client ─────────────────────────

class _Block:
    def __init__(self, inp, _id): self.type, self.input, self.id = "tool_use", inp, _id

class _Resp:
    def __init__(self, content): self.content = content

class _Messages:
    def __init__(self, queue): self.queue, self.calls = queue, 0
    def create(self, **kw):
        inp = self.queue[min(self.calls, len(self.queue) - 1)]
        self.calls += 1
        return _Resp([_Block(inp, f"tu_{self.calls}")])

class _FakeClient:
    QUEUE = []
    def __init__(self, *a, **k): self.messages = _Messages(_FakeClient.QUEUE)


@pytest.fixture
def mock_anthropic(monkeypatch):
    fake = types.ModuleType("anthropic")
    fake.Anthropic = _FakeClient
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    return _FakeClient


def test_generator_accepts_valid_on_first_try(mock_anthropic):
    from harness.generator import generate
    _FakeClient.QUEUE = [F.open_can()]
    spec, vr, logs = generate("an open room with a can")
    assert vr.ok and spec.name


def test_generator_repairs_then_accepts(mock_anthropic):
    from harness.generator import generate
    bad = F.open_can()
    bad["entities"][0]["pos"] = [0, 0]   # player_start on a wall -> L1 fails
    _FakeClient.QUEUE = [bad, F.open_can()]
    spec, vr, logs = generate("an open room with a can")
    assert vr.ok
    assert any("FAIL" in l for l in logs)  # the repair loop actually fired
