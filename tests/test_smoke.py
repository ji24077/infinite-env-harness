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


def test_gym_env_passes_official_check_env():
    from gymnasium.utils.env_checker import check_env
    check_env(make_from_spec(F.open_can()), skip_render_check=False)   # real Gymnasium interface


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


def test_generator_repair_exhaustion_raises(mock_anthropic):
    from harness.generator import generate
    bad = F.open_can(); bad["entities"][0]["pos"] = [0, 0]   # always L1-invalid
    _FakeClient.QUEUE = [bad]                                 # every attempt returns the bad spec
    with pytest.raises(RuntimeError):
        generate("an open room with a can")


# ── degenerate-level guards + controller mode ────────────────────────────────────

def test_plan_exceeding_time_limit_is_rejected():
    s = F.key_crate_return(); s["time_limit"] = 25          # plan is 32 > 25
    r = verify(s)
    assert not r.ok and "time_limit" in r.reason            # never certify an unbeatable level


def test_trivial_objective_rejected():
    s = F.open_can(); s["objective"] = [{"kind": "at_start"}]  # true at spawn
    assert not verify(s).ok


@pytest.mark.parametrize("mutate_spec,err", [
    (lambda s: s.update(objective=[{"kind": "collected_all_coins"}]), "coin"),
    (lambda s: s["entities"].append({"type": "exit", "id": "e2", "pos": [3, 3]}), "exit"),
    (lambda s: s.update(objective=[{"kind": "item_at", "item": "can1", "cell": [3, 3]}]), "crate"),
])
def test_l1_meaning_guards(mutate_spec, err):
    from pydantic import ValidationError
    s = F.open_can(); mutate_spec(s)
    with pytest.raises(ValidationError) as ex:
        EnvSpec(**s)
    assert err in str(ex.value)


def test_controller_deadzone_is_noop():
    import numpy as np
    env = make_from_spec(F.open_can(), action_mode="controller")
    env.reset()
    before = tuple(env.world.state.agent)
    env.step(np.zeros(6, dtype=np.float32))                  # no intent -> no move
    assert tuple(env.world.state.agent) == before


def test_contrast_timing_and_disagreements():
    c = E.run_contrast(F.occlusion_can(), use_vlm=False)
    assert c["code_time_us"] > 0 and c["perc_time_us"] > 0
    assert c["disagreements"] >= 1


def test_deadly_enemy_kills_on_contact():
    from harness.engine.world import World
    W, H = 12, 8
    g = [[1] * W for _ in range(H)]
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            g[y][x] = 0
    spec = {"name": "t", "width": W, "height": H, "tiles": g, "entities": [
        {"type": "player_start", "id": "p", "pos": [2, 3]},
        {"type": "enemy", "id": "e", "pos": [3, 3], "patrol": [[3, 3]]},   # stationary guard
        {"type": "exit", "id": "x", "pos": [9, 3]}],
        "objective": [{"kind": "reached_exit"}], "objective_text": "reach", "time_limit": 50}
    w = World(EnvSpec(**spec))
    w.step("right")                        # walk straight into the guard
    assert w.done and not w.won            # contact = death


def test_patrol_gauntlet_solvable_and_survivable():
    spec = F.patrol_gauntlet()
    vr = verify(spec)
    assert vr.ok                                           # a timed path exists
    env = make_from_spec(spec)
    out = run_episode(env, ScriptedOracle())
    assert out["won"]                                      # oracle actually survives the guard


def test_enemy_sealing_the_route_is_rejected():
    spec = F.patrol_gauntlet()
    spec["entities"][1]["patrol"] = [[8, 5]]               # guard parks on the only crossing
    spec["entities"][1]["pos"] = [8, 5]
    assert not verify(spec).ok                             # provably unsolvable -> rejected


def test_world_model_critic_flags_hallucinations():
    from harness import critic
    from harness.verifier import solve
    lvl = G.build_level(EnvSpec(**F.key_crate_return()))
    plan, _ = solve(lvl, EnvSpec(**F.key_crate_return()).objective)
    real = critic.rollout_from_plan(lvl, plan)
    assert critic.critique(lvl, real) == []            # a legal rollout has zero violations
    # inject a teleport -> must be flagged
    hall = list(real)
    hall[3] = G.GridState(agent=(real[3].agent[0] + 5, real[3].agent[1]),
                          held=real[3].held, crates=real[3].crates)
    assert len(critic.critique(lvl, hall)) >= 1
    assert critic.score(lvl, hall) < 1.0
