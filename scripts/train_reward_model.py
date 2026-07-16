"""
Train a pixel reward model on code-truth labels, and report its HONEST held-out disagreement
vs code truth. GI use-case #3, concretely — no hand-tuned threshold, no ground-truth-cell handout.

  uv run --extra rl python scripts/train_reward_model.py

Data: oracle rollouts on the can-containing environments; per frame the label is the code-truth
event `holding(can)` (from engine state). The HUD is cropped off so the model reads the SCENE,
not the predicate ticks. Train on some envs, evaluate on a HELD-OUT env (unseen layout).
Writes assets/reward_model.json.
"""
import json
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np

from harness import fixtures as F
from harness.dsl.schema import EnvSpec
from harness.gym_env import make_from_spec
from harness.rollout import run_episode
from harness.agents.scripted import ScriptedOracle
from harness.engine.world import TILE
from harness.reward_model import PixelRewardModel

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")
CAN_ENVS = ["open_can", "key_crate_return", "three_rooms", "occlusion_can"]
N_ROLLOUTS = 6               # per env; the LAST rollout of each env is held out (unseen episodes)


def collect_rollout(env_name, seed):
    """One oracle rollout -> (game-area frames, code-truth holding labels)."""
    spec = F.ALL[env_name]()
    can_id = EnvSpec(**spec).by_type("can")[0].id
    W, H = spec["width"], spec["height"]
    env = make_from_spec(spec)
    out = run_episode(env, ScriptedOracle(epsilon=0.2, seed=seed), collect_frames=True)
    frames, labels = [], []
    for i, fr in enumerate(out["frames"]):
        held = out["trace"][i - 1]["code_state"]["held"] if i > 0 else []
        frames.append(fr.crop((0, 0, W * TILE, H * TILE)))   # crop the HUD off (no leakage)
        labels.append(1 if can_id in held else 0)
    return frames, labels


def main():
    print("[reward-model] collecting code-truth-labeled frames (HUD cropped)...")
    train_f, train_y, test_f, test_y = [], [], [], []
    for name in CAN_ENVS:
        for r in range(N_ROLLOUTS):
            f, y = collect_rollout(name, seed=r)
            if r < N_ROLLOUTS - 1:                # held out: the last rollout of EACH env
                train_f += f; train_y += y
            else:
                test_f += f; test_y += y
    print(f"[reward-model] train frames={len(train_f)} (picked-up {sum(train_y)}) | "
          f"held-out (unseen episodes) frames={len(test_f)} (picked-up {sum(test_y)})")

    model = PixelRewardModel(size=96).fit(train_f, train_y, epochs=90)
    pred = model.predict(test_f).astype(int)
    truth = np.array(test_y)
    disagree = float((pred != truth).mean())

    # per-class (the pickup frames are the rare, interesting ones)
    pos = truth == 1
    pickup_acc = float((pred[pos] == 1).mean()) if pos.any() else float("nan")
    neg_acc = float((pred[~pos] == 0).mean()) if (~pos).any() else float("nan")

    result = {
        "held_out": "unseen episodes of the can environments",
        "held_out_frames": len(test_f),
        "disagreement_rate": round(disagree, 3),
        "pickup_recall": round(pickup_acc, 3),
        "not_picked_recall": round(neg_acc, 3),
        "note": "trained ONLY on code-truth labels; HUD cropped so it reads the scene, not ticks",
    }
    json.dump(result, open(os.path.join(ASSETS, "reward_model.json"), "w"), indent=2)
    print(f"[reward-model] HELD-OUT disagreement vs code truth: {disagree:.1%}  "
          f"(pickup recall {pickup_acc:.0%}, not-picked recall {neg_acc:.0%})")
    print("[reward-model] -> code truth is exact, label-free supervision a pixel model approximates.")
    print(f"[reward-model] wrote assets/reward_model.json")


if __name__ == "__main__":
    main()
