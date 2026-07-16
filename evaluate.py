"""
evaluate.py — the benchmark suite (GI's "evaluating a policy" use case), standalone.

  uv run python evaluate.py                 # scorecard over all cached, verified envs
  uv run python evaluate.py --epsilon 0.2   # oracle noise for the run

A policy (default: a noisy oracle) is run across every verified environment; the report gives
success rate and steps-vs-oracle efficiency, difficulty-stratified — the kind of eval you would
run a candidate policy against. (The honest code-truth -> pixels story is the trained pixel
reward model: uv run --extra rl python scripts/train_reward_model.py.)
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from harness import fixtures as F
from harness import eval as E


def load_cached(name):
    with open(os.path.join("specs", f"{name}.json")) as f:
        return json.load(f)["spec"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epsilon", type=float, default=0.12, help="oracle noise for the scorecard")
    args = ap.parse_args()

    specs = {n: load_cached(n) for n in F.ALL}
    print("=" * 66 + "\n  EVAL SCORECARD  (noisy oracle across verified envs)\n" + "=" * 66)
    sc = E.scorecard(specs, epsilon=args.epsilon, seed=1)
    print(E.format_scorecard(sc))


if __name__ == "__main__":
    main()
