"""
evaluate.py — the benchmark suite (GI's "evaluating a policy" use case), standalone.

  uv run python evaluate.py                     # scorecard over all cached envs + pixel contrast
  uv run python evaluate.py --agent scripted    # policy under test (default noisy oracle)
  uv run python evaluate.py --vlm --live         # swap the pixel detector for a Claude VLM judge
                                                 #   (needs ANTHROPIC_API_KEY; judges saved frames)

The code-vs-perception contrast is the empirical version of GI's own rationale: code-level
objectives are checked frame-exact against engine state, a perception model reading pixels is not.
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
    ap.add_argument("--vlm", action="store_true", help="use a Claude VLM judge for the contrast")
    ap.add_argument("--live", action="store_true", help="allow live API calls (required for --vlm)")
    args = ap.parse_args()

    specs = {n: load_cached(n) for n in F.ALL}

    print("=" * 66 + "\n  EVAL SCORECARD  (noisy oracle across verified envs)\n" + "=" * 66)
    sc = E.scorecard(specs, epsilon=args.epsilon, seed=1)
    print(E.format_scorecard(sc))

    print("\n" + "=" * 66 + "\n  CODE-TRUTH vs " +
          ("CLAUDE VLM" if (args.vlm and args.live) else "PIXEL DETECTOR") +
          "  PERCEPTION\n" + "=" * 66)
    use_vlm = args.vlm and args.live
    if args.vlm and not args.live:
        print("  --vlm requires --live (and ANTHROPIC_API_KEY); falling back to the pixel detector.")
    c = E.run_contrast(load_cached("occlusion_can"), use_vlm=use_vlm)
    E.render_contrast_strip(c, "assets/contrast.png")
    print(f"  scene '{c['spec_name']}': {c['n_frames']} frames")
    print(f"  code-truth  : first pickup = frame {c['code_first_true']} (exact)")
    print(f"  perception  : first pickup = frame {c['perc_first_true']}  "
          f"(latency {c['latency_frames']} frames; wrong on {c['disagreements']} frames)")
    if use_vlm:
        print(f"  timing      : code ~{c['code_time_us']} us/frame vs VLM ~{c['perc_time_s']} s/frame + $ per call")
    else:
        print(f"  timing      : code {c['code_time_us']} us/frame vs pixel {c['perc_time_us']} us/frame "
              f"({c['perc_time_us']/max(c['code_time_us'],1e-6):.0f}x)")
    print("  -> code-defined objectives are exact and ~free; pixel perception is fooled by occlusion.")
    print("  strip -> assets/contrast.png")


if __name__ == "__main__":
    main()
