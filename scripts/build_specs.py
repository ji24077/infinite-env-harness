"""Dump the canonical fixtures to specs/*.json with verification metadata.
These form the --offline cache (a reviewer with no API key still runs the full pipeline).
Run:  uv run python scripts/build_specs.py
"""
import json
import os

from harness import fixtures as F
from harness.verifier import verify

OUT = os.path.join(os.path.dirname(__file__), "..", "specs")


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, fn in F.ALL.items():
        spec = fn()
        vr = verify(spec)
        assert vr.ok, f"{name} failed to verify: {vr.stage} {vr.reason}"
        doc = {
            "spec": spec,
            "meta": {
                "command": F.DEMO_COMMANDS.get(name, ""),
                "difficulty": vr.difficulty,
                "oracle_plan_len": vr.plan_len,
                "oracle_plan": vr.plan,
                "verified": True,
            },
        }
        path = os.path.join(OUT, f"{name}.json")
        with open(path, "w") as f:
            json.dump(doc, f, indent=2)
        print(f"  wrote {name}.json  (difficulty={vr.difficulty}, plan={vr.plan_len})")


if __name__ == "__main__":
    main()
