"""
Evaluator — turns the environment factory into a benchmark suite (GI's "training AND
evaluating" use case): run a policy across a batch of verified envs and report success rate,
steps-vs-oracle efficiency, and a difficulty-stratified breakdown — the kind of eval report you
would run a candidate policy against.

(The honest code-truth -> pixels story lives in harness/reward_model.py, a real trained pixel
reward model; there is no hand-tuned strawman detector here.)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict

import numpy as np

from harness.verifier import verify
from harness.gym_env import make_from_spec
from harness.rollout import run_episode
from harness.agents.scripted import ScriptedOracle


# ── scorecard ─────────────────────────────────────────────────────────────────────

def scorecard(specs: Dict[str, dict], policy_factory: Callable[[], object] = None,
              epsilon: float = 0.12, seed: int = 0) -> dict:
    """specs: name -> spec dict. Returns per-env rows + aggregates (by difficulty)."""
    if policy_factory is None:
        policy_factory = lambda: ScriptedOracle(epsilon=epsilon, seed=seed)
    rows = []
    for name, spec in specs.items():
        vr = verify(spec)
        env = make_from_spec(spec)
        out = run_episode(env, policy_factory(), seed=seed)
        oracle = out["oracle_len"] or 1
        eff = round(oracle / max(1, out["steps"]), 3) if out["won"] else 0.0
        rows.append({
            "env": name, "difficulty": vr.difficulty, "won": out["won"],
            "steps": out["steps"], "oracle": oracle, "efficiency": eff,
            "reward": out["total_reward"],
        })
    by_diff = defaultdict(list)
    for r in rows:
        by_diff[r["difficulty"]].append(r)
    strata = {}
    for d, rs in by_diff.items():
        strata[d] = {
            "n": len(rs),
            "success_rate": round(sum(r["won"] for r in rs) / len(rs), 3),
            "mean_efficiency": round(np.mean([r["efficiency"] for r in rs]), 3),
        }
    agg = {
        "n": len(rows),
        "success_rate": round(sum(r["won"] for r in rows) / max(1, len(rows)), 3),
        "mean_efficiency": round(np.mean([r["efficiency"] for r in rows]), 3),
    }
    return {"rows": rows, "strata": strata, "aggregate": agg}


def format_scorecard(sc: dict) -> str:
    lines = ["  env                 difficulty  won  steps  oracle  eff",
             "  " + "-" * 58]
    for r in sc["rows"]:
        lines.append(f"  {r['env']:<19} {r['difficulty']:<10} "
                     f"{'Y' if r['won'] else 'N':^4} {r['steps']:>5} {r['oracle']:>6}  {r['efficiency']:.2f}")
    lines.append("  " + "-" * 58)
    for d in ("easy", "medium", "hard", "expert"):
        if d in sc["strata"]:
            s = sc["strata"][d]
            lines.append(f"  [{d:<6}] n={s['n']}  success={s['success_rate']:.0%}  mean_eff={s['mean_efficiency']:.2f}")
    a = sc["aggregate"]
    lines.append(f"  OVERALL   success={a['success_rate']:.0%}  mean_eff={a['mean_efficiency']:.2f}  (n={a['n']})")
    return "\n".join(lines)


# ── diversity report for the compounding mutation engine (mutate.evolve) ──────────

def diversity_report(records) -> dict:
    """Quantify how much variety evolve() actually produced: distinct objective signatures
    (+Shannon entropy), distinct entity-type multisets, difficulty spread, and lineage depth —
    the metrics that separate a real generator from a single-edit jitter of one template."""
    import math
    from collections import Counter
    objs, multisets, diffs = Counter(), set(), Counter()
    max_lineage = 0
    for r in records:
        spec = r["spec"]
        objs[tuple(sorted(p["kind"] for p in spec["objective"]))] += 1
        multisets.add(tuple(sorted(Counter(e["type"] for e in spec["entities"]).items())))
        diffs[r["difficulty"]] += 1
        max_lineage = max(max_lineage, r.get("lineage", 0))
    total = sum(objs.values()) or 1
    entropy = -sum((c / total) * math.log2(c / total) for c in objs.values())
    return {"n": len(records), "distinct_objectives": len(objs),
            "objective_entropy_bits": round(entropy, 2),
            "distinct_entity_multisets": len(multisets),
            "difficulty_hist": dict(diffs), "max_lineage_depth": max_lineage}


def format_diversity(d: dict) -> str:
    return (f"  environments={d['n']}   distinct objectives={d['distinct_objectives']} "
            f"(entropy {d['objective_entropy_bits']} bits)   "
            f"distinct entity-multisets={d['distinct_entity_multisets']}\n"
            f"  difficulty={d['difficulty_hist']}   max lineage depth={d['max_lineage_depth']}")
