# DESIGN — decisions & prior-art mapping

## The one load-bearing decision: DSL-as-environment, not code-as-environment

OMNI-EPIC generates environments as arbitrary Python. That is maximally expressive and minimally
verifiable — you cannot, in general, prove an arbitrary program produces a solvable level. We take
the opposite trade: generation targets a **small typed DSL** (`harness/dsl/schema.py`). The
expressiveness we lose, we buy back as **verifiability**:

- a **sound + complete BFS solver** over `(agent, inventory, crates)` decides solvability and
  returns a shortest **oracle plan** (`harness/verifier.py`);
- the oracle plan's **length is the difficulty label** — the solver *is* the labeler;
- the oracle **cost-to-go field** is the reward-shaping potential — the solver *is* the reward
  source.

Everything else follows from wanting that solver to stay honest.

## Why the semantics live in one module

`harness/engine/gridlogic.py` is imported by **both** the verifier (to search) and the runtime
engine (to step). So a plan the solver proves is, by construction, a plan the engine replays
frame-exact. Without this shared source of truth, "verified solvable" would be a claim about a
different system than the one you play. Rules are deliberately **conservative (sound)**: when a
push/pickup is ambiguous we take the more-restrictive branch, so the solver never over-claims
solvability. Over-strictness only triggers an extra repair-loop regeneration.

## Hybrid engine: physics off the critical path

Navigation/keys/doors/crates/pickups are **grid-authoritative and deterministic**. `pymunk`
simulates only **soft props** (a rolling ball) plus the L3 stability smoke test. This gives the
literal "physics engine" credential and a livelier render **without** putting tunneling / NaN /
LLM-authored-physics-explosions on the path that must never flake. The boundary is explicit and
intentional.

## Two-layer reliability (shape vs meaning)

Anthropic structured/tool-use guarantees the generated JSON's **shape**; it cannot guarantee a
coordinate is in-bounds or a key is reachable. So: **strict tool use → shape**, **verifier L1/L2/L3
→ meaning**, **repair loop → convergence** (structured failure fed back, ≤3 retries). The factory
raises rather than ship an unverified environment.

## The agent is an oracle, not the product

The product is *verified environments × code rewards × a standard RL interface*. The scripted
oracle (and, optionally, Claude state/pixel agents) exist to **prove environments are beatable**
and to **expose the pixel-vs-code gap** — not to be a good game-player. This is why the whole
offline demo needs no API key: a search oracle is a perfectly good solvability witness.

## Prior-art mapping

| Idea we use | Source | Where |
|---|---|---|
| Foundation model generates environments **and rewards** in code | OMNI-EPIC (Faldor et al., ICLR 2025) | `generator.py` (constrained to a DSL) |
| Self-verification + structured-feedback regeneration | Voyager (Wang et al. 2023), EnvGen (2024) | `generator.py` repair loop |
| Minimax-**regret** curation of a mutation curriculum | PAIRED (Dennis 2020), **ACCEL** (Parker-Holder 2022) | `mutate.py` (oracle-vs-greedy regret proxy) |
| Generate → **solver-verify** solvability, filter unplayable | Sokoban/Mario PCG (Todd 2023, MarioGPT 2023) | `verifier.py` L2 BFS gate |
| Standard RL env interface for a policy to mount on | Gymnasium / MiniGrid–Miniworld (Farama) | `gym_env.py` |
| Complement to neural world models (which lack code truth) | DIAMOND / Genie-class | positioning; `eval.py` contrast |

## What we deliberately did NOT build

- No RL policy training as the deliverable (that's GI's vision policy; we only ship a *learnability*
  PPO probe).
- No arbitrary-mechanic menagerie (pressure plates, teleporters, timed patrols) — they blow up the
  solver's state space (PSPACE) and would make the solvability guarantee hollow.
- No live-LLM dependency on the guaranteed demo path (generation is offline-cacheable; the VLM
  judge is an optional live upgrade over saved frames).
- No claim of end-to-end 3D transfer — we claim the **interface + pattern** that ports, honestly.
