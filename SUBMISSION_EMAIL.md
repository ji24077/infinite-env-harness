# Draft submission email

**To:** paula@generalintuition.com
**Subject:** Tech Challenge – [Your Name]

---

Hi Paula,

I built an **environment factory** for the challenge: a text command is compiled to a typed DSL,
then a search oracle **proves the environment solvable within its step budget** (rejecting and
regenerating otherwise), extracts an oracle plan, and reuses that one plan three ways — difficulty
label, replay witness, and reward-shaping signal — all exposed through a standard Gymnasium
interface. The result is reproducible end to end with no API key: a small off-the-shelf PPO mounted
on a generated env climbs the reward curve to an oracle-optimal solve, showing the environments
actually feed RL. (A short code-vs-pixel illustration and a rollout-legality-checker direction are
included and honestly scoped in the README.)

Repo: https://github.com/ji24077/infinite-env-harness — clone and run `uv run demo.py --offline`
(≈1 min, **no API key needed**); the README walks the whole pipeline in ~5 minutes. Happy to demo
it live for the research team.

Best,
[Your Name]
