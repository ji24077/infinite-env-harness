# Draft submission email

**To:** paula@generalintuition.com
**Subject:** Tech Challenge – [Your Name]

---

Hi Paula,

I built an **infinite environment factory** for the challenge: a text command is compiled to a
typed DSL, then a search oracle **proves the environment solvable**, extracts an oracle plan, and
**auto-labels its difficulty** — so every generated (and mutated) environment is guaranteed
beatable, with a **code-defined reward** exposed through a standard Gymnasium interface. The
headline turns your own premise into a number: on the same frames, a code-truth objective is
frame-exact while a pixel model is fooled ~10 frames early by an occlusion — and a small PPO
climbs the reward curve to prove the environments actually feed RL.

Repo: https://github.com/ji24077/infinite-env-harness — clone and run `uv run demo.py --offline` (≈1 min, **no API key needed**); the
README walks the whole pipeline in ~5 minutes. Happy to demo it live for the research team.

Best,
[Your Name]
