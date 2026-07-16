"""Generate README assets: a hero oracle-solve GIF, a diversity gallery, and the contrast strip.
Run:  uv run python scripts/build_assets.py
"""
import json
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from PIL import Image, ImageDraw

from harness import fixtures as F
from harness.verifier import verify
from harness.gym_env import make_from_spec
from harness.rollout import run_episode
from harness.agents.scripted import ScriptedOracle
from harness.engine import renderer as R
from harness import eval as E

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")


def hero_gif(name="three_rooms"):
    spec = F.ALL[name]()
    env = make_from_spec(spec)
    out = run_episode(env, ScriptedOracle(), collect_frames=True)
    for _ in range(10):
        out["frames"].append(R.to_pil(R.render_surface(env.world, tick=999)))
    R.save_gif(out["frames"], os.path.join(ASSETS, "hero.gif"), fps=8)
    print(f"  hero.gif ({name}, {out['steps']} steps, {len(out['frames'])} frames)")


def gallery():
    thumbs, tw = [], 320
    for name, fn in F.ALL.items():
        spec = fn()
        vr = verify(spec)
        env = make_from_spec(spec)
        img = R.to_pil(R.render_surface(env.world, tick=3))
        scale = tw / img.width
        img = img.resize((tw, int(img.height * scale)))
        card = Image.new("RGB", (tw, img.height + 30), (16, 16, 22))
        card.paste(img, (0, 30))
        d = ImageDraw.Draw(card)
        d.text((6, 4), f"{spec['name']}", fill=(210, 212, 228))
        d.text((6, 17), f"difficulty={vr.difficulty}  oracle={vr.plan_len} steps",
                fill=(120, 180, 255))
        thumbs.append(card)
    cols = 2
    rows = (len(thumbs) + cols - 1) // cols
    cw = max(t.width for t in thumbs)
    ch = max(t.height for t in thumbs)
    gap = 10
    board = Image.new("RGB", (cols * cw + (cols + 1) * gap, rows * ch + (rows + 1) * gap),
                      (8, 8, 12))
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        board.paste(t, (gap + c * (cw + gap), gap + r * (ch + gap)))
    board.save(os.path.join(ASSETS, "gallery.png"))
    print(f"  gallery.png ({len(thumbs)} envs)")


def contrast():
    c = E.run_contrast(F.occlusion_can(), use_vlm=False)
    E.render_contrast_strip(c, os.path.join(ASSETS, "contrast.png"))
    print(f"  contrast.png (latency {c['latency_frames']} frames, {c['disagreements']} wrong)")


if __name__ == "__main__":
    os.makedirs(ASSETS, exist_ok=True)
    hero_gif()
    gallery()
    contrast()
    print("assets built.")
