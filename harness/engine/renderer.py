"""
Headless renderer. Draws the World to a pygame-ce surface and exports frames as numpy
arrays / PNG bytes / animated GIFs. High-contrast tiles + optional coordinate tags — a
deliberate choice so a vision agent (and GI's future policy) has legible pixels to read.
"""

from __future__ import annotations

import io
import os
from typing import List, Optional

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import math
import pygame
from PIL import Image

from harness.engine.world import World, TILE

HUD_H = 56

PAL = {
    "bg": (12, 12, 16), "floor": (44, 46, 56), "floor2": (40, 42, 52),
    "wall": (84, 88, 110), "wall_top": (120, 126, 156),
    "hazard": (150, 40, 40), "hazard2": (185, 60, 55), "grass": (40, 92, 52),
    "player": (86, 180, 255), "player_dir": (255, 255, 255),
    "exit": (52, 224, 128), "exit_ring": (120, 255, 170),
    "key": (255, 214, 40), "door": (168, 96, 52), "door_open": (74, 52, 30),
    "coin": (255, 200, 30), "table": (150, 116, 66), "can": (200, 210, 220),
    "can_top": (235, 240, 248), "ball": (240, 130, 90), "crate": (176, 132, 74),
    "crate_edge": (120, 88, 44), "enemy": (224, 72, 72), "enemy_eye": (255, 210, 210),
    "coord": (78, 82, 100), "hud_bg": (18, 18, 26), "txt": (208, 210, 226),
    "accent": (86, 180, 255), "ok": (80, 220, 130), "no": (150, 154, 174),
}

_pygame_ready = False
_font_sm = _font_md = None


def _ensure_pygame():
    global _pygame_ready, _font_sm, _font_md
    if not _pygame_ready:
        pygame.init(); pygame.font.init()
        _font_sm = pygame.font.SysFont("monospace", 12)
        _font_md = pygame.font.SysFont("monospace", 17, bold=True)
        _pygame_ready = True


def render_surface(world: World, tick: int = 0, coords: bool = True) -> "pygame.Surface":
    _ensure_pygame()
    spec = world.spec
    w_px, h_px = spec.width * TILE, spec.height * TILE
    surf = pygame.Surface((w_px, h_px + HUD_H))
    surf.fill(PAL["bg"])

    # tiles
    for y, row in enumerate(spec.tiles):
        for x, t in enumerate(row):
            r = pygame.Rect(x*TILE, y*TILE, TILE, TILE)
            if t == 1:
                pygame.draw.rect(surf, PAL["wall"], r)
                pygame.draw.rect(surf, PAL["wall_top"], pygame.Rect(x*TILE, y*TILE, TILE, 4))
            elif t == 2:
                pygame.draw.rect(surf, PAL["hazard"], r)
                if (x + y + tick // 8) % 2 == 0:
                    pygame.draw.rect(surf, PAL["hazard2"], pygame.Rect(x*TILE+6, y*TILE+10, TILE-12, 5))
            elif t == 3:
                pygame.draw.rect(surf, PAL["grass"], r)
            else:
                pygame.draw.rect(surf, PAL["floor"] if (x+y) % 2 == 0 else PAL["floor2"], r)

    # faint coordinate tags (helps vision grounding)
    if coords:
        for x in range(0, spec.width, 2):
            surf.blit(_font_sm.render(str(x), True, PAL["coord"]), (x*TILE+2, 1))
        for y in range(0, spec.height, 2):
            surf.blit(_font_sm.render(str(y), True, PAL["coord"]), (1, y*TILE+1))

    held = world.state.held
    # doors
    for cell, keyid in world.level.doors.items():
        x, y = cell
        opened = keyid in held
        c = PAL["door_open"] if opened else PAL["door"]
        pygame.draw.rect(surf, c, pygame.Rect(x*TILE+3, y*TILE, TILE-6, TILE))
        if not opened:
            pygame.draw.circle(surf, (230, 200, 90), (x*TILE+TILE//2, y*TILE+TILE//2), 3)

    # tables + keys + coins + cans
    for e in spec.entities:
        x, y = e.pos
        cx, cy = x*TILE + TILE//2, y*TILE + TILE//2
        if e.type == "table":
            pygame.draw.rect(surf, PAL["table"], pygame.Rect(x*TILE+3, y*TILE+7, TILE-6, TILE-12))
            pygame.draw.rect(surf, PAL["crate_edge"], pygame.Rect(x*TILE+3, y*TILE+7, TILE-6, TILE-12), 1)
        elif e.type == "key" and e.id not in held:
            pygame.draw.circle(surf, PAL["key"], (cx-4, cy), 5)
            pygame.draw.rect(surf, PAL["key"], pygame.Rect(cx-1, cy-2, 10, 4))
        elif e.type == "coin" and e.id not in held:
            pygame.draw.circle(surf, PAL["coin"], (cx, cy), 6)
            pygame.draw.circle(surf, (190, 150, 10), (cx, cy), 6, 1)
        elif e.type == "can" and e.id not in held:
            pygame.draw.rect(surf, PAL["can"], pygame.Rect(cx-5, cy-9, 10, 16), border_radius=2)
            pygame.draw.rect(surf, PAL["can_top"], pygame.Rect(cx-5, cy-9, 10, 4), border_radius=2)

    # crates (grid pos)
    for cid, x, y in world.state.crates:
        pygame.draw.rect(surf, PAL["crate"], pygame.Rect(x*TILE+3, y*TILE+3, TILE-6, TILE-6))
        pygame.draw.rect(surf, PAL["crate_edge"], pygame.Rect(x*TILE+3, y*TILE+3, TILE-6, TILE-6), 2)
        pygame.draw.line(surf, PAL["crate_edge"], (x*TILE+3, y*TILE+3), (x*TILE+TILE-3, y*TILE+TILE-3), 1)

    # balls (physics pos)
    for b in world.balls.values():
        pygame.draw.circle(surf, PAL["ball"], (int(b.position.x), int(b.position.y)), int(TILE*0.32))
        pygame.draw.circle(surf, (255, 200, 170), (int(b.position.x)-3, int(b.position.y)-3), 3)

    # exit (pulsing)
    if world.level.exit:
        x, y = world.level.exit
        pulse = abs(math.sin(tick * 0.09)) * 0.5 + 0.5
        r = pygame.Rect(x*TILE+4, y*TILE+4, TILE-8, TILE-8)
        pygame.draw.rect(surf, PAL["exit"], r)
        ring = tuple(int(v*pulse) for v in PAL["exit_ring"])
        pygame.draw.rect(surf, ring, r, 2)

    # enemies (cosmetic patrol; can occlude items in the VLM-contrast demo)
    for e in spec.entities:
        if e.type == "enemy":
            x, y = e.pos
            if e.patrol and len(e.patrol) >= 2:
                idx = (tick // 6) % len(e.patrol)
                x, y = e.patrol[idx]
            cx, cy = x*TILE + TILE//2, y*TILE + TILE//2
            pygame.draw.circle(surf, PAL["enemy"], (cx, cy), TILE//2 - 4)
            pygame.draw.circle(surf, PAL["enemy_eye"], (cx-4, cy-3), 2)
            pygame.draw.circle(surf, PAL["enemy_eye"], (cx+4, cy-3), 2)

    # agent
    ax, ay = int(world.pose[0]), int(world.pose[1])
    pygame.draw.circle(surf, PAL["player"], (ax, ay), TILE//2 - 4)
    hx = ax + int(math.cos(world.pose[2]) * (TILE//2 - 3))
    hy = ay + int(math.sin(world.pose[2]) * (TILE//2 - 3))
    pygame.draw.line(surf, PAL["player_dir"], (ax, ay), (hx, hy), 2)

    _draw_hud(surf, world, w_px, h_px)
    return surf


def _draw_hud(surf, world, w_px, h_px):
    pygame.draw.rect(surf, PAL["hud_bg"], pygame.Rect(0, h_px, w_px, HUD_H))
    surf.blit(_font_md.render(f"step {world.step_count}/{world.spec.time_limit}", True, PAL["accent"]), (8, h_px+6))
    obj = world.spec.objective_text[:64]
    surf.blit(_font_sm.render(obj, True, PAL["txt"]), (8, h_px+34))
    # predicate ticks (code-truth)
    x = 150
    for label, val in world.predicate_states().items():
        short = label.split(":", 1)[1]
        col = PAL["ok"] if val else PAL["no"]
        mark = "OK" if val else ".."
        t = _font_sm.render(f"[{mark}] {short}", True, col)
        surf.blit(t, (x, h_px+6)); x += t.get_width() + 14


# ── exports ─────────────────────────────────────────────────────────────────────

def to_png_bytes(surf) -> bytes:
    raw = pygame.image.tobytes(surf, "RGB")
    img = Image.frombytes("RGB", surf.get_size(), raw)
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return buf.getvalue()


def to_pil(surf) -> "Image.Image":
    raw = pygame.image.tobytes(surf, "RGB")
    return Image.frombytes("RGB", surf.get_size(), raw)


def save_gif(frames: List["Image.Image"], path: str, fps: int = 8) -> None:
    if not frames:
        return
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0, optimize=True)
