"""
play.py — open a REAL window and play (or watch) a generated environment.

  uv run play.py                        # human plays a cached level (no API key)
  uv run play.py --env three_rooms      # pick a cached level (see specs/)
  uv run play.py --watch                # watch the scripted oracle solve it live
  uv run play.py "a room with a key, a locked door, and a can"   # generate live (needs API key)

Controls:  arrow keys / WASD move · SPACE wait (pass a tick to dodge patrols) · R reset · ESC quit.
Goal + progress are shown in the HUD; predicate ticks flip green as you satisfy the objective.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ["HARNESS_WINDOW"] = "1"      # tell the renderer to open a real window (not headless)

import pygame

from harness.dsl.schema import EnvSpec
from harness.engine.world import World
from harness.engine import renderer as R


def load_spec(env_name: str, command: str | None) -> EnvSpec:
    if command:
        from harness.generator import generate
        print(f"[play] generating from: {command!r}")
        spec, _vr, _ = generate(command)
        return spec
    path = os.path.join(os.path.dirname(__file__), "specs", f"{env_name}.json")
    with open(path) as f:
        return EnvSpec(**json.load(f)["spec"])


KEYMAP = {
    pygame.K_UP: "up", pygame.K_w: "up",
    pygame.K_DOWN: "down", pygame.K_s: "down",
    pygame.K_LEFT: "left", pygame.K_a: "left",
    pygame.K_RIGHT: "right", pygame.K_d: "right",
    pygame.K_SPACE: "wait",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", nargs="?", default=None, help="text command (needs API key)")
    ap.add_argument("--env", default="key_crate_return", help="cached env name (see specs/)")
    ap.add_argument("--watch", action="store_true", help="watch the oracle solve it")
    ap.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = until quit)")
    args = ap.parse_args()

    spec = load_spec(args.env, args.command)
    world = World(spec)

    # oracle plan for --watch
    plan = []
    if args.watch:
        from harness.verifier import solve
        plan, _ = solve(world.level, spec.objective)

    pygame.init()
    surf = R.render_surface(world)
    screen = pygame.display.set_mode(surf.get_size())
    mode = "WATCH oracle" if args.watch else "arrows/WASD to move"
    pygame.display.set_caption(f"{spec.name}   [{mode}]  —  {spec.objective_text}")
    clock = pygame.time.Clock()

    tick, plan_i, move_ms, frames = 0, 0, 0, 0
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_r:
                    world.reset(); tick = 0; plan_i = 0
                elif not args.watch and e.key in KEYMAP and not world.done:
                    world.step(KEYMAP[e.key]); tick += 1

        if args.watch and not world.done:
            move_ms += clock.get_time()
            if move_ms >= 220 and plan_i < len(plan):
                world.step(plan[plan_i]); plan_i += 1; tick += 1; move_ms = 0

        screen.blit(R.render_surface(world, tick=tick), (0, 0))
        pygame.display.flip()
        clock.tick(30)

        frames += 1
        if args.frames and frames >= args.frames:
            running = False

    result = "WON" if world.won else ("done" if world.done else "quit")
    print(f"[play] {result}  (steps {world.step_count})")
    pygame.quit()


if __name__ == "__main__":
    main()
