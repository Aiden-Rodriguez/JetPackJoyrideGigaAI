"""
jetpack.py — Jetpack Rectangle (pygame runner + frame/state sharing)

Run solo:
    python jetpack.py

Run with AI vision (launch both together):
    python jetpack.py          # terminal 1
    python vision.py           # terminal 2
"""

import sys
import json
import math
import random

import numpy as np
import pygame

from game_core import (
    GameCore,
    WIDTH, HEIGHT, FPS,
    PLAYER_W, PLAYER_H,
    clamp,
)


# ─────────────────────────────────────────────────────────────────────────────
# Runner-only config (colors, particles, exporting)
# ─────────────────────────────────────────────────────────────────────────────
BG_COLOR = (18, 18, 24)
UI_COLOR = (230, 230, 240)

PARTICLE_RATE = 80

# How often to write a frame for vision.py (every N game frames)
SHARE_EVERY_N_FRAMES = 2
SHARED_FRAME_PATH = "shared_frame.npy"
SHARED_STATE_PATH = "shared_state.json"


# ─────────────────────────────────────────────────────────────────────────────
# Small pygame helpers
# ─────────────────────────────────────────────────────────────────────────────
def draw_text(surface, text, pos, font, color=UI_COLOR):
    img = font.render(text, True, color)
    surface.blit(img, pos)


def export_frame(surface: pygame.Surface, path: str):
    """Convert pygame surface to RGB numpy array and save for vision.py to read."""
    rgb = pygame.surfarray.array3d(surface)          # (W, H, 3)
    rgb = np.transpose(rgb, (1, 0, 2))               # (H, W, 3)
    np.save(path, rgb)


def export_state(core: GameCore, path: str):
    """Write ground-truth object bboxes + score/speed/alive to JSON.

    This keeps your existing "collect_data.py" workflow working unchanged.
    """
    state = core.get_state()
    payload = {
        "objects": state.objects,
        "world_speed": state.world_speed,
        "score": state.score,
        "alive": state.alive,
    }
    with open(path, "w") as f:
        json.dump(payload, f)


# ─────────────────────────────────────────────────────────────────────────────
# Runner-only particles (purely visual)
# ─────────────────────────────────────────────────────────────────────────────
class Particle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = random.uniform(-140, -50)
        self.vy = random.uniform(-40, 40)
        self.life = random.uniform(0.25, 0.45)
        self.max_life = self.life
        self.size = random.uniform(3, 6)

    def update(self, dt):
        self.life -= dt
        self.x += self.vx * dt
        self.y += self.vy * dt

    def dead(self):
        return self.life <= 0

    def draw(self, surface):
        a = clamp(self.life / self.max_life, 0.0, 1.0)
        radius = max(1, int(self.size * a))
        s = pygame.Surface((radius * 2 + 2, radius * 2 + 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (255, 200, 80, int(220 * a)), (radius + 1, radius + 1), radius)
        surface.blit(s, (self.x - radius, self.y - radius))


# ─────────────────────────────────────────────────────────────────────────────
# Rendering of core objects (kept here, so core stays headless)
# ─────────────────────────────────────────────────────────────────────────────
def draw_player(surface: pygame.Surface, player_rect: pygame.Rect):
    body = pygame.Rect(0, 0, PLAYER_W, PLAYER_H)
    body.center = player_rect.center
    pygame.draw.rect(surface, (50, 205, 50), body, border_radius=8)
    helmet = pygame.Rect(body.x + 6, body.y + 8, body.w - 12, 14)
    pygame.draw.rect(surface, (180, 255, 180), helmet, border_radius=7)


def draw_zapper(surface: pygame.Surface, zapper):
    # We still use the zapper's endpoints/phase/angle, which live in core.
    theta_draw = -zapper.angle_deg
    w = int(zapper.length)
    h = 14

    surf = pygame.Surface((w + 8, h + 8), pygame.SRCALPHA)
    pygame.draw.rect(surf, (70, 160, 255), pygame.Rect(4, 4, w, h), border_radius=6)
    points = [
        (4 + t / 14 * w, 4 + h / 2 + math.sin(zapper.phase + t / 14 * 10) * 5)
        for t in range(15)
    ]
    pygame.draw.lines(surf, (220, 245, 255), False, points, 2)

    rotated = pygame.transform.rotate(surf, theta_draw)
    rr = rotated.get_rect(center=(int(zapper.x), int(zapper.y)))
    surface.blit(rotated, rr.topleft)


def draw_warning(surface: pygame.Surface, warning):
    alpha = int(160 + 80 * math.sin(warning.t * 18))
    warn_surf = pygame.Surface((WIDTH, 28), pygame.SRCALPHA)
    pygame.draw.rect(
        warn_surf,
        (255, 200, 0, alpha),
        pygame.Rect(WIDTH - 220, 0, 210, 28),
        border_radius=10,
    )
    surface.blit(warn_surf, (0, int(warning.y - 14)))


def draw_missile(surface: pygame.Surface, missile):
    pygame.draw.rect(surface, (255, 80, 80), missile.rect, border_radius=6)


# ─────────────────────────────────────────────────────────────────────────────
# Main pygame runner
# ─────────────────────────────────────────────────────────────────────────────
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Jetpack Rectangle")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("consolas", 20)
    big_font = pygame.font.SysFont("consolas", 34, bold=True)

    core = GameCore()
    core.reset()

    particles = []
    frame_count = 0

    running = True
    while running:
        # Note: RL will NOT use this runner. RL uses a fixed dt.
        dt = min(clock.tick(FPS) / 1000.0, 1 / 30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r and not core.player.alive:
                    core.reset()
                    particles.clear()

        # Map keys → action (binary thrust)
        keys = pygame.key.get_pressed()
        thrusting = core.player.alive and (
            keys[pygame.K_SPACE] or keys[pygame.K_UP] or keys[pygame.K_w]
        )

        # Step the *core* engine (single source of truth)
        core.step(dt=dt, thrusting=thrusting)

        # Visual-only particles
        if thrusting and core.player.alive:
            for _ in range(int(PARTICLE_RATE * dt)):
                particles.append(Particle(core.player.rect.left + 6, core.player.rect.bottom - 10))

        for p in particles:
            p.update(dt)
        particles = [p for p in particles if not p.dead()]

        # Render
        screen.fill(BG_COLOR)
        for z in core.zappers:
            draw_zapper(screen, z)
        for w in core.warnings:
            draw_warning(screen, w)
        for m in core.missiles:
            draw_missile(screen, m)
        for p in particles:
            p.draw(screen)
        draw_player(screen, core.player.rect)

        draw_text(screen, f"Score: {int(core.player.score):,}", (16, 12), font)
        draw_text(screen, f"Speed: {int(core.world_speed)}", (16, 36), font)
        if not core.player.alive:
            draw_text(screen, "CRASH!", (WIDTH // 2 - 70, HEIGHT // 2 - 60), big_font)

        pygame.display.flip()

        # Share frame + state for vision.py / collect_data.py
        frame_count += 1
        if frame_count % SHARE_EVERY_N_FRAMES == 0 and core.player.alive:
            export_frame(screen, SHARED_FRAME_PATH)
            export_state(core, SHARED_STATE_PATH)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()