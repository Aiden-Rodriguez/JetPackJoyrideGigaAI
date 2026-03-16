import sys
import math
import random
from typing import Optional

import pygame
from stable_baselines3 import PPO

from game_core import (
    GameCore,
    WIDTH,
    HEIGHT,
    FPS,
    PLAYER_W,
    PLAYER_H,
    clamp,
)

from jetpack_env import JetpackEnv


BG_COLOR = (18, 18, 24)
UI_COLOR = (230, 230, 240)
PARTICLE_RATE = 80


def draw_text(surface, text, pos, font, color=UI_COLOR):
    img = font.render(text, True, color)
    surface.blit(img, pos)


def center_draw_text(surface, text, pos, font, color=UI_COLOR):
    img = font.render(text, True, color)
    rect = img.get_rect(center=(int(pos[0]), int(pos[1])))
    surface.blit(img, rect)


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


def draw_player(surface, player_rect):
    body = pygame.Rect(0, 0, PLAYER_W, PLAYER_H)
    body.center = player_rect.center
    pygame.draw.rect(surface, (50, 205, 50), body, border_radius=8)
    helmet = pygame.Rect(body.x + 6, body.y + 8, body.w - 12, 14)
    pygame.draw.rect(surface, (180, 255, 180), helmet, border_radius=7)


def draw_zapper(surface, zapper):
    theta_draw = -zapper.angle_deg
    w = int(zapper.length)
    h = 14
    surf = pygame.Surface((w + 8, h + 8), pygame.SRCALPHA)
    pygame.draw.rect(surf, (70, 160, 255), pygame.Rect(4, 4, w, h), border_radius=6)
    points = [(4 + t / 14 * w, 4 + h / 2 + math.sin(zapper.phase + t / 14 * 10) * 5) for t in range(15)]
    pygame.draw.lines(surf, (220, 245, 255), False, points, 2)
    rotated = pygame.transform.rotate(surf, theta_draw)
    rr = rotated.get_rect(center=(int(zapper.x), int(zapper.y)))
    surface.blit(rotated, rr.topleft)


def draw_warning(surface, warning):
    alpha = int(160 + 80 * math.sin(warning.t * 18))
    warn_surf = pygame.Surface((WIDTH, 28), pygame.SRCALPHA)
    pygame.draw.rect(warn_surf, (255, 200, 0, alpha), pygame.Rect(WIDTH - 220, 0, 210, 28), border_radius=10)
    surface.blit(warn_surf, (0, int(warning.y - 14)))


def draw_missile(surface, missile):
    pygame.draw.rect(surface, (255, 80, 80), missile.rect, border_radius=6)


def main(
    model_path: str = "models/jetpack_ppo_v3.zip",
    seed: Optional[int] = None,
):
    if seed is None:
        seed = random.randint(0, 1000)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Jetpack Rectangle — PPO Agent")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("consolas", 20)
    big_font = pygame.font.SysFont("consolas", 34, bold=True)

    model = PPO.load(model_path)

    env = JetpackEnv(k_threats=3, seed=seed, fixed_dt=1.0 / 60.0)
    env.reset(seed=seed)
    core: GameCore = env.core

    particles = []
    fixed_dt = 1.0 / 60.0

    running = True
    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    core.reset(seed=seed)
                    particles.clear()
                if event.key == pygame.K_t:
                    seed = random.randint(0, 1000)
                    core.reset(seed=seed)
                    particles.clear()

        state = core.get_state()
        obs = env._make_obs(state.objects, state.world_speed)
        action, _ = model.predict(obs, deterministic=True)
        thrusting = bool(int(action) == 1) and core.player.alive

        core.step(dt=fixed_dt, thrusting=thrusting)

        if thrusting and core.player.alive:
            for _ in range(int(PARTICLE_RATE * fixed_dt)):
                particles.append(Particle(core.player.rect.left + 6, core.player.rect.bottom - 10))

        for p in particles:
            p.update(fixed_dt)
        particles = [p for p in particles if not p.dead()]

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
        draw_text(screen, f"Action: {'THRUST' if thrusting else 'COAST'}", (16, 60), font)

        if not core.player.alive:
            center_draw_text(screen, "CRASH!", (WIDTH // 2, HEIGHT // 2 - 60), big_font)
            center_draw_text(screen, "Press R to reset  |  T for new seed", (WIDTH // 2, HEIGHT // 2), big_font)

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()