"""
game.py — Jetpack Rectangle (standalone game + frame-sharing for AI vision)

Run solo:
    python game.py

Run with AI vision (launch both together):
    python game.py          # in terminal 1
    python vision.py        # in terminal 2

Frames are shared via a memory-mapped numpy array written to 'shared_frame.npy'
every N frames so vision.py can read them independently.
"""

import random
import sys
import math
import json
import numpy as np
import pygame

# Config
WIDTH, HEIGHT = 960, 540
FPS = 60

BG_COLOR = (18, 18, 24)
UI_COLOR = (230, 230, 240)

GROUND_MARGIN = 24
PLAY_TOP = GROUND_MARGIN
PLAY_BOTTOM = HEIGHT - GROUND_MARGIN

PLAYER_W, PLAYER_H = 38, 48
PLAYER_X = 140
GRAVITY = 1600.0
THRUST = 3000.0
MAX_FALL_SPEED = 950.0
MAX_RISE_SPEED = 650.0

SCROLL_SPEED = 320.0
SPEED_RAMP = 8.0

ZAPPER_MIN_LEN = 140
ZAPPER_MAX_LEN = 280
ZAPPER_THICKNESS = 14
ZAPPER_SPAWN_EVERY = (1.1, 1.8)

MISSILE_SPAWN_EVERY = (2.0, 3.8)
MISSILE_WARNING_TIME = 0.75
MISSILE_SPEED = 650.0
MISSILE_W, MISSILE_H = 44, 18

PARTICLE_RATE = 80

# How often to write a frame for vision.py (every N game frames)
SHARE_EVERY_N_FRAMES = 2
SHARED_FRAME_PATH = "shared_frame.npy"
SHARED_STATE_PATH = "shared_state.json"


# Helpers
def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def draw_text(surface, text, pos, font, color=UI_COLOR):
    img = font.render(text, True, color)
    surface.blit(img, pos)


def rects_collide(a, b):
    return a.colliderect(b)


def dist_point_to_segment_sq(px, py, ax, ay, bx, by):
    abx = bx - ax; aby = by - ay
    apx = px - ax; apy = py - ay
    ab_len_sq = abx * abx + aby * aby
    if ab_len_sq <= 1e-9:
        return (px - ax) ** 2 + (py - ay) ** 2
    t = clamp((apx * abx + apy * aby) / ab_len_sq, 0.0, 1.0)
    cx = ax + abx * t; cy = ay + aby * t
    return (px - cx) ** 2 + (py - cy) ** 2


# Game Classes
class Player:
    def __init__(self): self.reset()

    def reset(self):
        self.x = PLAYER_X; self.y = HEIGHT * 0.5
        self.vy = 0.0; self.alive = True; self.score = 0.0
        self.rect = pygame.Rect(0, 0, PLAYER_W, PLAYER_H)
        self._update_rect()

    def _update_rect(self): self.rect.center = (int(self.x), int(self.y))

    def update(self, dt, thrusting):
        if not self.alive:
            self.vy += GRAVITY * dt; self.y += self.vy * dt
            self._update_rect(); return
        ay = GRAVITY - THRUST if thrusting else GRAVITY
        self.vy = clamp(self.vy + ay * dt, -MAX_RISE_SPEED, MAX_FALL_SPEED)
        self.y += self.vy * dt
        top_limit = PLAY_TOP + PLAYER_H / 2
        bottom_limit = PLAY_BOTTOM - PLAYER_H / 2
        if self.y < top_limit: self.y = top_limit; self.vy = max(self.vy, 0)
        elif self.y > bottom_limit: self.y = bottom_limit; self.vy = min(self.vy, 0)
        self._update_rect(); self.score += dt * 10.0

    def draw(self, surface):
        body = pygame.Rect(0, 0, PLAYER_W, PLAYER_H)
        body.center = self.rect.center
        pygame.draw.rect(surface, (50, 205, 50), body, border_radius=8)
        helmet = pygame.Rect(body.x + 6, body.y + 8, body.w - 12, 14)
        pygame.draw.rect(surface, (180, 255, 180), helmet, border_radius=7)


class Zapper:
    def __init__(self, x, y, length):
        self.x = float(x); self.y = float(y)
        self.length = float(length); self.phase = random.random() * 6.28
        r = random.random()
        if r < 0.55: self.angle_deg = 0
        elif r < 0.8: self.angle_deg = 90
        else: self.angle_deg = random.uniform(30, 60) * (1 if random.random() < 0.5 else -1)

    def endpoints(self):
        theta = math.radians(self.angle_deg)
        dx = math.cos(theta) * (self.length * 0.5)
        dy = math.sin(theta) * (self.length * 0.5)
        return self.x - dx, self.y - dy, self.x + dx, self.y + dy

    def update(self, dt, world_speed):
        self.x -= world_speed * dt; self.phase += dt * 6.0

    def offscreen(self):
        ax, ay, bx, by = self.endpoints()
        return max(ax, bx) < -60

    def collides_player(self, player_rect):
        px, py = player_rect.center
        player_radius = min(player_rect.w, player_rect.h) * 0.45
        ax, ay, bx, by = self.endpoints()
        d2 = dist_point_to_segment_sq(px, py, ax, ay, bx, by)
        return d2 <= (ZAPPER_THICKNESS * 0.5 + player_radius) ** 2

    def draw(self, surface):
        theta_draw = -self.angle_deg
        w = int(self.length); h = int(ZAPPER_THICKNESS)
        surf = pygame.Surface((w + 8, h + 8), pygame.SRCALPHA)
        pygame.draw.rect(surf, (70, 160, 255), pygame.Rect(4, 4, w, h), border_radius=6)
        points = [(4 + t / 14 * w, 4 + h / 2 + math.sin(self.phase + t / 14 * 10) * 5)
                  for t in range(15)]
        pygame.draw.lines(surf, (220, 245, 255), False, points, 2)
        rotated = pygame.transform.rotate(surf, theta_draw)
        rr = rotated.get_rect(center=(int(self.x), int(self.y)))
        surface.blit(rotated, rr.topleft)


class MissileWarning:
    def __init__(self, y): self.y = y; self.t = 0
    def update(self, dt): self.t += dt
    def done(self): return self.t >= MISSILE_WARNING_TIME

    def draw(self, surface):
        alpha = int(160 + 80 * math.sin(self.t * 18))
        warn_surf = pygame.Surface((WIDTH, 28), pygame.SRCALPHA)
        pygame.draw.rect(warn_surf, (255, 200, 0, alpha),
                         pygame.Rect(WIDTH - 220, 0, 210, 28), border_radius=10)
        surface.blit(warn_surf, (0, int(self.y - 14)))


class Missile:
    def __init__(self, y): self.x = WIDTH + 30; self.y = y

    @property
    def rect(self):
        return pygame.Rect(int(self.x - MISSILE_W / 2), int(self.y - MISSILE_H / 2),
                           MISSILE_W, MISSILE_H)

    def update(self, dt): self.x -= MISSILE_SPEED * dt
    def offscreen(self): return self.x < -60

    def draw(self, surface):
        pygame.draw.rect(surface, (255, 80, 80), self.rect, border_radius=6)


class Particle:
    def __init__(self, x, y):
        self.x = x; self.y = y
        self.vx = random.uniform(-140, -50); self.vy = random.uniform(-40, 40)
        self.life = random.uniform(0.25, 0.45); self.max_life = self.life
        self.size = random.uniform(3, 6)

    def update(self, dt): self.life -= dt; self.x += self.vx * dt; self.y += self.vy * dt
    def dead(self): return self.life <= 0

    def draw(self, surface):
        a = clamp(self.life / self.max_life, 0.0, 1.0)
        radius = max(1, int(self.size * a))
        s = pygame.Surface((radius * 2 + 2, radius * 2 + 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (255, 200, 80, int(220 * a)), (radius + 1, radius + 1), radius)
        surface.blit(s, (self.x - radius, self.y - radius))


# Frame export helper
def export_frame(surface: pygame.Surface, path: str):
    """Convert pygame surface to BGR numpy array and save for vision.py to read."""
    rgb = pygame.surfarray.array3d(surface)          # (W, H, 3)
    rgb = np.transpose(rgb, (1, 0, 2))               # (H, W, 3)
    np.save(path, rgb)


def export_state(player, zappers, missiles, warnings, world_speed, path: str):
    """
    Write the ground-truth positions of all game objects to a JSON file.
    collect_data.py reads this instead of running color detection.

    Each object has:
        label       — class name
        x, y, w, h  — bounding box in pixel coords (top-left origin)
    """
    objects = []

    # Player
    r = player.rect
    objects.append({
        "label": "player",
        "x": r.left, "y": r.top, "w": r.width, "h": r.height,
    })

    # Zappers — use their actual rendered bounding box
    for z in zappers:
        ax, ay, bx, by = z.endpoints()
        x = int(min(ax, bx) - ZAPPER_THICKNESS)
        y = int(min(ay, by) - ZAPPER_THICKNESS)
        w = int(abs(bx - ax) + ZAPPER_THICKNESS * 2)
        h = int(abs(by - ay) + ZAPPER_THICKNESS * 2)
        objects.append({"label": "zapper", "x": x, "y": y, "w": w, "h": h})

    # Missiles
    for m in missiles:
        r = m.rect
        objects.append({
            "label": "missile",
            "x": r.left, "y": r.top, "w": r.width, "h": r.height,
        })

    # Warnings
    for wn in warnings:
        objects.append({
            "label": "warning",
            "x": WIDTH - 220, "y": int(wn.y - 14), "w": 210, "h": 28,
        })

    state = {
        "objects":     objects,
        "world_speed": world_speed,
        "score":       player.score,
        "alive":       player.alive,
    }

    with open(path, "w") as f:
        json.dump(state, f)


# Main
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Jetpack Rectangle")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("consolas", 20)
    big_font = pygame.font.SysFont("consolas", 34, bold=True)

    player = Player()
    zappers, warnings, missiles, particles = [], [], [], []
    zapper_timer = random.uniform(*ZAPPER_SPAWN_EVERY)
    missile_timer = random.uniform(*MISSILE_SPAWN_EVERY)
    world_speed = SCROLL_SPEED
    frame_count = 0

    running = True
    while running:
        dt = min(clock.tick(FPS) / 1000.0, 1 / 30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                if event.key == pygame.K_r and not player.alive:
                    player.reset()
                    zappers.clear(); warnings.clear()
                    missiles.clear(); particles.clear()
                    world_speed = SCROLL_SPEED

        keys = pygame.key.get_pressed()
        thrusting = player.alive and (
            keys[pygame.K_SPACE] or keys[pygame.K_UP] or keys[pygame.K_w])

        world_speed += SPEED_RAMP * dt
        player.update(dt, thrusting)

        if thrusting and player.alive:
            for _ in range(int(PARTICLE_RATE * dt)):
                particles.append(Particle(player.rect.left + 6, player.rect.bottom - 10))

        for p in particles: p.update(dt)
        particles = [p for p in particles if not p.dead()]

        if player.alive:
            zapper_timer -= dt
            if zapper_timer <= 0:
                zappers.append(Zapper(WIDTH + 40,
                    random.randint(PLAY_TOP + 40, PLAY_BOTTOM - 40),
                    random.randint(ZAPPER_MIN_LEN, ZAPPER_MAX_LEN)))
                zapper_timer = random.uniform(*ZAPPER_SPAWN_EVERY)

        for z in zappers: z.update(dt, world_speed)
        zappers = [z for z in zappers if not z.offscreen()]

        if player.alive:
            missile_timer -= dt
            if missile_timer <= 0:
                y = clamp(player.y + random.uniform(-90, 90), PLAY_TOP + 30, PLAY_BOTTOM - 30)
                warnings.append(MissileWarning(y))
                missile_timer = random.uniform(*MISSILE_SPAWN_EVERY)

        new_warnings = []
        for w in warnings:
            w.update(dt)
            if w.done(): missiles.append(Missile(w.y))
            else: new_warnings.append(w)
        warnings = new_warnings

        for m in missiles: m.update(dt)
        missiles = [m for m in missiles if not m.offscreen()]

        if player.alive:
            for z in zappers:
                if z.collides_player(player.rect): player.alive = False; break
        if player.alive:
            for m in missiles:
                if rects_collide(player.rect, m.rect): player.alive = False; break

        # Render
        screen.fill(BG_COLOR)
        for z in zappers: z.draw(screen)
        for w in warnings: w.draw(screen)
        for m in missiles: m.draw(screen)
        for p in particles: p.draw(screen)
        player.draw(screen)
        draw_text(screen, f"Score: {int(player.score):,}", (16, 12), font)
        draw_text(screen, f"Speed: {int(world_speed)}", (16, 36), font)
        if not player.alive:
            draw_text(screen, "CRASH!", (WIDTH // 2 - 70, HEIGHT // 2 - 60), big_font)

        pygame.display.flip()

        # Share frame + state for vision.py / collect_data.py
        # Only export when player is alive
        frame_count += 1
        if frame_count % SHARE_EVERY_N_FRAMES == 0 and player.alive:
            export_frame(screen, SHARED_FRAME_PATH)
            export_state(player, zappers, missiles, warnings, world_speed, SHARED_STATE_PATH)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()