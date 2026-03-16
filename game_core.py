"""
game_core.py
Core game logic
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pygame

WIDTH, HEIGHT = 960, 540
FPS = 60

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

def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def rects_collide(a: pygame.Rect, b: pygame.Rect) -> bool:
    return a.colliderect(b)


def dist_point_to_segment_sq(px, py, ax, ay, bx, by) -> float:
    """Distance-squared from point P to segment AB (used for zapper collision)."""
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    ab_len_sq = abx * abx + aby * aby
    if ab_len_sq <= 1e-9:
        return (px - ax) ** 2 + (py - ay) ** 2
    t = clamp((apx * abx + apy * aby) / ab_len_sq, 0.0, 1.0)
    cx = ax + abx * t
    cy = ay + aby * t
    return (px - cx) ** 2 + (py - cy) ** 2


class Player:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = PLAYER_X
        self.y = HEIGHT * 0.5
        self.vy = 0.0
        self.alive = True
        self.score = 0.0
        self.rect = pygame.Rect(0, 0, PLAYER_W, PLAYER_H)
        self._update_rect()

    def _update_rect(self):
        self.rect.center = (int(self.x), int(self.y))

    def update(self, dt: float, thrusting: bool):
        """Update player position/velocity exactly like the original game."""
        if not self.alive:
            self.vy += GRAVITY * dt
            self.y += self.vy * dt
            self._update_rect()
            return

        ay = GRAVITY - THRUST if thrusting else GRAVITY
        self.vy = clamp(self.vy + ay * dt, -MAX_RISE_SPEED, MAX_FALL_SPEED)
        self.y += self.vy * dt

        top_limit = PLAY_TOP + PLAYER_H / 2
        bottom_limit = PLAY_BOTTOM - PLAYER_H / 2

        if self.y < top_limit:
            self.y = top_limit
            self.vy = max(self.vy, 0)
        elif self.y > bottom_limit:
            self.y = bottom_limit
            self.vy = min(self.vy, 0)

        self._update_rect()
        self.score += dt * 10.0


class Zapper:
    def __init__(self, x: float, y: float, length: float, rng: random.Random):
        self.x = float(x)
        self.y = float(y)
        self.length = float(length)
        self.phase = rng.random() * 6.28

        r = rng.random()
        if r < 0.55:
            self.angle_deg = 0
        elif r < 0.8:
            self.angle_deg = 90
        else:
            self.angle_deg = rng.uniform(30, 60) * (1 if rng.random() < 0.5 else -1)

    def endpoints(self) -> Tuple[float, float, float, float]:
        theta = math.radians(self.angle_deg)
        dx = math.cos(theta) * (self.length * 0.5)
        dy = math.sin(theta) * (self.length * 0.5)
        return self.x - dx, self.y - dy, self.x + dx, self.y + dy

    def update(self, dt: float, world_speed: float):
        self.x -= world_speed * dt
        self.phase += dt * 6.0

    def offscreen(self) -> bool:
        ax, ay, bx, by = self.endpoints()
        return max(ax, bx) < -60

    def collides_player(self, player_rect: pygame.Rect) -> bool:
        px, py = player_rect.center
        player_radius = min(player_rect.w, player_rect.h) * 0.45
        ax, ay, bx, by = self.endpoints()
        d2 = dist_point_to_segment_sq(px, py, ax, ay, bx, by)
        return d2 <= (ZAPPER_THICKNESS * 0.5 + player_radius) ** 2

    def bbox(self) -> pygame.Rect:
        """Axis-aligned bbox approximation used for exporting / RL state."""
        ax, ay, bx, by = self.endpoints()
        x = int(min(ax, bx) - ZAPPER_THICKNESS)
        y = int(min(ay, by) - ZAPPER_THICKNESS)
        w = int(abs(bx - ax) + ZAPPER_THICKNESS * 2)
        h = int(abs(by - ay) + ZAPPER_THICKNESS * 2)
        return pygame.Rect(x, y, w, h)


class MissileWarning:
    def __init__(self, y: float):
        self.y = y
        self.t = 0.0

    def update(self, dt: float):
        self.t += dt

    def done(self) -> bool:
        return self.t >= MISSILE_WARNING_TIME


class Missile:
    def __init__(self, y: float):
        self.x = WIDTH + 30
        self.y = y

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(
            int(self.x - MISSILE_W / 2),
            int(self.y - MISSILE_H / 2),
            MISSILE_W,
            MISSILE_H,
        )

    def update(self, dt: float):
        self.x -= MISSILE_SPEED * dt

    def offscreen(self) -> bool:
        return self.x < -60


# Game engine
@dataclass
class CoreState:
    """Convenience container returned from `GameCore.step()` and `reset()`."""
    objects: List[Dict[str, Any]]
    world_speed: float
    score: float
    alive: bool


class GameCore:
    """Minimal engine that can be used by both pygame and RL.
    Action convention (for RL):
        0 = no thrust
        1 = thrust
    The pygame runner can pass a boolean `thrusting` too.
    """

    def __init__(self):
        self.rng = random.Random()
        self.player = Player()
        self.zappers: List[Zapper] = []
        self.warnings: List[MissileWarning] = []
        self.missiles: List[Missile] = []

        self.zapper_timer = 0.0
        self.missile_timer = 0.0
        self.world_speed = SCROLL_SPEED

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng.seed(seed)

        self.player.reset()
        self.zappers.clear()
        self.warnings.clear()
        self.missiles.clear()

        self.zapper_timer = self.rng.uniform(*ZAPPER_SPAWN_EVERY)
        self.missile_timer = self.rng.uniform(*MISSILE_SPAWN_EVERY)
        self.world_speed = SCROLL_SPEED

        return self.get_state()

    def step(self, dt: float, thrusting: bool) -> CoreState:
        """Advance the simulation by one step.

        This is the *single source of truth* for gameplay updates.
        """
        # 1) Speed ramp
        self.world_speed += SPEED_RAMP * dt

        # 2) Player physics
        self.player.update(dt, thrusting)

        # 3) Spawn and update zappers
        if self.player.alive:
            self.zapper_timer -= dt
            if self.zapper_timer <= 0:
                self.zappers.append(
                    Zapper(
                        WIDTH + 40,
                        self.rng.randint(PLAY_TOP + 40, PLAY_BOTTOM - 40),
                        self.rng.randint(ZAPPER_MIN_LEN, ZAPPER_MAX_LEN),
                        self.rng,
                    )
                )
                self.zapper_timer = self.rng.uniform(*ZAPPER_SPAWN_EVERY)

        for z in self.zappers:
            z.update(dt, self.world_speed)
        self.zappers = [z for z in self.zappers if not z.offscreen()]

        # 4) Spawn missile warnings
        if self.player.alive:
            self.missile_timer -= dt
            if self.missile_timer <= 0:
                y = clamp(
                    self.player.y + self.rng.uniform(-90, 90),
                    PLAY_TOP + 30,
                    PLAY_BOTTOM - 30,
                )
                self.warnings.append(MissileWarning(y))
                self.missile_timer = self.rng.uniform(*MISSILE_SPAWN_EVERY)

        # 5) Update warnings -> missiles
        new_warnings: List[MissileWarning] = []
        for w in self.warnings:
            w.update(dt)
            if w.done():
                self.missiles.append(Missile(w.y))
            else:
                new_warnings.append(w)
        self.warnings = new_warnings

        # 6) Update missiles
        for m in self.missiles:
            m.update(dt)
        self.missiles = [m for m in self.missiles if not m.offscreen()]

        # 7) Collisions -> death
        if self.player.alive:
            for z in self.zappers:
                if z.collides_player(self.player.rect):
                    self.player.alive = False
                    break

        if self.player.alive:
            for m in self.missiles:
                if rects_collide(self.player.rect, m.rect):
                    self.player.alive = False
                    break

        return self.get_state()

    def get_state(self) -> CoreState:
        """Export a ground-truth state in the same format your tooling expects."""
        objects: List[Dict[str, Any]] = []

        # Player
        r = self.player.rect
        objects.append({"label": "player", "x": r.left, "y": r.top, "w": r.width, "h": r.height})

        # Zappers
        for z in self.zappers:
            rr = z.bbox()
            objects.append({"label": "zapper", "x": rr.left, "y": rr.top, "w": rr.width, "h": rr.height})

        # Missiles
        for m in self.missiles:
            rr = m.rect
            objects.append({"label": "missile", "x": rr.left, "y": rr.top, "w": rr.width, "h": rr.height})

        # Warnings (same bbox as original export)
        for w in self.warnings:
            objects.append({"label": "warning", "x": WIDTH - 220, "y": int(w.y - 14), "w": 210, "h": 28})

        return CoreState(
            objects=objects,
            world_speed=self.world_speed,
            score=self.player.score,
            alive=self.player.alive,
        )