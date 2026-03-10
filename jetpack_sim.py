"""
jetpack_sim.py

A headless (optional-render) simulator that mirrors the logic in jetpack.py:
- Player physics with gravity + thrust
- Speed ramp
- Zapper / missile spawns
- Collision and death
- Score accumulation

This file is designed to integrate PPO training cleanly without needing
to run two processes or read/write shared_state.json.

The logic is intentionally kept close to jetpack.py's main loop so your
agent trains on the "real" game dynamics (minus rendering).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import pygame


# ──────────────────────────────────────────────
# Game constants (match jetpack.py)
# ──────────────────────────────────────────────
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


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def rects_collide(a: pygame.Rect, b: pygame.Rect) -> bool:
    return a.colliderect(b)


def dist_point_to_segment_sq(px, py, ax, ay, bx, by) -> float:
    """Same collision helper as jetpack.py."""
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


# ──────────────────────────────────────────────
# Game objects
# ──────────────────────────────────────────────
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
        """
        Mirrors jetpack.py player update:
        - ay = GRAVITY - THRUST if thrusting else GRAVITY
        - clamp vy, integrate y
        - clamp y between play bounds
        - score += dt * 10
        """
        if not self.alive:
            # After death, player falls; keep consistent with original.
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
    def __init__(self, x: float, y: float, length: float):
        self.x = float(x)
        self.y = float(y)
        self.length = float(length)
        self.phase = random.random() * 6.28

        # Keep same distribution as jetpack.py
        r = random.random()
        if r < 0.55:
            self.angle_deg = 0
        elif r < 0.8:
            self.angle_deg = 90
        else:
            self.angle_deg = random.uniform(30, 60) * (1 if random.random() < 0.5 else -1)

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
        """
        Mirrors your export_state() bounding box logic for zappers:
        It approximates the rotated zapper with an axis-aligned box.
        """
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


# ──────────────────────────────────────────────
# Simulator
# ──────────────────────────────────────────────
@dataclass
class StepResult:
    alive: bool
    score: float
    world_speed: float
    objects: List[Dict[str, Any]]  # (label, x,y,w,h) for observation building


class JetpackSim:
    """
    Headless simulator with an API designed for RL:
      reset(seed) -> StepResult
      step(action) -> (StepResult, reward, terminated)

    Action:
      0 = no thrust
      1 = thrust
    """
    def __init__(self, dt: float = 1.0 / FPS):
        self.dt = dt

        # RNG seedable for reproducibility in training
        self._rng = random.Random()

        # Game state
        self.player = Player()
        self.zappers: List[Zapper] = []
        self.warnings: List[MissileWarning] = []
        self.missiles: List[Missile] = []

        self.zapper_timer = 0.0
        self.missile_timer = 0.0
        self.world_speed = SCROLL_SPEED

        self.reset()

    def reset(self, seed: Optional[int] = None) -> StepResult:
        if seed is not None:
            self._rng.seed(seed)
            # Also seed global random for classes that used random.*
            random.seed(seed)

        self.player.reset()
        self.zappers.clear()
        self.warnings.clear()
        self.missiles.clear()

        self.zapper_timer = self._rng.uniform(*ZAPPER_SPAWN_EVERY)
        self.missile_timer = self._rng.uniform(*MISSILE_SPAWN_EVERY)
        self.world_speed = SCROLL_SPEED

        return self._build_result()

    def step(self, action: int) -> Tuple[StepResult, float, bool]:
        """
        Executes one simulator tick.

        Returns:
            result: StepResult containing current objects and state
            reward: survival reward (+1 per step alive, big negative on death)
            terminated: True if episode ended due to death
        """
        thrusting = (action == 1) and self.player.alive

        # Match game loop: speed ramp -> player update -> spawns -> obstacle updates -> collisions
        self.world_speed += SPEED_RAMP * self.dt
        prev_score = self.player.score

        self.player.update(self.dt, thrusting)

        # Spawn zappers
        if self.player.alive:
            self.zapper_timer -= self.dt
            if self.zapper_timer <= 0:
                self.zappers.append(
                    Zapper(
                        WIDTH + 40,
                        self._rng.randint(PLAY_TOP + 40, PLAY_BOTTOM - 40),
                        self._rng.randint(ZAPPER_MIN_LEN, ZAPPER_MAX_LEN),
                    )
                )
                self.zapper_timer = self._rng.uniform(*ZAPPER_SPAWN_EVERY)

        # Update zappers
        for z in self.zappers:
            z.update(self.dt, self.world_speed)
        self.zappers = [z for z in self.zappers if not z.offscreen()]

        # Spawn missile warnings
        if self.player.alive:
            self.missile_timer -= self.dt
            if self.missile_timer <= 0:
                y = clamp(
                    self.player.y + self._rng.uniform(-90, 90),
                    PLAY_TOP + 30,
                    PLAY_BOTTOM - 30,
                )
                self.warnings.append(MissileWarning(y))
                self.missile_timer = self._rng.uniform(*MISSILE_SPAWN_EVERY)

        # Progress warnings -> missiles
        new_warnings: List[MissileWarning] = []
        for w in self.warnings:
            w.update(self.dt)
            if w.done():
                self.missiles.append(Missile(w.y))
            else:
                new_warnings.append(w)
        self.warnings = new_warnings

        # Update missiles
        for m in self.missiles:
            m.update(self.dt)
        self.missiles = [m for m in self.missiles if not m.offscreen()]

        # Collisions
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

        # Reward: survive-as-long-as-possible
        # - +1 per step alive
        # - big negative on death to create a strong terminal signal
        alive = self.player.alive
        terminated = not alive

        # Keep the reward simple and stable:
        # survival reward dominates, score delta is a tiny shaping term.
        score_delta = self.player.score - prev_score
        reward = 1.0 + 0.05 * score_delta
        if terminated:
            reward = -100.0

        return self._build_result(), reward, terminated

    def _build_result(self) -> StepResult:
        """
        Build a result with a list of ground-truth bboxes, matching the
        spirit of shared_state.json objects :contentReference[oaicite:3]{index=3}.
        """
        objects: List[Dict[str, Any]] = []

        # Player bbox
        r = self.player.rect
        objects.append({"label": "player", "x": r.left, "y": r.top, "w": r.width, "h": r.height})

        # Zapper bboxes (axis-aligned approximation)
        for z in self.zappers:
            rr = z.bbox()
            objects.append({"label": "zapper", "x": rr.left, "y": rr.top, "w": rr.width, "h": rr.height})

        # Missile bboxes
        for m in self.missiles:
            rr = m.rect
            objects.append({"label": "missile", "x": rr.left, "y": rr.top, "w": rr.width, "h": rr.height})

        # Warning bboxes (same shape/placement idea as export_state())
        for w in self.warnings:
            objects.append({"label": "warning", "x": WIDTH - 220, "y": int(w.y - 14), "w": 210, "h": 28})

        return StepResult(
            alive=self.player.alive,
            score=self.player.score,
            world_speed=self.world_speed,
            objects=objects,
        )