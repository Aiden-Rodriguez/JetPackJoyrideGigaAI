"""
jetpack_env.py

Gymnasium environment wrapper around the *native* gameplay code in "game_core.py".

Key properties
--------------
- Uses GameCore as the single source of truth (no duplicate physics/spawn logic).
- Discrete(2) action space:
    0 = no thrust
    1 = thrust
- Observation is a compact vector built from ground-truth object bboxes.
  This trains far faster than pixels and matches your existing object-export
  format (label/x/y/w/h).

Install deps for RL:
    pip install gymnasium stable-baselines3 numpy
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from game_core import (
    GameCore,
    WIDTH,
    HEIGHT,
    PLAY_TOP,
    PLAY_BOTTOM,
    MAX_FALL_SPEED,
    MAX_RISE_SPEED,
    SCROLL_SPEED,
)


class JetpackEnv(gym.Env):
    """Jetpack RL environment.

    Observation design
    ------------------
    Convert the current scene into a fixed-size numeric vector.

    Vector layout:
      [player_y_norm, player_vy_norm, world_speed_norm,
       threats(K)*[dx, dy, w, h, is_zapper, is_missile],
       warning_present]
    """

    metadata = {"render_modes": ["none"], "render_fps": 60}

    def __init__(
        self,
        k_threats: int = 3,
        max_steps: int = 60 * 60,  # ~60 seconds at 60 FPS
        seed: Optional[int] = None,
        fixed_dt: float = 1.0 / 60.0,
    ):
        super().__init__()

        self.core = GameCore()
        self.k_threats = int(k_threats)
        self.max_steps = int(max_steps)
        self.seed_value = seed
        self.dt = float(fixed_dt)
        self.steps = 0

        # Binary action: 0/1
        self.action_space = spaces.Discrete(2)

        obs_dim = 3 + (self.k_threats * 6) + 1
        # Wide bounds; we normalize in _make_obs.
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(obs_dim,), dtype=np.float32
        )
        self._episode_rng = np.random.default_rng(seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # If Gym provides a seed, use it (reproducibility when desired).
        # Otherwise, do NOT reseed on reset; let the core RNG keep evolving,
        # producing different episodes each time.
        if seed is not None:
        # explicit seed overrides everything
            ep_seed = seed
        else:
            ep_seed = int(self._episode_rng.integers(0, 2**31 - 1))

        state = self.core.reset(seed=ep_seed)
        self.steps = 0
        obs = self._make_obs(state.objects, state.world_speed)

        info = {"score": state.score, "alive": state.alive}
        return obs, info

    def step(self, action: int):
        self.steps += 1

        # Convert action to thrust boolean
        thrusting = bool(int(action) == 1)

        prev_score = self.core.player.score
        state = self.core.step(dt=self.dt, thrusting=thrusting)

        # Reward: survive longer.
        # - +1 each step alive
        # - small shaping using score delta
        # - strong negative on death
        alive = state.alive
        terminated = not alive
        truncated = self.steps >= self.max_steps

        score_delta = state.score - prev_score
        reward = 1.0 + 0.05 * score_delta
        if terminated:
            reward = -100.0

        obs = self._make_obs(state.objects, state.world_speed)
        info = {
            "score": state.score,
            "alive": state.alive,
            "world_speed": state.world_speed,
        }

        return obs, float(reward), bool(terminated), bool(truncated), info

    # ──────────────────────────────────────────────────────────────────────
    # Observation builder
    # ──────────────────────────────────────────────────────────────────────
    def _make_obs(self, objects: List[Dict[str, Any]], world_speed: float) -> np.ndarray:
        """Convert object list (label/x/y/w/h) to a fixed-size vector."""

        # Extract player bbox
        player = next((o for o in objects if o["label"] == "player"), None)
        if player is None:
            # Should never happen, but keep robust.
            player_x = 140.0
            player_y = (PLAY_TOP + PLAY_BOTTOM) * 0.5
            player_vy = 0.0
        else:
            player_x = float(player["x"] + player["w"] * 0.5)
            player_y = float(player["y"] + player["h"] * 0.5)
            # vy is internal state (ground truth) – allowed for fast training.
            player_vy = float(self.core.player.vy)

        # Normalize primary scalars
        y_center = (PLAY_TOP + PLAY_BOTTOM) * 0.5
        y_half_range = (PLAY_BOTTOM - PLAY_TOP) * 0.5
        y_norm = (player_y - y_center) / y_half_range

        vy_norm = player_vy / max(MAX_FALL_SPEED, MAX_RISE_SPEED)

        # World speed slowly increases; keep it small.
        speed_norm = (world_speed - SCROLL_SPEED) / 1000.0

        # Gather threats
        warning_present = 0.0
        threats = []  # list of (dx, dy, bbox, label)

        for o in objects:
            label = o["label"]
            if label == "warning":
                warning_present = 1.0
                continue
            if label not in ("zapper", "missile"):
                continue

            cx = float(o["x"] + o["w"] * 0.5)
            cy = float(o["y"] + o["h"] * 0.5)
            dx = cx - player_x

            # Only consider upcoming hazards.
            if dx <= -10:
                continue

            dy = cy - player_y
            threats.append((dx, dy, o, label))

        # Nearest-first by dx
        threats.sort(key=lambda t: t[0])
        threats = threats[: self.k_threats]

        # Pack into fixed slots
        threat_vec: List[float] = []
        for i in range(self.k_threats):
            if i < len(threats):
                dx, dy, o, label = threats[i]
                dx_n = dx / WIDTH
                dy_n = dy / HEIGHT
                w_n = float(o["w"]) / WIDTH
                h_n = float(o["h"]) / HEIGHT
                is_zapper = 1.0 if label == "zapper" else 0.0
                is_missile = 1.0 if label == "missile" else 0.0
                threat_vec.extend([dx_n, dy_n, w_n, h_n, is_zapper, is_missile])
            else:
                threat_vec.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        obs = np.array([y_norm, vy_norm, speed_norm, *threat_vec, warning_present], dtype=np.float32)
        return obs

    def close(self):
        return