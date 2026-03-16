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
    PLAYER_W,
    MAX_FALL_SPEED,
    MAX_RISE_SPEED,
    SCROLL_SPEED,
)


class JetpackEnv(gym.Env):
    metadata = {"render_modes": ["none"], "render_fps": 60}

    def __init__(
        self,
        k_threats: int = 3,
        max_steps: int = 60 * 60 * 30,
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

        self.action_space = spaces.Discrete(2)

        obs_dim = 3 + 2 + (self.k_threats * 8) + 2
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(obs_dim,), dtype=np.float32
        )
        self._episode_rng = np.random.default_rng(seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if seed is not None:
            ep_seed = seed
        else:
            ep_seed = int(self._episode_rng.integers(0, 2**31 - 1))

        state = self.core.reset(seed=ep_seed)
        self.steps = 0
        obs = self._make_obs(state.objects, state.world_speed)
        return obs, {"score": state.score, "alive": state.alive}

    def step(self, action: int):
        self.steps += 1
        thrusting = bool(int(action) == 1)

        state = self.core.step(dt=self.dt, thrusting=thrusting)

        alive = state.alive
        terminated = not alive
        truncated = self.steps >= self.max_steps

        # Simple, stable reward:
        #   +1 every step alive   (survives longer → higher cumulative reward)
        #   -50 on death          (strong but not overwhelming terminal signal)
        # No score-delta shaping — it was just time * 10 * 0.05 = noise.
        reward = -50.0 if terminated else 1.0

        obs = self._make_obs(state.objects, state.world_speed)
        info = {
            "score": state.score,
            "alive": state.alive,
            "world_speed": state.world_speed,
        }

        return obs, float(reward), bool(terminated), bool(truncated), info

    def _make_obs(self, objects: List[Dict[str, Any]], world_speed: float) -> np.ndarray:
        play_range = float(PLAY_BOTTOM - PLAY_TOP)   # normalisation denominator

        player = next((o for o in objects if o["label"] == "player"), None)
        if player is None:
            player_x   = 140.0
            player_y   = (PLAY_TOP + PLAY_BOTTOM) * 0.5
            player_vy  = 0.0
            player_top = player_y - 24.0
            player_bot = player_y + 24.0
        else:
            player_x   = float(player["x"] + player["w"] * 0.5)
            player_y   = float(player["y"] + player["h"] * 0.5)
            player_vy  = float(self.core.player.vy)
            player_top = float(player["y"])
            player_bot = float(player["y"] + player["h"])

        y_center    = (PLAY_TOP + PLAY_BOTTOM) * 0.5
        y_half      = play_range * 0.5
        y_norm      = (player_y - y_center) / y_half
        vy_norm     = player_vy / max(MAX_FALL_SPEED, MAX_RISE_SPEED)
        speed_norm  = (world_speed - SCROLL_SPEED) / 1000.0

        dist_to_ceiling = max(0.0, player_top - PLAY_TOP)   / play_range
        dist_to_floor   = max(0.0, PLAY_BOTTOM - player_bot) / play_range

        warning_present = 0.0
        warning_dy_norm = 0.0
        threats = []

        for o in objects:
            label = o["label"]
            if label == "warning":
                warning_present = 1.0
                warn_cy = float(o["y"] + o["h"] * 0.5)
                warning_dy_norm = (warn_cy - player_y) / HEIGHT
                continue
            if label not in ("zapper", "missile"):
                continue

            cx = float(o["x"] + o["w"] * 0.5)
            cy = float(o["y"] + o["h"] * 0.5)
            dx = cx - player_x

            if dx <= -float(PLAYER_W):
                continue

            dy = cy - player_y
            threats.append((dx, dy, o, label))

        threats.sort(key=lambda t: t[0])
        threats = threats[: self.k_threats]

        threat_vec: List[float] = []
        for i in range(self.k_threats):
            if i < len(threats):
                dx, dy, o, label = threats[i]

                obs_top = float(o["y"])
                obs_bot = float(o["y"] + o["h"])

                # Space between the obstacle's top and the ceiling.
                # 0.0 → obstacle is flush with ceiling → cannot pass above.
                gap_above = max(0.0, obs_top - PLAY_TOP)    / play_range

                # Space between the floor and the obstacle's bottom.
                # 0.0 → obstacle is flush with floor → cannot pass below.
                gap_below = max(0.0, PLAY_BOTTOM - obs_bot) / play_range

                threat_vec.extend([
                    dx / WIDTH,
                    dy / HEIGHT,
                    float(o["w"]) / WIDTH,
                    float(o["h"]) / HEIGHT,
                    1.0 if label == "zapper"  else 0.0,
                    1.0 if label == "missile" else 0.0,
                    gap_above,
                    gap_below,
                ])
            else:
                threat_vec.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        obs = np.array(
            [y_norm, vy_norm, speed_norm,
             dist_to_ceiling, dist_to_floor,
             *threat_vec,
             warning_present, warning_dy_norm],
            dtype=np.float32,
        )
        return obs

    def close(self):
        return