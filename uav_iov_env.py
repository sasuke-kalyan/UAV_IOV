"""
Gymnasium environment: vehicle chooses among fixed UAVs each timestep.
Observation includes both UAV link previews; reward uses communication_model.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from communication_model import (
    AREA_SIZE,
    ENERGY_DRAIN_MAX,
    ENERGY_DRAIN_MIN,
    compute_link_metrics,
    shaped_reward,
)


class UAVIoVEnv(gym.Env):
    """UAV selection for a moving ground vehicle."""

    metadata = {"render_modes": []}

    NUM_UAVS = 2
    MAX_EPISODE_STEPS = 100

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(self.NUM_UAVS)
        # [vx, vy] normalized + per-UAV (rel_x, rel_y, signal, energy_norm)
        obs_dim = 2 + self.NUM_UAVS * 4
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self._rng = np.random.default_rng()
        self._step_count = 0
        self._vehicle_xy = np.zeros(2, dtype=np.float64)
        self._uav_positions = np.zeros((self.NUM_UAVS, 3), dtype=np.float64)
        self._uav_energies = np.zeros(self.NUM_UAVS, dtype=np.float64)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count = 0
        self._vehicle_xy = self._rng.uniform(0, AREA_SIZE, size=2)

        for i in range(self.NUM_UAVS):
            self._uav_positions[i] = [
                self._rng.uniform(0, AREA_SIZE),
                self._rng.uniform(0, AREA_SIZE),
                self._rng.integers(80, 201),
            ]
            self._uav_energies[i] = self._rng.integers(30, 51)

        return self._get_obs(), {}

    def step(self, action: int):
        action = int(action)
        if action < 0 or action >= self.NUM_UAVS:
            raise ValueError(f"Invalid action {action}")

        ux, uy, uz = self._uav_positions[action]
        metrics = compute_link_metrics(
            self._vehicle_xy[0],
            self._vehicle_xy[1],
            ux,
            uy,
            uz,
            rng=self._rng,
        )
        energy = float(self._uav_energies[action])
        reward = shaped_reward(
            metrics["Signal_Strength"],
            metrics["PDR"],
            metrics["Delay"],
            energy,
        )

        drain = self._rng.integers(ENERGY_DRAIN_MIN, ENERGY_DRAIN_MAX + 1)
        self._uav_energies[action] = max(0.0, energy - drain)

        self._move_vehicle()

        self._step_count += 1
        terminated = self._step_count >= self.MAX_EPISODE_STEPS
        truncated = False

        info = {
            "uav_id": f"U{action + 1}",
            "metrics": metrics,
            "energy": float(self._uav_energies[action]),
            "raw_reward": reward,
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def _move_vehicle(self):
        self._vehicle_xy[0] += self._rng.integers(-20, 21)
        self._vehicle_xy[1] += self._rng.integers(-20, 21)
        self._vehicle_xy = np.clip(self._vehicle_xy, 0.0, AREA_SIZE)

    def _get_obs(self) -> np.ndarray:
        features: list[float] = [
            self._vehicle_xy[0] / AREA_SIZE,
            self._vehicle_xy[1] / AREA_SIZE,
        ]

        for i in range(self.NUM_UAVS):
            ux, uy, uz = self._uav_positions[i]
            m = compute_link_metrics(
                self._vehicle_xy[0],
                self._vehicle_xy[1],
                ux,
                uy,
                uz,
                rng=self._rng,
            )
            rel_x = (ux - self._vehicle_xy[0]) / AREA_SIZE
            rel_y = (uy - self._vehicle_xy[1]) / AREA_SIZE
            features.extend(
                [
                    np.clip(rel_x, -1.0, 1.0) * 0.5 + 0.5,
                    np.clip(rel_y, -1.0, 1.0) * 0.5 + 0.5,
                    m["Signal_Strength"],
                    self._uav_energies[i] / 50.0,
                ]
            )

        return np.array(features, dtype=np.float32)
