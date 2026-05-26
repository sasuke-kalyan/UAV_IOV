"""
Shared UAV–vehicle link physics and reward (used by dataset, env, and RL).
"""

from __future__ import annotations

import numpy as np

import constraints as cons

COMMUNICATION_RANGE = 500.0
AREA_SIZE = 1000.0
ENERGY_DRAIN_MIN = 1
ENERGY_DRAIN_MAX = 3

# Shaping weights for constraint violations in RL reward
PENALTY_DELAY = 2.0
PENALTY_PDR = 1.5
PENALTY_ENERGY = 1.5
PENALTY_SIGNAL = 0.5


def compute_3d_distance(
    vehicle_x: float,
    vehicle_y: float,
    uav_x: float,
    uav_y: float,
    uav_z: float,
) -> float:
    return float(
        np.sqrt(
            (vehicle_x - uav_x) ** 2
            + (vehicle_y - uav_y) ** 2
            + uav_z**2
        )
    )


def compute_link_metrics(
    vehicle_x: float,
    vehicle_y: float,
    uav_x: float,
    uav_y: float,
    uav_z: float,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Distance, signal, delay, and PDR for one vehicle–UAV pair."""
    if rng is None:
        rng = np.random.default_rng()

    distance = compute_3d_distance(vehicle_x, vehicle_y, uav_x, uav_y, uav_z)

    if distance <= COMMUNICATION_RANGE:
        signal = max(0.1, 1.0 - (distance / COMMUNICATION_RANGE))
    else:
        signal = 0.05

    delay = min(100.0, distance / 10.0 + rng.integers(1, 10))
    pdr = max(50.0, signal * 100.0 - delay / 2.0)

    return {
        "Distance": round(distance, 2),
        "Signal_Strength": round(signal, 4),
        "Delay": round(delay, 2),
        "PDR": round(pdr, 2),
    }


def base_reward(signal: float, pdr: float, delay: float) -> float:
    return float(signal) * 100.0 + float(pdr) - float(delay)


def constraint_penalty(signal: float, pdr: float, delay: float, energy: float) -> float:
    return (
        PENALTY_DELAY * cons.violation_delay(delay)
        + PENALTY_PDR * cons.violation_pdr(pdr)
        + PENALTY_ENERGY * cons.violation_energy(energy)
        + PENALTY_SIGNAL * cons.violation_signal(signal)
    )


def shaped_reward(
    signal: float,
    pdr: float,
    delay: float,
    energy: float,
) -> float:
    """Objective minus QoS violation penalties (maximize)."""
    return base_reward(signal, pdr, delay) - constraint_penalty(
        signal, pdr, delay, energy
    )
