import numpy as np
import pandas as pd

from communication_model import AREA_SIZE, compute_link_metrics
from network_config import NUM_SAMPLES, NUM_UAVS, NUM_VEHICLES

_rng = np.random.default_rng(42)
_area = int(AREA_SIZE)

vehicle_positions = {
    f"V{i}": _rng.integers(50, _area - 50, size=2).tolist()
    for i in range(1, NUM_VEHICLES + 1)
}

uav_positions = {
    f"U{i}": [
        int(_rng.integers(0, _area)),
        int(_rng.integers(0, _area)),
        int(_rng.integers(80, 200)),
    ]
    for i in range(1, NUM_UAVS + 1)
}

uav_energy = {f"U{i}": int(_rng.integers(30, 51)) for i in range(1, NUM_UAVS + 1)}

data = []

for t in range(NUM_SAMPLES):
    for vehicle_id, (vehicle_x, vehicle_y) in list(vehicle_positions.items()):
        vehicle_x += np.random.randint(-20, 20)
        vehicle_y += np.random.randint(-20, 20)
        vehicle_x = max(0, min(vehicle_x, _area))
        vehicle_y = max(0, min(vehicle_y, _area))
        vehicle_positions[vehicle_id] = [vehicle_x, vehicle_y]

    for uav_id, (uav_x, uav_y, uav_z) in list(uav_positions.items()):
        uav_x += np.random.randint(-10, 11)
        uav_y += np.random.randint(-10, 11)
        uav_x = max(0, min(uav_x, _area))
        uav_y = max(0, min(uav_y, _area))
        uav_positions[uav_id] = [uav_x, uav_y, uav_z]
        uav_energy[uav_id] = max(10, uav_energy[uav_id] - np.random.randint(0, 2))

    for vehicle_id in vehicle_positions:
        vehicle_x, vehicle_y = vehicle_positions[vehicle_id]
        speed = np.random.randint(20, 80)

        for uav_id in uav_positions:
            uav_x, uav_y, uav_z = uav_positions[uav_id]
            metrics = compute_link_metrics(
                vehicle_x, vehicle_y, uav_x, uav_y, uav_z
            )

            data.append([
                t,
                vehicle_id,
                vehicle_x,
                vehicle_y,
                speed,
                uav_id,
                uav_x,
                uav_y,
                uav_z,
                round(metrics["Distance"], 2),
                metrics["Signal_Strength"],
                metrics["Delay"],
                metrics["PDR"],
                uav_energy[uav_id],
            ])

df = pd.DataFrame(data, columns=[
    "Timestamp",
    "Vehicle_ID",
    "Vehicle_X",
    "Vehicle_Y",
    "Speed",
    "UAV_ID",
    "UAV_X",
    "UAV_Y",
    "UAV_Z",
    "Distance",
    "Signal_Strength",
    "Delay",
    "PDR",
    "Energy",
])

df.to_csv("uav_iov_dataset.csv", index=False)

print("\nUAV-IoV Dataset Preview:\n")
print(df.head())
print(f"\nRows: {len(df)} ({NUM_VEHICLES} vehicles x {NUM_UAVS} UAVs x {NUM_SAMPLES} timesteps)")
print(f"Vehicles: {df['Vehicle_ID'].nunique()} / {NUM_VEHICLES}")
print(f"UAVs: {df['UAV_ID'].nunique()} / {NUM_UAVS}")
print("\nDataset Generated Successfully!")
