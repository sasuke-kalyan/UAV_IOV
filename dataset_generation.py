import numpy as np
import pandas as pd

# Number of samples
num_samples = 100

# Store dataset
data = []

# Initial vehicle positions
vehicle_positions = {
    "V1": [100, 200],
    "V2": [400, 500],
    "V3": [700, 300]
}

for t in range(num_samples):

    # Vehicle Information
    vehicle_id = f"V{np.random.randint(1,4)}"

    # Dynamic Vehicle Movement
    vehicle_x, vehicle_y = vehicle_positions[vehicle_id]

    vehicle_x += np.random.randint(-20, 20)
    vehicle_y += np.random.randint(-20, 20)

    # Boundary Control
    vehicle_x = max(0, min(vehicle_x, 1000))
    vehicle_y = max(0, min(vehicle_y, 1000))

    # Update Positions
    vehicle_positions[vehicle_id] = [vehicle_x, vehicle_y]

    # Vehicle Speed
    speed = np.random.randint(20, 80)

    # UAV Information
    uav_id = f"U{np.random.randint(1,3)}"

    uav_x = np.random.randint(0, 1000)
    uav_y = np.random.randint(0, 1000)

    # UAV Altitude (Z-axis)
    uav_z = np.random.randint(80, 200)

    # Communication Range
    communication_range = 500

    # 3D Distance Calculation
    distance = (
        (vehicle_x - uav_x) ** 2 +
        (vehicle_y - uav_y) ** 2 +
        (uav_z) ** 2
    ) ** 0.5

    # Signal Strength Based on Distance & Coverage
    if distance <= communication_range:

        signal_strength = round(
            max(0.1, 1 - (distance / communication_range)),
            2
        )

    else:
        # Out of communication range
        signal_strength = 0.05

    # Dynamic Delay Calculation
    delay = round(
        min(100, distance / 10 + np.random.randint(1, 10)),
        2
    )

    # Dynamic Packet Delivery Ratio (PDR)
    pdr = round(
        max(50, signal_strength * 100 - delay / 2),
        2
    )

    # UAV Energy
    energy = np.random.randint(10, 50)

    # Store row
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
        round(distance, 2),
        signal_strength,
        delay,
        pdr,
        energy
    ])

# Create DataFrame
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
    "Energy"
])

# Save Dataset
df.to_csv("uav_iov_dataset.csv", index=False)

# Preview Dataset
print("\nUAV-IoV Dataset Preview:\n")
print(df.head())

print("\nDataset Generated Successfully!")