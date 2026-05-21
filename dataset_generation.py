import numpy as np
import pandas as pd

# Number of samples
num_samples = 100

# Store dataset
data = []

for t in range(num_samples):

    vehicle_id = f"V{np.random.randint(1,4)}"

    vehicle_x = np.random.randint(0, 1000)
    vehicle_y = np.random.randint(0, 1000)

    speed = np.random.randint(20, 80)

    uav_id = f"U{np.random.randint(1,3)}"

    uav_x = np.random.randint(0, 1000)
    uav_y = np.random.randint(0, 1000)

    signal_strength = round(np.random.uniform(0.5, 1.0), 2)

    delay = np.random.randint(5, 50)

    pdr = np.random.randint(80, 100)

    energy = np.random.randint(10, 50)

    data.append([
        t,
        vehicle_id,
        vehicle_x,
        vehicle_y,
        speed,
        uav_id,
        uav_x,
        uav_y,
        signal_strength,
        delay,
        pdr,
        energy
    ])

# Create dataframe
df = pd.DataFrame(data, columns=[
    "Timestamp",
    "Vehicle_ID",
    "Vehicle_X",
    "Vehicle_Y",
    "Speed",
    "UAV_ID",
    "UAV_X",
    "UAV_Y",
    "Signal_Strength",
    "Delay",
    "PDR",
    "Energy"
])

# Save dataset
df.to_csv("uav_iov_dataset.csv", index=False)

print(df.head())