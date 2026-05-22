import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("uav_iov_dataset.csv")

print("\n======================================")
print(" Emergency Vehicle Priority System ")
print("======================================\n")

# Randomly assign emergency vehicles
vehicle_types = []

for i in range(len(df)):

    rand = np.random.randint(1, 10)

    if rand == 1:
        vehicle_types.append("Ambulance")

    elif rand == 2:
        vehicle_types.append("Police")

    else:
        vehicle_types.append("Normal")

# Add vehicle type column
df['Vehicle_Type'] = vehicle_types

# Apply priority communication
priority_scores = []

for index, row in df.iterrows():

    base_score = (
        (row['Signal_Strength'] * 50)
        + (row['PDR'] * 0.5)
        - (row['Delay'] * 0.4)
    )

    # Emergency priority boost
    if row['Vehicle_Type'] in ["Ambulance", "Police"]:

        priority_score = base_score + 30

    else:
        priority_score = base_score

    priority_scores.append(priority_score)

# Add priority score
df['Priority_Score'] = priority_scores

print("Emergency Vehicle Records:\n")

emergency_records = df[
    df['Vehicle_Type'] != "Normal"
]

print(emergency_records[
    [
        'Vehicle_ID',
        'Vehicle_Type',
        'UAV_ID',
        'Signal_Strength',
        'Delay',
        'Priority_Score'
    ]
].head())

print("\n======================================")
print(" Highest Priority Communication ")
print("======================================\n")

best_priority = df.loc[
    df['Priority_Score'].idxmax()
]

print(best_priority[
    [
        'Vehicle_ID',
        'Vehicle_Type',
        'UAV_ID',
        'Signal_Strength',
        'Delay',
        'Priority_Score'
    ]
])

print("\n======================================")
print(" Emergency Priority System Completed ")
print("======================================")