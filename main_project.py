import pandas as pd

# Load dataset
df = pd.read_csv("uav_iov_dataset.csv")

print("\n===================================")
print(" UAV-IoV Communication System ")
print("===================================\n")

# Dataset Preview
print("Dataset Preview:\n")
print(df.head())

# Total vehicles and UAVs
vehicles = df['Vehicle_ID'].nunique()
uavs = df['UAV_ID'].nunique()

print("\nTotal Vehicles:", vehicles)
print("Total UAVs:", uavs)

# Average Metrics
print("\nAverage Communication Metrics:\n")

print("Average Signal Strength:",
      round(df['Signal_Strength'].mean(), 2))

print("Average Delay:",
      round(df['Delay'].mean(), 2))

print("Average PDR:",
      round(df['PDR'].mean(), 2))

# Reliability Score
df['Reliability_Score'] = (
    (df['Signal_Strength'] * 50)
    + (df['PDR'] * 0.4)
    - (df['Delay'] * 0.5)
)

print("\nAverage Reliability Score:",
      round(df['Reliability_Score'].mean(), 2))

# RL Reward
df['Reward'] = (
    (df['Signal_Strength'] * 100)
    + df['PDR']
    - df['Delay']
)

print("Average RL Reward:",
      round(df['Reward'].mean(), 2))

print("\n===================================")
print(" System Analysis Completed ")
print("===================================")