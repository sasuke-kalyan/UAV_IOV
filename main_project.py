import pandas as pd

# Load Dataset
df = pd.read_csv("uav_iov_dataset.csv")

print("\n==========================================")
print("   UAV-IoV Communication System Analysis")
print("==========================================\n")

# Dataset Preview
print("Dataset Preview:\n")
print(df.head())

# Basic Statistics
print("\n==========================================")
print(" Basic Network Statistics ")
print("==========================================\n")

# Total Vehicles
total_vehicles = df['Vehicle_ID'].nunique()

# Total UAVs
total_uavs = df['UAV_ID'].nunique()

print("Total Vehicles :", total_vehicles)
print("Total UAVs     :", total_uavs)

# Average Metrics
print("\n==========================================")
print(" Average Communication Metrics ")
print("==========================================\n")

avg_distance = round(df['Distance'].mean(), 2)
avg_signal = round(df['Signal_Strength'].mean(), 2)
avg_delay = round(df['Delay'].mean(), 2)
avg_pdr = round(df['PDR'].mean(), 2)
avg_energy = round(df['Energy'].mean(), 2)

print("Average Distance          :", avg_distance)
print("Average Signal Strength   :", avg_signal)
print("Average Delay             :", avg_delay)
print("Average PDR               :", avg_pdr)
print("Average Energy            :", avg_energy)

# Reliability Score Calculation
print("\n==========================================")
print(" Reliability Analysis ")
print("==========================================\n")

df['Reliability_Score'] = (
    (df['Signal_Strength'] * 50)
    + (df['PDR'] * 0.5)
    - (df['Delay'] * 0.4)
)

avg_reliability = round(df['Reliability_Score'].mean(), 2)

print("Average Reliability Score :", avg_reliability)

# Reinforcement Learning Reward
print("\n==========================================")
print(" RL Optimization Analysis ")
print("==========================================\n")

df['Reward'] = (
    (df['Signal_Strength'] * 100)
    + df['PDR']
    - df['Delay']
)

avg_reward = round(df['Reward'].mean(), 2)

print("Average RL Reward         :", avg_reward)

# Best Communication Record
print("\n==========================================")
print(" Best Communication Record ")
print("==========================================\n")

best_record = df.loc[df['Reliability_Score'].idxmax()]

print(best_record)

# Communication Quality Classification
print("\n==========================================")
print(" Communication Quality ")
print("==========================================\n")

def classify_signal(signal):

    if signal >= 0.7:
        return "Strong"

    elif signal >= 0.4:
        return "Medium"

    else:
        return "Weak"

df['Communication_Quality'] = df['Signal_Strength'].apply(classify_signal)

quality_counts = df['Communication_Quality'].value_counts()

print(quality_counts)

# Final Summary
print("\n==========================================")
print(" Final System Summary ")
print("==========================================\n")

print("✔ Dynamic Vehicle Mobility Implemented")
print("✔ 3D UAV Communication Implemented")
print("✔ Distance-Based Signal Modeling Implemented")
print("✔ Communication Coverage Range Implemented")
print("✔ Dynamic Delay & PDR Implemented")
print("✔ Reliability Evaluation Completed")
print("✔ RL-Based Optimization (heuristic reward on dataset)")
print("✔ Neural PPO: run  python train_ppo.py  then  python evaluate_ppo.py")
print("✔ GAT routing: run  python train_gat.py  then  python gat_routing.py")

print("\n==========================================")
print(" UAV-IoV System Analysis Completed ")
print("==========================================")