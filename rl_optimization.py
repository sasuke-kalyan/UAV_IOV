import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("uav_iov_dataset.csv")

# Reward calculation
rewards = []

for index, row in df.iterrows():

    signal = row['Signal_Strength']
    delay = row['Delay']
    pdr = row['PDR']

    # Basic reward logic
    reward = (signal * 100) + pdr - delay

    rewards.append(reward)

# Add rewards to dataset
df['Reward'] = rewards

# Average reward per vehicle
avg_reward = df.groupby('Vehicle_ID')['Reward'].mean()

print("\nRL OPTIMIZATION REWARD ANALYSIS\n")
print(avg_reward)

# Plot graph
plt.figure(figsize=(7,5))

avg_reward.plot(kind='bar')

plt.title("Average RL Reward per Vehicle")
plt.xlabel("Vehicle ID")
plt.ylabel("Reward")

plt.show()