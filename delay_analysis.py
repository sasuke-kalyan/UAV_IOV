import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("uav_iov_dataset.csv")

# Average delay per vehicle
avg_delay = df.groupby('Vehicle_ID')['Delay'].mean()

print("\nAVERAGE DELAY ANALYSIS\n")
print(avg_delay)

# Plot graph
plt.figure(figsize=(7,5))

avg_delay.plot(kind='bar')

plt.title("Average Communication Delay per Vehicle")
plt.xlabel("Vehicle ID")
plt.ylabel("Average Delay")

plt.show()