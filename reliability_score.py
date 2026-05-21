import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("uav_iov_dataset.csv")

# Reliability score calculation
df['Reliability_Score'] = (
    (df['Signal_Strength'] * 50) +
    (df['PDR'] * 0.4) -
    (df['Delay'] * 0.5)
)

# Average reliability per vehicle
avg_reliability = df.groupby('Vehicle_ID')['Reliability_Score'].mean()

print("\nRELIABILITY SCORE ANALYSIS\n")
print(avg_reliability)

# Plot graph
plt.figure(figsize=(7,5))

avg_reliability.plot(kind='bar')

plt.title("Average Communication Reliability Score")
plt.xlabel("Vehicle ID")
plt.ylabel("Reliability Score")

plt.show()