import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("uav_iov_dataset.csv")

# Average PDR per vehicle
avg_pdr = df.groupby('Vehicle_ID')['PDR'].mean()

print("\nAVERAGE PDR ANALYSIS\n")
print(avg_pdr)

# Plot graph
plt.figure(figsize=(7,5))

avg_pdr.plot(kind='bar')

plt.title("Average Packet Delivery Ratio per Vehicle")
plt.xlabel("Vehicle ID")
plt.ylabel("Average PDR")

plt.show()