import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("uav_iov_dataset.csv")

# Reliability Score
df['Reliability_Score'] = (
    (df['Signal_Strength'] * 50)
    + (df['PDR'] * 0.5)
    - (df['Delay'] * 0.4)
)

# -------------------------------
# SIGNAL STRENGTH GRAPH
# -------------------------------

plt.figure(figsize=(8,5))

plt.plot(df['Timestamp'],
         df['Signal_Strength'])

plt.title("Signal Strength Over Time")
plt.xlabel("Timestamp")
plt.ylabel("Signal Strength")

plt.grid(True)

plt.show()

# -------------------------------
# DELAY GRAPH
# -------------------------------

plt.figure(figsize=(8,5))

plt.plot(df['Timestamp'],
         df['Delay'])

plt.title("Communication Delay Over Time")
plt.xlabel("Timestamp")
plt.ylabel("Delay")

plt.grid(True)

plt.show()

# -------------------------------
# PDR GRAPH
# -------------------------------

plt.figure(figsize=(8,5))

plt.plot(df['Timestamp'],
         df['PDR'])

plt.title("Packet Delivery Ratio Over Time")
plt.xlabel("Timestamp")
plt.ylabel("PDR")

plt.grid(True)

plt.show()

# -------------------------------
# RELIABILITY SCORE GRAPH
# -------------------------------

plt.figure(figsize=(8,5))

plt.plot(df['Timestamp'],
         df['Reliability_Score'])

plt.title("Reliability Score Over Time")
plt.xlabel("Timestamp")
plt.ylabel("Reliability Score")

plt.grid(True)

plt.show()