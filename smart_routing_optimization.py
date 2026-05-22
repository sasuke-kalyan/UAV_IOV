import pandas as pd

# Load dataset
df = pd.read_csv("uav_iov_dataset.csv")

print("\n======================================")
print(" AI-Based Smart Routing Optimization ")
print("======================================\n")

# Reliability Score
df['Reliability_Score'] = (
    (df['Signal_Strength'] * 50)
    + (df['PDR'] * 0.5)
    - (df['Delay'] * 0.4)
)

# Routing Score
df['Routing_Score'] = (

    (df['Signal_Strength'] * 40)

    + (df['PDR'] * 0.3)

    + (df['Energy'] * 0.2)

    - (df['Delay'] * 0.5)

)

print("Routing Analysis Started...\n")

# Best Route Per Vehicle
best_routes = df.loc[
    df.groupby('Vehicle_ID')['Routing_Score'].idxmax()
]

# Display Best Routes
for index, row in best_routes.iterrows():

    print(f"Vehicle : {row['Vehicle_ID']}")
    print(f"Best UAV : {row['UAV_ID']}")

    print(f"Signal Strength : {row['Signal_Strength']}")
    print(f"Delay : {row['Delay']}")
    print(f"PDR : {row['PDR']}")
    print(f"Energy : {row['Energy']}")

    print(f"Routing Score : {round(row['Routing_Score'],2)}")

    print("--------------------------------------")

print("\n======================================")
print(" Network-Wide Routing Statistics ")
print("======================================\n")

# Average Routing Score
avg_route = round(df['Routing_Score'].mean(), 2)

# Best UAV Overall
best_uav = (
    df.groupby('UAV_ID')['Routing_Score']
    .mean()
    .idxmax()
)

print(f"Average Routing Score : {avg_route}")
print(f"Best Overall UAV      : {best_uav}")

print("\n======================================")
print(" Smart Routing Optimization Completed ")
print("======================================")