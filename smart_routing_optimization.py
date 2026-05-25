import pandas as pd

from lagrangian_solver import base_routing_score, add_lagrangian_columns, LagrangianMultipliers

# Load dataset
df = pd.read_csv("uav_iov_dataset.csv")

print("\n======================================")
print(" Lagrangian Smart Routing Optimization ")
print("======================================\n")

# Routing objective (same as before)
df["Routing_Score"] = df.apply(base_routing_score, axis=1)

# One pass with final-like lambdas (or run 3-5 dual iters locally)
lam = LagrangianMultipliers(delay=2.0, pdr=1.5, energy=1.5, signal=0.5)
df = add_lagrangian_columns(df, lam, objective_col="Routing_Score")

print("Routing with constraints (sample):\n")
print(
    df[
        [
            "Vehicle_ID",
            "UAV_ID",
            "Routing_Score",
            "Lagrangian_Score",
            "Violation_Delay",
            "Violation_Energy",
        ]
    ].head(8)
)

# Best route per vehicle: maximize Lagrangian_Score
best_routes = df.loc[df.groupby("Vehicle_ID")["Lagrangian_Score"].idxmax()]

print("\nBest constrained route per vehicle:\n")
for _, row in best_routes.iterrows():
    print(f"Vehicle : {row['Vehicle_ID']}")
    print(f"Best UAV : {row['UAV_ID']}")
    print(f"Routing Score : {round(row['Routing_Score'], 2)}")
    print(f"Lagrangian Score : {round(row['Lagrangian_Score'], 2)}")
    print(f"Delay : {row['Delay']} | PDR : {row['PDR']} | Energy : {row['Energy']}")
    print("--------------------------------------")

print("\n======================================")
print(" Network-Wide Routing Statistics ")
print("======================================\n")

avg_route = round(df["Routing_Score"].mean(), 2)
avg_lag = round(df["Lagrangian_Score"].mean(), 2)
best_uav = df.groupby("UAV_ID")["Lagrangian_Score"].mean().idxmax()

print(f"Average Routing Score     : {avg_route}")
print(f"Average Lagrangian Score  : {avg_lag}")
print(f"Best Overall UAV (Lagr.)  : {best_uav}")

print("\n======================================")
print(" Lagrangian Smart Routing Completed ")
print("======================================")
