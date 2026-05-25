import pandas as pd
import matplotlib.pyplot as plt

from lagrangian_solver import base_reward, run_primal_dual

# Load dataset
df = pd.read_csv("uav_iov_dataset.csv")

# Original reward (for comparison)
df["Reward"] = df.apply(base_reward, axis=1)

print("\n======================================")
print(" Lagrangian RL Reward Optimization ")
print("======================================\n")

# Primal-dual Lagrangian on full dataset
df_lag, final_lambda = run_primal_dual(df, objective_col="Reward")

print("\nFinal multipliers:")
print(final_lambda)

print("\nSample rows (Reward vs Lagrangian_Score):\n")
print(
    df_lag[
        [
            "Vehicle_ID",
            "UAV_ID",
            "Delay",
            "PDR",
            "Energy",
            "Reward",
            "Lagrangian_Score",
            "Violation_Delay",
            "Violation_PDR",
            "Violation_Energy",
        ]
    ].head(10)
)

# Compare average reward per vehicle: unconstrained vs Lagrangian
avg_reward = df_lag.groupby("Vehicle_ID")["Reward"].mean()
avg_lag = df_lag.groupby("Vehicle_ID")["Lagrangian_Score"].mean()

print("\nAverage unconstrained Reward per vehicle:\n")
print(avg_reward)

print("\nAverage Lagrangian_Score per vehicle:\n")
print(avg_lag)

# Plot (Lagrangian score per vehicle)
plt.figure(figsize=(7, 5))
avg_lag.plot(kind="bar")
plt.title("Average Lagrangian Score per Vehicle")
plt.xlabel("Vehicle ID")
plt.ylabel("Lagrangian Score")
plt.tight_layout()
plt.show()

print("\nLagrangian RL optimization completed.\n")