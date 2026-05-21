import pandas as pd

# Load dataset
df = pd.read_csv("uav_iov_dataset.csv")

print("\nBEST UAV SELECTION RESULTS\n")

# Group by vehicle
vehicles = df['Vehicle_ID'].unique()

for vehicle in vehicles:

    # Get rows for current vehicle
    vehicle_data = df[df['Vehicle_ID'] == vehicle]

    # Find row with maximum signal strength
    best_row = vehicle_data.loc[
        vehicle_data['Signal_Strength'].idxmax()
    ]

    print(f"Vehicle: {vehicle}")
    print(f"Best UAV: {best_row['UAV_ID']}")
    print(f"Signal Strength: {best_row['Signal_Strength']}")
    print(f"Delay: {best_row['Delay']}")
    print(f"PDR: {best_row['PDR']}")
    print("-" * 40)