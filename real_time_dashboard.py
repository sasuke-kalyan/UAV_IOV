import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("uav_iov_dataset.csv")

# Create figure
plt.figure(figsize=(8,8))

# Run only first 20 timestamps
for i in range(20):

    # Clear previous frame
    plt.clf()

    # Current row
    row = df.iloc[i]

    # Vehicle Position
    vx = row['Vehicle_X']
    vy = row['Vehicle_Y']

    # UAV Position
    ux = row['UAV_X']
    uy = row['UAV_Y']

    # Signal Strength
    signal = row['Signal_Strength']

    # Communication Quality
    if signal >= 0.7:
        quality = "Strong"

    elif signal >= 0.4:
        quality = "Medium"

    else:
        quality = "Weak"

    # Plot Vehicle
    plt.scatter(
        vx,
        vy,
        s=120,
        label='Vehicle'
    )

    # Plot UAV
    plt.scatter(
        ux,
        uy,
        s=200,
        marker='^',
        label='UAV'
    )

    # Communication Link
    plt.plot(
        [vx, ux],
        [vy, uy]
    )

    # Vehicle Label
    plt.text(vx + 10, vy + 10,
             row['Vehicle_ID'])

    # UAV Label
    plt.text(ux + 10, uy + 10,
             row['UAV_ID'])

    # Title
    plt.title(
        f"Real-Time UAV-IoV Communication\n"
        f"Signal Strength: {signal} | "
        f"Quality: {quality}"
    )

    # Axis Limits
    plt.xlim(0, 1000)
    plt.ylim(0, 1000)

    # Grid
    plt.grid(True)

    # Legend
    plt.legend()

    # Pause for animation
    plt.pause(0.5)

# Automatically close after completion
plt.close()

# Final show
plt.show()