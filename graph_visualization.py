import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("uav_iov_dataset.csv")

# Create graph
G = nx.Graph()

# Take first 20 rows
sample_data = df.head(20)

# Add nodes and weighted edges
for index, row in sample_data.iterrows():

    vehicle = row['Vehicle_ID']
    uav = row['UAV_ID']
    signal = row['Signal_Strength']

    # Add nodes
    G.add_node(vehicle, type='vehicle')
    G.add_node(uav, type='uav')

    # Add weighted communication edge
    G.add_edge(vehicle, uav, weight=signal)

# Create graph layout
pos = nx.spring_layout(G)

# Draw graph
plt.figure(figsize=(10,7))

nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=3000,
    font_size=10
)

# Edge labels
edge_labels = nx.get_edge_attributes(G, 'weight')

nx.draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=edge_labels
)

plt.title("Weighted UAV-IoV Communication Graph")

plt.show()