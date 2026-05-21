import matplotlib.pyplot as plt

# Sample comparison values

metrics = ['Delay', 'PDR', 'Reliability', 'RL Reward']

before_optimization = [40, 75, 50, 90]

after_optimization = [20, 92, 80, 150]

# Create graph
x = range(len(metrics))

plt.figure(figsize=(8,5))

plt.bar(x, before_optimization, width=0.4, label='Before Optimization')

plt.bar(
    [i + 0.4 for i in x],
    after_optimization,
    width=0.4,
    label='After Optimization'
)

plt.xticks([i + 0.2 for i in x], metrics)

plt.title("Final Performance Comparison")
plt.ylabel("Performance Values")

plt.legend()

plt.show()