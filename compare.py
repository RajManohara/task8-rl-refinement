import pandas as pd
import matplotlib.pyplot as plt

baseline = pd.read_csv("results/baseline_rewards.csv")
learning = pd.read_csv("results/learning_rewards.csv")
architecture = pd.read_csv("results/architecture_rewards.csv")
exploration = pd.read_csv("results/exploration_rewards.csv")

baseline["average"] = baseline["reward"].rolling(10).mean()
learning["average"] = learning["reward"].rolling(10).mean()
architecture["average"] = architecture["reward"].rolling(10).mean()
exploration["average"] = exploration["reward"].rolling(10).mean()

plt.figure(figsize=(10, 6))

plt.plot(
    baseline["episode"],
    baseline["average"],
    label="Baseline"
)

plt.plot(
    learning["episode"],
    learning["average"],
    label="Learning Parameters"
)

plt.plot(
    architecture["episode"],
    architecture["average"],
    label="256x256 Architecture"
)

plt.plot(
    exploration["episode"],
    exploration["average"],
    label="Exploration Decay"
)

plt.xlabel("Episode")
plt.ylabel("10 Episode Average Reward")
plt.title("DDPG Refinement Results")
plt.legend()
plt.grid()

plt.savefig("results/comparison.png")
plt.close()

print("Saved results/comparison.png")