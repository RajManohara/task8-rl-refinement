import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import csv
import os

from ddpg import DDPG, ReplayBuffer

episodes = 100
batch_size = 64
actor_lr = 0.0003
critic_lr = 0.001
gamma = 0.99
tau = 0.005

start_noise = 0.2
min_noise = 0.02
noise_decay = 0.97

env = gym.make("Pendulum-v1")

state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

agent = DDPG(
    state_size,
    action_size,
    max_action,
    actor_lr,
    critic_lr,
    gamma,
    tau
)

buffer = ReplayBuffer()

rewards = []

for episode in range(episodes):

    state, info = env.reset()
    total_reward = 0
    done = False

    noise_amount = max(
        min_noise,
        start_noise * (noise_decay ** episode)
    )

    while not done:

        action = agent.get_action(state)

        noise = np.random.normal(
            0,
            noise_amount,
            size=action_size
        )

        action = action + noise

        action = np.clip(
            action,
            env.action_space.low,
            env.action_space.high
        )

        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        buffer.add(
            state,
            action,
            reward,
            next_state,
            done
        )

        agent.train(buffer, batch_size)

        state = next_state
        total_reward += reward

    rewards.append(total_reward)

    print(
        "Episode:",
        episode + 1,
        "Reward:",
        round(total_reward, 2),
        "Noise:",
        round(noise_amount, 3)
    )

env.close()

os.makedirs("results", exist_ok=True)

with open("results/exploration_rewards.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["episode", "reward"])

    for i in range(len(rewards)):
        writer.writerow([i + 1, rewards[i]])

average_reward = np.mean(rewards)
last_20_average = np.mean(rewards[-20:])
best_reward = np.max(rewards)

print()
print("Average reward:", round(average_reward, 2))
print("Last 20 average:", round(last_20_average, 2))
print("Best reward:", round(best_reward, 2))

moving_average = []

for i in range(len(rewards)):
    start = max(0, i - 9)
    average = np.mean(rewards[start:i + 1])
    moving_average.append(average)

plt.plot(rewards, label="Episode Reward")
plt.plot(moving_average, label="10 Episode Average")

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Exploration DDPG")
plt.legend()

plt.savefig("results/exploration.png")
plt.close()

torch.save(
    agent.actor.state_dict(),
    "results/exploration_actor.pth"
)

torch.save(
    agent.critic.state_dict(),
    "results/exploration_critic.pth"
)