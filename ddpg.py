import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


class Actor(nn.Module):
    def __init__(self, state_size, action_size, max_action):
        super().__init__()

        self.layer1 = nn.Linear(state_size, 400)
        self.layer2 = nn.Linear(400, 300)
        self.output = nn.Linear(300, action_size)

        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))

        action = torch.tanh(self.output(x))
        action = action * self.max_action

        return action


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()

        self.layer1 = nn.Linear(state_size, 400)
        self.layer2 = nn.Linear(400 + action_size, 300)
        self.output = nn.Linear(300, 1)

    def forward(self, state, action):
        x = F.relu(self.layer1(state))

        x = torch.cat((x, action), dim=1)

        x = F.relu(self.layer2(x))
        value = self.output(x)

        return value


class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = []
        self.max_size = max_size

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)

        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)

        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for experience in batch:
            state, action, reward, next_state, done = experience

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def size(self):
        return len(self.buffer)


class DDPG:
    def __init__(
        self,
        state_size,
        action_size,
        max_action,
        actor_lr=0.0001,
        critic_lr=0.001,
        gamma=0.99,
        tau=0.001
    ):
        self.actor = Actor(state_size, action_size, max_action)
        self.actor_target = Actor(state_size, action_size, max_action)

        self.critic = Critic(state_size, action_size)
        self.critic_target = Critic(state_size, action_size)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=actor_lr
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=critic_lr
        )

        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action = self.actor(state)

        return action.numpy()[0]

    def train(self, replay_buffer, batch_size=64):
        if replay_buffer.size() < batch_size:
            return

        states, actions, rewards, next_states, dones = replay_buffer.sample(
            batch_size
        )

        next_actions = self.actor_target(next_states)

        target_q = self.critic_target(next_states, next_actions)

        target = rewards + self.gamma * target_q * (1 - dones)

        current_q = self.critic(states, actions)

        critic_loss = F.mse_loss(current_q, target.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_actions = self.actor(states)

        actor_loss = -self.critic(states, actor_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for target_param, param in zip(
            self.actor_target.parameters(),
            self.actor.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data
                + (1 - self.tau) * target_param.data
            )

        for target_param, param in zip(
            self.critic_target.parameters(),
            self.critic.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data
                + (1 - self.tau) * target_param.data
            )


if __name__ == "__main__":

    state_size = 3
    action_size = 1
    max_action = 2.0

    actor = Actor(state_size, action_size, max_action)
    critic = Critic(state_size, action_size)

    state = torch.randn(1, state_size)

    action = actor(state)
    value = critic(state, action)

    print("State:", state)
    print("Action:", action)
    print("Critic value:", value)

    replay_buffer = ReplayBuffer()

    replay_buffer.add(
        [0.1, 0.2, 0.3],
        [0.5],
        1.0,
        [0.2, 0.3, 0.4],
        False
    )

    replay_buffer.add(
        [0.4, 0.5, 0.6],
        [-0.2],
        0.5,
        [0.5, 0.6, 0.7],
        False
    )

    print("Replay buffer size:", replay_buffer.size())

    batch = replay_buffer.sample(2)

    states = batch[0]

    print("Sampled states:")
    print(states)