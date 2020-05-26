import itertools
import typing

import numpy as np

import gym
import torch

from .common import returns_from


class Reinforce(torch.nn.Module):
    def __init__(self, env):
        super(Reinforce, self).__init__()

        self.env = env
        observation_size = env.observation_space.shape[0]

        if type(env.action_space) is not gym.spaces.discrete.Discrete:
            raise NotImplemented("Continuous action spaces yet supported")
        control_size = int(env.action_space.n)

        self.lay1 = torch.nn.Linear(observation_size, 50)
        self.act1 = torch.nn.ReLU()
        self.lay2 = torch.nn.Linear(50, 30)
        self.act2 = torch.nn.ReLU()
        self.lay3 = torch.nn.Linear(30, control_size)
        self.act3 = torch.nn.Softmax(dim=1)

        self.discount_factor = 0.99
        self.learning_rate = 0.01

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.optimizer.zero_grad()

    def forward(self, x: torch.tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        x = self.lay1(x)
        x = self.act1(x)
        x = self.lay2(x)
        x = self.act2(x)
        x = self.lay3(x)
        x = self.act3(x)
        return x

    def act(self, x: np.ndarray):
        action, _ = self.forward(torch.from_numpy(x).float().unsqueeze(0))
        return action.argmax().item()

    def train(self, episodes: int, batch_size: int, episode_steps: int = 1000):
        policy_losses = []
        total_episode_rewards = []

        for ep in range(episodes + 1):
            observation = self.env.reset()
            episode_rewards = []
            episode_log_probabilities = []
            for t in itertools.count(episode_steps):
                action = self.forward(torch.from_numpy(observation).float().unsqueeze(0))
                
                action_distribution = torch.distributions.Categorical(action)
                sampled_action = action_distribution.sample()
                observation, reward, done, _ = self.env.step(sampled_action.item())

                episode_log_probabilities.append(action_distribution.log_prob(sampled_action))
                episode_rewards.append(reward)

                if done:
                    returns = returns_from(episode_rewards, self.discount_factor)
                    for ret, log_prob in zip(returns, episode_log_probabilities):
                        advantage = ret
                        policy_losses.append(-advantage * log_prob)
                    total_episode_rewards.append(sum(episode_rewards))
                    break

            if ep > 0 and ep % batch_size == 0:
                self.optimizer.zero_grad()
                batch_policy_loss = torch.cat(policy_losses).sum() / batch_size
                batch_policy_loss.backward(retain_graph=True)

                batch_loss = round(batch_policy_loss.item(), 2)

                self.optimizer.step()

                mean_reward = round(torch.tensor(total_episode_rewards).mean().item(), 1)
                median_reward = round(torch.tensor(total_episode_rewards).median().item(), 1)
                percentage = round(ep / episodes * 100, 1)
                print(f"Progress: {percentage}%. Batch loss: {batch_loss}. Episode rewards (mean/median): {mean_reward}/{median_reward}.")
                total_episode_rewards = []
                policy_losses = []

