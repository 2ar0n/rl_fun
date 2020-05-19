import itertools
import typing

import numpy as np

import gym
import torch

from .common import returns_from


class ActorCritic(torch.nn.Module):
    def __init__(self, env):
        super(ActorCritic, self).__init__()

        self.env = env
        observation_size = env.observation_space.shape[0]
        if type(env.action_space) is gym.spaces.discrete.Discrete:
            control_size = int(env.action_space.n)
        else:
            control_size = env.action_space.shape[0]
            raise NotImplemented("Continuous action spaces not yet supported")

        self.lay1 = torch.nn.Linear(observation_size, 50)
        self.act1 = torch.nn.ReLU()
        self.lay2 = torch.nn.Linear(50, 30)
        self.act2 = torch.nn.ReLU()
        self.policy_head = torch.nn.Linear(30, control_size)
        self.policy_act = torch.nn.Softmax(dim=1)
        self.value_head = torch.nn.Linear(30, 1)
        self.value_act = torch.nn.Tanh()

        self.discount_factor = 0.99
        self.learning_rate = 0.01

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.optimizer.zero_grad()

    def forward(self, x: torch.tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        x = self.lay1(x)
        x = self.act1(x)
        x = self.lay2(x)
        x = self.act2(x)
        action = self.policy_act(self.policy_head(x))
        value = self.value_act(self.value_head(x))
        return action, value

    def train(self, episodes: int, batch_size: int, episode_steps: int = 1000):
        policy_losses = []
        value_losses = []
        total_episode_rewards = []

        for ep in range(episodes + 1):
            observation = self.env.reset()
            episode_rewards = []
            episode_log_probabilities = []
            episode_estimated_values = []
            for t in itertools.count(episode_steps):
                action, value = self.forward(torch.from_numpy(observation).float().unsqueeze(0))
                
                action_distribution = torch.distributions.Categorical(action)
                sampled_action = action_distribution.sample()
                observation, reward, done, _ = self.env.step(sampled_action.item())

                episode_log_probabilities.append(action_distribution.log_prob(sampled_action))
                episode_estimated_values.append(value)
                episode_rewards.append(reward)

                if done:
                    returns = returns_from(episode_rewards, self.discount_factor)
                    for ret, log_prob, value in zip(returns, episode_log_probabilities, episode_estimated_values):
                        advantage = ret
                        policy_losses.append(-advantage * log_prob)
                        value_losses.append((advantage - value) ** 2 / len(episode_estimated_values))
                    total_episode_rewards.append(sum(episode_rewards))
                    break

            if ep > 0 and ep % batch_size == 0:
                self.optimizer.zero_grad()
                batch_policy_loss = torch.cat(policy_losses).sum() / batch_size
                batch_policy_loss.backward(retain_graph=True)

                batch_value_loss = torch.cat(value_losses).sum() / batch_size
                batch_value_loss.backward()

                batch_loss = round(batch_policy_loss.item() + batch_value_loss.item(), 2)

                self.optimizer.step()

                mean_reward = round(torch.tensor(total_episode_rewards).mean().item(), 1)
                median_reward = round(torch.tensor(total_episode_rewards).median().item(), 1)
                percentage = round(ep / episodes * 100, 1)
                print(f"Progress: {percentage}%. Batch loss: {batch_loss}. Episode rewards (mean/median): {mean_reward}/{median_reward}.")
                total_episode_rewards = []
                policy_losses = []
                value_losses = []
