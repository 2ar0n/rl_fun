import itertools
import typing

import numpy as np

import gym
import torch

from .common import returns_from


class PPO(torch.nn.Module):
    def __init__(self, env):
        super(PPO, self).__init__()

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
        self.epsilon = 0.2
        self.max_batch_replays = 4

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
        number_batches = episodes // batch_size
        for batch_idx in range(number_batches):
            episodes_batch = self.generate_episodes(batch_size, episode_steps)

            for replay_idx in range(self.max_batch_replays):
                batch_policy_loss = []
                batch_value_loss = []
                batch_rewards = []
                for episode in episodes_batch:
                    latest_policy_logprob = []
                    estimated_values = []
                    observations, actions, action_probs, rewards, estimated_values = tuple(episode)
                    batch_rewards.append(sum(rewards))
                    returns = returns_from(rewards, self.discount_factor)
                    for observation, old_policy_action, old_policy_action_prob, return_value, old_policy_value in zip(observations, actions, action_probs, returns, estimated_values):
                        action, value = self.forward(torch.from_numpy(observation).float().unsqueeze(0))
                        current_policy_action_prob = torch.distributions.Categorical(action).probs.squeeze(0)[old_policy_action]
                        old_policy_action_prob = old_policy_action_prob.squeeze(0)[old_policy_action].detach()

                        advantage = return_value - old_policy_value.detach()
                        if advantage >= 0:
                            batch_policy_loss.append(- min(current_policy_action_prob/old_policy_action_prob, 1 + self.epsilon) * advantage)
                        else:
                            batch_policy_loss.append(- max(current_policy_action_prob/old_policy_action_prob, 1 - self.epsilon) * advantage)
                        batch_value_loss.append((return_value - value) ** 2 / len(returns))

                self.optimizer.zero_grad()
                batch_policy_loss = torch.cat(batch_policy_loss).sum() / batch_size
                batch_policy_loss.backward(retain_graph=True)

                batch_value_loss = torch.cat(batch_value_loss).sum() / batch_size
                batch_value_loss.backward()

                batch_loss = round(batch_policy_loss.item() + batch_value_loss.item(), 2)

                self.optimizer.step()

                mean_reward = round(torch.tensor(batch_rewards).mean().item(), 1)
                median_reward = round(torch.tensor(batch_rewards).median().item(), 1)
                percentage = round((replay_idx + batch_idx * self.max_batch_replays) / (number_batches * self.max_batch_replays) * 100, 1)
                print(f"Progress: {percentage}%. Batch loss: {batch_loss}. Episode rewards (mean/median): {mean_reward}/{median_reward}.")

    def generate_episodes(self, batch_size: int, episode_steps: int = 1000):
        episodes = []
        with torch.no_grad():
            for ep in range(batch_size + 1):
                observation = self.env.reset()
                observations = []
                rewards = []
                actions = []
                action_probs = []
                estimated_values = []
                for t in itertools.count(episode_steps):
                    action, value = self.forward(torch.from_numpy(observation).float().unsqueeze(0))
                    action_distribution = torch.distributions.Categorical(action)
                    sampled_action = action_distribution.sample().item()

                    new_observation, reward, done, _ = self.env.step(sampled_action)

                    observations.append(observation)
                    actions.append(sampled_action)
                    action_probs.append(action_distribution.probs)
                    estimated_values.append(value)
                    rewards.append(reward)
                    observation = new_observation
                    if done:
                        break
                episodes.append([observations, actions, action_probs, rewards, estimated_values])
        return episodes