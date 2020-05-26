import itertools
import typing

import numpy as np

import gym
import torch

from .common import returns_from


class ActorCriticContinuous(torch.nn.Module):
    def __init__(self, env, show_debug_msg: bool = True):
        super(ActorCriticContinuous, self).__init__()

        self.show_debug_msg = show_debug_msg
        self.env = env
        observation_size = env.observation_space.shape[0]

        if type(env.action_space) is gym.spaces.discrete.Discrete:
            raise RuntimeError("Discrete action space not supported")

        control_size = env.action_space.shape[0]

        self.lay1 = torch.nn.Linear(observation_size, 50)
        self.act1 = torch.nn.ReLU()
        self.lay2 = torch.nn.Linear(50, 30)
        self.act2 = torch.nn.ReLU()
        self.policy_mean_head = torch.nn.Linear(30, control_size)
        self.policy_std_head = torch.nn.Linear(30, control_size)
        self.policy_std_act = torch.nn.Softplus()
        self.value_head = torch.nn.Linear(30, 1)
        self.value_act = torch.nn.Tanh()

        self.discount_factor = 0.99
        self.learning_rate = 0.01

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.optimizer.zero_grad()

    def forward(self, x: torch.tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        x = self.act1(self.lay1(x))
        x = self.act2(self.lay2(x))
        action_mean = self.policy_mean_head(x)
        action_std = self.policy_std_act(self.policy_std_head(x)) + 0.01
        action = torch.cat((action_mean, action_std), dim=1)
        value = self.value_act(self.value_head(x))
        return action, value

    def act(self, x: np.ndarray):
        action, _ = self.forward(torch.from_numpy(x).float().unsqueeze(0))
        return action.argmax().item()

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
                
                control_size = int(action.size(1) / 2)
                mean = action[0,0:control_size]
                std = action[0,control_size:control_size*2]
                action_distribution = torch.distributions.normal.Normal(mean, std)
                sampled_action = action_distribution.sample()

                observation, reward, done, _ = self.env.step(sampled_action.numpy())

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

                if self.show_debug_msg:
                    mean_reward = round(torch.tensor(total_episode_rewards).mean().item(), 1)
                    median_reward = round(torch.tensor(total_episode_rewards).median().item(), 1)
                    percentage = round(ep / episodes * 100, 1)
                    print(f"Progress: {percentage}%. Batch loss: {batch_loss}. Episode rewards (mean/median): {mean_reward}/{median_reward}.")
                total_episode_rewards = []
                policy_losses = []
                value_losses = []
