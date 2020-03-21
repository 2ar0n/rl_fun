import gym
import torch
from torch import nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class ActorCritic:
    def __init__(self, input, hidden_layers, output, hidden_layers_value):
        temp = [input] + hidden_layers + [output]
        layers = []
        for i in range(0, len(temp) - 1):
            layers.append(nn.Linear(temp[i], temp[i+1]))
            if i < len(temp) - 2:
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Softmax())
        self.policy = nn.Sequential(*layers).float()

        temp = [input] + hidden_layers_value + [1]
        layers = []
        for i in range(0, len(temp) - 1):
            layers.append(nn.Linear(temp[i], temp[i+1]))
            if i < len(temp) - 2:
                layers.append(nn.ReLU())
            else:
                pass
        self.value = nn.Sequential(*layers).float()

        self.probs = []
        self.rewards = []
        self.estimated_values = []

        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=0.01)
        self.optimizer_policy.zero_grad()   # zero the gradient buffers
        self.optimizer_value = optim.Adam(self.policy.parameters(), lr=0.01)
        self.optimizer_value.zero_grad()   # zero the gradient buffers
        self.discount_factor = 0.99

    def forward_train(self, observation):
        inference = self.policy.forward(observation)
        m = Categorical(inference)
        action = m.sample()
        self.probs.append(m.log_prob(action))
        estimated_value = self.value.forward(observation)
        self.estimated_values.append(estimated_value)
        return action

    def forward(self, observation):
        inference = self.policy.forward(obs)
        return torch.argmax(inference)

    def add_reward(self, reward):
        self.rewards.append(reward)

    def train_episode(self):
        R = 0.0
        returns = []
        for r in self.rewards[::-1]:
            R = R * self.discount_factor + r
            returns.append(R)
        returns.reverse()
        returns = torch.tensor(returns)
        eps = np.finfo(np.float32).eps.item()
        returns = (returns - returns.mean()) / (returns.std() + eps)

        policy_loss = []
        action_value_loss = []
        for idx in range(0, len(self.probs)):
            log_prob_action = self.probs[idx]
            target_return = returns[idx]
            estimated_value = self.estimated_values[idx]

            advantage = target_return.item() - estimated_value.item()
            policy_loss.append(-advantage * log_prob_action)

            loss_fn = torch.nn.MSELoss()
            action_value_loss.append(loss_fn(target_return, estimated_value))

        self.optimizer_policy.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer_policy.step()

        self.optimizer_value.zero_grad()
        action_value_loss = torch.stack(action_value_loss).sum()
        action_value_loss.backward()
        self.optimizer_value.step()

        self.probs = []
        self.rewards = []
        self.estimated_values = []


def main():
    pass

if __name__ == "__main__":
    main()