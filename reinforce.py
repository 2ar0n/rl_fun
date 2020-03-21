import gym
import torch
from torch import nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class Reinforce:
    def __init__(self, input, hidden_layers, output):
        temp = [input] + hidden_layers + [output]

        layers = []
        for i in range(0, len(temp) - 1):
            layers.append(nn.Linear(temp[i], temp[i+1]))
            if i < len(temp) - 2:
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Softmax())
        self.policy = nn.Sequential(*layers).float()

        self.probs = []
        self.rewards = []
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.01)
        self.optimizer.zero_grad()   # zero the gradient buffers
        self.discount_factor = 0.99

    def forward_train(self, observation):
        inference = self.policy.forward(observation)
        m = Categorical(inference)
        action = m.sample()
        self.probs.append(m.log_prob(action))
        return action

    def forward(self, observation):
        inference = self.policy.forward(obs)
        return torch.argmax(inference)

    def add_reward(self, reward):
        self.rewards.append(reward)

    def train_episode(self):
        values = []
        R = 0.0
        for r in self.rewards[::-1]:
            R = R * self.discount_factor + r
            values.append(R)
        values.reverse()
        values = torch.tensor(values)
        eps = np.finfo(np.float32).eps.item()
        values = (values - values.mean()) / (values.std() + eps)
        loss = []
        for v, prob in zip(values, self.probs):
            loss.append(- v * prob)
        policy_loss = torch.cat(loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()   # zero the gradient buffers
        self.probs = []
        self.rewards = []


def main():
    pass

if __name__ == "__main__":
    main()