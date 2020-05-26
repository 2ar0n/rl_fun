import argparse
import sys

import gym
import torch

import agents

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--env", help="the open ai env to use")
parser.add_argument("-a", "--agent", help="the agent to use")
parser.add_argument("-w", "--model-weight", help="the path to the weight of the model to load")
parser.add_argument("-ep", "--episodes", type=int, default=2000, help="the number of episodes to run")
parser.add_argument("-r", "--render", type=bool, default=False, help="if should render the environment")
args = parser.parse_args()

try:
    env = gym.make(args.env)
except:
    print(f"Could not find env {args.env}")
    sys.exit(1)

agent = agents.create_agent(args.agent, env)

try:
    agent.load_state_dict(torch.load(args.model_weight))
except:
    print(f"Could not load model weights at: {args.model_weight}")
    sys.exit(1)

sums_of_reward = []
for ep in range(args.episodes):
    sum_of_reward = 0
    observation = env.reset()
    for t in range(1000):
        if args.render:
            env.render()
        action = agent.act(observation)
        observation, reward, done, _ = env.step(action)
        sum_of_reward += reward
        if done:
            break
    print(f"Sum of episode rewards: {sum_of_reward}")
    sums_of_reward.append(sum_of_reward)
env.close()

avg_sum_of_reward = round(torch.tensor(sums_of_reward).mean().item(), 1)
print(f"Average sum of episode rewards: {avg_sum_of_reward}")
