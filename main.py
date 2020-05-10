import argparse
import json
import sys

import gym
import torch

import reinforce
import actor_critic

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--env", help="the open ai env to use")
parser.add_argument("-a", "--agent", help="the agent to use")
parser.add_argument("--episodes", type=int, default=2000, help="the number of episodes to run")
parser.add_argument("--render", default=False, help="if should render the environment")
args = parser.parse_args()

try:
    env = gym.make(args.env)
except:
    print(f"Could not find env {args.env}")
    sys.exit(1)

observation_size = env.observation_space.shape[0]
control_size = int(env.action_space.n)

if args.agent == "reinforce":
    net = reinforce.Reinforce(observation_size, [128], control_size)
elif args.agent == "actor-critic":
    net = actor_critic.ActorCritic(observation_size, [64], control_size, [64, 10])
else:
    print(f"Agent type {args.agent} not found")
    sys.exit(1)


NUM_EPISODES = args.episodes
MAX_EPISODE_STEPS = 1000
for ep in range(NUM_EPISODES):
    percentage = round(ep / NUM_EPISODES * 100, 1)
    observation = env.reset()
    total_returns = 0
    for t in range(MAX_EPISODE_STEPS):
        if args.render:
            env.render()

        obs = torch.from_numpy(observation).float().unsqueeze(0)
        action = net.forward_train(obs)
        observation, reward, done, _ = env.step(action.item())
        net.add_reward(reward)

        total_returns += reward
        if done:
            loss = net.train_episode()
            print(f"Progress: {ep}/{NUM_EPISODES} ({percentage}%); Episode duration: {t+1} steps; Total return: {total_returns}; Loss: {loss}")
            break

env.close()
