import argparse
import json
import sys

import gym
import torch

import agents

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--env", help="the open ai env to use")
parser.add_argument("-a", "--agent", help="the agent to use")
parser.add_argument("-ep", "--episodes", type=int, default=2000, help="the number of episodes to run")
parser.add_argument("-b", "--batch-size", type=int, default=50, help="the batch size")
args = parser.parse_args()

try:
    env = gym.make(args.env)
except:
    print(f"Could not find env {args.env}")
    sys.exit(1)

if args.agent == "reinforce":
    agent = agents.Reinforce(env)
elif args.agent == "actor-critic":
    agent = agents.ActorCritic(env)
elif args.agent == "ppo":
    agent = agents.PPO(env)
else:
    print(f"Agent type {args.agent} not found")
    sys.exit(1)

agent.train(args.episodes, args.batch_size)
env.close()
