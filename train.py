import argparse
import sys
import datetime

import gym
import torch

import agents

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--env", help="the open ai env to use")
parser.add_argument("-a", "--agent", help="the agent to use")
parser.add_argument("-ep", "--episodes", type=int, default=2000, help="the number of episodes to run")
parser.add_argument("-b", "--batch-size", type=int, default=50, help="the batch size")
parser.add_argument("-t", "--target-dir", default="weights", help="the model weights will be saved to this directory")
args = parser.parse_args()

try:
    env = gym.make(args.env)
except:
    print(f"Could not find env {args.env}")
    sys.exit(1)

agent = agents.create_agent(args.agent, env)

agent.train(args.episodes, args.batch_size)
env.close()

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
file_name = f"{args.target_dir}/{args.agent}_{args.env}_{timestamp}.pt"
torch.save(agent.state_dict(), file_name)
print(f"Saved model weights to {file_name}")
