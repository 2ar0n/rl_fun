# RL fun

This repository is yet another one dedicated to the implementation of various reinforcement learning algorithm for didactic purpose. It is designed around OpenAIs gym environment.

# Dependencies

- pytorch
- OpenAi gym

You can either install into your system with `pip install torch gym` or get the docker image with `docker pull 2ar0n/rl_fun:latest`.

# Train and evaluate an agent

For example:

`python train.py --env=CartPole-v0 --agent=reinforce`

Model weights will be saved to the `weights` directory per default.
To evaluate a model on a given environment, run:

`python evaluate.py --env=CartPole-v0 --agent=reinforce --model-weight <path_to_weight_file> --render=true`
