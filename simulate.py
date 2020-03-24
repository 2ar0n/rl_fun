import gym
import torch
import reinforce
import actor_critic

env = gym.make('CartPole-v0')
observation_size = env.observation_space.shape[0]
control_size = int(env.action_space.n)

# net = reinforce.Reinforce(observation_size, [128], control_size)
net = actor_critic.ActorCritic(observation_size, [128], control_size, [128, 10])

NUM_EPISODES = 1000
MAX_EPISODE_STEPS = 1000
RENDER_START_PERCENTAGE = 15
for i_episode in range(NUM_EPISODES):
    percentage = round(i_episode / NUM_EPISODES * 100, 1)
    observation = env.reset()
    for t in range(MAX_EPISODE_STEPS):
        if percentage >= RENDER_START_PERCENTAGE:
            env.render()

        obs = torch.from_numpy(observation).float().unsqueeze(0)
        action = net.forward_train(obs)
        observation, reward, done, _ = env.step(action.item())
        net.add_reward(reward)

        if done:
            print(f"Episode finished after {t+1} timesteps")
            print(f"Percentage: {percentage}")
            break
    net.train_episode()

env.close()