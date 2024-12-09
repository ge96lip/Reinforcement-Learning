# Copyright [2024] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 2 for EL2805 - Reinforcement Learning.

# Load packages
import numpy as np
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from DQN_agent import RandomAgent, DQNAgent
import warnings
from utils import running_average

warnings.simplefilter(action='ignore', category=FutureWarning)


# Import and initialize the discrete Lunar Lander Environment
env = gym.make('LunarLander-v3')
print("env made")
# If you want to render the environment while training run instead:
# env = gym.make('LunarLander-v2', render_mode = "human")


env.reset()

# Parameters
N_episodes = 600                          # Number of episodes
discount_factor = 0.99                       # Value of the discount factor
n_ep_running_average = 50                    # Running average of 50 episodes
n_actions = env.action_space.n               # Number of available actions
dim_state = len(env.observation_space.high)  # State dimensionality
buffer_size = 3000
# We will use these variables to compute the average episodic reward and
# the average number of steps per episode
episode_reward_list = []       # this list contains the total reward per episode
episode_number_of_steps = []   # this list contains the number of steps per episode

# Random agent initialization
N = 32
L = 5000
C = 10
print("retraining every: ", C)
agent = DQNAgent(n_actions, dim_state, gamma=discount_factor, buffer_size=buffer_size) # buffer_size=L, batch_size=N, gamma=discount_factor, update_target = C RandomAgent(n_actions)

### Training process
patience = 100  # Number of episodes to wait for improvement
min_delta = 1e-2  # Minimum change in reward to count as improvement
best_reward = -float('inf')  # Initialize with the worst possible reward
no_improvement = 0
# trange is an alternative to range in python, from the tqdm library
# It shows a nice progression bar that you can update with useful information
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

for episode in EPISODES:
    # Reset enviroment data and initialize variables
    done, truncated = False, False
    state = env.reset()[0]
    total_episode_reward = 0.
    t = 0
    while not (done or truncated):
        # Take a random action
        action = agent.forward(state)

        # Get next state and reward
        next_state, reward, done, truncated, _ = env.step(action)
        
        # Store the experience in the replay buffer
        agent.replay_buffer.add((state, action, reward, next_state, done))
        
        # Train the agent
        agent.backward()

        # Update episode reward
        total_episode_reward += reward

        # Update state for next iteration
        state = next_state
        t+= 1
    # Decay epsilon
    agent.epsilon = max(agent.epsilon_min, (agent.epsilon * agent.epsilon_decay))
    
    # Update the target network every few episodes
    if episode % agent.update_target == 0:
        agent.update_target_network()
        
    # Append episode reward and total number of steps
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)
    avg_reward = running_average(episode_reward_list, n_ep_running_average)[-1]

    # Early stopping logic
    if avg_reward > best_reward + min_delta:
        best_reward = avg_reward
        no_improvement = 0  # Reset counter
    else:
        no_improvement += 1

    """if no_improvement >= patience and avg_reward >= 55:
        print(f"Early stopping at episode {episode} - No improvement in the last {patience} episodes.")
        break
    if avg_reward >= 60: 
        print(f"Early stopping at episode {episode} - Got average reward over 60.")
        break"""

    # Update progress bar
    EPISODES.set_description(
        f"Episode {episode} - Reward/Steps: {total_episode_reward:.1f}/{t} - Avg. Reward/Steps: {avg_reward:.1f}/{t}")

    
# Save the trained Q-network
name = f'neural-network-groundtruth-Buffer{buffer_size}'
torch.save(agent.network, './weights/'+name+'.pth')
# Close environment
env.close()

# Plot Rewards and steps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))

# Use the length of episode_reward_list for the x-axis
x_range = range(1, len(episode_reward_list) + 1)

ax[0].plot(x_range, episode_reward_list, label='Episode reward')
ax[0].plot(x_range, running_average(
    episode_reward_list, n_ep_running_average), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward vs Episodes')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot(x_range, episode_number_of_steps, label='Steps per episode')
ax[1].plot(x_range, running_average(
    episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes')
ax[1].legend()
ax[1].grid(alpha=0.3)

plt.savefig('./figures/'+name+'.jpg')
plt.show()