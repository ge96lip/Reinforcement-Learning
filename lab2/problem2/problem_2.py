# Copyright [2024] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 2 for EL2805 - Reinforcement Learning.


# Load packages
import numpy as np
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from DDPG_agent import RandomAgent, DDPGAgent
import warnings
from utils import running_average
from training1 import random_training, ddpg_training, ddpg_training1
warnings.simplefilter(action='ignore', category=FutureWarning)



# Import and initialize Mountain Car Environment
env = gym.make('LunarLanderContinuous-v3')
# If you want to render the environment while training run instead:
# env = gym.make('LunarLanderContinuous-v2', render_mode = "human")

env.reset()

# Parameters
N_episodes = 1000               # Number of episodes to run for training
discount_factor = 0.95         # Value of gamma
n_ep_running_average = 50      # Running average of 50 episodes
m = len(env.action_space.high) # dimensionality of the action
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = env.action_space.high[0]
print("m is: ", m, "action dim is: ", action_dim)
buffer_size=5000
agent = DDPGAgent(state_dim, action_dim, max_action, gamma=0.99, buffer_size=buffer_size)
# Reward

# Agent initialization
#agent = RandomAgent(m)

# Training process

#episode_reward_list, episode_number_of_steps = random_training(N_episodes, env, agent, n_ep_running_average)
episode_reward_list, episode_number_of_steps = ddpg_training(N_episodes, env, agent, n_ep_running_average)
# Close environment
env.close()


# Plot Rewards and steps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
ax[0].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_reward_list, n_ep_running_average), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward vs Episodes')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, N_episodes+1)], episode_number_of_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes')
ax[1].legend()
ax[1].grid(alpha=0.3)
name = f'./figures/neural-network-2-memory{buffer_size}'
plt.savefig(name + '.png')
plt.show()
