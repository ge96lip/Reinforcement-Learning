# Copyright [2024] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 2 for EL2805 - Reinforcement Learning.


# Load packages
import numpy as np
import gymnasium as gym
import torch
from tqdm import trange
import warnings
from classes import DQNNetwork
warnings.simplefilter(action='ignore', category=FutureWarning)


def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y



# Import and initialize Mountain Car Environment
env = gym.make('LunarLander-v3')
# Load model
try:
    model = torch.load('neural-network-1.pth')
    #input_dim = env.observation_space.shape[0]  # Adjust to your environment
    #output_dim = env.action_space.n            # Adjust to your environment

    #model = DQNNetwork(input_dim, output_dim, 128)
    #model.load_state_dict(torch.load('neural-network-1.pth'))
    #model.eval()
    print('Network model: {}'.format(model))
except:
    print('File neural-network-1.pth not found!')
    exit(-1)
# If you want to render the environment while training run instead:
# env = gym.make('LunarLander-v2', render_mode = "human")

env.reset()

# Parameters
N_EPISODES = 50            # Number of episodes to run for trainings
CONFIDENCE_PASS = 50

# Reward
episode_reward_list = []  # Used to store episodes reward

# Simulate episodes
print('Checking solution...')
EPISODES = trange(N_EPISODES, desc='Episode: ', leave=True)
for i in EPISODES:
    EPISODES.set_description("Episode {}".format(i))
    # Reset enviroment data
    done, truncated = False, False
    state = env.reset()[0]
    total_episode_reward = 0.
    while not (done or truncated):
        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        q_values = model(torch.tensor(state))
        _, action = torch.max(q_values, dim=0)
        next_state, reward, done, truncated, _ = env.step(action.item())

        # Update episode reward
        total_episode_reward += reward

        # Update state for next iteration
        state = next_state

    # Append episode reward
    episode_reward_list.append(total_episode_reward)


# Close environment 
env.close()


avg_reward = np.mean(episode_reward_list)
confidence = np.std(episode_reward_list) * 1.96 / np.sqrt(N_EPISODES)


print('Policy achieves an average total reward of {:.1f} +/- {:.1f} with confidence 95%.'.format(
                avg_reward,
                confidence))

if avg_reward - confidence >= CONFIDENCE_PASS:
    print('Your policy passed the test!')
else:
    print("Your policy did not pass the test! The average reward of your policy needs to be greater than {} with 95% confidence".format(CONFIDENCE_PASS))