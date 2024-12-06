# Copyright [2024] [KTH Royal Institute of Technology]
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 1 for EL2805 - Reinforcement Learning.

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Import and initialize Mountain Car Environment
#env = gym.make('MountainCar-v0', render_mode='human')
env = gym.make('MountainCar-v0')
env.reset()
n_actions = env.action_space.n      # Number of actions
low, high = env.observation_space.low, env.observation_space.high

# Parameters
N_episodes = 1000               # Number of episodes to run for training
discount_factor = 0.99          # Gamma
learning_rate = 0.1             # Alpha (base learning rate)
epsilon = 1                   # Exploration rate
lambda_ = 0.9                   # Lambda for SARSA(λ)
momentum = 0.9                  # Momentum for SGD updates
order = 2                       # Order for Fourier basis
n_features = (order + 1) ** len(low)  # Number of Fourier basis features

# Initialize weights, eligibility traces, and velocity for momentum
weights = np.random.randn(n_features)  # Weight vector
eligibility_traces = np.random.randn(n_features)  # Eligibility trace vector

# Reward tracking
episode_reward_list = []

# Functions
def running_average(x, N):
    ''' Compute the running mean of the last N elements of a vector x '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

def scale_state_variables(s, low=env.observation_space.low, high=env.observation_space.high):
    ''' Rescale s to the box [0,1]^2 '''
    return (s - low) / (high - low)

def fourier_basis(state, order=2):
    # Generate terms for each dimension
    terms = [np.arange(order + 1) for _ in range(len(state))]

    # Create all combinations of frequencies
    frequencies = np.array(np.meshgrid(*terms)).T.reshape(-1, len(state))

    # Compute Fourier basis
    return np.cos(np.pi * np.dot(frequencies, state))


def epsilon_greedy_action(weights, state, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)  # Random action (exploration)
    else:
        q_values = [np.dot(fourier_basis(state), weights) for _ in range(n_actions)]
        return np.argmax(q_values)  # Best action (exploitation)

# Track simulation number
simulation_number = 0

action_labels = ["Left", "Neutral", "Right"]

# Training process
for episode in range(N_episodes):
    # Reset environment and initialize variables
    done = False
    truncated = False
    state = env.reset()[0]
    state = scale_state_variables(state)
    action = epsilon_greedy_action(weights, state, epsilon)
    total_episode_reward = 0

    # Reset eligibility traces
    eligibility_traces = np.zeros_like(weights)

    while not (done or truncated):
        # Take action and observe result
        next_state, reward, done, truncated, _ = env.step(action)
        next_state = scale_state_variables(next_state)

        # Visualize every 100th simulation (if desired)
        if simulation_number % 100 == 0:
            env.render()  # This might slow down training
            print(f"Simulation {simulation_number}: Action taken: {action_labels[action]}")



        # Initialize TD error
        td_error = reward
        features = fourier_basis(state)

        next_action = epsilon_greedy_action(weights, next_state, epsilon)
        next_features = fourier_basis(next_state)

        for i in features:
            td_error -= weights[int(i)]
            eligibility_traces += features

        if done:
            weights += learning_rate * features * eligibility_traces  # Update weights
            break

        # next_action = epsilon_greedy_action(weights, next_state, epsilon)
        # next_features = fourier_basis(next_state)
        for i in next_features:
            td_error += discount_factor * weights[int(i)]
        
        weights += learning_rate * td_error * eligibility_traces  # Update weights
        eligibility_traces *= discount_factor * lambda_  # Update eligibility traces
        eligibility_traces = np.clip(eligibility_traces, -5, 5)  # Clip traces
        state, action = next_state, next_action  # Update state and action

        # Update episode reward
        total_episode_reward += reward

    # Update simulation number
    simulation_number += 1

    # Append episode reward
    episode_reward_list.append(total_episode_reward)

    # Decay epsilon
    epsilon = max(0.01, epsilon * 0.99)

    # Print progress every 100 episodes
    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(episode_reward_list[-100:])
        print(f"Episode {episode + 1}: Average Reward: {avg_reward:.2f}")

# Plot Rewards
plt.plot([i for i in range(1, N_episodes + 1)], episode_reward_list, label='Episode reward')
plt.plot([i for i in range(1, N_episodes + 1)], running_average(episode_reward_list, 10), label='Average episode reward')
plt.xlabel('Episodes')
plt.ylabel('Total reward')
plt.title('Total Reward vs Episodes')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Close environment
env.close()
