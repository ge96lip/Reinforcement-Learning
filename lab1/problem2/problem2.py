# Copyright [2024] [KTH Royal Institute of Technology]
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 1 for EL2805 - Reinforcement Learning.
# Carlotta Sophia Hölzle || 20020521-8167 || csholzle@kth.se
# Jannis || jannise@kth.se || 20010411-1653

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import sem
import pickle

show_plots = False

# Set random seed for reproducibility
np.random.seed(42)

# Import and initialize Mountain Car Environment
# env = gym.make('MountainCar-v0', render_mode='human') # You can add render_mode='human' to see the rendering
env = gym.make('MountainCar-v0')
env.reset()
n_actions = env.action_space.n  # Number of actions
low, high = env.observation_space.low, env.observation_space.high

# Parameters
N_episodes = 200  # Number of episodes to run for training
discount_factor = 1  # Gamma
lambda_ = 0.6  # Lambda for SARSA(λ) eligibility traces
base_learning_rate = 0.001  # Base alpha (learning rate)
epsilon = 0.0001  # Exploration rate
exploration_decay = 0.9  # Decay rate for epsilon
learning_rate_decay = 0.95  # Decay rate for learning rate
momentum = 0.96  # Momentum for SGD updates
order = 2  # Order for Fourier basis
n_features = (order + 1) ** len(low)  # Number of Fourier basis features
include_constant = True

# # Initialize weights, eligibility traces, and velocity for momentum
# weights = np.zeros((n_actions, n_features))  # Shape: (n_actions, n_features)
# eligibility_traces = np.zeros((n_actions, n_features))  # Shape: (n_actions, n_features)
# velocity = np.zeros_like(weights)  # For Nesterov momentum

# # Reward tracking
# episode_reward_list = []

# # Evaluate the impact of including/excluding η = [0,0] 
# if include_constant:
#     print("Including η = [0,0] in Fourier basis.")
# else:
#     print("Excluding η = [0,0] in Fourier basis.")


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

def fourier_basis(state, order=2, include_constant=True):
    terms = [np.arange(order + 1) for _ in range(len(state))]
    frequencies = np.array(np.meshgrid(*terms)).T.reshape(-1, len(state))
    if not include_constant:
        frequencies = frequencies[np.sum(frequencies, axis=1) != 0]  # Exclude constant
    norms = np.linalg.norm(frequencies, axis=1)
    scaled_frequencies = np.where(norms == 0, 1, norms)  # Avoid division by zero
    return np.cos(np.pi * np.dot(frequencies, state)) / scaled_frequencies

def epsilon_greedy_action(weights, state, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)  # Random action (exploration)
    else:
        q_values = [np.dot(fourier_basis(state), weights[a]) for a in range(n_actions)]
        return np.argmax(q_values)  # Best action (exploitation)
    
# Function to compute the value function
def compute_value_function(weights):
    resolution = 50  # Number of points in each dimension
    state_space = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(state_space, state_space)
    value_function = np.zeros_like(X)

    for i in range(resolution):
        for j in range(resolution):
            state = np.array([X[i, j], Y[i, j]])
            value_function[i, j] = np.max(
                [np.dot(fourier_basis(state), weights[a]) for a in range(n_actions)]
            )
    return X, Y, value_function

# Function to compute the optimal policy
def compute_optimal_policy(weights):
    resolution = 50  # Number of points in each dimension
    state_space = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(state_space, state_space)
    optimal_policy = np.zeros_like(X)

    for i in range(resolution):
        for j in range(resolution):
            state = np.array([X[i, j], Y[i, j]])
            optimal_policy[i, j] = np.argmax(
                [np.dot(fourier_basis(state), weights[a]) for a in range(n_actions)]
            )
    return X, Y, optimal_policy


def train(env, N_episodes, discount_factor, lambda_, base_learning_rate, epsilon, exploration_decay, learning_rate_decay, momentum, order, include_constant, show_plots, custom_exploration):
    # Initialize weights, eligibility traces, and velocity for momentum
    weights = np.zeros((n_actions, n_features))  # Shape: (n_actions, n_features)
    eligibility_traces = np.zeros((n_actions, n_features))  # Shape: (n_actions, n_features)
    velocity = np.zeros_like(weights)  # For Nesterov momentum
    eligibility_traces = np.zeros_like(weights)

    # Reward tracking
    episode_reward_list = []

    # Evaluate the impact of including/excluding η = [0,0] 
    if include_constant:
        print("Including η = [0,0] in Fourier basis.")
    else:
        print("Excluding η = [0,0] in Fourier basis.")


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
        #eligibility_traces = np.zeros_like(weights)

        while not (done or truncated):
            # Take action and observe result
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = scale_state_variables(next_state)

            # Compute features and TD error
            features = fourier_basis(state)
            next_features = fourier_basis(next_state)
            td_error = reward + discount_factor * np.dot(next_features, weights[action]) - np.dot(features, weights[action])

            # Update eligibility traces
            eligibility_traces *= discount_factor * lambda_
            eligibility_traces[action] += features

            # Clip eligibility traces
            eligibility_traces = np.clip(eligibility_traces, -5, 5)

            # SGD with Nesterov acceleration
            velocity = momentum * velocity + base_learning_rate * td_error * eligibility_traces
            weights += momentum * velocity + base_learning_rate * td_error * eligibility_traces

            # Update state and action
            state, action = next_state, epsilon_greedy_action(weights, next_state, epsilon)

            # Update episode reward
            total_episode_reward += reward

        # Simulate episode 100
            if episode == 100:
                env.render()

        # Append episode reward
        episode_reward_list.append(total_episode_reward)

        # Decay epsilon and learning rate
        if custom_exploration:
            epsilon = max(0.01, epsilon * exploration_decay)  # Reduce exploration during training
        base_learning_rate *= learning_rate_decay  # Reduce learning rate during training

        # Print progress every 100 episodes
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_reward_list[-100:])
            print(f"Episode {episode + 1}: Average Reward: {avg_reward:.2f}")


    if show_plots:
        # Plot the value function
        X, Y, value_function = compute_value_function(weights)
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, value_function, cmap='viridis')
        ax.set_title("Value Function")
        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")
        ax.set_zlabel("Value")
        plt.show()

        # Plot the optimal policy
        X, Y, optimal_policy = compute_optimal_policy(weights)
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, optimal_policy, cmap='plasma')
        ax.set_title("Optimal Policy")
        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")
        ax.set_zlabel("Action")
        plt.show()

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

    return episode_reward_list, weights, eligibility_traces


def analyze_learning_rate_alpha(alphas, env, N_episodes, discount_factor, lambda_, epsilon, exploration_decay, learning_rate_decay, momentum, order, include_constant, show_plots, custom_exploration):
    """Analyze the effect of learning rate (α) on average total reward."""
    avg_rewards = []
    std_errs = []

    for alpha in alphas:
        episode_rewards ,_,_  = train(env, N_episodes, discount_factor, lambda_, alpha, epsilon, exploration_decay, learning_rate_decay, momentum, order, include_constant, show_plots, custom_exploration)
        avg_rewards.append(np.mean(episode_rewards))
        std_errs.append(sem(episode_rewards))

    # Plotting results
    plt.errorbar(alphas, avg_rewards, yerr=std_errs, fmt='-o', label='Average reward')
    plt.xlabel("Learning Rate (α)")
    plt.ylabel("Average Total Reward")
    plt.title("Effect of Learning Rate (α) on Policy Performance")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def analyze_eligibility_trace_lambda(lambdas, num_runs=50):
    """Analyze the effect of λ on average total reward."""
    avg_rewards = []
    std_errs = []

    for lambda_ in lambdas:
        total_rewards = []

        total_rewards, _, _ = train(env, N_episodes, discount_factor, lambda_, base_learning_rate, epsilon, exploration_decay, learning_rate_decay, momentum, order, include_constant, show_plots, custom_exploration)

        avg_rewards.append(np.mean(total_rewards))
        std_errs.append(sem(total_rewards))

    # Plotting results
    plt.errorbar(lambdas, avg_rewards, yerr=std_errs, fmt='-o', label='Average reward')
    plt.xlabel("Eligibility Trace Parameter (λ)")
    plt.ylabel("Average Total Reward")
    plt.title("Effect of Eligibility Trace Parameter (λ) on Policy Performance")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()



#Train the agent
show_plots = True
custom_exploration = False
#N_episodes = 50
rewards, weights, eligibitlity =  train(env, N_episodes, discount_factor, lambda_, base_learning_rate, epsilon, exploration_decay, learning_rate_decay, momentum, order, include_constant, show_plots, custom_exploration)
show_plots = False
# Create a dictionary to store W and N
#data = {'W': weights, 'N': eligibitlity}


# # Save the dictionary to a file using pickle
# file_name = "weights.pkl"
# with open(file_name, 'wb') as file:
#     pickle.dump(data, file)

#print(f"Matrices W and N saved successfully to {file_name}.")

# Define hyperparameter ranges
# alphas = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.05, 0.1]
# #alphas = np.linspace(0.001, 0.1, 10)
# lambdas = [0.1, 0.3, 0.5, 0.7, 0.9]
# lambdas = np.linspace(0.1, 1, 20)
# N_episodes = 50

# Perform the analyses
#_,_,_ = analyze_learning_rate_alpha(alphas, env, N_episodes, discount_factor, lambda_, epsilon, exploration_decay, learning_rate_decay, momentum, order, include_constant, show_plots, custom_exploration)
#analyze_eligibility_trace_lambda(lambdas, num_runs=10)