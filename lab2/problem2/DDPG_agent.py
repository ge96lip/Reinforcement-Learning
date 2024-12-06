# Copyright [2024] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 2 for EL2805 - Reinforcement Learning.


# Load packages
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from classes import ExperienceReplayBuffer, ActorNetwork, CriticNetwork
from DDPG_soft_updates import soft_updates

class Agent(object):
    ''' Base agent class

        Args:
            n_actions (int): actions dimensionality

        Attributes:
            n_actions (int): where we store the dimensionality of an action
    '''
    def __init__(self, n_actions: int):
        self.n_actions = n_actions

    def forward(self, state: np.ndarray):
        ''' Performs a forward computation '''
        pass

    def backward(self):
        ''' Performs a backward pass on the network '''
        pass


class RandomAgent(Agent):
    ''' Agent taking actions uniformly at random, child of the class Agent'''
    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__(n_actions)

    def forward(self, state: np.ndarray) -> np.ndarray:
        ''' Compute a random action in [-1, 1]

            Returns:
                action (np.ndarray): array of float values containing the
                    action. The dimensionality is equal to self.n_actions from
                    the parent class Agent.
        '''
        return np.clip(-1 + 2 * np.random.rand(self.n_actions), -1, 1)
    
class DDPGAgent(Agent):
    def __init__(self, state_dim, action_dim, max_action, buffer_size=30000, batch_size=64, gamma=0.99, tau=1e-3, actor_lr=5e-5, critic_lr=5e-4, d=2):
        super(DDPGAgent, self).__init__(action_dim)
        self.state_dim = state_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.update_frequency = d  # Frequency for target network update
        
        self.t = 0  # Step counter for periodic updates
        
        # Replay Buffer
        self.replay_buffer = ExperienceReplayBuffer(buffer_size)
        
        # Actor and Critic Networks
        self.actor = ActorNetwork(state_dim, action_dim)
        self.actor_target = ActorNetwork(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        self.critic = CriticNetwork(state_dim, action_dim)
        self.critic_target = CriticNetwork(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Noise parameters
        self.noise_mu = 0.0
        self.noise_sigma = 0.2

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = self.actor(state).detach().cpu().numpy().flatten()
        noise = np.random.normal(self.noise_mu, self.noise_sigma, size=self.n_actions)
        action = action + noise
        return np.clip(action, -self.max_action, self.max_action)

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample a batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # Compute target Q values
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_Q = rewards + (1 - dones) * self.gamma * self.critic_target(next_states, next_actions)

        # Update Critic
        current_Q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_Q, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # Update Actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # Increment step counter
        self.t += 1

        # Update target networks periodically
        if self.t % self.update_frequency == 0:
            soft_updates(self.actor, self.actor_target, self.tau)
            soft_updates(self.critic, self.critic_target, self.tau)