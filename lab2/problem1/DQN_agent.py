# Copyright [2024] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 2 for EL2805 - Reinforcement Learning.


# Load packages
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from classes import DQNNetwork, ExperienceReplayBuffer

class Agent(object):
    ''' Base agent class, used as a parent class

        Args:
            n_actions (int): number of actions

        Attributes:
            n_actions (int): where we store the number of actions
            last_action (int): last action taken by the agent
    '''
    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.last_action = None

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

    def forward(self, state: np.ndarray) -> int:
        ''' Compute an action uniformly at random across n_actions possible
            choices

            Returns:
                action (int): the random action
        '''
        self.last_action = np.random.randint(0, self.n_actions)
        return self.last_action
    
class DQNAgent(Agent):
    def __init__(self, n_actions, state_dim, buffer_size=10000, batch_size=32, gamma=0.99, lr=1e-3, epsilon=1, epsilon_min=0.01, epsilon_decay=0.995, hidden_size = 128, update_target = 10):
        super().__init__(n_actions)
        self.state_dim = state_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.update_target = update_target
        
        # Initialize the DQN and target networks
        self.network = DQNNetwork(input_size=state_dim, output_size=n_actions, hidden_size = hidden_size)
        # Reinitialize the network with the same architecture
        #self.network = DQNNetwork(input_size=state_dim, output_size=n_actions)

        # Load the saved weights
        #self.network.load_state_dict(torch.load('./weights/neural-network-1.pth'))
        
        self.target_network = DQNNetwork(input_size=state_dim, output_size=n_actions, hidden_size=hidden_size)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()

        # Optimizer and loss function
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # Experience replay buffer
        self.replay_buffer = ExperienceReplayBuffer(buffer_size)

    def forward(self, state: np.ndarray):
        """Selects an action using epsilon-greedy strategy."""
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.n_actions)  # Explore: random action
        else:
            state_tensor = torch.tensor([state], dtype=torch.float32)
            with torch.no_grad():
                action = self.network(state_tensor).argmax().item()  # Exploit: max Q-value
        self.last_action = action
        return action

    def backward(self):
        """Samples a batch from the replay buffer and updates the network."""
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough samples to train
        
        # Sample a batch of experiences
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Compute Q-values for the current states
        q_values = self.network(states).gather(1, actions).squeeze()

        # Compute target Q-values for the next states
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute the loss and perform a backward pass
        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0) # Clip gradients to avoid exploding gradients
        self.optimizer.step()

    def update_target_network(self):
        """Updates the target network weights."""
        self.target_network.load_state_dict(self.network.state_dict())