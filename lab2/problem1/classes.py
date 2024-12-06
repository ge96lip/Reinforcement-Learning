import numpy as np
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim

class ExperienceReplayBuffer:
    """Replay buffer for storing experiences.
    The experience replay buffer stores past experiences so that the agent can
    learn from them later.
    By sampling randomly from these experiences, the agent avoids overfitting to
    the most recent
    transitions and helps stabilize training.
    - The buffer size is limited, and older experiences are discarded to make
    room for new ones.
    - Experiences are stored as tuples of (state, action, reward, next_state,
    done).
    - A batch of experiences is sampled randomly during each training step for
    updating the Q-values."""
    def __init__(self, maximum_length):
        self.buffer = deque(maxlen=maximum_length) # Using deque ensures efficient removal of oldest elements
        
    def add(self, experience):
        """Add a new experience to the buffer"""
        
        self.buffer.append(experience)
        
    def __len__(self):
        """Return the current size of the buffer"""
        
        return len(self.buffer)
    
    def sample(self, n):
        """Randomly sample a batch of experiences"""
        
        if n > len(self.buffer):
            raise IndexError('Sample size exceeds buffer size!')
        
        indices = np.random.choice(len(self.buffer), size=n, replace=False) # Random sampling
        batch = [self.buffer[i] for i in indices] # Create a batch from sampledindices
        return zip(*batch) # Unzip batch into state, action, reward, next_state,and done

class MyNetwork(nn.Module):
    """Feedforward neural network that approximates the Q-function.
    The network takes the current state as input and outputs Q-values for all
    possible actions.
    The action corresponding to the highest Q-value is considered the optimal
    action.
    - The input size corresponds to the state dimension of the environment.
    - The network has one hidden layer with 64 neurons and ReLU activation.
    - The output layer has one neuron per action (Q-values for each action)."""
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_layer = nn.Linear(input_size, 64) # First layer: state -> hidden layer
        self.hidden_layer = nn.Linear(64, 64) # Second layer: hidden -> hidden layer
        self.output_layer = nn.Linear(64, output_size) # Output layer: hidden -> Q-values
        self.activation = nn.ReLU() # ReLU activation function for hidden layers
    
    def forward(self, x):
        """Define forward pass"""
        
        x = self.activation(self.input_layer(x)) # Apply input layer and ReLU
        x = self.activation(self.hidden_layer(x)) # Apply hidden layer and ReLU
        return self.output_layer(x) # Return Q-values for all actions
    

class DQNNetwork(nn.Module):
    """A simple feedforward neural network for DQN."""
    def __init__(self, input_size, output_size, hidden_size):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return self.fc3(x) 