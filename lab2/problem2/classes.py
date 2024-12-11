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
    def get_all_states(self):
        """Retrieve all states stored in the buffer."""
        return [experience[0] for experience in self.buffer]
    

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_1=400, hidden_2=200):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()  # Constrain actions to [-1, 1]

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.tanh(self.fc3(x))
    
    
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_1=400, hidden_2=200):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_1)
        self.fc2 = nn.Linear(hidden_1 + action_dim, hidden_2)
        self.fc3 = nn.Linear(hidden_2, 1)
        self.relu = nn.ReLU()

    def forward(self, state, action):
        x = self.relu(self.fc1(state))
        x = torch.cat([x, action], dim=1)  # Concatenate state and action
        x = self.relu(self.fc2(x))
        return self.fc3(x)