import numpy as np
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

def orthogonal_init(m):
    if isinstance(m, nn.Linear):
        init.orthogonal_(m.weight, gain=init.calculate_gain("relu"))
        if m.bias is not None:
            init.constant_(m.bias, 0)


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_1=400, hidden_2=200):
        super(CriticNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, 1)
        )
        self.network.apply(orthogonal_init)  # Apply orthogonal initialization


    def forward(self, state):
        return self.network(state)

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_1=400, hidden_2=200):
        super(ActorNetwork, self).__init__()

        # Shared input layer
        self.shared_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_1),
            nn.ReLU()
        )

        # Mean (mu) head
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, action_dim),
            nn.Tanh()  # Clipped between -1 and 1
        )

        # Variance (sigma^2) head
        self.sigma_head = nn.Sequential(
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, action_dim),
            nn.Sigmoid()  # Ensure variance is positive
        )

        # Apply orthogonal initialization
        self.shared_layer.apply(orthogonal_init)
        self.mu_head.apply(orthogonal_init)
        self.sigma_head.apply(orthogonal_init)

    def forward(self, state):
        shared_output = self.shared_layer(state)
        mu = self.mu_head(shared_output)
        sigma = self.sigma_head(shared_output)
        return mu, sigma
    