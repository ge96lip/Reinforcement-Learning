# Load packages
import numpy as np
from Networks import CriticNetwork, ActorNetwork
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import torch.nn.init as init


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

class PPOAgent(Agent):
    def __init__(self, state_dim, action_dim, max_action, gamma=0.99, actor_lr=5e-5, critic_lr=5e-4, epsilon=0.2, M=10):
        super(PPOAgent, self).__init__(action_dim)
        self.state_dim = state_dim
        self.max_action = max_action
        self.gamma = gamma

        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs_old = []
        self.dones = []
        self.trajectory = []


        self.epsilon = epsilon  # Clipping parameter
        self.M = M
        
        # Actor and Critic Networks
        self.actor = ActorNetwork(state_dim, action_dim)
        #self.actor_target = ActorNetwork(state_dim, action_dim)
        #self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        self.critic = CriticNetwork(state_dim, action_dim)
        #self.critic_target = CriticNetwork(state_dim, action_dim)
        #self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        

    # def select_action(self, state):
    #     mu, var = self.actor(torch.tensor(state, dtype=torch.float32))
    #     var = var + 1e-6  # Add small constant to avoid division by zero
    #     mu = mu.detach().numpy()
    #     std = torch.sqrt(var).detach().numpy()
    #     action = np.clip(np.random.normal(mu, std), -1, 1).flatten()
    #     return action

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        mu, std = self.actor.forward(state)
        #std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        
        # Sample an action and compute log probability
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)  # Sum over action dimensions
        
        # Clip the action if necessary (e.g., for environments with bounded actions)
        action = torch.clamp(action, -self.max_action, self.max_action)
        
        return action.detach().numpy(), log_prob.detach().numpy()
    
    def store_transition(self, state, action, reward, next_state, done, log_prob):
        self.trajectory.append((state, action, reward, next_state, done, log_prob))

    def clear_trajectory(self):
        self.trajectory = []

    def clear_buffer(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs_old = []
        self.dones = []

    # def compute_returns_and_advantages(self, rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    #     """
    #     Compute discounted returns and advantages using GAE.
    #     Args:
    #         rewards (torch.Tensor): Rewards for the trajectory.
    #         values (torch.Tensor): Value estimates for the states in the trajectory.
    #         next_values (torch.Tensor): Value estimates for the next states.
    #         dones (torch.Tensor): Done flags for the trajectory.
    #         gamma (float): Discount factor for rewards.
    #         lam (float): GAE lambda for advantage estimation.
    #     Returns:
    #         returns (torch.Tensor): Discounted returns.
    #         advantages (torch.Tensor): Advantage estimates.
    #     """
    #     returns = []
    #     advantages = []
    #     advantage = 0
    #     for t in reversed(range(len(rewards))):
    #         td_error = rewards[t] + gamma * (1 - dones[t]) * next_values[t] - values[t]
    #         advantage = td_error + gamma * lam * (1 - dones[t]) * advantage
    #         advantages.insert(0, advantage)
    #         returns.insert(0, advantage + values[t])  # GAE advantage + baseline value
        
    #     returns = torch.tensor(returns, dtype=torch.float32)
    #     advantages = torch.tensor(advantages, dtype=torch.float32)
    #     return returns, advantages

    def compute_returns_and_advantages(self, rewards, values, dones, gamma=0.99, lam=0.95):
        """
        Compute discounted returns and Generalized Advantage Estimation (GAE).
        
        Args:
            rewards (torch.Tensor): Rewards for the trajectory.
            values (torch.Tensor): Value estimates for the states.
            dones (torch.Tensor): Boolean flags indicating terminal states.
            gamma (float): Discount factor.
            lam (float): Lambda for GAE.
        
        Returns:
            returns (torch.Tensor): Discounted returns.
            advantages (torch.Tensor): Advantage estimates.
        """
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        next_value = 0.0  # V(s') = 0 for terminal state
        advantage = 0.0

        for t in reversed(range(len(rewards))):
            # Mask terminal states
            mask = 1.0 - dones[t]

            # Temporal Difference error (δ_t)
            td_error = rewards[t] + gamma * next_value * mask - values[t]

            # GAE Advantage calculation
            advantage = td_error + gamma * lam * mask * advantage # I added a lambda term to stabalize the training
            advantages[t] = advantage

            # Compute the return
            next_value = values[t]
            returns[t] = advantage + values[t]  # GAE advantage + baseline value

        return returns, advantages
        

    def train(self, episode, normal_update=True):
        # Unpack trajectory
        states, actions, rewards, next_states, dones, log_probs_old = zip(*self.trajectory)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        #print("log_probs_old:", log_probs_old)
        log_probs_old = np.array(log_probs_old, dtype=np.float32)
        log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32)

        # # Compute returns and advantages
        # with torch.no_grad():
        #     values = self.critic(states).squeeze()  # V(s) from the critic
        #     next_values = torch.cat((values[1:], torch.tensor([0.])))  # V(s') for next states
        #     returns, advantages = self.compute_returns_and_advantages(rewards, values, next_values, dones)

        with torch.no_grad():
            values = self.critic.forward(states).squeeze()  # Critic value estimates V(s)
            next_values = torch.cat((values[1:], torch.zeros(1)))  # Append 0 for the final state
            returns, advantages = self.compute_returns_and_advantages(rewards, values, dones)

        # Normalize advantages for numerical stability
        #advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        # Perform multiple epochs of updates
        for _ in range(self.M):
            # ----- Update Critic -----
            self.critic_optimizer.zero_grad()
            value_preds = self.critic.forward(states).squeeze()
            critic_loss = nn.MSELoss()(value_preds, returns)
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.critic_optimizer.step()

            if episode % 2 == 0 or normal_update == True:
                # ----- Update Actor -----
                self.actor_optimizer.zero_grad()
                mu, std = self.actor.forward(states)
                #std = torch.exp(log_std)
                dist = torch.distributions.Normal(mu, std)
                log_probs_new = dist.log_prob(actions).sum(axis=-1)

                # Compute the ratio r(θ) = π_θ(a|s) / π_θ_old(a|s)
                ratios = torch.exp(log_probs_new - log_probs_old)

                # Compute surrogate losses
                surrogate1 = ratios * advantages
                surrogate2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
                policy_loss = -torch.mean(torch.min(surrogate1, surrogate2))

                # Backpropagation for the actor
                policy_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                self.actor_optimizer.step()

 #   def train(self):
        # ''' Perform a single training step for the agent.'''

        # # Retrieve all data from the buffer
        # states, actions, rewards, next_states, dones = self.replay_buffer.get_all()

        # # Convert to tensors
        # states = torch.tensor(states, dtype=torch.float32)
        # actions = torch.tensor(actions, dtype=torch.float32)
        # rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        # dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        
        # # Step 1: Compute the target returns G
        # target_returns = []
        # running_return = 0
        # for reward, done in zip(reversed(rewards), reversed(dones)):
        #     running_return = reward + (self.gamma * running_return * (1 - done))
        #     target_returns.insert(0, running_return)
        # target_returns = torch.tensor(target_returns, dtype=torch.float32)
        # target_returns = (target_returns - target_returns.mean()) / (target_returns.std() + 1e-6)

        # # Step 2: Compute advantages Ψ_i = G_i - V_ω(s_i)
        # values = self.critic(states).detach()
        # advantages = target_returns - values

        # # Step 3: Store old log probabilities π_old(a|s)
        # with torch.no_grad():
        #     mu_old, var_old = self.actor(states)
        #     dist_old = torch.distributions.Normal(mu_old, torch.sqrt(var_old))
        #     log_probs_old = dist_old.log_prob(actions).sum(axis=-1)

        # # Step 4: Perform M gradient updates
        # for _ in range(self.M):
        #     # ----- Update Critic (Value Network) -----
        #     self.critic_optimizer.zero_grad()
        #     value_preds = self.critic(states)
        #     critic_loss = nn.MSELoss()(value_preds, target_returns)  # MSE loss between V(s) and target G
        #     critic_loss.backward()
        #     nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        #     self.critic_optimizer.step()

        #     # ----- Update Actor (Policy Network) -----
        #     self.actor_optimizer.zero_grad()
        #     # Compute new log probabilities π_θ(a|s)
        #     mu_new, var_new = self.actor(states)
        #     dist_new = torch.distributions.Normal(mu_new, torch.sqrt(var_new))
        #     log_probs_new = dist_new.log_prob(actions).sum(axis=-1)

        #     # Compute ratio r_θ
        #     ratios = torch.exp(log_probs_new - log_probs_old)

        #     # Clipped Surrogate Loss
        #     surrogate1 = ratios * advantages.squeeze()
        #     surrogate2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages.squeeze()
        #     policy_loss = -torch.mean(torch.min(surrogate1, surrogate2))

        #     # Backpropagation for the actor
        #     policy_loss.backward()
        #     nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        #     self.actor_optimizer.step()





class RandomAgent(Agent):
    ''' Agent taking actions uniformly at random, child of the class Agent'''
    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__(n_actions)

    def forward(self, state: np.ndarray) -> np.ndarray:
        ''' Compute a random action in [-1, 1]

            Returns:
                action (np.ndarray): array of float values containing the
                    action. The dimensionality is equal to self.n_actions from
                    the parent class Agent
        '''
        return np.clip(-1 + 2 * np.random.rand(self.n_actions), -1, 1)
