from lab1.problem1.utils import decide_random
from collections import defaultdict
from tqdm import trange
import numpy as np 

INITIAL_STATE = ((0, 0), (6, 5), "NOKEYS")

class SARSA(): 
    
    def __init__(
            self,
            env, 
            epsilon: float | str,
            discount: float, 
            alpha: float | None = None,
            q_init: float = 0,
            delta: float = 1,
    ):
        self.env = env
        self.discount = discount
        self.epsilon = epsilon
        self.alpha = alpha
        self.q_init = q_init
        self.delta = delta
        self._q = [
            (self.q_init if not env.terminal_state(env.states[state]) else 0) *
            np.ones(len(self.env.possible_actions(env.states[state])))
            for state in self.env.states
        ]
        #print(self.env.possible_actions)
        self._n = []
        for state_idx, state in self.env.states.items():
            try:
                # Get valid actions for the current state
                valid_actions = self.env.possible_actions(state)
                num_actions = len(valid_actions)

                # Log the state, valid actions, and their count
                #print(f"State Index: {state_idx}, State: {state}, Valid Actions: {valid_actions}, Number of Actions: {num_actions}")

                # Initialize the counts array for the valid actions
                self._n.append(np.zeros(num_actions))
            except Exception as e:
                # Log any issues encountered during initialization
                print(f"Error initializing state {state_idx} ({state}): {e}")
                raise
        
    def q(self, state, action):
        s = self.env.map[state]
        a = self.possible_actions(state, action)
        q = self._q[s][a]
        return q
    
    
    def compute_action(
            self,
            state,
            explore: bool = True,
    ):
        
        valid_actions = self.env.possible_actions(state)

        epsilon = self.epsilon
        if explore and decide_random(self.env.random_with_seed, epsilon):
            # random_number = random.randint(0, len(valid_actions) - 1)
            
            # action = valid_actions[random_number]
            action = self.env.random_with_seed.choice(valid_actions)
        else:
            s = self.env.map[state]
            v = max(self._q[s])
            a = self.env.random_with_seed.choice(np.asarray(self._q[s] == v).nonzero()[0])      # random among those with max Q-value
            action = valid_actions[a]

        return action
    
    def train(
            self,
            n_episodes: int,
            decrease_epsilon: bool = False,
    ):
        
        episodes = trange(1, n_episodes + 1, desc='Episode: ', leave=True)
        initial_state_values = []
        
        for episode in episodes:
            # Reset environment data and initialize variables
            done = False
            state = self.env.initial_state
            episode_reward, episode_length = 0, 0
            if decrease_epsilon: 
                temp = 1 / (episode ** self.delta)
                self.epsilon = temp
            # Run episode
            while not done:
                # Interact with the environment
                action = self.compute_action(state=state, explore=True)
                next_state, reward, done, _ = self.env.step(action, state)
                # Update policy
                last_state = {
                    "episode": episode,
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "done": done
                }
                self.update(last_state)

                episode_reward += reward
                episode_length += 1

                # Update state
                state = next_state
            
            #print(self.v(INITIAL_STATE))
            s_index = self.env.map[INITIAL_STATE]
            v = max(self._q[s_index])
            initial_state_values.append(v)
            
        #return stats, initial_state_values 
        return initial_state_values
        
    def update(self, last_state):
        state = last_state["state"]
        action = last_state["action"]
        reward = last_state["reward"]
        next_state = last_state["next_state"]
        valid_actions = self.env.possible_actions(state)
        valid_actions_next = self.env.possible_actions(next_state)

        # Get indices
        s = self.env.map[state]
        a = valid_actions.index(action)
        s_next = self.env.map[next_state]
        next_action = self.compute_action(state=next_state, explore=True)
        a_next = valid_actions_next.index(next_action)
        
        # Update Q-function
        self._n[s][a] += 1
        step_size = 1 / (self._n[s][a] ** self.alpha)
        self._q[s][a] += step_size * (reward + self.discount * (self._q[s_next][a_next]) - self._q[s][a])
        return {}