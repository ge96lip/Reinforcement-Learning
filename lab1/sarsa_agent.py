from utils import decide_random, calculate_moving_average
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
        # Initialize self._n with detailed debugging
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
    
    def _a_idx(self, state, action):
        #print("original action is: ", action)
        valid_actions = self.env.possible_actions(state)
        
        # Find the index of the action in the valid_actions list
        try:
            action_index = valid_actions.index(action)
            #print("Action index is: ", action_index)
            return action_index
        except ValueError:
            print("Action not found in valid actions!")
            return -1 
    
    def v(self, state):
        s = self.env.map[state]
        v = max(self._q[s])
        return v
    
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
            v = self.v(state)
            a = self.env.random_with_seed.choice(np.asarray(self._q[s] == v).nonzero()[0])      # random among those with max Q-value
            action = valid_actions[a]

        return action
    
    def train(
            self,
            n_episodes: int,
            decrease_epsilon: bool = False,
    ):
        
        stats = defaultdict(list)
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
                update_stats = self.update(last_state)

                for k, v in update_stats.items():
                    stats[k].append(v)
                episode_reward += reward
                episode_length += 1

                # Update state
                state = next_state
            stats["episode_reward"].append(episode_reward)
            stats["episode_length"].append(episode_length)
            
            episodes.set_description(
                f"Episode {episode} - "
                f"Avg reward: {(calculate_moving_average(stats["episode_reward"])[-1]):.1f} - "
                f"Avg length: {(calculate_moving_average(stats["episode_length"])[-1]):.1f}"
            )
            
            #print(self.v(INITIAL_STATE))
            initial_state_values.append(self.v(INITIAL_STATE))
            
        return stats, initial_state_values 
        
    def update(self, last_state):
        state = last_state["state"]
        action = last_state["action"]
        reward = last_state["reward"]
        next_state = last_state["next_state"]

        # Get indices
        s = self.env.map[state]
        a = self._a_idx(state, action)
        s_next = self.env.map[next_state]
        next_action = self.compute_action(state=next_state, explore=True)
        a_next = self._a_idx(next_state, next_action)
        
        # Update Q-function
        self._n[s][a] += 1
        step_size = 1 / (self._n[s][a] ** self.alpha)
        self._q[s][a] += step_size * (reward + self.discount * (self._q[s_next][a_next]) - self._q[s][a])
        return {}