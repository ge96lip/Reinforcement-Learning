import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display
import scipy.stats as stats

from q_agent import QLearning
from sarsa_agent import SARSA
import random

# Implemented methods
methods = ['DynProg', 'ValIter']

# Some colours
LIGHT_RED    = '#FFC4CC'
LIGHT_GREEN  = '#95FD99'
BLACK        = '#000000'
WHITE        = '#FFFFFF'
LIGHT_PURPLE = '#E8D0FF'
LIGHT_ORANGE = '#FAE0C3'

    


def decide_random(rng, probability):
    #rand = np.random.uniform()
    #return (rand < probability)
    temp = rng.binomial(n=1, p=probability)
    return (temp == 1)

    # return np.random.binomial(n=1, p=probability) == 1
class Maze:

    # Actions
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_RIGHT = 2
    MOVE_LEFT = 3
    STAY = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    
    STEP_REWARD = 0         #TODO
    KEY_REWARD = 1
    GOAL_REWARD = 1         #TODO
    IMPOSSIBLE_REWARD =  0  #TODO
    LOSS_REWARD = -50      #TODO


    def __init__(self, 
                 maze, 
                 seed,
                 horizon: int | None = None,
                 allow_minotaur_stay: bool = False, 
                 expected_life: float = 1, 
                 minotaur_chase: bool = False, 
                 keys: bool = False
                 ):
        """ Constructor of the environment Maze.
        """
        self.maze = maze
        self.horizon = horizon
        self.temp_position = (-1, -1)
        self.poise_probability = (1/expected_life)
        self.minotaur_chase = minotaur_chase
        self.keys = keys
        self.finite_horizon = horizon
        #self.key_position             = (0,7)
        self.start_position           = ((0,0), (6,5))
        self.allow_minotaur_stay      = allow_minotaur_stay
        self.actions                  = self.__actions()
        self.states, self.map         = self.__states()
        self.n_actions                = len(self.actions)
        self.n_states                 = len(self.states)
        self._initial_state = ((0,0), (6,5), "NOKEYS")
        self._rng = np.random.RandomState(seed)
        self.moves_cache = dict()
        assert not (self.poise_probability > 0 and self.finite_horizon is not None)    # poison only for discounted MDPs

        

    def __actions(self):
        actions = dict()
        actions[self.STAY]       = (0, 0)
        actions[self.MOVE_LEFT]  = (0, -1)
        actions[self.MOVE_RIGHT] = (0, 1)
        actions[self.MOVE_UP]    = (-1, 0)
        actions[self.MOVE_DOWN]  = (1, 0)
        return actions

    def __states(self):
        states = dict()
        map = dict()
        s = 0

        # Generate all possible states
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                if self.maze[i, j] != 1:  # Agent cannot be in a wall
                    for k in range(self.maze.shape[0]):
                        for l in range(self.maze.shape[1]):
                            for has_key in ["NOKEYS", "KEYS"]:
                                states[s] = ((i, j), (k, l), has_key)
                                map[((i, j), (k, l), has_key)] = s
                                s += 1

        # Helper function to check if a state is non-terminal
        def non_terminal_state(state):
            player_position, minotaur_position, progress = state
            eaten = player_position == minotaur_position
            exited = progress == "KEYS" and player_position == (6, 5)
            return not eaten and not exited

        # Filter out terminal states
        filtered_states = {key: value for key, value in states.items() if non_terminal_state(value)}

        # Reindex states and map
        reindexed_states = {idx: state for idx, state in enumerate(filtered_states.values())}
        reindexed_map = {state: idx for idx, state in enumerate(filtered_states.values())}

        s = len(reindexed_states)  # Update the counter to reflect the size of `reindexed_states`

        # Add the two special states
        win_state = ((self.temp_position), (self.temp_position), "WIN")
        eaten_state = ((self.temp_position), (self.temp_position), "EATEN")

        reindexed_states[s] = win_state
        reindexed_map[win_state] = s
        s += 1

        reindexed_states[s] = eaten_state
        reindexed_map[eaten_state] = s
        s += 1

        # Return the results
        return reindexed_states, reindexed_map

    def get_position(self, agent_position, action, is_player): 
        x, y = agent_position

        if action == self.MOVE_UP or action == 0:
            x -= 1
        elif action == self.MOVE_DOWN or action == 1:
            x += 1
        elif action == self.MOVE_LEFT or action == 3:
            y -= 1
        elif action == self.MOVE_RIGHT or action == 2:
            y += 1
        elif action == self.STAY or action == 4:
            pass
        else:
            raise ValueError(f"Invalid move {action}")
        return (x, y)
    
    def __move_minotaur(self, state, actions_minotaur, only_valid = True): 
        minotaur_actions = []
        
        # Check if the current state is terminal
        if self.terminal_state(state):
            minotaur_actions.append(self.STAY)
        elif not self.minotaur_chase: 
            if np.random.rand() < 0.35:
                minotaur_actions = []
                mx, my = self.states[state][1][0], self.states[state][1][1]  # Minotaur's position
                ax, ay = self.states[state][0][0], self.states[state][0][1]  # Player's position
                
                # Determine the chase direction
                if ax > mx:
                    minotaur_actions.append(self.MOVE_RIGHT)
                elif ax < mx:
                    minotaur_actions.append(self.MOVE_LEFT)
                if ay > my:
                    minotaur_actions.append(self.MOVE_DOWN)
                elif ay < my:
                    minotaur_actions.append(self.MOVE_UP)
                
                # If no specific direction is determined, add all potential moves
                if not minotaur_actions:
                    minotaur_actions = [self.MOVE_UP, self.MOVE_DOWN, self.MOVE_LEFT, self.MOVE_RIGHT]
            # Validate actions to ensure they do not result in the Minotaur leaving the environment
            valid_actions = []
            for action in minotaur_actions:
                # Calculate the new position after applying the action
                new_position = tuple(map(sum, zip((mx, my), self.actions[action])))
                
                # Check if the new position is within the maze boundaries
                if all(0 <= new_position[i] < self.maze.shape[i] for i in range(len(new_position))):
                    valid_actions.append(action)
            
            minotaur_actions = valid_actions
        else: 
            # Use provided actions, validating them as well
            for action in actions_minotaur:
                
                new_state = tuple(map(sum, zip(state[1], self.actions[action])))
                if all(0 <= new_state[i] < self.maze.shape[i] for i in range(len(new_state))):
                    minotaur_actions.append(action)

        # Choose a random valid action
        if minotaur_actions:  # Ensure there are valid actions before choosing
            random_number = random.randint(0, len(minotaur_actions) - 1)
            return minotaur_actions[random_number]
        else:
            return self.STAY 

    def _valid_minotaur_moves(self, state, chase):
        valid_moves = self._chase_minotaur_moves(state) if chase else self._random_minotaur_moves(state)
        return valid_moves

    def _random_minotaur_moves(self, state):
        random_moves = []

        if self.terminal_state(state):
            random_moves.append(self.STAY)
        else:
            _, minotaur_position, _ = state
            x_minotaur, y_minotaur = minotaur_position
            if self.allow_minotaur_stay:
                random_moves.append(self.STAY)
            if x_minotaur - 1 >= 0:
                random_moves.append(self.MOVE_UP)
            if x_minotaur + 1 <  self.maze.shape[0]:
                random_moves.append(self.MOVE_DOWN)
            if y_minotaur - 1 >= 0:
                random_moves.append(self.MOVE_LEFT)
            if y_minotaur + 1 < self.maze.shape[1]:
                random_moves.append(self.MOVE_RIGHT)

        return random_moves

    def _chase_minotaur_moves(self, state):
        chase_moves = []

        if self.terminal_state(state):
            chase_moves.append(self.STAY)
        else:
            player_position, minotaur_position, _ = state
            x_player, y_player = player_position
            x_minotaur, y_minotaur = minotaur_position

            delta_x = x_player - x_minotaur
            delta_y = y_player - y_minotaur
            assert abs(delta_x) > 0 or abs(delta_y) > 0  # otherwise it should be eaten (terminal state)
            
            if delta_x != 0 and (delta_y == 0 or abs(delta_x) <= abs(delta_y)):
                if delta_x < 0:
                    chase_moves.append(self.MOVE_UP)
                else:
                    chase_moves.append(self.MOVE_DOWN)
            if delta_y != 0 and (delta_x == 0 or abs(delta_y) <= abs(delta_x)):
                if delta_y < 0:
                    chase_moves.append(self.MOVE_LEFT)
                else:
                    chase_moves.append(self.MOVE_RIGHT)

        return chase_moves

    def reward(self, previous_state, current_state, action):

        reward = self._reward(previous_state, current_state)

        return reward
    
    def _reward(self, state, next_state):
        _, _, progress = state
        _, _, next_progress = next_state

        # terminal state (absorbing): nothing happens
        if self.terminal_state(state):
            reward = 0

        elif next_progress != "EATEN" and \
                progress == "NOKEYS" and next_progress == "KEYS":
            reward = self.KEY_REWARD
        elif next_progress !=  "EATEN" and \
                progress == "KEYS" and next_progress == "WIN":
            reward = self.GOAL_REWARD
        else:
            reward = self.STEP_REWARD

        return reward
    
    def _next_state(self, state, action, minotaur_move = None):
        
        player_position, minotaur_position, progress = state

        if self.terminal_state(state):
            print("terminal state")
            pass 
        else:
            if minotaur_move is None:
                #actions_minotaur = [self.MOVE_DOWN, self.MOVE_UP, self.MOVE_RIGHT, self.MOVE_LEFT] # Possible moves for the Minotaur
                #if self.allow_minotaur_stay:
                 #   actions_minotaur.append(self.STAY)+  chase = self.minotaur_chase and decide_random(self._probability_chase_move)
                chase = self.minotaur_chase and decide_random(self._rng, 0.35)
                action_minotaur = self._valid_minotaur_moves(state, chase)
                
                #chase = self.minotaur_chase and np.random.rand() < 0.35
                #valid_minotaur_moves = self._valid_minotaur_moves(state, chase=chase)
                action_minotaur = self._rng.choice(action_minotaur)

            next_player_position = self.get_position(player_position, action, True)
            next_minotaur_position = self.get_position(minotaur_position, action_minotaur, False)

            x, y = next_player_position
            if next_player_position == next_minotaur_position:
                #print("progress EATEN")
                state = (self.temp_position, self.temp_position, "EATEN")
            elif progress == "KEYS" and next_player_position == (6,5):
                #print("progress win!")
                state = (self.temp_position, self.temp_position, "WIN")
            elif progress == "NOKEYS" and next_player_position == (0,7):
                state = (next_player_position, next_minotaur_position, "KEYS")
            else:
                state = (next_player_position, next_minotaur_position, progress)

        return state
    
    def terminal_state(self, state):
        _, _, progress = state
        if progress in ["EATEN", "WIN"]: 
            terminal = True
        else: 
            terminal = False  
        return terminal
    
    
    def possible_actions(self, state):
        valid_moves = [self.STAY]

        if not self.terminal_state(state):
            player_position, _, _ = state
            x, y = player_position

            x_tmp = x - 1
            if x_tmp >= 0 and self.maze[x_tmp, y] != 1:
                valid_moves.append(self.MOVE_UP)

            x_tmp = x + 1
            if x_tmp < self.maze.shape[0] and self.maze[x_tmp, y] != 1:
                valid_moves.append(self.MOVE_DOWN)

            y_tmp = y - 1
            if y_tmp >= 0 and self.maze[x, y_tmp] != 1:
                valid_moves.append(self.MOVE_LEFT)

            y_tmp = y + 1
            if y_tmp < self.maze.shape[1] and self.maze[x, y_tmp] != 1:
                valid_moves.append(self.MOVE_RIGHT)

        return valid_moves
    
    def _horizon_reached(self):
        # random time horizon geometrically distributed
        if self.poise_probability > 0:
            horizon_reached = decide_random(self._rng, self.poise_probability)
        else:
            print("Poise Probability needs to be bigger than 0")
        return horizon_reached

    
    def reset(self):
        self._current_state = self._initial_state
        return self._current_state
    
    def step(self, action, previous_state):
        
        # update state
        new_state = self._next_state(previous_state, action)

        reward = self.reward(previous_state, new_state, action)
        
        horizon_reached = self._horizon_reached()
        terminal = self.terminal_state(new_state)
        done = horizon_reached or terminal
        win = new_state[2] == "WIN"
       
        return new_state, reward, done, win 

    
def minotaur_maze_exit_probability(environment, agent):
    n_episodes = 100
    n_wins = 0
    for episode in range(1, n_episodes+1):
        done = False
        time_step = 0
        #environment.seed(episode)
        state = environment.reset()
        while not done:
            action = agent.compute_action(state=state, explore=False)
            state, _, done, won = environment.step(action, state)
            time_step += 1
            n_wins += 1 if won else 0
    exit_probability = n_wins / n_episodes
    return exit_probability

NUM_EPISODES = 5000
INITIAL_STATE = ((0, 0), (6, 5), "NOKEYS")
maze = np.array([
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 2, 0, 0]])

# Initialize the environment
expected_life = 50 
gamma = 1- (1/expected_life)


epsilon1 = 0.1
alpha1 = (2/3)
env1 = Maze(maze, seed= 6, expected_life=expected_life, minotaur_chase=True, keys=True)
agent_q_learning = QLearning(
        env=env1,
        discount=gamma,
        alpha= alpha1,
        epsilon=epsilon1,
        q_init=0.1,
    )
#_, values_epsilon1 = agent_q_learning.train(NUM_EPISODES)
#exit_probability = minotaur_maze_exit_probability(env1, agent_q_learning)
#print(f"Exit_probability for QLearning Agent: ", exit_probability)

epsilon2 = 0.1
alpha2 = (2/3)
env2 = Maze(maze, seed= 6, expected_life=expected_life, minotaur_chase=True, keys=True)
agent_sarsa = SARSA(
        env=env2,
        discount=gamma,
        alpha=alpha2,
        epsilon=epsilon2,
        q_init=1,
    )
_, values_epsilon2 = agent_sarsa.train(NUM_EPISODES)
exit_probability = minotaur_maze_exit_probability(env2, agent_sarsa)
print(f"Exit_probability for SARSA agent: ", exit_probability)

plt.figure(figsize=(10, 6))
#plt.plot(range(NUM_EPISODES), values_epsilon1, label=f"$\\alpha = {alpha1}$")
plt.plot(range(NUM_EPISODES), values_epsilon2, label=f"$\\alpha = {alpha2}$")
plt.xlabel("Episodes")
plt.ylabel(f"Value Function $V(s_0)$")
plt.title("Convergence of Value Function Over Episodes with different alpha values")
plt.legend()
plt.grid()
plt.show()

"""
env1 = Maze(maze, seed=6, expected_life=expected_life, minotaur_chase=True, keys=True)
env2 = Maze(maze, seed=6, expected_life=expected_life, minotaur_chase=True, keys=True)

epsilon1 = 0.1
epsilon2 = 0.3

values_epsilon1 = train_agent(env1, epsilon1, NUM_EPISODES)
values_epsilon2 = train_agent(env2, epsilon2, NUM_EPISODES)

# Plot the value function over episodes
plt.figure(figsize=(10, 6))
plt.plot(range(NUM_EPISODES), values_epsilon1, label=f"$\epsilon = {epsilon1}$")
plt.plot(range(NUM_EPISODES), values_epsilon2, label=f"$\epsilon = {epsilon2}$")
plt.xlabel("Episodes")
plt.ylabel(f"Value Function $V(s_0)$")
plt.title("Convergence of Value Function Over Episodes")
plt.legend()
plt.grid()
plt.show()
"""