import numpy as np
from utils import decide_random

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


    def __init__(self, 
                 maze, 
                 seed,
                 allow_minotaur_stay: bool = False, 
                 expected_life: float = 1, 
                 minotaur_chase: bool = False, 
                 keys: bool = False
                 ):

        self.maze = maze
        self.temp_position = (-1, -1)
        self.poise_probability = (1/expected_life)
        self.minotaur_chase = minotaur_chase
        self.allow_minotaur_stay      = allow_minotaur_stay
        self.actions                  = self.__actions()
        self.states, self.map         = self.__states()
        self.n_actions                = len(self.actions)
        self.n_states                 = len(self.states)
        self.initial_state = ((0,0), (6,5), "NOKEYS")
        self.random_with_seed = np.random.RandomState(seed)
        

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

    def get_position(self, agent_position, action): 
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
            pass
        return (x, y)

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
    
    def reward(self, previous_state, next_state):
        _, _, progress = previous_state
        _, _, next_progress = next_state

        # terminal state (absorbing): nothing happens
        if self.terminal_state(previous_state):
            reward = 0
        elif next_progress != "EATEN" and progress == "NOKEYS" and next_progress == "KEYS":
            reward = self.KEY_REWARD
        elif next_progress !=  "EATEN" and progress == "KEYS" and next_progress == "WIN":
            reward = self.GOAL_REWARD
        else:
            reward = self.STEP_REWARD

        return reward
    
    def _next_state(self, state, action):
        
        player_position, minotaur_position, progress = state

        if self.terminal_state(state):
            print("terminal state")
            pass 
        
        else:
            
            chase = self.minotaur_chase and decide_random(self.random_with_seed, 0.35)
            action_minotaur = self._valid_minotaur_moves(state, chase)
            
            #chase = self.minotaur_chase and np.random.rand() < 0.35
            #valid_minotaur_moves = self._valid_minotaur_moves(state, chase=chase)
            action_minotaur = self.random_with_seed.choice(action_minotaur)

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
        if state[2] in ["EATEN", "WIN"]: 
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
        if self.poise_probability > 0:
            horizon_reached = decide_random(self.random_with_seed, self.poise_probability)
        else:
            print("Poise Probability needs to be bigger than 0")
        return horizon_reached
    
    def step(self, action, previous_state):
        
        # update state
        new_state = self._next_state(previous_state, action)

        reward = self.reward(previous_state, new_state)
        
        horizon_reached = self._horizon_reached()
        terminal = self.terminal_state(new_state)
        done = horizon_reached or terminal
        win = new_state[2] == "WIN"
       
        return new_state, reward, done, win 

    
def minotaur_maze_exit_probability(environment, agent):
    n_episodes = 100
    n_wins = 0
    for _ in range(1, n_episodes+1):
        done = False
        time_step = 0
        state = environment.initial_state
        while not done:
            action = agent.compute_action(state=state, explore=False)
            state, _, done, won = environment.step(action, state)
            time_step += 1
            n_wins += 1 if won else 0
    exit_probability = n_wins / n_episodes
    return exit_probability



"""
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
env1 = Maze(maze, seed= 6, expected_life=expected_life, minotaur_chase=True, keys=True)
agent_sarsa = SARSA(
        env=env1,
        discount=gamma,
        alpha=alpha2,
        epsilon=epsilon2,
        q_init=0.01,
        delta = 0.99
    )
_, values_epsilon1 = agent_sarsa.train(NUM_EPISODES, decrease_epsilon=True)
exit_probability = minotaur_maze_exit_probability(env1, agent_sarsa)
print(f"Exit_probability for SARSA agent: ", exit_probability)


epsilon2 = 0.1
alpha2 = (2/3)
env2 = Maze(maze, seed= 6, expected_life=expected_life, minotaur_chase=True, keys=True)
agent_sarsa = SARSA(
        env=env2,
        discount=gamma,
        alpha=alpha2,
        epsilon=epsilon2,
        q_init=0.01,
        delta = 0.3
    )
_, values_epsilon2 = agent_sarsa.train(NUM_EPISODES, decrease_epsilon=True)
exit_probability = minotaur_maze_exit_probability(env2, agent_sarsa)
print(f"Exit_probability for SARSA agent: ", exit_probability)

plt.figure(figsize=(10, 6))
plt.plot(range(NUM_EPISODES), values_epsilon1, label=f"$\\delta = {1}$")
plt.plot(range(NUM_EPISODES), values_epsilon2, label=f"$\\delta = {0.3}$")
plt.xlabel("Episodes")
plt.ylabel(f"Value Function $V(s_0)$")
plt.title("Convergence of Value Function Over Episodes with different alpha values")
plt.legend()
plt.grid()
plt.show()

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