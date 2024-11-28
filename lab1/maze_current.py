import random
import numpy as np



# Implemented methods
methods = ['DynProg', 'ValIter']


class Maze:

    # Actions
    STAY       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

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
    GOAL_REWARD = 1         #TODO
    MINOTAUR_REWARD = -1      #TODO


    def __init__(self, maze, allow_minotaur_stay):
        """ Constructor of the environment Maze.
        """
        self.maze                     = maze
        self.allow_minotaur_stay      = allow_minotaur_stay
        self.actions                  = self.__actions()
        self.states, self.map         = self.__states()
        self.n_actions                = len(self.actions)
        self.n_states                 = len(self.states)
        self.transition_probabilities = self.__transitions()
        self.rewards                  = self.__rewards()
        

    def __actions(self):
        actions = dict()
        actions[self.STAY]       = (0, 0)
        actions[self.MOVE_LEFT]  = (0,-1)
        actions[self.MOVE_RIGHT] = (0, 1)
        actions[self.MOVE_UP]    = (-1,0)
        actions[self.MOVE_DOWN]  = (1,0)
        return actions

    def __states(self):
        
        states = dict()
        map = dict()
        s = 0
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                for k in range(self.maze.shape[0]):
                    for l in range(self.maze.shape[1]):
                        if self.maze[i,j] != 1:
                            states[s] = ((i,j), (k,l))
                            map[((i,j), (k,l))] = s
                            s += 1
        
        states[s] = 'Eaten'
        map['Eaten'] = s
        s += 1
        
        states[s] = 'Win'
        map['Win'] = s
        
        return states, map
    
    
    def __move_chase(self, probability, actions_minotaur, state):
         
        if np.random.rand() < probability: 
            actions_minotaur = []
            mx, my = self.states[state][1][0], self.states[state][1][1]
            ax, ay = self.states[state][0][0], self.states[state][0][1]
            if ax > mx:
                actions_minotaur.append('RIGHT')
            elif ax < mx:
                actions_minotaur.append('LEFT')
            if ay > my:
                actions_minotaur.append('DOWN')
            elif ay < my:
                actions_minotaur.append('UP')
            if not actions_minotaur:
                actions_minotaur = ['UP', 'DOWN', 'LEFT', 'RIGHT']
                
        return actions_minotaur
    
    def __move_minotaur(self, state, actions_minotaur, pick_random): 
        
        if not pick_random: 
            actions_minotaur = self.__move_chase(0.35, actions_minotaur, state)
         
        rows_minotaur, cols_minotaur = [], []
        for i in range(len(actions_minotaur)):
            # Is the minotaur getting out of the limits of the maze?
            impossible_action_minotaur = (self.states[state][1][0] + actions_minotaur[i][0] == -1) or \
                                            (self.states[state][1][0] + actions_minotaur[i][0] == self.maze.shape[0]) or \
                                            (self.states[state][1][1] + actions_minotaur[i][1] == -1) or \
                                            (self.states[state][1][1] + actions_minotaur[i][1] == self.maze.shape[1])
            if not impossible_action_minotaur:
                rows_minotaur.append(self.states[state][1][0] + actions_minotaur[i][0])
                cols_minotaur.append(self.states[state][1][1] + actions_minotaur[i][1]) 
                
        return rows_minotaur, cols_minotaur

    def move(self, state, action, pick_random=True):
        """
        Makes a step in the maze, given a current state and an action.
        If the action is STAY or an inadmissible action, the player stays in place.

        :param state: Current state of the system.
        :param action: Action taken by the player.
        :param pick_random: If True, the Minotaur's movements are randomized.
        :return: List of possible next states ((x, y), (x', y')) in the maze.
        """
        # Handle terminal states (game over)
        if self.states[state] in ['Eaten', 'Win']:
            return [self.states[state]]

        # Player's next position based on the action
        row_player, col_player = (
            self.states[state][0][0] + self.actions[action][0],
            self.states[state][0][1] + self.actions[action][1]
        )

        # Check if the player's action is impossible
        is_impossible_action = (
            row_player < 0 or row_player >= self.maze.shape[0] or
            col_player < 0 or col_player >= self.maze.shape[1] or
            self.maze[row_player, col_player] == 1  # Wall check
        )

        # Minotaur's possible moves
        minotaur_actions = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        if self.allow_minotaur_stay:
            minotaur_actions.append([0, 0])
        rows_minotaur, cols_minotaur = self.__move_minotaur(state, minotaur_actions, pick_random)

        # Helper function to determine the resulting state
        def determine_state(player_pos, minotaur_pos):
            if player_pos == minotaur_pos:
                return 'Eaten'
            elif player_pos == tuple(np.where(self.maze == 2)):  # Exit position
                return 'Win'
            else:
                return (player_pos, minotaur_pos)

        # Generate next states
        player_pos = (self.states[state][0][0], self.states[state][0][1]) if is_impossible_action else (row_player, col_player)
        next_states = [
            determine_state(player_pos, (rows_minotaur[i], cols_minotaur[i]))
            for i in range(len(rows_minotaur))
        ]

        return next_states

    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states, self.n_states, self.n_actions)
        transition_probabilities = np.zeros(dimensions)

        # Compute the transition probabilities
        for s in range(self.n_states):
            for a in range(self.n_actions):
                current_state = self.states[s]

                # Handle terminal states: 'Eaten' and 'Win'
                if current_state == 'Eaten':
                    transition_probabilities[self.map['Eaten'], s, a] = 1.0
                    continue
                elif current_state == 'Win':
                    transition_probabilities[self.map['Win'], s, a] = 1.0
                    continue

                # Get next possible states for the given action
                next_states = self.move(s, a)

                # Distribute probability equally among all valid next states
                prob = 1.0 / len(next_states)
                # print probability 
                for next_state in next_states:
                    transition_probabilities[self.map[next_state], s, a] = prob
                    
        return transition_probabilities

    def __rewards(self):
        
        """ Computes the rewards for every state action pair """

        rewards = np.zeros((self.n_states, self.n_actions))
        
        for s in range(self.n_states):
            for a in range(self.n_actions):
                
                if self.states[s] == 'Eaten': # The player has been eaten
                    rewards[s, a] = self.MINOTAUR_REWARD
                
                elif self.states[s] == 'Win': # The player has won
                    rewards[s, a] = self.GOAL_REWARD
                
                else:                
                    next_states = self.move(s,a)
                    #next_s = next_states[0] # The reward does not depend on the next position of the minotaur, we just consider the first one
                    #print("next_s is: ", next_s)
                    rewards[s, a] = self.STEP_REWARD
        return rewards
    
    def simulate(self, start, policy, method):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods)
            raise NameError(error)

        path = list()
        if method == 'DynProg':  
            horizon = policy.shape[1] # Deduce the horizon from the policy shape
            t = 0  # Initialize current time
            s = self.map[start] # Initialize current state 
            path.append(start) # Add the starting position in the maze to the path
            while t < horizon - 1:
                #next_s = self.__move(s, policy[s, t]) # Move to next state given the policy and the current state
                #path.append(self.states[next_s])
                #t +=1
                #s = next_s
                
                a = policy[s, t] # Move to next state given the policy and the current state
                next_states = self.move(s, a) 
                random_number = random.randint(0,len(next_states)-1)
                next_s = next_states[random_number]
                #next_s = next_states[0]
                path.append(next_s) # Add the next state to the path
                t +=1 # Update time and state for next iteration
                s = self.map[next_s]
                
                
        if method == 'ValIter':
            t = 0 # Initialize current state, next state and time
            s = self.map[start]
            # Add the starting position in the maze to the path
            path.append(start)
            while True: 
                if np.random.rand() < 1/30:
                    path.append('Poisoned') 
                    break

                a = policy[s]
                next_states = self.move(s, a)  # Move to next state given the policy and the current state
                random_number = random.randint(0,len(next_states)-1)
                next_s = next_states[random_number]
                path.append(next_s)
                if next_s == 'Win' or next_s == 'Eaten':
                    break
                # Update time for next iteration
                s = self.map[next_s]
                t +=1
                
            horizon = t   
        return [path, horizon]
    