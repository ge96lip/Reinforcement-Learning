import random 
import numpy as np
import time
from IPython import display
import matplotlib.pyplot as plt
import scipy.stats as stats
from maze_current import Maze
from methods import dynamic_programming, value_iteration

methods = ['DynProg', 'ValIter']
# Some colours
LIGHT_RED    = '#FFC4CC'
LIGHT_GREEN  = '#95FD99'
BLACK        = '#000000'
WHITE        = '#FFFFFF'
LIGHT_PURPLE = '#E8D0FF'
LIGHT_ORANGE = '#FAE0C3'


def survival_rate_horizon(env, start, policy):
    states = set()
    num_states = dict()
    prob_states = dict()

    states.add(start)
    num_states[start] = 1
    prob_states[start] = 1
    
    horizon = policy.shape[1]

    t = 0
    while t < horizon:
        next_states = set()
        next_num = dict()
        next_prob = dict()
        next_total = 0

        #print("States at {}: unique {}, total {}".format(t, len(states), total_num_states))
        for s in states:
            
            s_count = num_states[s]
            s_prob = prob_states[s]
            s = env.map[s]
            action = policy[s, t]
            new_states = env.move(s, action)

            each_ns_prob = s_prob / len(new_states)

            for ns in new_states:
                next_num[ns] = next_num.get(ns, 0) + s_count
                next_prob[ns] = next_prob.get(ns, 0) + each_ns_prob

                next_states.add(ns)
                next_total += s_count

        # nothing moving, break
        if next_states == states:
            break

        states = next_states
        num_states = next_num
        prob_states = next_prob
        t += 1

    won = 0
    dead = 0
    for state, prob in prob_states.items():
        if state == 'Eaten':
            dead += prob
        elif state == 'Win':
            won += prob

    print("T = {}, win {:06.2%}, dead {:06.2%}".format(horizon-1, won, dead))

    return won

def survival_rates_dynprog(maze, T_range, minotaur_stay = False):
    """
    possible kwargs are minotaur_stay, avoid_minotaur, and min_path
    """
    env = Maze(maze, minotaur_stay)

    sr = []
    policies = {}

    start = ((0,0), (6,5))

    for T in T_range:
        V, policy = dynamic_programming(env, T)
        rate = survival_rate_horizon(env, start, policy)
        sr.append(rate)
        policies[T] = policy 
    return sr, policies


def survival_rate_valiter(maze, mean_lifetime, minotaur_stay = False):
   
    env = Maze(maze, allow_minotaur_stay=minotaur_stay)

    # Discount Factor
    gamma = 1 - (1/mean_lifetime)
    # Accuracy treshold 
    epsilon = 0.0001
    V, policy = value_iteration(env, gamma, epsilon)

    start = ((0,0), (6,5))
    
    won = 0
    total_path_len = 0
    simulation_num = 10000
    for i in range(simulation_num):
        path = env.simulate(start, policy, "ValIter")
        last_state = path[0][-1]
        #print("mapped: ", self.map[last_state])
        if last_state == 'Win':
            won += 1
        total_path_len += len(path[0])
    
    rate = won/simulation_num
    avg_path_len = total_path_len/simulation_num

    print("Survived {:%}".format(rate))
    print("Avg. lifetime ", avg_path_len-1)

    return rate
    
def animate_solution(maze, path):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -1: LIGHT_RED, -2: LIGHT_PURPLE}
    
    rows, cols = maze.shape # Size of the maze
    fig = plt.figure(1, figsize=(cols, rows)) # Create figure of the size of the maze

    # Remove the axis ticks and add title
    ax = plt.gca()
    ax.set_title('Policy simulation')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)]

    # Create a table to color
    grid = plt.table(
        cellText = None, 
        cellColours = colored_maze, 
        cellLoc = 'center', 
        loc = (0,0), 
        edges = 'closed'
    )
    
    # Modify the height and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)
    for i in range(0, len(path)):
        if i < len(path): 
            if(type(path[i][0]) != str):
                grid.get_celld()[(path[i][0])].get_text().set_text('Player')
                grid.get_celld()[(path[i][1])].get_text().set_text('Minotaur')
        if path[i-1] != 'Eaten' and path[i-1] != 'Win':
            grid.get_celld()[(path[i-1][0])].set_facecolor(col_map[maze[path[i-1][0]]])
            grid.get_celld()[(path[i-1][1])].set_facecolor(col_map[maze[path[i-1][1]]])
            if i < len(path)-1:
                if(type(path[i][0]) != str):
                    grid.get_celld()[(path[i-1][0])].get_text().set_text('')
                    grid.get_celld()[(path[i-1][1])].get_text().set_text('')
        if path[i] != 'Eaten' and path[i] != 'Win':
            grid.get_celld()[(path[i][0])].set_facecolor(col_map[-2]) # Position of the player
            grid.get_celld()[(path[i][1])].set_facecolor(col_map[-1]) # Position of the minotaur
        display.display(fig)
        time.sleep(0.1)
        display.clear_output(wait = True)
        if i < len(path)-1:
            if(type(path[i][0]) != str):
                grid.get_celld()[(path[i][0])].get_text().set_text('')
                grid.get_celld()[(path[i][1])].get_text().set_text('')
