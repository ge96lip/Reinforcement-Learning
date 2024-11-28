from typing import NamedTuple
import numpy as np
    
def calculate_moving_average(values, window_size: int = 50):
    overlap_counts = np.concatenate((np.arange(1, window_size), np.full(len(values), window_size)))
    sliding_window = np.ones(window_size)
    moving_avg = (np.convolve(values, sliding_window) / overlap_counts)[:-(window_size - 1)]
    return moving_avg

def decide_random(rng, probability):
    #rand = np.random.uniform()
    #return (rand < probability)
    temp = rng.binomial(n=1, p=probability)
    return (temp == 1)