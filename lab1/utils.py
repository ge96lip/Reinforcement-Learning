from typing import NamedTuple
import numpy as np

class Experience(NamedTuple):
    episode: int
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    
def running_average(data, window_length: int = 50):
    overlap_length = np.concatenate((np.arange(1, window_length), window_length * np.ones(len(data))))
    window = np.ones(window_length)
    averages = (np.convolve(data, window) / overlap_length)[:-(window_length-1)]
    assert len(averages) == len(data)
    return averages

def decide_random(rng, probability):
    #rand = np.random.uniform()
    #return (rand < probability)
    temp = rng.binomial(n=1, p=probability)
    return (temp == 1)