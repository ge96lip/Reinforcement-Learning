

def decide_random(rng, probability):
    #rand = np.random.uniform()
    #return (rand < probability)
    temp = rng.binomial(n=1, p=probability)
    return (temp == 1)