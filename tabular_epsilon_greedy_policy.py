import numpy as np
import random

# implementing epsilon greedy policy

def tabular_epsilon_greedy_policy(Q, eps, state):

    if random.uniform(0.0, 1.0) < eps:
        action = np.random.choice(4,1)
        return action
    else:
        action = np.argmax(Q[state])
        return action 
