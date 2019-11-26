import numpy as np

#Q learning 

class QLearning(object):
    # Initialize a Qlearning object
    # alpha is the "learning_rate"
    def __init__(self, num_states, num_actions, alpha=0.9):
         # initialize Q values to something
        self.Q = np.zeros((num_states, num_actions))
        self.alpha = alpha
        self.gamma = 0.9
    
    # updates the Q value table
    # with a (state, action, reward, next_state) from one step in the environment
    # done is a bool indicating if the episode terminated at this step
    # you can return anything from this function to help with plotting/debugging
    def update(self, state, action, reward, next_state, done):
        max_next_Q = max([self.Q[next_state, next_action] for next_action in range(4)])
        if (done and reward<=-100):
            max_next_Q = 0.0

        self.Q[state, action] = self.Q[state, action] + self.alpha*(reward + self.gamma*max_next_Q - self.Q[state, action])
