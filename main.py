from grid_world import *
from qlearning_train import qlearning_train
from tabular_epsilon_greedy_policy import tabular_epsilon_greedy_policy
from QLearning import QLearning 
import numpy as np
import matplotlib.pyplot as plt

def main():

    env = GridWorld(MAP4)
    qlearning_policy = QLearning(env.get_num_states(), env.get_num_actions())

    num_episodes = 1000
    eps = 0.1 
    qlearnt = qlearning_train(env, qlearning_policy, num_episodes, eps)

    state = env.reset()
    env.print()
    done = False
    eps_test = 0.0
    while not done:
        input("press enter:")
        action = tabular_epsilon_greedy_policy(qlearnt.Q, eps_test, state)
        state, reward, done = env.step(action)
        env.print()

    Qmatrix = np.max(qlearnt.Q, axis = 1)
    Qmatrix = Qmatrix.reshape(6,13)
    plt.imshow(Qmatrix)
    plt.colorbar()
    plt.title("Q Value Matrix plot trained for 100 episodes (MAP 4)")
    plt.show()
    
if __name__ == "__main__":
    main()




