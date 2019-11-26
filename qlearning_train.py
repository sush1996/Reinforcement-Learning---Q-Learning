import matplotlib.pyplot as plt
from tabular_epsilon_greedy_policy import tabular_epsilon_greedy_policy
import numpy as np

#Training thr Q network

def qlearning_train(env, qlearning, num_episodes, eps):

    epsilon_rewards_plot = []
    Qval_start_plot = [0]*num_episodes
    count2 = 0
    for i in range(num_episodes):
        
        curr_state = env.reset()
        sum_of_rewards = 0
        steps = 0
        done = False
        
        Qval_start_plot[i] = max([qlearning.Q[curr_state, a] for a in range(4)]) 
        
        while not done:
            action = tabular_epsilon_greedy_policy(qlearning.Q, eps, curr_state)
            next_state, reward, done = env.step(action)
            qlearning.update(curr_state, action, reward, next_state, done)
            curr_state = next_state
            sum_of_rewards+=reward
            steps = steps+1
        
        epsilon_rewards_plot+= [sum_of_rewards]
        
        if sum_of_rewards>0:
            count2+=1

        print("episode:",i, "steps", steps, "sum_of_rewards", sum_of_rewards)            
    
    Qval_start_plot = Qval_start_plot + [max([qlearning.Q[0, a] for a in range(4)])]
    OptimalQ_start_plot = [Qval_start_plot[-1]]*(num_episodes+1)
    
    plt.plot(Qval_start_plot, label = "Q value for state 0 at each episode")
    plt.plot(OptimalQ_start_plot, label = "Optimal Q value for state 0")
    plt.title("Convergence of Q values to its optimal value")
    plt.xlabel("Episodes")
    plt.ylabel("Q values")
    plt.legend()
    plt.show()
    
    plt.plot(epsilon_rewards_plot)
    plt.title("Total Reward for each Episode for MAP 4 (Epsilon Greedy Policy, eps = 0.1)")
    plt.xlabel("Number of Episodes")
    plt.ylabel("Total Sum of Rewards for each Episode")
    plt.show()
    
    print("Number of Times Reaching the Goal using Epsilon Greedy Policy:", count2)
    
    return qlearning
