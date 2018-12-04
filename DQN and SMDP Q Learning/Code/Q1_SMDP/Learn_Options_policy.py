# For learning options optimal policy


import numpy as np
import gym
import gym_gridworld
import itertools
from collections import defaultdict
import sys
from gym import wrappers
import dill
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn import preprocessing


# Creates epsilon greedy policy
def epsilon_greedy_policy(Q, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA

        # Taking random action if all are same
        if np.allclose(Q[observation], Q[observation][0]):
            best_action = np.random.choice(4, 1, p=[0.25, 0.25, 0.25, 0.25])[0]
        else:
            best_action = np.argmax(Q[observation])

        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

# Q Learning algorithm implementation
def q_learning(env, num_episodes=500, iterations=50, gamma=0.9, alpha=0.1, epsilon=0.1):
    env.reset()
    env.epsilon = epsilon
    env.gamma = gamma
    env.alpha = alpha

    Q = defaultdict(lambda: np.zeros(env.n_actions))

    number_of_steps = np.zeros(num_episodes)
    total_return = np.zeros(num_episodes)

    for itr in range(iterations):

        Q.clear()

        policy = epsilon_greedy_policy(Q, epsilon, env.n_actions)
        figcount = 0
        options_goal = [[25, 56, 77, 103], [103, 25, 56, 77]]
        options_start_states = [[np.append(np.arange(0, 25, 1), 103), np.arange(25, 56, 1), np.arange(56, 77, 1), np.arange(77, 103, 1)],
                                [np.arange(0, 26, 1), np.arange(
                                    26, 57, 1), np.arange(57, 78, 1), np.arange(78, 104, 1)]]

        for op in np.arange(4):

            for i_episode in range(num_episodes):
                dis_return = 0
                steps_per_episode = 0

                if (i_episode + 1) % 100 == 0:
                    print("\nIteration: {} Episode {}/{}.".format(itr,
                                                              i_episode + 1, num_episodes))

                observation = env.reset()  # Start state
                env.state = np.random.choice(options_start_states[Option][op])
                env.start_state = env.state
                env.terminal_state = options_goal[Option][op]
                observation = env.state

                for i in itertools.count():  # Till the end of episode
                    action_prob = policy(observation)
                    a = np.random.choice(
                        [i for i in range(len(action_prob))], p=action_prob)  # Action selection

                    
                    next_observation, reward, done, _ = env.step(a)  # Taking action

                    env.steps = i
                    dis_return += reward*gamma**i  # Updating return
                    env.dis_return = dis_return
                    steps_per_episode += env.options_length

                    if ENABLE_RENDERING:  # Rendering
                        env.render(mode='human')
                        # env.figurecount = figcount
                        # figcount += 1

                    if (next_observation not in options_start_states[Option][op]) and (next_observation != options_goal[Option][op]) and not done:
                        # print("Outside the room")
                        break

                    # Finding next best action from next state
                    best_next_a = np.argmax(Q[next_observation])
                    # Q Learning update
                    Q[observation][a] += alpha*(reward + gamma
                                                * Q[next_observation][best_next_a] - Q[observation][a])

                    if done:
                        print("Total discounted return is :", dis_return)
                        print("Total steps taken is :", i)
                        env.dis_return = 0
                        env.steps = 0
                        break

                    


                    observation = next_observation
            # print("Total steps taken is :", steps_per_episode)
            # Updating Number of steps
            number_of_steps[i_episode] += steps_per_episode
            # Updating return
            total_return[i_episode] += gamma**steps_per_episode

    number_of_steps /= iterations
    total_return /= iterations  # Updating return

    return Q, number_of_steps, total_return


def plot_avg_steps(num_episodes, avg_steps):
    x = np.arange(num_episodes)
    plt.clf()
    plt.plot(x, avg_steps)
    plt.title(
        "SMDP Q-Learning Average steps per episode for goal " + problemis[pb])
    plt.xlabel("Episodes")
    plt.ylabel("Average steps per episode")
    plt.xlim((0, num_episodes))
    plt.savefig(figure_save[0] + problemis[pb] +
                "_episode_" + str(num_episodes)+'.png', dpi=300)
    plt.ylim((0, 1000))
    plt.savefig(figure_save[1] + problemis[pb] +
                "_episode_" + str(num_episodes)+'.png', dpi=300)
    # plt.show()


def plot_total_return(num_episodes, total_return):
    x = np.arange(num_episodes)
    plt.clf()
    plt.plot(x, total_return)
    plt.title(
        "SMDP Q-Learning Average Return per episode for goal " + problemis[pb])
    plt.xlabel("Episodes")
    plt.ylabel("Return")
    plt.xlim((0, num_episodes))
    plt.savefig(figure_save[2] + problemis[pb] +
                "_episodes_" + str(num_episodes)+'.png', dpi=300)
    # plt.show()


# Setting some values for Option 1 & Option 2
Option = 1
if Option == 1:
    mapFiles = ["map1.txt", "map2.txt"]
    problemis = ["G1", "G2"]
    filesave = ["O1.pkl", "O2.pkl"]
    figtitle = ["O1", "O2"]
    filesave_option = [["V_O1.pkl", "P_O1.pkl"],
                       ["V_O2.pkl", "P_O2.pkl"]]
    np_saves = ['Q_saves/Q1_avg_num_of_steps_',
                'Q_saves/Q1_avg_num_of_steps_']
    figure_save = ['Q1_Avg_steps_Prob_',
                   'Q1_Avg_steps_cut_Prob_', 'Q1_Return_Prob_']

else:
    mapFiles = ["map3.txt", "map4.txt"]
    problemis = ["G1", "G2"]
    filesave = ["O1.pkl", "O2.pkl"]
    filesave_option = [["Q2_G1_O1.pkl", "Q2_G1_O2.pkl"],
                       ["Q2_G2_O1.pkl", "Q2_G2_O2.pkl"]]
    figtitle = ["O1", "O2"]
    np_saves = ['Q_saves/Q2_avg_num_of_steps_',
                'Q_saves/Q2_avg_num_of_steps_']
    figure_save = ['Q2_Avg_steps_Prob_',
                   'Q2_Avg_steps_cut_Prob_', 'Q2_Return_Prob_']


alpha = [0.12, 0.25] # As described in paper

num_episodes = 1000
iterations = 1
ENABLE_RENDERING = False
load_option_Q_values = False
save_option_Q_values = True

for pb in range(2):
    env = gym.make("GridWorld-v0")
    env.mapFile = mapFiles[pb]
    env.saveFile = False
    env.action_space = gym.spaces.Discrete(4)  # spaces.Discrete(4)
    env.n_actions = 4
    env.reset()
    env.first_time = False
    # env.draw_circles = True
    env.figtitle = figtitle[pb]
    Option = pb

    Q, number_of_steps, total_return = q_learning(env, num_episodes=num_episodes,
                                                  iterations=iterations, gamma=0.9, alpha=alpha[pb], epsilon=0.3)

    dill.dump(Q, open(filesave[pb], 'wb'))

    # np.save(np_saves[pb]+problemis[pb], number_of_steps)
    # np.save(np_saves[pb]+problemis[pb], total_return)

    # plot_avg_steps(num_episodes, number_of_steps)
    # plot_total_return(num_episodes, total_return)

    dill.dump(Q, open(filesave[pb], 'wb'))

    # Finding V values
    Q = dill.load(open(filesave[pb], 'rb'))
    V = np.zeros(len(Q)) # For storing V values
    optimal_policy = np.zeros(len(Q),dtype='int8') # For storing optimal policy

    for j in range(len(Q)):
        V[j] = np.max(Q[j])
        optimal_policy[j] = np.argmax(Q[j])

    V = preprocessing.minmax_scale(V, feature_range=(0, 0.5))
    for g in [25, 56, 77, 103]:
        V[g] = 0
    if pb == 0: # Setting optimal action for hallway states according to option
        optimal_policy[25] = 1
        optimal_policy[56] = 2
        optimal_policy[77] = 3
        optimal_policy[103] = 0
    else:
        optimal_policy[25] = 3
        optimal_policy[56] = 0
        optimal_policy[77] = 1
        optimal_policy[103] = 2

    dill.dump(V, open(filesave_option[pb][0], 'wb'))
    dill.dump(optimal_policy, open(filesave_option[pb][1], 'wb'))

    env.V = V # passing value function to environment 
    env.optimal_policy = optimal_policy # passing optimal policy to environment
    env.terminal_state = 1000
    env.draw_circles = True
    env.draw_arrows = False
    env.figtitle = figtitle[pb]+'_'+str(num_episodes)
    env.render(mode='human')

    env.draw_circles = False
    env.draw_arrows = True
    env.render(mode='human')
