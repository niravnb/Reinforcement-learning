# Intra option Q Learning

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
            best_action = np.random.choice(
                6, 1, p=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6])[0]
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
    Q[env.terminal_state] = np.ones(env.n_actions)
    Q_hat = defaultdict(lambda: np.zeros(env.n_actions))

    number_of_steps = np.zeros(num_episodes)
    total_return = np.zeros(num_episodes)
    total_return_steps = np.zeros(num_episodes)


    for itr in range(iterations):

        Q.clear()
        Q_hat.clear()
        Q[env.terminal_state] = np.ones(env.n_actions)
        Q_hat[env.terminal_state] = np.ones(env.n_actions)

        policy = epsilon_greedy_policy(Q, epsilon, env.n_actions)
        figcount = 0

        for i_episode in range(num_episodes):
            dis_return = 0
            steps_per_episode = 0

            if (i_episode + 1) % 100 == 0:
                print("\nIteration: {} Episode {}/{}.".format(itr,
                                                          i_episode + 1, num_episodes))
            observation = env.reset()  # Start state
            if Question == 3: # Setting start state according to Question
                env.state = 0
                env.start_state = 0
            else:
                env.state = 90
                env.start_state = 90

            for i in itertools.count():  # Till the end of episode
                action_prob = policy(observation)
                a = np.random.choice(
                    [i for i in range(len(action_prob))], p=action_prob)  # Action selection

                env.options_Q = Q # passing Q values to environment
                env.options_Q_hat = Q_hat  # passing Q_hat values to environment
                if a > 3:
                    # passing option optimal policy to environment
                    env.options_poilcy = dill.load(open(option_poilcy[a-4], 'rb'))
                next_observation, reward, done, _ = env.step(a)  # Taking option and also updating Q and Q_hat values accordingly
                Q = env.options_Q # Replacing updated Q values
                Q_hat = env.options_Q_hat # Replacing updated Q_hat values
                


                env.steps = i
                dis_return += reward*gamma**i  # Updating return
                env.dis_return = dis_return
                steps_per_episode += env.options_length

                if ENABLE_RENDERING:  # Rendering
                    env.render(mode='human')
                    # env.figurecount = figcount
                    # figcount += 1

                if done:
                    print("Total discounted return is :",
                          gamma**steps_per_episode)
                    env.dis_return = 0
                    env.steps = 0
                    break

                observation = next_observation
            print("Total steps taken is :", steps_per_episode)
            # Updating Number of steps
            number_of_steps[i_episode] += steps_per_episode
            # Updating return
            total_return[i_episode] += dis_return
            total_return_steps[i_episode] += gamma**steps_per_episode

    number_of_steps /= iterations
    total_return /= iterations  # Updating return
    total_return_steps /= iterations

    return Q, number_of_steps, total_return, total_return_steps


def plot_avg_steps(num_episodes, avg_steps):
    x = np.arange(num_episodes)
    plt.clf()
    plt.plot(x, avg_steps)
    plt.title(
        "SMDP Q-Learning Average steps per episode for goal " + problemis[pb])
    plt.xlabel("Episodes")
    plt.ylabel("Average steps per episode")
    # plt.xlim((0, num_episodes))
    plt.ylim((0, max(avg_steps)+20))
    plt.savefig(figure_save[0] + problemis[pb] +"_episode_" + str(num_episodes)+'.png', dpi=300)
    plt.xscale('log', basex=10)
    plt.savefig(figure_save[1] + problemis[pb] + "_episode_" +str(num_episodes)+'.png', dpi=300)
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
    plt.savefig(figure_save[2] + problemis[pb] + "_episodes_" +str(num_episodes)+'.png', dpi=300)
    # plt.show()


def plot_total_return_steps(num_episodes, total_return):
    x = np.arange(num_episodes)
    plt.clf()
    plt.plot(x, total_return)
    plt.title(
        "SMDP Q-Learning Average Return per episode for goal " + problemis[pb])
    plt.xlabel("Episodes")
    plt.ylabel("Return")
    plt.xlim((0, num_episodes))
    plt.savefig(figure_save[2] + problemis[pb] +
                "steps_episode_" + str(num_episodes)+'.png', dpi=300)
    # plt.show()


# Setting some values for Q3 & Q4 where start state is centre of room 4
option_poilcy = ["P_O1.pkl", "P_O2.pkl"]
Question = 3
for Question in [3, 4]:

    if Question == 3:
        mapFiles = ["map1.txt", "map2.txt"]
        problemis = ["G1", "G2"]
        filesave = ["Q3_G1.pkl", "Q3_G2.pkl"]
        figtitle = ["Q3_G1", "Q3_G2"]
        filesave_option = [["Q3_G1_O1.pkl", "Q3_G1_O2.pkl"],
                        ["Q3_G2_O1.pkl", "Q3_G2_O2.pkl"]]
        np_saves = ['Q_saves/Q3_avg_num_of_steps_', 'Q_saves/Q3_avg_rewards_']
        figure_save = ['Q3_Avg_steps_Prob_',
                    'Q3_Avg_steps_cut_Prob_', 'Q3_Return_Prob_']

    else:
        mapFiles = ["map3.txt", "map4.txt"]
        problemis = ["G1", "G2"]
        filesave = ["Q4_G1.pkl", "Q4_G2.pkl"]
        filesave_option = [["Q4_G1_O1.pkl", "Q4_G1_O2.pkl"],
                        ["Q4_G2_O1.pkl", "Q4_G2_O2.pkl"]]
        figtitle = ["Q4_G1", "Q4_G2"]
        np_saves = ['Q_saves/Q4_avg_num_of_steps_', 'Q_saves/Q4_avg_rewards_']
        figure_save = ['Q4_Avg_steps_Prob_',
                    'Q4_Avg_steps_cut_Prob_', 'Q4_Return_Prob_']

    alpha = [0.12, 0.25]

    num_episodes = 1000
    iterations = 10
    ENABLE_RENDERING = False
    load_option_Q_values = False
    save_option_Q_values = True
    lines_circles = 1
    plot_graphs = 1


    for pb in range(2):
        env = gym.make("GridWorld-v0")
        env.mapFile = mapFiles[pb]
        env.saveFile = False
        env.reset()
        env.first_time = False
        # env.draw_circles = True
        # env.figtitle = figtitle[pb]
        env.intra_options = True

        Q, number_of_steps, total_return, total_return_steps = q_learning(env, num_episodes=num_episodes,
                                                    iterations=iterations, gamma=0.9, alpha=alpha[pb], epsilon=0.1)

        dill.dump(Q, open(filesave[pb], 'wb'))

        if plot_graphs:
            np.save(np_saves[0]+problemis[pb], number_of_steps)
            np.save(np_saves[1]+problemis[pb], total_return)
            np.save(np_saves[1]+'_steps_'+problemis[pb], total_return_steps)

            number_of_steps = np.load(np_saves[0]+problemis[pb]+".npy")
            total_return = np.load(np_saves[1]+problemis[pb] + ".npy")
            total_return_steps = np.load(
                np_saves[1]+'_steps_'+problemis[pb] + ".npy")

            plot_avg_steps(num_episodes, number_of_steps)
            plot_total_return(num_episodes, total_return)
            plot_total_return_steps(num_episodes, total_return_steps)


        # # Finding V values
        Q = dill.load(open(filesave[pb], 'rb'))
        if lines_circles:
            env.my_state = env.start_state
            V = np.zeros(104)
            optimal_policy = np.zeros(104)
            optimal_policy += -1

            for k,v in Q.items():
                if k < 104:
                    if not(np.allclose(v, v[0])):
                        optimal_policy[k] = np.argmax(v)
                    V[k] = np.max(v)
                    

            V = preprocessing.minmax_scale(V, feature_range=(0, 0.5)) 
            env.V = V # passing V values to environment 
            env.optimal_policy = optimal_policy # passing optimal policy to environment
            env.draw_circles = True
            env.draw_arrows = False
            env.figtitle = figtitle[pb]+'_'+str(num_episodes)
            env.render(mode='human')

            env.draw_circles = False
            env.draw_arrows = True
            env.render(mode='human')
