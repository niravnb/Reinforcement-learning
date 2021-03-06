# Taken some reference & idea from below github soucre: Some GridWorld environments for OpenAI Gym
# https://github.com/opocaj92/GridWorldEnvs 

import numpy as np
import gym
import gym_gridworld
import itertools
from collections import defaultdict
import sys
from gym import wrappers
import dill
import CreateMovie as movie
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


pklfiles = ['QL_PuddleAQ500.pkl', 'QL_PuddleBQ500.pkl', 'QL_PuddleCQ500.pkl']
mapFiles = ["map1.txt", "map2.txt", "map3.txt"]
figuretitle = ['Q Learning Puddle World Problem A',
               'Q Learning Puddle World Problem B', 'Q Learning Puddle World Problem C']
plotsave = ['QL_Avg_steps_A_', 'QL_Avg_steps_B_', 'QL_Avg_steps_C_']
problemis = ["A", "B", "C"]
moviefilename = ["QL_movieA", "QL_movieB", "QL_movieC"]


# Creates epsilon greedy policy
def epsilon_greedy_policy(Q, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA

        # Taking random action if all are same
        if np.allclose(Q[observation], Q[observation][0]):
            best_action = np.random.choice(
                4, 1, p=[0.25, 0.25, 0.25, 0.25])[0]
        else:
            best_action = np.argmax(Q[observation])

        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

# Q Learning algorithm implementation
def q_learning(env, num_episodes=500, iterations=50, gamma=0.9, alpha=0.1, epsilon=0.1):
    env.reset()

    # env.render(mode='human')
    Q = defaultdict(lambda: np.zeros(env.n_actions))

    for itr in range(iterations):
        number_of_steps = np.zeros(num_episodes)
        total_return = np.zeros(num_episodes)

        Q.clear()

        policy = epsilon_greedy_policy(Q, epsilon, env.n_actions)
        figcount = 0

        for i_episode in range(num_episodes):
            dis_return = 0

            if (i_episode + 1) % 100 == 0:
                print("\nIteration: {} Episode {}/{}.".format(itr,
                                                                i_episode + 1, num_episodes))
            observation = env.reset()  # Start state

            for i in itertools.count():  # Till the end of episode
                action_prob = policy(observation)
                a = np.random.choice(
                    [i for i in range(len(action_prob))], p=action_prob)  # Action selection

                next_observation, reward, done, _ = env.step(
                    a)  # Taking action

                env.steps = i
                dis_return += reward  # Updating return
                env.dis_return = dis_return

                if ENABLE_RENDERING:  # Rendering
                    env.render(mode='human')
                    env.figurecount = figcount
                    figcount += 1

                # Finding next best action from next state
                best_next_a = np.argmax(Q[next_observation])
                Q[observation][a] += alpha *(reward + gamma*Q[next_observation][best_next_a] - Q[observation][a])  # Q Learning update

                if done:
                    # print("Total discounted return is :", dis_return)
                    env.dis_return = 0
                    env.steps = 0
                    break

                observation = next_observation
            # print("Total steps taken is :", i)
            number_of_steps[i_episode] = i  # Updating Number of steps
            total_return[i_episode] = dis_return  # Updating return


        np.save('Q_saves/Q_avg_num_of_steps_for_problem_' +problemis[grids]+"_itr_"+str(itr), number_of_steps)
        np.save('Q_saves/Q_total_return_per_episode_for_problem_' +problemis[grids]+"_itr_"+str(itr), total_return)

    return Q


def plot_avg_steps(num_episodes, avg_steps):
    x = np.arange(num_episodes)
    plt.clf()
    plt.plot(x, avg_steps)
    plt.title("Q Learning Average number of steps to goal for problem " +
              problemis[grids])
    plt.xlabel("Episodes")
    plt.ylabel("Average number of steps")
    plt.xlim((0, num_episodes))
    plt.savefig("QL_Avg_steps_"+problemis[grids] +
                str(num_episodes)+'.png', dpi=300)
    # plt.show()


def plot_total_return(num_episodes, total_return):
    x = np.arange(num_episodes)
    plt.clf()
    plt.plot(x, total_return)
    plt.title("Q Learning Return per episode for problem " +
              problemis[grids])
    plt.xlabel("Episodes")
    plt.ylabel("Return")
    plt.xlim((0, num_episodes))
    plt.savefig("QL_Return_"+problemis[grids] +
                str(num_episodes)+'.png', dpi=300)
    # plt.show()

# find optimal policy
def find_optimal_policy(Q):
    optimal_policy = defaultdict(lambda: np.zeros(1))
    for k, v in Q.items():
        if np.allclose(v, v[0]):
            # np.random.choice(4, 1, p=[0.25, 0.25, 0.25, 0.25])[0]
            optimal_policy[k] = 1
        else:
            optimal_policy[k] = np.argmax(v)
    return optimal_policy

# Simulate on learned optimal policy
def Simulate_Q_Learning(optimal_policy, num_episodes=10):
    figcount = 0
    env.saveFile = ENABLE_RECORDING

    for i in range(num_episodes):
        dis_return = 0
        observation = env.reset()

        for i in itertools.count():

            a = int(optimal_policy[observation])
            next_observation, reward, done, _ = env.step(a)

            env.steps = i
            dis_return += reward
            env.dis_return = dis_return

            if ENABLE_RENDERING:
                env.render(mode='human')
                env.figurecount = figcount
                figcount += 1

            if done:
                env.dis_return = 0
                env.steps = 0
                break

            observation = next_observation



for grids in range(3):

    env = gym.make("GridWorld-v0")
    filename = pklfiles[grids]
    ENABLE_RECORDING = False
    ENABLE_RENDERING = False
    env.saveFile = ENABLE_RECORDING
    env.mapFile = mapFiles[grids]
    env.figtitle = figuretitle[grids]

    if problemis[grids] == "C":
        env.westerly_wind = False

    num_episodes = 500
    iterations = 50

    env.reset()
    env.first_time = False

    Q = q_learning(env, num_episodes=num_episodes, iterations=iterations,gamma=0.9, alpha=0.1, epsilon=0.1)




    if ENABLE_RECORDING:
        movie.CreateMovie(moviefilename[grids], 5)


    # Plotting
    avg_number_of_steps = np.zeros(num_episodes)
    total_return_per_episode = np.zeros(num_episodes)

    for i in range(iterations):
        avg_number_of_steps += np.load('Q_saves/Q_avg_num_of_steps_for_problem_' +
                                       problemis[grids]+"_itr_"+str(i)+".npy")
        total_return_per_episode += np.load('Q_saves/Q_total_return_per_episode_for_problem_' +
                                            problemis[grids]+"_itr_"+str(i)+".npy")

    avg_number_of_steps /= iterations
    total_return_per_episode /= iterations
    np.save('Q_saves/Q_avg_num_of_steps_for_problem_' +
            problemis[grids], avg_number_of_steps)
    np.save('Q_saves/Q_total_return_per_episode_for_problem_' +
            problemis[grids], total_return_per_episode)


    plot_avg_steps(num_episodes, avg_number_of_steps)
    plot_total_return(num_episodes, total_return_per_episode)

    # Finding optimal policy based on Q
    optimal_policy = find_optimal_policy(Q)
    env.optimal_policy = optimal_policy

    dill.dump_session(filename)
    
    # env.saveFile = True
    # Drawing arrows of optimal policy
    env.draw_arrows = True
    env.render(mode='human')

    # For recording video of simulation
    # env.draw_arrows = False
    ENABLE_RENDERING = True
    ENABLE_RECORDING = True

    if ENABLE_RECORDING:
        Simulate_Q_Learning(optimal_policy, 10)
        movie.CreateMovie(moviefilename[grids], 5)

# Plotting all 3 problems graph togeather
import Q_learning_plot
