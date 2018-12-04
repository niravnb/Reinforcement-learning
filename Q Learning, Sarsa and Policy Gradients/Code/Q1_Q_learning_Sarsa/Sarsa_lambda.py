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


pklfiles = ['SL_PuddleAQ500.pkl', 'SL_PuddleBQ500.pkl', 'SL_PuddleCQ500.pkl']
mapFiles = ["map1.txt", "map2.txt", "map3.txt"]
figuretitle = ['Sarsa lambda Puddle World Problem A',
               'Sarsa lambda Puddle World Problem B', 'Sarsa lambda Puddle World Problem C']
plotsave = ['SL_Avg_steps_A_', 'SL_Avg_steps_B_', 'SL_Avg_steps_C_']
problemis = ["A", "B", "C"]
moviefilename = ["SL_movieA", "SL_movieB", "SL_movieC"]


def plot_avg_steps(num_episodes, avg_steps, slambda):
    x = np.arange(num_episodes)
    plt.clf()
    plt.plot(x, avg_steps)
    plt.title("Sarsa lambda Average number of steps to goal for problem " +
                problemis[grids]+" lambda "+str(slambda))
    plt.xlabel("Episodes")
    plt.ylabel("Average number of steps")
    plt.xlim((0, num_episodes))
    plt.savefig("SL_Avg_steps_"+problemis[grids]+"_lambda_"+str(slambda) +
                str(num_episodes)+'.png', dpi=300)
    # plt.show()

def plot_total_return(num_episodes, total_return, slambda):
    x = np.arange(num_episodes)
    plt.clf()
    plt.plot(x, total_return)
    plt.title("Sarsa lambda Average Return per episode for problem " +
                problemis[grids]+" lambda "+str(slambda))
    plt.xlabel("Episodes")
    plt.ylabel("Average Return")
    plt.xlim((0, num_episodes))
    plt.savefig("SL_return_"+problemis[grids]+"_lambda_"+str(slambda) +
                str(num_episodes)+'.png', dpi=300)
    # plt.show()

# find optimal policy
def find_optimal_policy(Q):
    optimal_policy = defaultdict(lambda: np.zeros(1))
    for k, v in Q.items():
        if np.allclose(v, v[0]):
            optimal_policy[k] = 1
            # optimal_policy[k] = np.random.choice(4, 1, p=[0.25, 0.25, 0.25, 0.25])[0]
        else:
            optimal_policy[k] = np.argmax(v)
    return optimal_policy

# Simulate on learned optimal policy
def Simulate_sarsa(optimal_policy, num_episodes=10):
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

# Sarsa lambda algorithm implementation
def sarsa_lambda(env, num_episodes=1000, iterations=50, gamma=0.9, slambda=0.1, alpha=0.1, epsilon=0.1):
    env.reset()
    Q = defaultdict(lambda: np.zeros(env.n_actions))        
    et = defaultdict(lambda: np.zeros(env.n_actions))


    for itr in range(iterations):
        number_of_steps = np.zeros(num_episodes)
        total_return = np.zeros(num_episodes)        
        Q.clear()
        et.clear()

        policy = epsilon_greedy_policy(Q, epsilon, env.n_actions)
        figcount = 0

        for i_episode in range(num_episodes):
            dis_return = 0

            if (i_episode + 1) % 100 == 0:
                print("\nIteration: {} Episode {}/{}.".format(itr,
                                                                i_episode + 1, num_episodes))

            # Reset the environment and pick the first action
            observation = env.reset()
            action_prob = policy(observation)
            a = np.random.choice(
                [i for i in range(len(action_prob))], p=action_prob)

            for i in itertools.count():  # Till the end of episode
                # TAKE A STEP
                next_observation, reward, done, _ = env.step(a)

                env.steps = i
                dis_return += reward  # Updating return
                env.dis_return = dis_return

                if ENABLE_RENDERING:  # Rendering
                    env.render(mode='human')
                    env.figurecount = figcount
                    figcount += 1

                
                next_action_probs = policy(next_observation)
                next_action = np.random.choice(
                    np.arange(len(next_action_probs)), p=next_action_probs) # Next action
                TDError = reward + gamma * \
                    Q[next_observation][next_action] - Q[observation][a] # TD Error
                et[observation][a] += 1  # Accumulating traces

                for k, _ in Q.items():
                    for actions in range(4):
                        Q[k][actions] += alpha*TDError*et[k][actions] # Sarsa lambda Q update
                        et[k][actions] = gamma*slambda*et[k][actions] # Elegibility trace update

                if done:
                    # print("Total discounted return is :", dis_return)
                    env.dis_return = 0
                    env.steps = 0
                    break

                observation = next_observation
                a = next_action
            # print("Total steps taken is :", i)
            number_of_steps[i_episode] = i  # Updating Number of steps
            total_return[i_episode] = dis_return  # Updating return

        np.save('SL_saves/SL_avg_num_of_steps_for_problem_' +
                problemis[grids]+"_lambda_"+str(slambda)+"_itr_"+str(itr), number_of_steps)
        np.save('SL_saves/SL_total_return_per_episode_for_problem_' +
                problemis[grids]+"_lambda_"+str(slambda)+"_itr_"+str(itr), total_return)

    return Q

for grids in range(3):

    env = gym.make("GridWorld-v0")
    filename = pklfiles[grids]
    ENABLE_RECORDING = False
    ENABLE_RENDERING = False
    env.saveFile = ENABLE_RECORDING
    env.mapFile = mapFiles[grids]

    if problemis[grids] == "C":
        env.westerly_wind = False

    num_episodes = 30
    iterations = 25

    env.reset()
    env.first_time = False

    sarsa_lambda_values = [0, 0.3, 0.5, 0.9, 0.99, 1.0]

    for l in range(len(sarsa_lambda_values)):
        env.figtitle = figuretitle[grids]+" lambda "+str(sarsa_lambda_values[l])
        

        Q = sarsa_lambda(env, num_episodes=num_episodes, iterations=iterations,
                gamma=0.9, slambda=sarsa_lambda_values[l] ,alpha=0.1, epsilon=0.1)

        if ENABLE_RECORDING:
            movie.CreateMovie(moviefilename[grids], 5)

        # Plotting
        avg_number_of_steps = np.zeros(num_episodes)
        total_return_per_episode = np.zeros(num_episodes)

        for i in range(iterations):
            avg_number_of_steps += np.load('SL_saves/SL_avg_num_of_steps_for_problem_' +
                                           problemis[grids]+"_lambda_"+str(sarsa_lambda_values[l])+"_itr_"+str(i)+".npy")
            total_return_per_episode += np.load('SL_saves/SL_total_return_per_episode_for_problem_' +
                                                problemis[grids]+"_lambda_"+str(sarsa_lambda_values[l])+"_itr_"+str(i)+".npy")

        avg_number_of_steps /= iterations
        total_return_per_episode /= iterations
        np.save('SL_saves/SL_avg_num_of_steps_for_problem_' +
                problemis[grids]+"_lambda_"+str(sarsa_lambda_values[l]), avg_number_of_steps)
        np.save('SL_saves/SL_total_return_per_episode_for_problem_' +
                problemis[grids]+"_lambda_"+str(sarsa_lambda_values[l]), total_return_per_episode)

        # plot_avg_steps(num_episodes, avg_number_of_steps,
        #                sarsa_lambda_values[l])
        # plot_total_return(
        #     num_episodes, total_return_per_episode, sarsa_lambda_values[l])

        # Finding optimal policy based on Q
        optimal_policy = find_optimal_policy(Q)
        env.optimal_policy = optimal_policy

        # Drawing arrows of optimal policy
        env.draw_arrows = True
        env.render(mode='human')

        # # For recording video of simulation
        # env.draw_arrows = False
        ENABLE_RENDERING = True
        ENABLE_RECORDING = True

        if ENABLE_RECORDING:
            Simulate_sarsa(optimal_policy, 5)
            movie.CreateMovie(
                moviefilename[grids]+"_lambda_"+str(sarsa_lambda_values[l]), 5)

# Plotting all 3 problems graph togeather
import Sarsa_lambda_plot
