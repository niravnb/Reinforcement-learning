
import numpy as np
import gym
import gym_gridworld
import itertools
from collections import defaultdict
import sys
from gym import wrappers
import dill
# import CreateMovie as movie
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn import preprocessing

num_episodes = 1000
avg_steps_labels = ['SMDP: Room 1 Top left', 'SMDP: Room 4 Center',
                    'Intra-Option: Room 1 Top left', 'Intra-Option: Room 4 center']
avg_steps_loads_G1 = ['Q_saves/Q1_avg_num_of_steps_G1.npy', 'Q_saves/Q2_avg_num_of_steps_G1.npy',
                   'Q_saves/Q3_avg_num_of_steps_G1.npy', 'Q_saves/Q4_avg_num_of_steps_G1.npy']
avg_steps_loads_G2 = ['Q_saves/Q1_avg_num_of_steps_G2.npy', 'Q_saves/Q2_avg_num_of_steps_G2.npy',
                   'Q_saves/Q3_avg_num_of_steps_G2.npy', 'Q_saves/Q4_avg_num_of_steps_G2.npy']


avg_return_load_G1 = ['Q_saves/Q1_avg_return_G1.npy', 'Q_saves/Q2_avg_return_G1.npy',
                      'Q_saves/Q3_avg_rewards_G1.npy', 'Q_saves/Q4_avg_rewards_G1.npy']
avg_return_load_G2 = ['Q_saves/Q1_avg_return_G2.npy', 'Q_saves/Q2_avg_return_G2.npy',
                      'Q_saves/Q3_avg_rewards_G2.npy', 'Q_saves/Q4_avg_rewards_G2.npy']

avg_return_steps_load_G1 = ['Q_saves/Q1_avg_return__steps_G1.npy', 'Q_saves/Q2_avg_return__steps_G1.npy',
                            'Q_saves/Q3_avg_rewards__steps_G1.npy', 'Q_saves/Q4_avg_rewards__steps_G1.npy']
avg_return_steps_load_G2 = ['Q_saves/Q1_avg_return__steps_G2.npy', 'Q_saves/Q2_avg_return__steps_G2.npy',
                            'Q_saves/Q3_avg_rewards__steps_G2.npy', 'Q_saves/Q4_avg_rewards__steps_G2.npy']

# avg_steps_labels[1], avg_steps_labels[2] = avg_steps_labels[2], avg_steps_labels[1]
# avg_steps_loads_G1[1], avg_steps_loads_G1[2] = avg_steps_loads_G1[2], avg_steps_loads_G1[1]
# avg_steps_loads_G2[1], avg_steps_loads_G2[2] = avg_steps_loads_G2[2], avg_steps_loads_G2[1]
# avg_return_steps_load_G1[1], avg_return_steps_load_G1[2] = avg_return_steps_load_G1[2], avg_return_steps_load_G1[1]
# avg_return_steps_load_G2[1], avg_return_steps_load_G2[2] = avg_return_steps_load_G2[2], avg_return_steps_load_G2[1]



def plot_avg_steps(num_episodes, avg_steps_loads, avg_steps_labels,savefile,goal):
    x = np.arange(num_episodes)
    plt.clf()
    for i in range(len(avg_steps_loads)):
        avg_steps = np.load(avg_steps_loads[i])
        plt.plot(x, avg_steps, label=avg_steps_labels[i])
    plt.title(
        "Average steps per episode for goal G"+goal)
    plt.xlabel("Episodes")
    plt.ylabel("Average steps per episode")
    plt.legend(loc=0)
    plt.xscale('log', basex=10)
    plt.savefig(savefile+'.png', dpi=300)
    # plt.show()


def plot_total_return(num_episodes, total_return_loads, labels, savefile, goal):
    x = np.arange(num_episodes)
    plt.clf()
    for i in range(len(total_return_loads)):
        total_return = np.load(total_return_loads[i])
        plt.plot(x, total_return, label=labels[i])

    plt.title(
        "Average Total return per episode for goal G" + goal)
    plt.xlabel("Episodes")
    plt.ylabel("Return")
    plt.legend(loc=0)
    # plt.xlim((0, num_episodes))
    plt.xscale('log', basex=10)
    plt.savefig(savefile+'.png', dpi=300)
    # # plt.show()



plot_avg_steps(num_episodes, avg_steps_loads_G1, avg_steps_labels,'Avg_steps_G1','1')
plot_avg_steps(num_episodes, avg_steps_loads_G2,
               avg_steps_labels, 'Avg_steps_G2','2')

plot_total_return(num_episodes, avg_return_load_G1,
                  avg_steps_labels, 'Total_return_G1', '1')
plot_total_return(num_episodes, avg_return_load_G2,
                  avg_steps_labels, 'Total_return_G2', '2')


plot_total_return(num_episodes, avg_return_steps_load_G1,
                        avg_steps_labels, 'Total_return_steps_G1', '1')
plot_total_return(num_episodes, avg_return_steps_load_G2,
                        avg_steps_labels, 'Total_return_steps_G2', '2')

