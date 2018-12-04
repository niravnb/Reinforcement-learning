# Plot comparing variation in learning curves by removing i) Target network 
# ii) Experience replay memory and iii) Both experience replay and target network.


import gym
import random

import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import Experiment_DQN_wo_replay
import Experiment_DQN_wo_target
import Experiment_DQN_wo_both

methods = ['wo_target', 'wo_replay', 'wo_both']
labels = ['w/o Target', 'w/o Replay', 'w/o both']

plt.clf()
i=0
for method in methods:
    avg_100_steps = np.load('saves/avg_100_steps_'+str(method)+'.npy')
    x = np.arange(len(avg_100_steps))
    plt.plot(x, avg_100_steps,label=str(labels[i]))
    i+=1
plt.title("Average total reward of last 100-episodes")
plt.xlabel("Episodes")
plt.ylabel("Average total reward of last 100-episodes")
plt.legend(loc=0)
plt.savefig('Avg_rewards_wos.png',dpi=300)

plt.clf()
i=0
for method in methods:
    step_counts = np.load('saves/step_counts_'+str(method)+'.npy')
    x = np.arange(len(step_counts))
    plt.plot(x, step_counts,label=str(labels[i]))
    i+=1
plt.title("Episode Lengths")
plt.xlabel("Episodes")
plt.ylabel("Episode Length")
plt.legend(loc=0)
plt.savefig('Episode_lengths_wos.png', dpi=300)

x = np.arange(100)
plt.clf()
i=0
for method in methods:
    results = np.load('saves/results_'+str(method)+'.npy')
    x = np.arange(len(results))
    plt.plot(x, results,label=str(labels[i]))
    i+=1
plt.title("Learned agent episodes length")
plt.xlabel("Episodes")
plt.ylabel("Episode Length")
plt.xlim((0, 100))
plt.legend(loc=0)
plt.savefig('Learned_Episode_lengths_wos.png', dpi=300)
