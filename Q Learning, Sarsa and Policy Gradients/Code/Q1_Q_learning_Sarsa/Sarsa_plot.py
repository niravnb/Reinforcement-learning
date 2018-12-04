import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

problemis = ["A", "B", "C"]

# For plotting all three problems togeather
def plot_avg_steps():
    plt.clf()
    pa = np.load('S_saves/S_avg_num_of_steps_for_problem_' +
                                 problemis[0]+'.npy')
    pb = np.load('S_saves/S_avg_num_of_steps_for_problem_' +
                 problemis[1]+'.npy')
    pc = np.load('S_saves/S_avg_num_of_steps_for_problem_' +
                 problemis[2]+'.npy')
    num_episodes = len(pa)
    x = np.arange(num_episodes)
    plt.plot(x, pa)
    plt.plot(x, pb)
    plt.plot(x, pc)
    plt.legend(["Goal A","Goal B","Goal C"],loc=0)
    plt.title("Sarsa Average number of steps to goal")
    plt.xlabel("Episodes")
    plt.ylabel("Average number of steps")
    plt.xlim((0, num_episodes))
    plt.savefig("Sarsa_Combined_Avg_steps_"+
                str(num_episodes)+'.png', dpi=300)
    # plt.show()


def plot_total_return():
    plt.clf()
    pa = np.load('S_saves/S_total_return_per_episode_for_problem_' +
                 problemis[0]+'.npy')
    pb = np.load('S_saves/S_total_return_per_episode_for_problem_' +
                 problemis[1]+'.npy')
    pc = np.load('S_saves/S_total_return_per_episode_for_problem_' +
                 problemis[2]+'.npy')
    num_episodes = len(pa)
    x = np.arange(num_episodes)
    plt.plot(x, pa)
    plt.plot(x, pb)
    plt.plot(x, pc)
    plt.legend(["Goal A", "Goal B", "Goal C"], loc=0)

    plt.title("Sarsa Average return per episode")
    plt.xlabel("Episodes")
    plt.ylabel("Average return")
    plt.xlim((0, num_episodes))
    plt.savefig("Sarsa_Combined_return_"+
                str(num_episodes)+'.png', dpi=300)
    # plt.show()


plot_avg_steps()
plot_total_return()


