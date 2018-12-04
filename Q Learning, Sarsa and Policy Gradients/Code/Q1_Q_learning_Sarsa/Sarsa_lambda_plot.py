import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

problemis = ["A", "B", "C"]
sarsa_lambda_values = [0, 0.3, 0.5, 0.9, 0.99, 1.0]


# For plotting all three problems togeather
def plot_avg_steps(problem):
    plt.clf()
    pa = np.load('SL_saves/SL_avg_num_of_steps_for_problem_' +
                 problem+"_lambda_"+str(sarsa_lambda_values[0])+'.npy')
    num_episodes = len(pa)
    x = np.arange(num_episodes)
    for i in range(len(sarsa_lambda_values)):
        pa = np.load('SL_saves/SL_avg_num_of_steps_for_problem_' +
                     problem+"_lambda_"+str(sarsa_lambda_values[i])+'.npy')
        plt.plot(x, pa)

    plt.legend(["Lambda: 0", "Lambda: 0.3", "Lambda: 0.5",
                "Lambda: 0.9", "Lambda: 0.99", "Lambda: 1"], loc=0)
    plt.title("Sarsa lambda Average number of steps to goal for problem "+problem)
    plt.xlabel("Episodes")
    plt.ylabel("Average number of steps")
    plt.xlim((0, num_episodes))
    plt.savefig("Sarsa_lambda_Combined_Avg_steps_"+problem+"_"+
                str(num_episodes)+'.png', dpi=300)
    # plt.show()


def plot_total_return(problem):
    plt.clf()
    pa = np.load('SL_saves/SL_total_return_per_episode_for_problem_' +
                 problem+"_lambda_"+str(sarsa_lambda_values[0])+'.npy')
    num_episodes = len(pa)
    x = np.arange(num_episodes)

    for i in range(len(sarsa_lambda_values)):
        pa = np.load('SL_saves/SL_total_return_per_episode_for_problem_' +
                     problem+"_lambda_"+str(sarsa_lambda_values[i])+'.npy')
        plt.plot(x, pa)

    plt.legend(["Lambda: 0", "Lambda: 0.3", "Lambda: 0.5",
                "Lambda: 0.9", "Lambda: 0.99", "Lambda: 1"], loc=0)

    plt.title("Sarsa lambda Average return per episode for problem "+problem)
    plt.xlabel("Episodes")
    plt.ylabel("Average return")
    plt.xlim((0, num_episodes))
    plt.savefig("Sarsa_lambda_Combined_return_"+problem+"_"+
                str(num_episodes)+'.png', dpi=300)
    # plt.show()


for i in range(len(problemis)):
    plot_avg_steps(problemis[i])
    plot_total_return(problemis[i])



