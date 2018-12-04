# For plotting Average return Vs Iterations for different values of hyperparameter

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


batch_size_list = [100, 10]
alpha_list = [0.1, 0.9]
gamma_list = [0.1, 0.9]
iteration = 100


def plot_avg_ret():
    plt.clf()
    # matplotlib.rcParams['figure.figsize'] = (10, 16)
    plt.rcParams["figure.figsize"] = (10,20)
    x = np.arange(iteration)
    for alpha in alpha_list:
        for batch_size in batch_size_list:
            for gamma in gamma_list:
                pa = np.load("avg_episode_rewards_lr_"+str(alpha)+"_gma_" +
                            str(gamma)+"_bs_"+str(batch_size)+'.npy')
                plt.plot(x, pa,label="lr:"+str(alpha)+" gamma:"+str(gamma)+" batch size:"+str(batch_size))
    plt.legend(loc=0,frameon=False)
    plt.title("Average return per batch size")
    plt.xlabel("Iterations")
    plt.ylabel("Average Return")
    plt.xlim((0, iteration))
    plt.savefig("Avg_return_chakra.png", dpi=300)
    plt.show()


plot_avg_ret()

