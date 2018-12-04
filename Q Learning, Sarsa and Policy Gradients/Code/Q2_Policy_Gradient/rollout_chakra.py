#!/usr/bin/env python

# import click
# click.disable_unicode_literals_warning = True
import numpy as np
import gym
import itertools
import dill
from gym import wrappers


def include_bias(ob):
    return [ob[0], ob[1], 1]


def chakra_get_action(theta, ob, rng=np.random):
    ob_1 = include_bias(ob)
    mean = theta.dot(ob_1)
    return rng.normal(loc=mean, scale=1.)


def get_mean(theta, ob):
    ob_1 = include_bias(ob)
    mean = theta.dot(ob_1)
    return mean

def compute_logPi(action, theta, ob):
    ob_1 = include_bias(ob)
    mean = theta.dot(ob_1)
    diff = action - mean

    return -0.5 * np.log(2 * np.pi) * theta.shape[0] - 0.5 * np.sum(np.square(diff))


def compute_grad_logPi(action, theta, state):
    # I*(a - m)*state
    
    state_1 = include_bias(state)
    mean = theta.dot(state_1)
    diff = np.array(action - mean)
    grad = np.outer(diff, state_1)
    return grad

# @click.command() # Not working on Python 3 on Mac OS 
# @click.argument("env_id", type=str, default="chakra")


def main(env_id):
    # Register the environment
    rng = np.random.RandomState(42)

    if env_id == 'chakra':
        from rlpa2 import chakra
        env = gym.make('chakra-v0')
        # env = wrappers.Monitor(env,"videos/chakra",force=True)

        # get_action = chakra_get_action()
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
    else:
        raise ValueError(
            "Unsupported environment: must be 'chakra' ")

    env.seed(42)
    timestep_limit = env.spec.timestep_limit

    # alpha = 0.5
    # gamma = 0.9
    # batch_size = 100

    batch_size_list = [100]
    alpha_list = [0.9]
    gamma_list = [0.9]
    iteration = 100

    for batch_size in batch_size_list:
        for alpha in alpha_list:
            for gamma in gamma_list:
                # Initialize parameters
                theta = rng.normal(scale=0.01, size=(action_dim, obs_dim + 1))

                # Store baselines for each time step.
                baselines = np.zeros(timestep_limit)
                episode_len = 0
                avg_episode_rewards = []

                # Policy Implementation
                for itr in range(iteration):
                    # n_samples = 0
                    grad = np.zeros_like(theta)
                    episode_rewards = []

                    # Storing cumulative returns for each time step
                    all_returns_for_baseline = [[] for _ in range(timestep_limit)]
                
                    # Iterating over Batch Size
                    for current_batch in range(batch_size): 
                        
                        # Collect a new trajectory
                        rewards = []
                        states = []
                        actions = []
                        ob = env.reset()

                        for i in itertools.count():
                            action = chakra_get_action(theta, ob, rng=rng)
                            next_ob, rew, reached, _ = env.step(action)
                            states.append(ob)
                            actions.append(action)
                            rewards.append(rew)
                            ob = next_ob
                            # if itr >= iteration -2:
                                # env.render()
                            if reached:
                                # print("\n reached and reward is", rew)
                                episode_len = i
                                env.done = False
                                break

                        # print("Episode reward: %.2f Length: %d" %(np.sum(rewards), episode_len))

                        # Going back in time to compute returns and Compute accumulate gradient
                        for t in range(episode_len):
                            tt = t
                            G_t = 0
                            while tt < episode_len:
                                G_t += gamma**(tt-t)*rewards[tt]
                                tt += 1
                            all_returns_for_baseline[t].append(G_t)
                            avdantage = G_t - baselines[t]
                            grad_temp = compute_grad_logPi(actions[t], theta, states[t])
                            grad_temp *= avdantage
                            grad += grad_temp

                        episode_rewards.append(np.sum(rewards))

                        # Updating Baseline from earlier trajectory
                        baselines_temp = np.zeros(len(all_returns_for_baseline))
                        for t in range(len(all_returns_for_baseline)):
                            if len(all_returns_for_baseline[t]) > 0:
                                baselines_temp[t] = np.mean(all_returns_for_baseline[t])
                        
                        baselines = baselines_temp
                            
                    # Normalizing Gradient
                    grad = grad / (np.linalg.norm(grad) + 1e-8)
                    theta += alpha*grad  # /batch_size
                    avg_episode_rewards.append(np.mean(episode_rewards))
                    print("\n Iteration: %d Average Return: %.2f |theta|_2: %.2f" %
                        (itr, avg_episode_rewards[itr], np.linalg.norm(theta)))
                    print("Theta", theta)
                
                # dill.dump_session("chakra.pkl")
                np.save("chakra",theta)
                np.save("avg_episode_rewards_lr_"+str(alpha)+"_gma_"+str(gamma)+"_bs_"+str(batch_size), avg_episode_rewards)


if __name__ == "__main__":
    main('chakra')

    # Loading trained theta
    theta = np.load("chakra.npy")
    from rlpa2 import chakra
    env = gym.make('chakra-v0')
    rng = np.random.RandomState(42)

    # For calculating value function
    value_fun_lst = []
    state_space = []
    for i in range(20):
        value = 0
        ob = env.reset()
        state_space.append(ob)
        for i in itertools.count():
            action = chakra_get_action(theta,ob, rng=rng)
            next_ob, rew, reached, _ = env.step(action)
            value += 0.9*i*rew
            ob = next_ob
            env.render()
            if reached:
                env.done = False
                break
        value_fun_lst.append(value)
    
    print(state_space)
    print(value_fun_lst)
    np.save("chakra_state_space",state_space)
    np.save("chakra_val_fun",value_fun_lst)


    # For Saving policy trajectries
    for epi in range(20):
        ob = env.reset()
        trajectory = []
        for i in itertools.count():
            trajectory.append(ob)
            action = chakra_get_action(theta, ob, rng=rng)
            next_ob, rew, reached, _ = env.step(action)
            ob = next_ob
            env.render()
            if reached:
                print("\n reached and reward is", rew)
                episode_len = i
                env.done = False
                np.save("chakra_traj_"+str(epi),trajectory)
                break
    
    
