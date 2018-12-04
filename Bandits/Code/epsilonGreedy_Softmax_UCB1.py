# Referred Shangtong Zhang, Python implementation of Reinforcement Learning: An Introduction on Github
# For how Bandit class blueprint is used to organize code into reusable peices for different algorithm
# i.e e-greedy, softmax & UCB1


# coding: utf-8

# In[1]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import dill


# In[2]:


# for saving variables
filename = 'globalsave.pkl'
dill.load_session(filename)


# In[3]:


# for reproducibility
np.random.seed(1234)


# In[4]:


# Bandit class maintains properties of bandit environment,
# with methods to get action as per algorithm and recieve rewards from environment
class Bandit:
    def __init__(self,num_arms=10,epsilon=0.,initial=0.,UCBPar=None,
                 softmax=False,mean=0,stdev=1,temperature=1,MEAPar=1):
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.UCBPar = UCBPar
        self.softmax = softmax
        self.mean = mean
        self.stdev = stdev
        self.temperature = temperature

        self.time = 0
        self.indices = np.arange(self.num_arms)

        self.q_star = np.random.normal(self.mean,self.stdev,self.num_arms) # True means of arms
        self.optimal_action = np.argmax(self.q_star) # optimal arm

        self.q_hat = np.repeat(initial,self.num_arms) # Estimated means of arm
        self.pull_count = np.zeros(self.num_arms) # Number of times each arm is pulled


    # get an action for this bandit, depending on the Parameter, like UCB, Softmax, e-Greedy
    def getAction(self):
        #Explore
        if self.epsilon > 0:
            if np.random.binomial(1,self.epsilon) == 1:
                return np.random.choice(self.indices) # Returning random arm

        #Exploit
        if self.UCBPar is not None:
            UCBEstimate = self.q_hat + self.UCBPar * np.sqrt(np.log(self.time + 1) / (np.asarray(self.pull_count) + 1))
            return np.argmax(UCBEstimate) # Returning arm which has highest UCBEstimate

        if self.softmax:
            expEst = np.exp(self.q_hat / self.temperature)
            self.actionProb = expEst / np.sum(expEst)
            return np.random.choice(self.indices,p=self.actionProb) # Returning arm choose according to probability distribution actionProb

        return np.argmax(self.q_hat) # If none of the above condition is met, then return arm with highest mean estimated


    # take an action, receive reward from environment & update mean estimates for corresponding action
    def takeAction(self,action):
        reward = np.random.normal(self.q_star[action],self.stdev)
        self.time += 1
        self.pull_count[action] += 1

        self.q_hat[action] += (reward - self.q_hat[action]) / self.pull_count[action] # Upading mean estimate for action


        return reward


# In[4]:


# Plot of Rewards distribution for each arm taken from standard normal (Gaussian) distribution
def reward_distribution():
#     matplotlib.rcParams['figure.figsize'] = (10, 6)
    fig, axes = plt.subplots(1,10,sharex=True,sharey=True)
    axes = axes.ravel()
    bandit_greedy = Bandit(epsilon=0)
    for k,m in enumerate(bandit_greedy.q_star):
        ax = axes[k]
        ax.violinplot(np.random.normal(m,bandit_greedy.stdev,500),showmeans=True)
        ax.set_xticks([])
        ax.set_xlabel('$q^*({0})$'.format(k+1))
        ax.set_xlim(ax.get_xlim())
        ax.plot([0,2],[0,0],'--', color='black')
        if k == 0:
            ax.set_ylabel('Reward distribution')
        if k == 4:    
            ax.set_title('Rewards taken from standard normal (Gaussian) distribution')
    plt.savefig('Greedy_Reward_dist.png',dpi=300)
    
reward_distribution()    


# In[6]:


# Performs Bandit simulation for given Bandit class object & 
# returns average rewards, optimal arm pulls averaged over provided number of runs
def banditSimulation(nBandits, time, bandits):
    optimal_arm_pulls = [np.zeros(time, dtype='float') for _ in range(len(bandits))]
    averageRewards = [np.zeros(time, dtype='float') for _ in range(len(bandits))]

    for banditIndex,bandit in enumerate(bandits): # Bandit class objects
        for i in range(nBandits): # Number of different bandit problems
            for t in range(time): # time steps
                action = bandit[i].getAction()
                reward = bandit[i].takeAction(action)
                averageRewards[banditIndex][t] += reward
                if action == bandit[i].optimal_action: # increment optimal_arm_pulls if current arm pulled is optimal
                    optimal_arm_pulls[banditIndex][t] += 1
        
        averageRewards[banditIndex] /= nBandits # Taking average 
        optimal_arm_pulls[banditIndex] /= nBandits*0.01 # calculating percentage of optimal arm pulls
        
    return optimal_arm_pulls, averageRewards


# In[7]:


# Runs epsilon Greedy algorithm for different epsilons & returns average rewards, % optimal arm pulls
def epsilonGreedy(nBandits,time):
    epsilons = [0, 0.01, 0.05, 0.1]
    bandits = []
    for epIndex, ep in enumerate(epsilons):
        bandits.append([Bandit(epsilon=ep) for _ in range(nBandits)]) # Bandit class object with required parameter

    optimal_arm_pulls, averageRewards = banditSimulation(nBandits,time,bandits)
    
    return optimal_arm_pulls, averageRewards,epsilons

    
greedy_optimal_arm_pulls, greedy_averageRewards,epsilons = epsilonGreedy(2000,1000)


# In[5]:


# Plot of Average rewards & % optimal arm pulls for epsilon greedy algorithm with different values of epsilon
def plot_greedy(optimal_arm_pulls, averageRewards,epsilons):
    colors = ['green', 'red', 'blue', 'black']
#     matplotlib.rcParams['figure.figsize'] = (10, 6)

    figureIndex = 0
    plt.figure(figureIndex)
    figureIndex += 1
    i = 0
    for ep, rewards in zip(epsilons,averageRewards):
        plt.plot(rewards,label='epsilon '+str(ep),color=colors[i])
        i += 1
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.legend(loc=4)
    plt.title('Averages over 2000 10-Armed Bandit problems')
    plt.savefig('Greedy_Avg_Reward.png',dpi=300)
    
    plt.figure(figureIndex)
    figureIndex += 1
    i = 0
    for ep, count in zip(epsilons,optimal_arm_pulls):
        plt.plot(count,label='epsilon '+str(ep),color=colors[i])
        i += 1
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.ylim([0,100])
    plt.legend(loc=4)
    plt.title('Averages over 2000 10-Armed Bandit problems')
    plt.savefig('Greedy_per_optimal_pulls.png',dpi=300) 

plot_greedy(greedy_optimal_arm_pulls, greedy_averageRewards,epsilons)


# In[15]:


# Runs softmax algorithm for different temperature & returns average rewards, % optimal arm pulls
def softmax(nBandits, time):
    bandits =[[], [], [], []]
    bandits[0] = [Bandit(softmax=True, temperature=0.01) for _ in range(0, nBandits)] # Bandit class object with required parameter
    bandits[1] = [Bandit(softmax=True, temperature=0.1) for _ in range(0, nBandits)]
    bandits[2] = [Bandit(softmax=True, temperature=0.3) for _ in range(0, nBandits)]
    bandits[3] = [Bandit(softmax=True, temperature=1) for _ in range(0, nBandits)]
    
    optimal_arm_pulls, averageRewards = banditSimulation(nBandits, time, bandits)
    
    return optimal_arm_pulls, averageRewards

softmax_optimal_arm_pulls, softmax_averageRewards = softmax(2000,1000)


# In[6]:


# Plot of Average rewards & % optimal arm pulls for softmax algorithm with different values of temperature
def plot_softmax(optimal_arm_pulls, averageRewards):
    labels = ['temperature = 0.01','temperature = 0.1','temperature = 0.3','temperature = 1']
   
    colors = ['green', 'red', 'blue', 'black']
#     matplotlib.rcParams['figure.figsize'] = (10, 6)

    figureIndex = 0
    plt.figure(figureIndex)
    figureIndex += 1
    for i in range(0, len(averageRewards)):
        plt.plot(averageRewards[i], label=labels[i],color=colors[i])
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.legend(loc=4)
    plt.title('Averages over 2000 10-Armed Bandit problems')
    plt.savefig('Softmax_Avg_Reward.png',dpi=300)

    plt.figure(figureIndex)
    figureIndex += 1
    for i in range(0, len(optimal_arm_pulls)):
        plt.plot(optimal_arm_pulls[i], label=labels[i],color=colors[i])
    plt.xlabel('Steps')
    plt.ylim([0,100])
    plt.ylabel('% Optimal action')
    plt.legend(loc=4)
    plt.title('Averages over 2000 10-Armed Bandit problems')
    plt.savefig('Softmax_per_optimal_pulls.png',dpi=300) 
    
plot_softmax(softmax_optimal_arm_pulls, softmax_averageRewards)


# In[17]:


# Runs UCB1 algorithm with softmax & epsilon greedy for comparison & returns average rewards, % optimal arm pulls
def ucb(nBandits,time):
    bandits = [[],[],[]]
    bandits[0] = [Bandit(UCBPar=2) for _ in range(nBandits)] # Bandit class object with required parameter
    bandits[1] = [Bandit(epsilon=0.1) for _ in range(nBandits)]
    bandits[2] = [Bandit(softmax=True, temperature=0.3) for _ in range(0, nBandits)]

    optimal_arm_pulls, averageRewards = banditSimulation(nBandits, time, bandits)
    
    return optimal_arm_pulls, averageRewards


ucb_optimal_arm_pulls, ucb_averageRewards = ucb(2000, 1000)


# In[7]:


# Plot of Average rewards & % optimal arm pulls for UCB algorithm & compares with softmax and epsilon greedy
def plot_ucb(optimal_arm_pulls, averageRewards):
#     matplotlib.rcParams['figure.figsize'] = (10, 6)
    figureIndex = 0
    plt.figure(figureIndex)
    figureIndex += 1

    plt.plot(averageRewards[0], label='UCB c = 2',color='red')
    plt.plot(averageRewards[1], label='epsilon greedy epsilon = 0.1',color='black')
    plt.plot(averageRewards[2], label='softmax temperature = 0.3',color='green')
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.legend(loc=4)
    plt.title('Averages over 2000 10-Armed Bandit problems')
    plt.savefig('UCB_Avg_Reward.png',dpi=300)
    
    plt.figure(figureIndex)
    figureIndex += 1
    plt.plot(optimal_arm_pulls[0], label='UCB c = 2',color='red')
    plt.plot(optimal_arm_pulls[1], label='epsilon greedy epsilon = 0.1',color='black')
    plt.plot(optimal_arm_pulls[2], label='softmax temperature = 0.3',color='green')
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.ylim([0,100])
    plt.legend(loc=4)
    plt.title('Averages over 2000 10-Armed Bandit problems')
    plt.savefig('UCB_per_optimal_pulls.png',dpi=300) 
    
plot_ucb(ucb_optimal_arm_pulls, ucb_averageRewards)


# In[19]:


# Runs epsilon Greedy, UCB & softmax algorithm for 1000 arms & returns average rewards, % optimal arm pulls
def allBanditAlgo(nBandits,time):
    bandits = [[],[],[]]
    bandits[0] = [Bandit(num_arms=1000,UCBPar=2) for _ in range(nBandits)] # Bandit class object with required parameter
    bandits[1] = [Bandit(num_arms=1000,epsilon=0.1) for _ in range(nBandits)]
    bandits[2] = [Bandit(num_arms=1000,softmax=True, temperature=0.3) for _ in range(0, nBandits)]

    optimal_arm_pulls, averageRewards = banditSimulation(nBandits, time, bandits)
    
    return optimal_arm_pulls, averageRewards


all_optimal_arm_pulls, all_averageRewards = allBanditAlgo(2000, 10000)


# In[9]:


# Plot of Average rewards & % optimal arm pulls for epsilon greedy, UCB & softmax algorithm for 1000 arms
def plot_all(optimal_arm_pulls, averageRewards):
#     matplotlib.rcParams['figure.figsize'] = (10, 6)

    figureIndex = 0
    plt.figure(figureIndex)
    figureIndex += 1
    plt.plot(averageRewards[0], label='UCB c = 2',color='red')
    plt.plot(averageRewards[1], label='epsilon greedy epsilon = 0.1',color='black')
    plt.plot(averageRewards[2], label='softmax temperature = 0.3',color='green')
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.legend(loc=4)
    plt.title('Averages over 2000 1000-Armed Bandit problems')
    plt.savefig('All_Avg_Reward.png',dpi=300)
    
    plt.figure(figureIndex)
    figureIndex += 1
    plt.plot(optimal_arm_pulls[0], label='UCB c = 2',color='red')
    plt.plot(optimal_arm_pulls[1], label='epsilon greedy epsilon = 0.1',color='black')
    plt.plot(optimal_arm_pulls[2], label='softmax temperature = 0.3',color='green')
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.ylim([0,100])
    plt.legend(loc=0)
    plt.title('Averages over 2000 1000-Armed Bandit problems')
    plt.savefig('All_per_optimal_pulls.png',dpi=300) 
    
plot_all(all_optimal_arm_pulls, all_averageRewards)


# In[21]:


# Saving Average rewards, % optimal arm pulls for all the above algorithms
filename = 'globalsave.pkl'
dill.dump_session(filename)

