
# Median Elimination Algorithm
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import dill
from IPython.display import HTML, display
import pandas as pd


# In[2]:


# for reproducibility
np.random.seed(1234)


# In[3]:


filename = 'MEA.pkl'
dill.load_session(filename)


# In[4]:


# Median Elimination Algorithm, with provided epsilon, delta & returns and array of
# ['epsilon', 'delta', 'total_pulls', 'per_optimal_retained','optimal_arm_not_retained', 'outside_epsilon_range']
def MEA(num_arms=10, nBandits=2000, epsilon=0.6, delta=0.3):
    optimal_arm_retained = 0
    optimal_arm_not_retained = 0
    outside_epsilon_range = 0
    total_pulls = 0

    for _ in range(nBandits): # repeat for different bandit problems
        arm_indicies = set(np.arange(num_arms))
        q_star = np.random.normal(10, 1, num_arms) # true arm means
        optimal_arm = np.argmax(q_star) # optimal arm
        epsilon_l = epsilon / 4
        delta_l = delta / 2
        l = 0
        total_pulls = 0

        while (len(arm_indicies) != 1): # repeat till only one arm is left
            q_hat = dict(zip(list(arm_indicies), np.repeat(0., len(arm_indicies))))
            num_samples = int(np.log(3 / delta_l) * 2 / (epsilon_l ** 2)) # calculating number of samples to take for each arm
            for key in q_hat:
                for _ in range(num_samples):
                    q_hat[key] += np.random.normal(q_star[key], 1) # updating mean estimates
                    total_pulls += 1 # incrementing total pulls

            median = np.median(list(q_hat.values())) # finding median of mean estimates
            to_remove_indices = []
            for key, value in q_hat.items():
                if value < median:
                    to_remove_indices.append(key) # finding which arm to remove

            arm_indicies -= set(to_remove_indices) # removing half arms

            epsilon_l *= 3 / 4
            delta_l *= 1 / 2
            l += 1

        if arm_indicies == set([optimal_arm]):
            optimal_arm_retained += 1 # counting number of times optimal arm is retained 
        else:
            optimal_arm_not_retained += 1 # counting number of times optimal arm is not retained 
            temp = q_star[optimal_arm] - q_star[arm_indicies.pop()]
            if temp > epsilon:
                outside_epsilon_range += 1 # counting number of times arm retained is outside epsilon range
        

    return [epsilon, delta, total_pulls, optimal_arm_retained*100/nBandits, optimal_arm_not_retained, outside_epsilon_range]



# In[5]:


# Running MEA for different values of epsilon & delta
experiments = [[0.9,0.9],[0.9,0.1],[0.5,0.9],[0.5,0.5],[0.5,0.1],[0.1,0.9],[0.1,0.1],[0.2,0.02],[0.2,0.98]]
results = []

for ind,exp in enumerate(experiments):
    results.append(MEA(num_arms=10, nBandits=2000, epsilon=exp[0], delta=exp[1]))


# In[6]:


# Saving experiment results
filename = 'MEA.pkl'
dill.dump_session(filename)


# In[6]:


def tableIt(data):
    print(pd.DataFrame(data))


# In[74]:


columns=['epsilon', 'delta', 'total_pulls', 'per_optimal_retained','optimal_arm_not_retained', 'outside_epsilon_range']
print(columns)
print(tableIt(results))


# In[79]:
# Plotting 3D scatter of epsilon, delta & total samples
from mpl_toolkits import mplot3d
matplotlib.rcParams['figure.figsize'] = (10, 6)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
arr = np.array(results,dtype='float')

# Data for a three-dimensional line
zline = arr.T[2][:]
xline = arr.T[0][:]
yline = arr.T[1][:]
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.set_xlabel('epsilon')
ax.set_ylabel('delta')
ax.set_zlabel('Total Samples')
ax.scatter(xline, yline, zline, c='r', marker='o')
plt.savefig('MEA.png',dpi=300)


# In[85]:

# Plotting Total samples taken by MEA
matplotlib.rcParams['figure.figsize'] = (10, 6)
fig = plt.figure()
arr = np.array(results,dtype='float')

zline = arr.T[2][:]
xline = arr.T[0][:]
plt.xlim([0,1])
plt.xlabel('epsilon')
plt.ylabel('Total Samples')
plt.title('Total samples taken by MEA for delta = 0.9 & 0.1 for each epsilon (less value for 0.9 & more for 0.1)')
plt.plot(xline, zline, c='r', marker='o')
plt.savefig('MEA_Samples.png',dpi=300)


# In[86]:

# Plotting % optimal arm retained by MEA
matplotlib.rcParams['figure.figsize'] = (10, 6)
fig = plt.figure()
arr = np.array(results,dtype='float')

zline = arr.T[3][:]
xline = arr.T[0][:]
plt.xlim([0,1])
plt.xlabel('epsilon')
plt.ylabel('% optimal arm retained')
plt.title('% optimal arm retained by MEA for delta = 0.9 & 0.1 for each epsilon (less value for 0.9 & more for 0.1)')
plt.plot(xline, zline, c='r', marker='o')
plt.savefig('MEA_per_Optimal.png',dpi=300)


# In[88]:

# Plotting # times optimal arm not retained by MEA
matplotlib.rcParams['figure.figsize'] = (10, 6)
fig = plt.figure()
arr = np.array(results,dtype='float')

zline = arr.T[4][:]
xline = arr.T[0][:]
plt.xlim([0,1])
plt.xlabel('epsilon')
plt.ylabel('# times optimal arm not retained')
plt.title('# times optimal arm not retained by MEA for delta = 0.9 & 0.1 for each epsilon (less value for 0.9 & more for 0.1)')
plt.plot(xline, zline, c='r', marker='o')
plt.savefig('MEA_not_Optimal.png',dpi=300)

