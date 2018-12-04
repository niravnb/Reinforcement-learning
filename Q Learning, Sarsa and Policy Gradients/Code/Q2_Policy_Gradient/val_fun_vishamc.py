# For Plotting State Value from trained policy 

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

xlist = np.linspace(-1.0, 1.0, 100)
ylist = np.linspace(-1.0, 1.0, 100)
X, Y = np.meshgrid(xlist, ylist)
Z = (.5*X**2 + 5*Y**2)
# print(Z)

plt.figure()
cp = plt.contour(X, Y, Z)
plt.clabel(cp, inline=True,
           fontsize=10)
state_space = np.load("vishamc_state_space.npy")
val = np.load("vishamc_val_fun.npy")
val = np.around(val, decimals=2)
x = []
y = []
for j in range(len(state_space)):
    x.append(state_space[j][0])
    y.append(state_space[j][1])
plt.scatter(x, y,marker='o')

for i, txt in enumerate(val):
    plt.annotate(txt, (x[i], y[i]), ha='center', size='large')

plt.title('State values function of learned agent for vishamC')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("vishamc_val.png", dpi=300)
plt.show()

