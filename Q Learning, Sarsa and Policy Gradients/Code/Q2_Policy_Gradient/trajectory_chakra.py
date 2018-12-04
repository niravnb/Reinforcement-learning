# For plotting Policy trajectories of learned agent

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

xlist = np.linspace(-1.0, 1.0, 100)
ylist = np.linspace(-1.0, 1.0, 100)
X, Y = np.meshgrid(xlist, ylist)
Z = np.sqrt(X**2 + Y**2)
# print(Z)


plt.figure()
cp = plt.contour(X, Y, Z)
plt.clabel(cp, inline=True,
           fontsize=10)
for i in range(15):
    tmp = np.load("chakra_traj_"+str(i)+".npy")
    x = []
    y = []
    for j in range(len(tmp)):
        x.append(tmp[j][0])
        y.append(tmp[j][1])
    plt.plot(x,y)
plt.title('Policy trajectories of learned agent for chakra')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("chakra_traj.png", dpi=300)
# plt.show()

