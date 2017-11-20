import numpy as np

mean = np.zeros((78, 3))
median = np.zeros((78, 3))
i = 0
for tau in range(0, 6, 1):
    for c in range(13):
        with open("../outputs/output_tau{}.0_c{}.txt".format(tau, c * 0.25), 'r') as f:
            lines = f.readlines()
            mn = lines[12][:-1]
            md = lines[13][:-1]
            mean[i] = (tau, c * 0.25, float(mn.split(" ")[4]))
            median[i] = (tau, c * 0.25, float(md.split(" ")[2]))
            i += 1

print(mean[:,0].reshape(6,13))
print(mean[:,1].reshape(6,13))
print(mean[:,2].reshape(6,13))
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# Axes3D.plot_wireframe(mean[:,0],mean[:,1],mean[:,2])

