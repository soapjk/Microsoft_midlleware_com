from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import pyplot as plt

fig = plt.figure()
ax = Axes3D(fig)
x = np.arange(-20, 20, 0.1)
y = np.arange(-200, 200, 0.1)
X, Y = np.meshgrid(x, y)  # 网格的创建，这个是关键
Z = np.power((X-2),4)+np.power((X-2*Y),2)
plt.xlabel('x')
plt.ylabel('y')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
plt.show()
