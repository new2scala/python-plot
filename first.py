import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

import numpy as np

def f(x,y):
    return x**2 + y**2

def poly_reg(px1, py1, px2, py2, xdiff, ydiff):
    # x + y = 1
    x1 = np.arange(px1, py1, step)
    y1 = 1 - (x1 - xdiff) + ydiff
    # 2x - y = -1
    x2 = np.arange(px1, py1, step)
    y2 = 2 * (x2 - xdiff) + 1 + ydiff
    # x + y = 4
    x3 = np.arange(px2, py2, step)
    y3 = 4 - (x3 - xdiff) + ydiff
    # # x - 2y = 1
    x4 = np.arange(px2, py2, step)
    y4 = ((x4 - xdiff) - 1) / 2 + ydiff

    x_reg = np.concatenate((x1, x2, x3, x4))
    y_reg = np.concatenate((y1, y2, y3, y4))
    z_reg_proj = np.array([f(x, y) for x, y in zip(x_reg, y_reg)])
    return x_reg, y_reg, z_reg_proj


step = 0.05

x = y = np.arange(-4.0, 4.0, step)
X, Y = np.meshgrid(x, y)
zs = np.array([f(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)
ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)

x_reg1, y_reg1, z_reg_proj1 = poly_reg(0, 1, 1, 3, 0, 0)
ax.plot(x_reg1, y_reg1, '.')
ax.plot(x_reg1, y_reg1, z_reg_proj1, '.')

x_reg2, y_reg2, z_reg_proj2 = poly_reg(-1, 0, 0, 2, -1, -1)
ax.plot(x_reg2, y_reg2, '--')
ax.plot(x_reg2, y_reg2, z_reg_proj2, '--')
#ax.plot(x2, y2, '.')
#ax.plot(x3, y3, '.')
#ax.plot(x4, y4, '.')

plt.show()