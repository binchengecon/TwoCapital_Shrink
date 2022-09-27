import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


# set up a figure twice as wide as it is tall
fig = plt.figure(figsize=plt.figaspect(0.5))

# =============
# First subplot
# =============
# set up the axes for the first plot
ax = fig.add_subplot(1, 2, 1, projection='3d')

# plot a 3D surface like in the example mplot3d/surface3d_demo

ceartharray = np.linspace(0.3725, 10, 2)
tv = np.linspace(0, 600, 100000)


ceartharray_meshgrid, tv_meshgrid = np.meshgrid(ceartharray, tv, indexing="ij")

Tarray = np.zeros(ceartharray_meshgrid.shape)

surf = ax.plot_surface(ceartharray_meshgrid, tv_meshgrid, Tarray, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

plt.show()


# ceartharray = np.linspace(0.3725, 10, 2)
# tv = np.linspace(0, 600, 100000)

# ceartharray_meshgrid, tv_meshgrid = np.meshgrid(ceartharray, tv, indexing="ij")

# Tarray = np.zeros(ceartharray_meshgrid.shape)
# Carray = np.zeros(ceartharray_meshgrid.shape)

# fig = plt.figure(figsize=(16, 16))

# ax = fig.add_subplot(121, projection='3d')

# ax.plot(ceartharray, tv, Tarray)

# plt.show()
