
from scipy.interpolate import LinearNDInterpolator

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng()

x = rng.random(10) - 0.5

y = rng.random(10) - 0.5

z = np.hypot(x, y)

X = np.linspace(min(x), max(x))

Y = np.linspace(min(y), max(y))

X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation

interp = LinearNDInterpolator(list(zip(x, y)), z)

Z = interp(X, Y)

plt.pcolormesh(X, Y, Z, shading='auto')

plt.plot(x, y, "ok", label="input point")

plt.legend()

plt.colorbar()

plt.axis("equal")

plt.savefig("./abatement/pdf_2tech/interpolate/example_linear.pdf")
plt.savefig("./abatement/pdf_2tech/interpolate/example_linear.png")

plt.clf()







from scipy.interpolate import NearestNDInterpolator

import matplotlib.pyplot as plt

# rng = np.random.default_rng()

# x = rng.random(10) - 0.5

# y = rng.random(10) - 0.5

# z = np.hypot(x, y)

# X = np.linspace(min(x), max(x))

# Y = np.linspace(min(y), max(y))

# X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation

interp = NearestNDInterpolator(list(zip(x, y)), z)

Z = interp(X, Y)

plt.pcolormesh(X, Y, Z, shading='auto')

plt.plot(x, y, "ok", label="input point")

plt.legend()

plt.colorbar()

plt.axis("equal")

plt.savefig("./abatement/pdf_2tech/interpolate/example_near.pdf")
plt.savefig("./abatement/pdf_2tech/interpolate/example_near.png")