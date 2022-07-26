from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import NearestNDInterpolator
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(4,9+0.2,0.2)
y = np.arange(0,4+0.2,0.2)
q = np.arange(-5.5,0+0.2,0.2)

x_mat, y_mat, q_mat = np.meshgrid(x,y,q)
x_mat_1d = x_mat.ravel(order='F')
y_mat_1d = y_mat.ravel(order='F')
q_mat_1d = q_mat.ravel(order='F')

z_1d = np.sum((x_mat_1d**2, y_mat_1d**2,q_mat_1d**2),axis=0)



X = np.arange(4,9+0.1,0.1)
Y = np.arange(0,4+0.1,0.1)
Q = np.arange(-5.5,0+0.1,0.1)
X, Y, Q = np.meshgrid(X, Y, Q)  # 2D grid for interpolation

interp1 = LinearNDInterpolator(list(zip(x_mat_1d, y_mat_1d, q_mat_1d)), z_1d)
Z1 = interp1(X, Y, Q)

interp2 = NearestNDInterpolator(list(zip(x_mat_1d, y_mat_1d, q_mat_1d)), z_1d)
Z2 = interp2(X, Y, Q)

print(Z2.max())

plt.pcolormesh(X[:,:,0], Y[:,:,0], Z1[:,:,0], shading='auto')
# plt.plot(x_mat[:,:,0], y_mat[:,:,0], "ok", label="input point")
plt.plot(x_mat_1d, y_mat_1d,  label="input point", marker=".", markersize = 10)
plt.legend()
plt.colorbar()
plt.axis("equal")
plt.savefig("./abatement/pdf_2tech/interpolate/test_linear.pdf")
plt.clf()

plt.pcolormesh(X[:,:,0], Y[:,:,0], Z2[:,:,0], shading='auto')
# plt.plot(x_mat[:,:,0], y_mat[:,:,0], "ok", label="input point")
plt.plot(x_mat_1d, y_mat_1d,  label="input point", marker=".", markersize = 10)
plt.legend()
plt.colorbar()
plt.axis("equal")
plt.savefig("./abatement/pdf_2tech/interpolate/test_near.pdf")
