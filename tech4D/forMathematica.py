from supportfunctions import *
import numpy as np

# Parameters as defined in the paper
delta = 0.01
A_d = 0.12
A_g = 0.15

alpha_d = -0.02
alpha_g = -0.02
sigma_d = 0.016
sigma_g = 0.016

varsigma = 1.2 * 1.86 / 1000
phi_d = 8.
phi_g = 8.



###### damage
gamma_1 = 0.00017675
gamma_2 = 2. * 0.0022


y_bar = 2
beta_f = 1.86 / 1000

# Grids Specification
# Coarse Grids
Y_min = 1e-8
Y_max = 2.5
# range of capital
K_min = 4.00
K_max = 7.5
R_min = 0.14
R_max = 0.99
# R_max = 0.50
# hR = 0.05
hK = 0.10
hR = 0.01
hY = 0.05 # make sure it is float instead of int

# R = np.arange(R_min, R_max + hR, hR)
# nR = len(R)
Y = np.arange(Y_min, Y_max + hY, hY)
nY = len(Y)
K = np.arange(K_min, K_max + hK, hK)
nK = len(K)
R = np.arange(R_min, R_max + hR, hR)
nR = len(R)
# lam = np.arange(lam_min, lam_max + hlam, hlam)
# nlam = len(lam)



(K_mat, R_mat, Y_mat) = np.meshgrid(K, R, Y, indexing = 'ij')
stateSpace = np.hstack([K_mat.reshape(-1,1,order = 'F'), R_mat.reshape(-1,1,order = 'F'), Y_mat.reshape(-1, 1, order='F')])

# For PETSc
K_mat_1d =K_mat.ravel(order='F')
R_mat_1d = R_mat.ravel(order='F')
Y_mat_1d = Y_mat.ravel(order='F')
lowerLims = np.array([K_min, R_min, Y_min], dtype=np.float64)
upperLims = np.array([K_max, R_max, Y_max], dtype=np.float64)





v0 = K_mat - (gamma_1 + gamma_2 * Y_mat)

dK = finiteDiff(v0,0,1,hK)
# dK[dK < 1e-16] = 1e-16
dR = finiteDiff(v0,1,1,hR)
# dR[dR < 1e-8] = 1e-8
dY = finiteDiff(v0,2,1,hY)
######## second order
ddK = finiteDiff(v0,0,2,hK)
ddR = finiteDiff(v0,1,2,hR)
ddY = finiteDiff(v0,2,2,hY)


i_d = np.zeros(K_mat.shape)
i_g = np.zeros(R_mat.shape)
consumption_0 = A_d * (1 - R_mat) + A_g * R_mat
consumption = consumption_0
mc = delta / consumption
i_d = 1 -  mc / (dK - R_mat *  dR)
i_d /= phi_d
# i_d[i_d < 0] = 0

i_g = 1 - mc / (dK + (1 - R_mat) * dR)
i_g /= phi_g



