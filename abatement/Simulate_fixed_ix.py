import numpy as np
import pandas as pd
import sys
print(sys.path)

sys.path.append('./src')

import pickle
import plotly.graph_objects as go
import plotly.offline as pyo
import matplotlib as mpl
import matplotlib.pyplot as plt
import SolveLinSys
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import CubicSpline
from matplotlib.backends.backend_pdf import PdfPages
from src.supportfunctions import finiteDiff_3D
import os

from matplotlib.backends.backend_pdf import PdfPages
from scipy import optimize

def PDESolver(stateSpace, A, B_r, B_f, B_k, C_rr, C_ff, C_kk, D, v0, ε = 1, tol = -10, smartguess = False, solverType = 'False Transient'):

    if solverType == 'False Transient':
        A = A.reshape(-1,1,order = 'F')
        B = np.hstack([B_r.reshape(-1,1,order = 'F'),B_f.reshape(-1,1,order = 'F'),B_k.reshape(-1,1,order = 'F')])
        C = np.hstack([C_rr.reshape(-1,1,order = 'F'), C_ff.reshape(-1,1,order = 'F'), C_kk.reshape(-1,1,order = 'F')])
        D = D.reshape(-1,1,order = 'F')
        v0 = v0.reshape(-1,1,order = 'F')
        out = SolveLinSys.solveFT(stateSpace, A, B, C, D, v0, ε, tol)

        return out

    elif solverType == 'Feyman Kac':
        
        if smartguess:
            iters = 1
        else:
            iters = 4000000
            
        A = A.reshape(-1, 1, order='F')
        B = np.hstack([B_r.reshape(-1, 1, order='F'), B_f.reshape(-1, 1, order='F'), B_k.reshape(-1, 1, order='F')])
        C = np.hstack([C_rr.reshape(-1, 1, order='F'), C_ff.reshape(-1, 1, order='F'), C_kk.reshape(-1, 1, order='F')])
        D = D.reshape(-1, 1, order='F')
        v0 = v0.reshape(-1, 1, order='F')
        out = SolveLinSys.solveFK(stateSpace, A, B, C, D, v0, iters)

        return out

def decompose(v0, stateSpace, states=(), controls=(), args=()):
    i_star, e_star, x_star = controls
    K_mat, Y_mat, L_mat = states
    delta, alpha, kappa, mu_k, sigma_k, gamma_1, gamma_2, theta_ell, pi_c_o, sigma_y,  theta, vartheta_bar, lambda_bar = args
    
    j_star = 1 - e_star / (alpha * lambda_bar * np.exp(K_mat))
    j_star[j_star <= 1e-16] = 1e-16
    
    mc = delta / (alpha - i_star - alpha * vartheta_bar * j_star**theta - x_star)
    dG  = gamma_1 + gamma_2 * Y_mat
    ddG = gamma_2
    
    tol = 1e-7
    episode = 0
    epsilon = 0.1
    max_iter = 3000
    error = 1.
    
#     while error > tol and episode < max_iter:
    
        
#         dvdy = finiteDiff_3D(v0, 1, 1, hY)
#         ddvdyy = finiteDiff_3D(v0, 1, 2, hY)
    
    A = np.zeros(K_mat.shape)
    B_1 = np.zeros(K_mat.shape)
    B_2 = np.sum(theta_ell * pi_c_o, axis=0) 
    B_3 = np.zeros(K_mat.shape)

    C_1 = np.zeros(K_mat.shape)
    C_2 = e_star * sigma_y**2
    C_3 = np.zeros(K_mat.shape)

    D = mc * theta * vartheta_bar / (lambda_bar * np.exp(K_mat)) * j_star**(theta-1) - dG * np.sum(theta_ell * pi_c_o, axis=0) - ddG * sigma_y**2 * e_star

    out = PDESolver(stateSpace, A, B_1, B_2, B_3, C_1, C_2, C_3, D, v0, epsilon, solverType="Feyman Kac")
    v  = out[2].reshape(v0.shape, order="F")
        
#         rhs_error = A * v0 + B_2 * dvdy + C_2 * ddvdyy + D
#         error = np.max(np.abs(v-v0) / epsilon)
        
#         v0 = v
#         episode += 1
        
        
    
#     print(episode, error)
    dvdy = finiteDiff_3D(v, 1, 1, hY)
    ddvdyy = finiteDiff_3D(v, 1, 2, hY)
    RHS = - dvdy * np.sum(pi_c_o * theta_ell, axis=0) - ddvdyy * sigma_y**2 * e_star + dG * np.sum(theta_ell * pi_c_o, axis=0) + ddG * sigma_y**2 * e_star
    LHS = mc * theta * vartheta_bar /(lambda_bar * np.exp(K_mat)) * j_star**(theta-1)
    diff = np.max(abs(RHS - LHS))
    
    ME_base = RHS
    return v, ME_base, diff

def Damage_Intensity(Yt, y_bar_lower=1.5):
    r_1 = 1.5
    r_2 = 2.5
    Intensity = r_1 * (np.exp(r_2 / 2 * (Yt - y_bar_lower)**2) -1) * (Yt > y_bar_lower)
    return Intensity

def maxover_psi0(initial=(np.log(85/0.115), 1.1, -3.7),T0=0, T=40, dt=1/12):

    def simulate_fixed_ix(parameters):

        K_0, Y_0, L_0 = initial
        # print("psi_0 value = {:}".format(parameters))

        psi_0 = parameters
        a = kappa/delta
        b = -(1+alpha*kappa)/delta
        c = alpha/delta-1

        i = (-b - np.sqrt(b**2-4*a*c))/(2*a)
        x = 0.004 * alpha * np.exp(K_0)

        def mu_K(i_x):
            return mu_k + i_x - 0.5 * kappa * i_x ** 2  - 0.5 * sigma_k ** 2
        
        def mu_L(Xt, state):
            return -zeta + psi_0 * (Xt * (np.exp(state[0] - state[2]) ) )**psi_1 - 0.5 * sigma_g**2
        
        years  = np.arange(T0, T0 + T + dt, dt)
        pers   = len(years)

        hist      = np.zeros([pers, 3])
        i_hist    = np.zeros([pers])
        x_hist    = np.zeros([pers])

        mu_K_hist = np.zeros([pers])
        mu_L_hist = np.zeros([pers])

        for tm in range(pers):
            if tm == 0:
                # initial points
                hist[0,:] = [K_0, Y_0, L_0] # logL
                i_hist[0] = i
                x_hist[0] = x
                mu_K_hist[0] = mu_K(i_hist[0])
                mu_L_hist[0] = mu_L(x_hist[0], hist[0,:])
    #             if pre_damage:

                
    #             gt_dmg[0] = damage_func(hist[0, :])
    #             ME_base_t[0] = ME_base_func(hist[0, :])
                
    #             scc_hist[0] = scc_func(hist[0, :])

            else:
                # other periods
                # print(hist[tm-1,:])
                i_hist[tm] = i
                x_hist[tm] = 0.004 * alpha * np.exp(hist[tm-1,0])

                

                mu_K_hist[tm] = mu_K(i_hist[tm])
                mu_L_hist[tm] = mu_L(x_hist[tm], hist[tm-1, :])
                # print("check{:}".format(mu_L_hist[tm]))
                hist[tm,0] = hist[tm-1,0] + mu_K_hist[tm] * dt #logK
                hist[tm,2] = hist[tm-1,2] + mu_L_hist[tm] * dt # logλ

        true_tech_intensity = np.exp(hist[:, 2]) 
        true_tech_prob = 1 - np.exp(- np.cumsum(np.insert(true_tech_intensity * dt, 0, 0) ))[:-1]
        
        temp = 1e3*np.abs(true_tech_prob[-1]-0.995)
        # print("iteration result = {:}".format(true_tech_prob[-1]))
        return temp

    
    x0 = 0.0005

    result = optimize.minimize(simulate_fixed_ix,x0,tol=1e-8,method='Nelder-Mead')

    psi0 =result.x

    return psi0


def hitting995_psi0(initial=(np.log(85/0.115), 1.1, -3.7),T0=0, T=40, dt=1/12):


    K_0, Y_0, L_0 = initial
    # print("psi_0 value = {:}".format(parameters))
    years  = np.arange(T0, T0 + T + dt, dt)
    pers   = len(years)

    i_hist    = np.zeros([20*pers])

    true_tech_intensity = np.exp(L_0)*np.ones_like(i_hist) 
    true_tech_prob = 1 - np.exp(- np.cumsum(np.insert(true_tech_intensity * dt, 0, 0) ))[:-1]
    
    time_greater995 = np.where(true_tech_prob>0.995)
    
    # .min()[0]/pers*T
    # print(time_greater995[0][0]/pers*T)

    # return true_tech_prob
    return time_greater995[0][0]/pers

delta = 0.01
alpha = 0.115
kappa = 6.667
mu_k  = -0.043
sigma_k = 0.0095
beta_f = 1.86/1000
sigma_y = 1.2 * 1.86 / 1000
zeta = 0.0
psi_1 = 1/2
sigma_g = 0.016
gamma_1 = 1.7675 / 1000
gamma_2 = 0.0022 * 2
gamma_3_list = np.linspace(0., 1./3., 20)
y_bar = 2.
y_bar_lower = 1.5


# Tech
theta = 3
lambda_bar = 0.1206
vartheta_bar = 0.0453

lambda_bar_first = lambda_bar / 2.
vartheta_bar_first = vartheta_bar / 2.

lambda_bar_second = 1e-3
vartheta_bar_second = 0.

K_min = 4.00
K_max = 9.00
hK    = 0.20
K     = np.arange(K_min, K_max + hK, hK)
nK    = len(K)
Y_min = 0.
Y_max = 5.
hY    = 0.20 # make sure it is float instead of int
Y     = np.arange(Y_min, Y_max + hY, hY)
nY    = len(Y)
L_min = - 5.
L_max = - 0.
hL    = 0.2
L     = np.arange(L_min, L_max,  hL)
nL    = len(L)

id_2 = np.abs(Y - y_bar).argmin()
Y_min_short = 0.
Y_max_short = 3.
Y_short     = np.arange(Y_min_short, Y_max_short + hY, hY)
nY_short    = len(Y_short)
# print("bY_short={:d}".format(nY_short))
(K_mat, Y_mat, L_mat) = np.meshgrid(K, Y_short, L, indexing="ij")

stateSpace = np.hstack([K_mat.reshape(-1,1,order = 'F'), Y_mat.reshape(-1,1,order = 'F'), L_mat.reshape(-1, 1, order='F')])



PDF_Dir = "./abatement/pdf_2tech/"

if not os.path.exists(PDF_Dir):
    os.mkdir(PDF_Dir)


Ig_Initial = np.array([1/80,1/120,1/160])
psi_0 = np.zeros_like(Ig_Initial)
time1 = np.zeros_like(Ig_Initial)
time2  = np.zeros_like(Ig_Initial)

# psi_0[0]=maxover_psi0(initial=(np.log(85), 1.1, ),T0=0, T=80, dt=1/12)
# psi_0[1]=maxover_psi0(initial=(np.log(85), 1.1, np.log(1/120)),T0=0, T=80, dt=1/12)
# psi_0[2]=maxover_psi0(initial=(np.log(85), 1.1, np.log(1/160)),T0=0, T=80, dt=1/12)

for i in range(len(Ig_Initial)):

    psi_0[i]=maxover_psi0(initial=(np.log(85/0.115), 1.1, np.log(Ig_Initial[i])),T0=0, T=80, dt=1/12)

print("I_g series = {:}".format(Ig_Initial))
print("psi_0 series = {:}".format(psi_0))


# time1 = -np.log(1-0.995)/Ig_Initial*(1/12)
# print("time1 series = {:}".format(time1))

for i in range(len(Ig_Initial)):

    time2[i]=hitting995_psi0(initial=(np.log(85/0.115), 1.1, np.log(Ig_Initial[i])),T0=0, T=80, dt=1/12)

print("time2 series = {:}".format(time2))


