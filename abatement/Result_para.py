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

font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : 18}

plt.rc('font', **font)  # pass in the font dict as kwargs

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

def simulate_post(grid = (), model_args = (), controls = (), initial=(np.log(85/0.115), 1.1), T0=0, T=40, dt=1/12):

#     T = 20
#     dt = 1/12
    K, Y = grid
    K_min, K_max, Y_min, Y_max = min(K), max(K), min(Y), max(Y)
    hK, hY = K[1] - K[0], Y[1] - Y[0]
    (K_mat, Y_mat) = np.meshgrid(K, Y, indexing = 'ij')

    delta, mu_k, kappa,sigma_k, beta_f = model_args
    i, e = controls

    method = 'Linear'
    years = np.arange(T0, T0 + T + dt, dt)
    pers = len(years)

    # setting up grids
    stateSpace = np.hstack([
        K_mat.reshape(-1,1,order = "F"), 
        Y_mat.reshape(-1,1,order = "F"),
    ])

    # some parameters remaiend unchanged across runs
    gamma_1 = 0.00017675
    gamma_2 = 2. * 0.0022
    gamma_bar = 2
    beta_f = 1.86 / 1000
    sigma_y = 1.2 * 1.86 / 1000

    gridpoints = (K, Y)
    i_func = RegularGridInterpolator(gridpoints, i)
    e_func = RegularGridInterpolator(gridpoints, e)

    def get_i(x):
        return i_func(x)

    def get_e(x):
        return e_func(x)


#     Y_0 = 1.1
#     K_0 = np.log(85 / 0.115)
    K_0, Y_0 = initial

    def mu_K(i_x):
        return max(mu_k + i_x - 0.5 * kappa * i_x ** 2  - 0.5 * sigma_k ** 2, 1e-15)

    hist = np.zeros([pers, 2])

    i_hist = np.zeros([pers])
    e_hist = np.zeros([pers])

    mu_k_hist = np.zeros([pers])

    for tm in range(pers):
        if tm == 0:
            # initial points
            hist[0,:] = [K_0, Y_0] # logL
            i_hist[0] = get_i(hist[0, :])
            e_hist[0] = get_e(hist[0, :])
            mu_k_hist[0] = mu_K(i_hist[0])

        else:
            # other periods
            i_hist[tm] = get_i(hist[tm-1,:])
            e_hist[tm] = get_e(hist[tm-1,:])

            mu_k_hist[tm] = mu_K(i_hist[tm])

            hist[tm,0] = hist[tm-1,0] + mu_k_hist[tm-1] * dt #logK
            hist[tm,1] = hist[tm-1,1] + e_hist[tm] * (beta_f * dt)

    return dict(
        states= hist, 
        i = i_hist, 
        e = e_hist,
        years=years
    )

def simulate_pre(
    grid = (), model_args = (), controls = (), initial=(np.log(85/0.115), 1.1, np.log(1/120)), 
    T0=0, T=40, dt=1/12,
    printing=False):

    K, Y, L = grid

    if printing==True:
        print("K_min={},K_max={},Y_min={},Y_max={},L_min={},L_max={}" .format(K.min(),K.max(),Y.min(),Y.max(),L.min(),L.max()))

    K_min, K_max, Y_min, Y_max, L_min, L_max = min(K), max(K), min(Y), max(Y), min(L), max(L)
    hK, hY = K[1] - K[0], Y[1] - Y[0]
    (K_mat, Y_mat, L_mat) = np.meshgrid(K, Y, L, indexing = 'ij')

    delta, mu_k, kappa, sigma_k, beta_f, zeta, psi_0, psi_1, sigma_g, theta, lambda_bar, vartheta_bar, = model_args
    ii, ee, xx, g_tech, g_damage, pi_c = controls
    n_climate = len(pi_c)

    method = 'linear'
    years  = np.arange(T0, T0 + T + dt, dt)
    pers   = len(years)
       

    # setting up grids
    stateSpace = np.hstack([
        K_mat.reshape(-1,1,order = "F"), 
        Y_mat.reshape(-1,1,order = "F"),
        L_mat.reshape(-1,1,order = "F"),
    ])

    # some parameters remaiend unchanged across runs
    gamma_1  = 0.00017675
    gamma_2  = 2. * 0.0022
    beta_f   = 1.86 / 1000
    sigma_y  = 1.2 * 1.86 / 1000
    
    theta_ell = pd.read_csv("./data/model144.csv", header=None).to_numpy()[:, 0]/1000.
    pi_c_o = np.ones(len(theta_ell)) / len(theta_ell)
    pi_c_o = np.array([temp * np.ones(K_mat.shape) for temp in pi_c_o])
    theta_ell = np.array([temp * np.ones(K_mat.shape) for temp in theta_ell])
    args = (delta, alpha, kappa, mu_k, sigma_k, gamma_1, gamma_2, theta_ell, pi_c_o, sigma_y,  theta, vartheta_bar, lambda_bar)

#     v, ME_base, diff = decompose(v0, stateSpace, (K_mat, Y_mat, L_mat), (ii, ee, xx), args=args)

    gridpoints = (K, Y, L)

    i_func = RegularGridInterpolator(gridpoints, ii)
    e_func = RegularGridInterpolator(gridpoints, ee)
    x_func = RegularGridInterpolator(gridpoints, xx)
    tech_func = RegularGridInterpolator(gridpoints, g_tech)
#     ME_base_func = RegularGridInterpolator(gridpoints, ME_base)
    
#     if pre_damage:
    n_damage = len(g_damage)

    damage_func_list = []
    for i in range(n_damage):
        func_i = RegularGridInterpolator(gridpoints, g_damage[i])
        damage_func_list.append(func_i)
        
    climate_func_list = []
    for i in range(n_climate):
        func_i = RegularGridInterpolator(gridpoints, pi_c[i])
        climate_func_list.append(func_i)


    def get_i(x):
        return i_func(x)

    def get_e(x):
        return e_func(x)
    
    def get_x(x):
        return x_func(x)


#     K_0 = np.log(85 / 0.115)
#     Y_0 = 1.1
#     L_0 = -3.7
    
    K_0, Y_0, L_0 = initial

    def mu_K(i_x):
        return mu_k + i_x - 0.5 * kappa * i_x ** 2  - 0.5 * sigma_k ** 2
    
    def mu_L(Xt, state):
        return -zeta + psi_0 * (Xt * (np.exp(state[0] - state[2]) ) )**psi_1 - 0.5 * sigma_g**2
    
    
    hist      = np.zeros([pers, 3])
    i_hist    = np.zeros([pers])
    e_hist    = np.zeros([pers])
    x_hist    = np.zeros([pers])
    scc_hist  = np.zeros([pers])
    gt_tech   = np.zeros([pers])
#     if pre_damage:
    gt_dmg    = np.zeros([n_damage, pers])
    pi_c_t = np.zeros([n_climate, pers])
    
#     ME_base_t = np.zeros([pers])

    mu_K_hist = np.zeros([pers])
    mu_L_hist = np.zeros([pers])

    for tm in range(pers):
        if tm == 0:

            # initial points
            hist[0,:] = [K_0, Y_0, L_0] # logL
            i_hist[0] = get_i(hist[0, :])
            e_hist[0] = get_e(hist[0, :])
            x_hist[0] = get_x(hist[0, :])
            mu_K_hist[0] = mu_K(i_hist[0])
            mu_L_hist[0] = mu_L(x_hist[0], hist[0,:])
            gt_tech[0] = tech_func(hist[0, :])
#             if pre_damage:
            for i in range(n_damage):
                damage_func = damage_func_list[i]
                gt_dmg[i, 0] = damage_func(hist[0, :])
            
            for i in range(n_climate):
                climate_func = climate_func_list[i]
                pi_c_t[i, 0] = climate_func(hist[0, :])
            
#             gt_dmg[0] = damage_func(hist[0, :])
#             ME_base_t[0] = ME_base_func(hist[0, :])
            
#             scc_hist[0] = scc_func(hist[0, :])

        else:
            # other periods
            # print(hist[tm-1,:])
            i_hist[tm] = get_i(hist[tm-1,:])
            e_hist[tm] = get_e(hist[tm-1,:])
            x_hist[tm] = get_x(hist[tm-1,:])
            gt_tech[tm] = tech_func(hist[tm-1,:])
#             if pre_damage:
            for i in range(n_damage):
                damage_func = damage_func_list[i]
                gt_dmg[i, tm] = damage_func(hist[tm-1, :])

            for i in range(n_climate):
                climate_func = climate_func_list[i]
                pi_c_t[i, tm] = climate_func(hist[tm -1, :])
                
#             ME_base_t[tm] = ME_base_func(hist[tm-1, :])
            

            mu_K_hist[tm] = mu_K(i_hist[tm])
            mu_L_hist[tm] = mu_L(x_hist[tm], hist[tm-1, :])

            hist[tm,0] = hist[tm-1,0] + mu_K_hist[tm] * dt #logK
            hist[tm,1] = hist[tm-1,1] + beta_f * e_hist[tm] * dt
            hist[tm,2] = hist[tm-1,2] + mu_L_hist[tm] * dt # logλ

        if printing==True:
            print("time={}, K={},Y={},L={},mu_K={},mu_Y={},mu_L={},ii={},ee={},xx={}" .format(tm, hist[tm,0],hist[tm,1],hist[tm,2],mu_K_hist[tm],beta_f * e_hist[tm],mu_L_hist[tm],ii.max(),ee.max(),xx.max()))
        
    
    
        # using Kt instead of K0
    jt = 1 - e_hist/ (alpha * lambda_bar * np.exp(hist[:, 0]))
    jt[jt <= 1e-16] = 1e-16
    LHS = theta * vartheta_bar / lambda_bar * jt**(theta -1)
    MC = delta / (alpha  - i_hist - alpha * vartheta_bar * jt**theta - x_hist)

    
    scc_hist = LHS * 1000
#     scc_0 = ME_base_t / MC * 1000 * np.exp(hist[:, 0])
    
    distorted_tech_intensity = np.exp(hist[:, 2]) * gt_tech
    distorted_tech_prob = 1 - np.exp(- np.cumsum(np.insert(distorted_tech_intensity * dt, 0, 0) ))[:-1]

    true_tech_intensity = np.exp(hist[:, 2]) 
    true_tech_prob = 1 - np.exp(- np.cumsum(np.insert(true_tech_intensity * dt, 0, 0) ))[:-1]
        
#     if pre_damage:
    damage_intensity = Damage_Intensity(hist[:, 1])
    distorted_damage_intensity = np.mean(gt_dmg, axis=0) * damage_intensity
    distorted_damage_prob = 1 - np.exp(- np.cumsum(np.insert(distorted_damage_intensity * dt, 0, 0) ))[:-1]
    
    true_damage_intensity =  damage_intensity
    true_damage_prob = 1 - np.exp(- np.cumsum(np.insert(true_damage_intensity * dt, 0, 0) ))[:-1]

    
    res = dict(
        states= hist, 
        i = i_hist * np.exp(hist[:, 0]), 
        e = e_hist,
        # x = x_hist * np.exp(hist[:, 0]),
        x = x_hist * np.exp(hist[:, 0]),
        scc = scc_hist,
#         scc0 = scc_0,
        gt_tech = gt_tech,
        gt_dmg = gt_dmg,
        distorted_damage_prob=distorted_damage_prob,
        distorted_tech_prob=distorted_tech_prob,
        pic_t = pi_c_t,
#         ME_base = ME_base_t,
        jt = jt,
        LHS = LHS,
        years=years,
        true_tech_prob = true_tech_prob,
        true_damage_prob = true_damage_prob
    )
    
#     if pre_damage:
#         res["gt_dmg"] = gt_dmg
    
    return res

def Damage_Intensity(Yt, y_bar_lower=1.5):
    r_1 = 1.5
    r_2 = 2.5
    Intensity = r_1 * (np.exp(r_2 / 2 * (Yt - y_bar_lower)**2) -1) * (Yt > y_bar_lower)
    return Intensity

def graph(psi_0,psi_1,Ig_initial = 1/120):



    def model_solution_extraction(xi_a,xi_g,psi_0,psi_1):
    
        Data_Dir = "./abatement/data_2tech/"

        File_Dir = "Compare_Psi0_Psi1_xi_a_{}_xi_g_{}_psi_0_{}_psi_1_{}_" .format(xi_a,xi_g,psi_0,psi_1)

        # Tech I, pre damage, 25 years technlogy jump
        with open(Data_Dir + File_Dir+"model_tech1_pre_damage", "rb") as f:
            tech1 = pickle.load(f)
            

        with open(Data_Dir + File_Dir+"model_tech3_pre_damage", "rb") as f:
            tech3 = pickle.load(f)

        model_args = (delta, mu_k, kappa,sigma_k, beta_f, zeta, psi_0, psi_1, sigma_g, theta, lambda_bar, vartheta_bar)
        i = tech1["i_star"]
        e = tech1["e_star"]
        x = tech1["x_star"]
        pi_c = tech1["pi_c"]
        g_tech = tech1["g_tech"]
        g_damage =  tech1["g_damage"]
        
        
        # g_damage = np.ones((1, nK, nY, nL))
        res = simulate_pre(grid = (K, Y_short, L), model_args = model_args, 
                                    controls = (i,e,x, g_tech, g_damage, pi_c), 
                                    T0=0, T=IntPeriod, dt=timespan,printing=False)
        return res



    def graph_solution_extraction(res1,res2,res3):



        PDF_Dir = "./abatement/pdf_2tech/"
        File_Dir = "attemp_newpsi0_psi0_{}_psi1_{}_" .format(psi_0,psi_1)


        if not os.path.exists(PDF_Dir):
            os.mkdir(PDF_Dir)
                
        pdf_pages2 = PdfPages(PDF_Dir+File_Dir+"Mitigation_RD_Investment" +'.pdf')

        figwidth = 10

        fig, axs = plt.subplots(1, 1, sharex=True, figsize=(2  * figwidth, 2 *figwidth))  


        axs.plot(res1["years"][res1["states"][:, 1]<1.5], (res1["x"]/(alpha*np.exp(res1["states"][:,0])))[res1["states"][:, 1]<1.5],label=r'$\xi_p=\xi_g=0.025$',linewidth=7.0)
        axs.plot(res2["years"][res2["states"][:, 1]<1.5], (res2["x"]/(alpha*np.exp(res2["states"][:,0])))[res2["states"][:, 1]<1.5],label=r'$\xi_p=\xi_g=0.050$',linewidth=7.0)
        axs.plot(res3["years"][res3["states"][:, 1]<1.5], (res3["x"]/(alpha*np.exp(res3["states"][:,0])))[res3["states"][:, 1]<1.5],label='baseline',linewidth=7.0)
        axs.set_xlabel('Years')
        axs.set_ylabel("R&D investment (as a ratio of total GDP)")
        axs.set_title("R&D investment (as a ratio of total GDP)")
        axs.grid(linestyle=':')
        axs.legend(loc='upper left')

        pdf_pages2.savefig(fig)
        plt.close()

        pdf_pages2.close()          

        pdf_pages = PdfPages(PDF_Dir+File_Dir+'Mitigation_Years_'+str(IntPeriod)+'.pdf')

        figwidth = 10

        # fig1, axs1 = plt.subplots(3, 1, sharex=True, figsize=(2  * figwidth, 2 *figwidth))  
        fig1, axs1 = plt.subplots(3, 1, sharex=False, figsize=(2  * figwidth, 2 *figwidth))  



        axs.plot(res1["years"][res1["states"][:, 1]<1.5], (res1["x"]/(alpha*np.exp(res1["states"][:,0])))[res1["states"][:, 1]<1.5],label=r'$\xi_p=\xi_g=0.025$',linewidth=7.0)
        axs.plot(res2["years"][res2["states"][:, 1]<1.5], (res2["x"]/(alpha*np.exp(res2["states"][:,0])))[res2["states"][:, 1]<1.5],label=r'$\xi_p=\xi_g=0.050$',linewidth=7.0)
        axs.plot(res3["years"][res3["states"][:, 1]<1.5], (res3["x"]/(alpha*np.exp(res3["states"][:,0])))[res3["states"][:, 1]<1.5],label='baseline',linewidth=7.0)
        axs1[0].set_xlabel('Years')
        axs1[0].set_ylabel("R&D investment (as a ratio of total GDP)")
        axs1[0].set_title("R&D investment (as a ratio of total GDP)")
        axs1[0].grid(linestyle=':')
        axs1[0].legend(loc='upper left')


        axs1[1].plot(res1["years"][res1["states"][:, 1]<1.5], res1["i"][res1["states"][:, 1]<1.5],label=r'$\xi_p=\xi_g=0.025$',linewidth=7.0)
        axs1[1].plot(res2["years"][res2["states"][:, 1]<1.5], res2["i"][res2["states"][:, 1]<1.5],label=r'$\xi_p=\xi_g=0.050$',linewidth=7.0)
        axs1[1].plot(res3["years"][res3["states"][:, 1]<1.5], res3["i"][res3["states"][:, 1]<1.5],label='baseline',linewidth=7.0)
        axs1[1].set_xlabel('Years')
        axs1[1].set_ylabel("Capital investment")
        axs1[1].set_title("Capital investment")
        axs1[1].grid(linestyle=':')
        axs1[1].legend(loc='upper left')


        axs1[2].plot(res1["years"][res1["states"][:, 1]<1.5], res1["e"][res1["states"][:, 1]<1.5],label=r'$\xi_p=\xi_g=0.025$',linewidth=7.0)
        axs1[2].plot(res2["years"][res2["states"][:, 1]<1.5], res2["e"][res2["states"][:, 1]<1.5],label=r'$\xi_p=\xi_g=0.050$',linewidth=7.0)
        axs1[2].plot(res3["years"][res3["states"][:, 1]<1.5], res3["e"][res3["states"][:, 1]<1.5],label='baseline',linewidth=7.0)
        axs1[2].set_xlabel('Years')
        axs1[2].set_ylabel("Emission")
        axs1[2].set_title("Emission")
        axs1[2].grid(linestyle=':')
        axs1[2].legend(loc='upper left')


        pdf_pages.savefig(fig1)
        plt.close()

        # fig2, axs2 = plt.subplots(2, 1, sharex=True, figsize=(2  * figwidth, 2 *figwidth))  
        fig2, axs2 = plt.subplots(2, 1, sharex=False, figsize=(2  * figwidth, 2 *figwidth))  


        axs2[0].plot(res1["years"][res1["states"][:, 1]<1.5], res1["states"][:, 1][res1["states"][:, 1]<1.5],label=r'$\xi_p=\xi_g=0.025$',linewidth=7.0)
        axs2[0].plot(res2["years"][res2["states"][:, 1]<1.5], res2["states"][:, 1][res2["states"][:, 1]<1.5],label=r'$\xi_p=\xi_g=0.050$',linewidth=7.0)
        axs2[0].plot(res3["years"][res3["states"][:, 1]<1.5], res3["states"][:, 1][res3["states"][:, 1]<1.5],label='baseline',linewidth=7.0)
        axs2[0].set_xlabel('Years')
        axs2[0].set_ylabel("Temperature anomaly")
        axs2[0].set_title("Temperature anomaly")
        axs2[0].grid(linestyle=':')
        axs2[0].legend(loc='upper left')

        axs2[1].plot(res1["years"][res1["states"][:, 1]<1.5], np.exp(res1["states"][:, 2])[res1["states"][:, 1]<1.5],label=r'$\xi_p=\xi_g=0.025$',linewidth=7.0)
        axs2[1].plot(res2["years"][res2["states"][:, 1]<1.5], np.exp(res2["states"][:, 2])[res2["states"][:, 1]<1.5],label=r'$\xi_p=\xi_g=0.050$',linewidth=7.0)
        axs2[1].plot(res3["years"][res3["states"][:, 1]<1.5], np.exp(res3["states"][:, 2])[res3["states"][:, 1]<1.5],label='baseline',linewidth=7.0)
        axs2[1].set_xlabel('Years')
        axs2[1].set_ylabel("Technology jump intensity")
        axs2[1].set_title("Technology jump intensity")
        axs2[1].grid(linestyle=':')
        axs2[1].legend(loc='upper left')


        pdf_pages.savefig(fig2)
        plt.close()


        fig3, axs3 = plt.subplots(2, 1, sharex=True, figsize=(2  * figwidth, 2 *figwidth))  


        axs3[0].plot(res1["years"], res1["distorted_tech_prob"],label=r'$\xi_p=\xi_g=0.025$',linewidth=7.0)
        axs3[0].plot(res2["years"], res2["distorted_tech_prob"],label=r'$\xi_p=\xi_g=0.050$',linewidth=7.0)
        axs3[0].plot(res3["years"], res3["distorted_tech_prob"],label='baseline',linewidth=7.0)
        axs3[0].set_xlabel('Years')
        axs3[0].set_ylabel("Distorted probability of first technology jump")
        axs3[0].set_title("Distorted probability of first technology jump")
        axs3[0].grid(linestyle=':')
        axs3[0].legend(loc='upper left')

        axs3[1].plot(res1["years"], res1["distorted_damage_prob"],label=r'$\xi_p=\xi_g=0.025$',linewidth=7.0)
        axs3[1].plot(res2["years"], res2["distorted_damage_prob"],label=r'$\xi_p=\xi_g=0.050$',linewidth=7.0)
        axs3[1].plot(res3["years"], res3["distorted_damage_prob"],label='baseline',linewidth=7.0)
        axs3[1].set_xlabel('Years')
        axs3[1].set_ylabel("Distorted probability of damage changes")
        axs3[1].set_title("Distorted probability of damage changes")
        axs3[1].grid(linestyle=':')
        axs3[1].legend(loc='upper left')

        pdf_pages.savefig(fig3)
        plt.close()

        fig4, axs4 = plt.subplots(2, 1, sharex=False, figsize=(2  * figwidth, 2 *figwidth))  

        pi_d_o = np.ones(len(gamma_3_list)) / len(gamma_3_list)
        pi_d_distorted = pi_d_o*res2["gt_dmg"][:,-1]

        axs4[0].hist(gamma_3_list, weights=pi_d_o,label='baseline',alpha=0.5,bins=20)
        axs4[0].hist(gamma_3_list, weights=pi_d_distorted,label=r'$\xi_p=\xi_g=0.050$',alpha=0.5,bins=20)
        axs4[0].set_xlabel(r'$\gamma$')
        axs4[0].set_ylabel("Probability distortion for damage function")
        axs4[0].set_title("Probability distortion for damage function")
        axs4[0].grid(linestyle=':')
        axs4[0].legend(loc='upper left')


        theta_ell = pd.read_csv('./data/model144.csv', header=None).to_numpy()[:, 0]
        # pi_c_o    = np.ones_like(theta_ell)/len(theta_ell)
        # pi_c_distorted = pi_c_o*res2["gt_tech"][:,-1]
        theta_ell_dist = theta_ell*res2["pic_t"][:,-1]

        axs4[1].hist(theta_ell,bins=10,density=True,label='baseline',alpha=0.5)
        axs4[1].hist(theta_ell,bins=10,density=True,weights=res2["pic_t"][:,-1],stacked=True,label=r'$\xi_p=\xi_g=0.050$',alpha=0.5)
        axs4[1].set_xlabel(r'$\theta_\ell,$ Climate Sensitivity')
        axs4[1].set_ylabel("Probability distortion for climate sensitivity models")
        axs4[1].set_title("Probability distortion for climate sensitivity models")
        axs4[1].grid(linestyle=':')
        axs4[1].set_xlim(1, 3)
        axs4[1].legend(loc='upper left')


        pdf_pages.savefig(fig4)
        plt.close()

        # 

        fig5, axs5 = plt.subplots(2, 1, sharex=True, figsize=(2  * figwidth, 2 *figwidth))  


        axs5[0].plot(res1["years"], res1["true_tech_prob"],label=r'$\xi_p=\xi_g=0.025$',linewidth=7.0)
        axs5[0].plot(res2["years"], res2["true_tech_prob"],label=r'$\xi_p=\xi_g=0.050$',linewidth=7.0)
        axs5[0].plot(res3["years"], res3["true_tech_prob"],label='baseline',linewidth=7.0)
        axs5[0].set_xlabel("Years")
        axs5[0].set_ylabel("True probability of first technology jump")
        axs5[0].set_title("True probability of first technology jump")
        axs5[0].grid(linestyle=':')
        axs5[0].legend(loc='upper left')


        axs5[1].plot(res1["years"], res1["true_damage_prob"],label=r'$\xi_p=\xi_g=0.025$',linewidth=7.0)
        axs5[1].plot(res2["years"], res2["true_damage_prob"],label=r'$\xi_p=\xi_g=0.050$',linewidth=7.0)
        axs5[1].plot(res3["years"], res3["true_damage_prob"],label='baseline',linewidth=7.0)
        axs5[1].set_xlabel("Years")
        axs5[1].set_ylabel("True probability of damage changes")
        axs5[1].set_title("True probability of damage changes")
        axs5[1].grid(linestyle=':')
        axs5[1].legend(loc='upper left')


        pdf_pages.savefig(fig5)
        plt.close()

        pdf_pages.close()          
    
    



    res1= model_solution_extraction(2/10000.,0.025,psi_0,psi_1)
    res2= model_solution_extraction(2/10000.,0.050,psi_0,psi_1)
    # res1= model_solution_extraction(1000.,1000.,psi_0,psi_1)
    # res2= model_solution_extraction(1000.,1000.,psi_0,psi_1)
    res3= model_solution_extraction(1000.,1000.,psi_0,psi_1)

    graph_solution_extraction(res1,res2,res3)
# psi_0_grid = np.array([0.003,0.006,0.009])


def graph2(psi_0_meshgrid,psi_1_meshgrid,Ig_initial = 1/120):

    "Only work for xi_a=1000"



    def model_solution_extraction(xi_a,xi_g,psi_0,psi_1):
    
        Data_Dir = "./abatement/data_2tech/"

        File_Dir = "Compare_Psi0_Psi1_xi_a_{}_xi_g_{}_psi_0_{}_psi_1_{}_" .format(xi_a,xi_g,psi_0,psi_1)

        with open(Data_Dir + File_Dir+"model_tech1_pre_damage", "rb") as f:
            tech1 = pickle.load(f)
            

        model_args = (delta, mu_k, kappa,sigma_k, beta_f, zeta, psi_0, psi_1, sigma_g, theta, lambda_bar, vartheta_bar)
        i = tech1["i_star"]
        e = tech1["e_star"]
        x = tech1["x_star"]
        pi_c = tech1["pi_c"]
        g_tech = tech1["g_tech"]
        g_damage =  tech1["g_damage"]
        
        
        # g_damage = np.ones((1, nK, nY, nL))
        res = simulate_pre(grid = (K, Y_short, L), model_args = model_args, 
                                    controls = (i,e,x, g_tech, g_damage, pi_c), 
                                    T0=0, T=IntPeriod, dt=timespan,printing=False)
        return res


    def graph_solution_extraction(res):



        PDF_Dir = "./abatement/pdf_2tech/"
        File_Dir = "Compare_Psi0_8_Psi1_81012_" .format(psi_0,psi_1)
        
        pdf_pages = PdfPages(PDF_Dir+File_Dir+'Years_'+str(IntPeriod)+'.pdf')

        figwidth = 10

        # fig1, axs1 = plt.subplots(3, 1, sharex=True, figsize=(2  * figwidth, 2 *figwidth))  
        fig1, axs1 = plt.subplots(3, 1, sharex=False, figsize=(2  * figwidth, 2 *figwidth))  



        for k in range(len(psi_0_meshgrid_1d)):
            axs1[0].plot(res[k]["years"][res[k]["states"][:, 1]<1.5], (res[k]["x"]/(alpha*np.exp(res[k]["states"][:,0])))[res[k]["states"][:, 1]<1.5],label=r'$\psi_0=$'+str(psi_0_meshgrid_1d[k])+'$\psi_1=$'+str(psi_1_meshgrid_1d[k]),linewidth=7.0)
            axs1[0].set_xlabel('Years')
            axs1[0].set_ylabel("R&D investment (as a ratio of total GDP)")
            axs1[0].set_title("R&D investment (as a ratio of total GDP)")
            axs1[0].grid(linestyle=':')
            axs1[0].legend(loc='upper left')


            axs1[1].plot(res[k]["years"][res[k]["states"][:, 1]<1.5], res[k]["i"][res[k]["states"][:, 1]<1.5],label=r'$\psi_0=$'+str(psi_0_meshgrid_1d[k])+'$\psi_1=$'+str(psi_1_meshgrid_1d[k]),linewidth=7.0)
            axs1[1].set_xlabel('Years')
            axs1[1].set_ylabel("Capital investment")
            axs1[1].set_title("Capital investment")
            axs1[1].grid(linestyle=':')
            axs1[1].legend(loc='upper left')


            axs1[2].plot(res[k]["years"][res[k]["states"][:, 1]<1.5], res[k]["e"][res[k]["states"][:, 1]<1.5],label=r'$\psi_0=$'+str(psi_0_meshgrid_1d[k])+'$\psi_1=$'+str(psi_1_meshgrid_1d[k]),linewidth=7.0)
            # axs1[2].plot(res2["years"][res2["states"][:, 1]<1.5], res2["e"][res2["states"][:, 1]<1.5],label=r'$\xi_p=\xi_g=0.050$',linewidth=7.0)
            # axs1[2].plot(res3["years"][res3["states"][:, 1]<1.5], res3["e"][res3["states"][:, 1]<1.5],label='baseline',linewidth=7.0)
            axs1[2].set_xlabel('Years')
            axs1[2].set_ylabel("Emission")
            axs1[2].set_title("Emission")
            axs1[2].grid(linestyle=':')
            axs1[2].legend(loc='upper left')


        pdf_pages.savefig(fig1)
        plt.close()

        # fig2, axs2 = plt.subplots(2, 1, sharex=True, figsize=(2  * figwidth, 2 *figwidth))  
        fig2, axs2 = plt.subplots(2, 1, sharex=False, figsize=(2  * figwidth, 2 *figwidth))  

        for k in range(len(psi_0_meshgrid_1d)):




            axs2[0].plot(res[k]["years"][res[k]["states"][:, 1]<1.5], res[k]["states"][:, 1][res[k]["states"][:, 1]<1.5],label=r'$\psi_0=$'+str(psi_0_meshgrid_1d[k])+'$\psi_1=$'+str(psi_1_meshgrid_1d[k]),linewidth=7.0)
            # axs2[0].plot(res2["years"][res2["states"][:, 1]<1.5], res2["states"][:, 1][res2["states"][:, 1]<1.5],label=r'$\xi_p=\xi_g=0.050$',linewidth=7.0)
            # axs2[0].plot(res3["years"][res3["states"][:, 1]<1.5], res3["states"][:, 1][res3["states"][:, 1]<1.5],label='baseline',linewidth=7.0)
            axs2[0].set_xlabel('Years')
            axs2[0].set_ylabel("Temperature anomaly")
            axs2[0].set_title("Temperature anomaly")
            axs2[0].grid(linestyle=':')
            axs2[0].legend(loc='upper left')

            axs2[1].plot(res[k]["years"][res[k]["states"][:, 1]<1.5], np.exp(res[k]["states"][:, 2])[res[k]["states"][:, 1]<1.5],label=r'$\psi_0=$'+str(psi_0_meshgrid_1d[k])+'$\psi_1=$'+str(psi_1_meshgrid_1d[k]),linewidth=7.0)
            # axs2[1].plot(res2["years"][res2["states"][:, 1]<1.5], np.exp(res2["states"][:, 2])[res2["states"][:, 1]<1.5],label=r'$\xi_p=\xi_g=0.050$',linewidth=7.0)
            # axs2[1].plot(res3["years"][res3["states"][:, 1]<1.5], np.exp(res3["states"][:, 2])[res3["states"][:, 1]<1.5],label='baseline',linewidth=7.0)
            axs2[1].set_xlabel('Years')
            axs2[1].set_ylabel("Technology jump intensity")
            axs2[1].set_title("Technology jump intensity")
            axs2[1].grid(linestyle=':')
            axs2[1].legend(loc='upper left')


        pdf_pages.savefig(fig2)
        plt.close()


        fig3, axs3 = plt.subplots(2, 1, sharex=False, figsize=(2  * figwidth, 2 *figwidth))  

        for k in range(len(psi_0_meshgrid_1d)):



            axs3[0].plot(res[k]["years"], res[k]["distorted_tech_prob"],label=r'$\psi_0=$'+str(psi_0_meshgrid_1d[k])+'$\psi_1=$'+str(psi_1_meshgrid_1d[k]),linewidth=7.0)
            # axs3[0].plot(res2["years"], res2["distorted_tech_prob"],label=r'$\xi_p=\xi_g=0.050$',linewidth=7.0)
            # axs3[0].plot(res3["years"], res3["distorted_tech_prob"],label='baseline',linewidth=7.0)
            axs3[0].set_xlabel('Years')
            axs3[0].set_ylabel("Distorted probability of first technology jump")
            axs3[0].set_title("Distorted probability of first technology jump")
            axs3[0].grid(linestyle=':')
            axs3[0].legend(loc='upper left')

            axs3[1].plot(res[k]["years"], res[k]["distorted_damage_prob"],label=r'$\psi_0=$'+str(psi_0_meshgrid_1d[k])+'$\psi_1=$'+str(psi_1_meshgrid_1d[k]),linewidth=7.0)
            # axs3[1].plot(res2["years"], res2["distorted_damage_prob"],label=r'$\xi_p=\xi_g=0.050$',linewidth=7.0)
            # axs3[1].plot(res3["years"], res3["distorted_damage_prob"],label='baseline',linewidth=7.0)
            axs3[1].set_xlabel('Years')
            axs3[1].set_ylabel("Distorted probability of damage changes")
            axs3[1].set_title("Distorted probability of damage changes")
            axs3[1].grid(linestyle=':')
            axs3[1].legend(loc='upper left')

        pdf_pages.savefig(fig3)
        plt.close()

        fig5, axs5 = plt.subplots(2, 1, sharex=False, figsize=(2  * figwidth, 2 *figwidth))  

        for k in range(len(psi_0_meshgrid_1d)):



            axs5[0].plot(res[k]["years"], res[k]["true_tech_prob"],label=r'$\psi_0=$'+str(psi_0_meshgrid_1d[k])+'$\psi_1=$'+str(psi_1_meshgrid_1d[k]),linewidth=7.0)
            # axs5[0].plot(res2["years"], res2["true_tech_prob"],label=r'$\xi_p=\xi_g=0.050$',linewidth=7.0)
            # axs5[0].plot(res3["years"], res3["true_tech_prob"],label='baseline',linewidth=7.0)
            axs5[0].set_xlabel("Years")
            axs5[0].set_ylabel("True probability of first technology jump")
            axs5[0].set_title("True probability of first technology jump")
            axs5[0].grid(linestyle=':')
            axs5[0].legend(loc='upper left')


            axs5[1].plot(res[k]["years"], res[k]["true_damage_prob"],label=r'$\psi_0=$'+str(psi_0_meshgrid_1d[k])+'$\psi_1=$'+str(psi_1_meshgrid_1d[k]),linewidth=7.0)
            # axs5[1].plot(res2["years"], res2["true_damage_prob"],label=r'$\xi_p=\xi_g=0.050$',linewidth=7.0)
            # axs5[1].plot(res3["years"], res3["true_damage_prob"],label='baseline',linewidth=7.0)
            axs5[1].set_xlabel("Years")
            axs5[1].set_ylabel("True probability of damage changes")
            axs5[1].set_title("True probability of damage changes")
            axs5[1].grid(linestyle=':')
            axs5[1].legend(loc='upper left')


        pdf_pages.savefig(fig5)
        plt.close()

        pdf_pages.close()          
    
    
    res = []
    for psi_0,psi_1 in zip(psi_0_meshgrid_1d,psi_1_meshgrid_1d):
        res.append(model_solution_extraction(1000.,1000.,psi_0,psi_1))


    graph_solution_extraction(res)


# psi_0_grid = np.array([0.003,0.006,0.009])

delta = 0.01
alpha = 0.115
kappa = 6.667
mu_k  = -0.043
sigma_k = 0.0095
beta_f = 1.86/1000
sigma_y = 1.2 * 1.86 / 1000
zeta = 0.0
# psi_0 = 0.00025
# psi_1 = 1/2
sigma_g = 0.016
gamma_1 = 1.7675 / 1000
gamma_2 = 0.0022 * 2
gamma_3_list = np.linspace(0., 1./3., 6)
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






IntPeriod = 50
timespan = 1/12

# psi_0_grid = np.array([0.006,0.009])
# # # psi_0_grid = np.array([0.009])
# # # psi_1_grid = np.array([.5,.7,.9])
# psi_1_grid = np.array([.3,.4])

psi_0_grid = np.array([0.008, 0.010, 0.012])
psi_1_grid = np.array([.8])

psi_0_meshgrid,psi_1_meshgrid = np.meshgrid(psi_0_grid,psi_1_grid)

psi_0_meshgrid_1d =psi_0_meshgrid.ravel(order='F')
psi_1_meshgrid_1d = psi_1_meshgrid.ravel(order='F')



# for psi_0,psi_1 in zip(psi_0_meshgrid_1d,psi_1_meshgrid_1d):
#     print(psi_0,psi_1)
#     graph(psi_0,psi_1)

graph2(psi_0_meshgrid,psi_1_meshgrid)
