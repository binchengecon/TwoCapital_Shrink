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
import argparse


parser = argparse.ArgumentParser(description="xi_r values")
parser.add_argument("--dataname",type=str,default="2jump_step_0.2_0.2_0.2_LR_0.008_Psi01Comparison")
parser.add_argument("--pdfname",type=str,default="mercury")
parser.add_argument("--xi_a", type=float, default=1000.)
parser.add_argument("--xi_g", type=float, default=1000.)
parser.add_argument("--hXarr",nargs='+',type=float,default=(0.2, 0.2, 0.2))
parser.add_argument("--Xminarr",nargs='+',type=float,default=(4.00, 0.0 ,-5.5 ,0.0))
parser.add_argument("--Xmaxarr",nargs='+',type=float,default=(9.00, 4.0 ,0.0, 3.0))
parser.add_argument("--psi0arr",nargs='+',type=float,default=(0.005, 0.008 ,0.010, 0.012))
parser.add_argument("--psi1arr",nargs='+',type=float,default=(0.5, 0.6, 0.7 ,0.8))
# parser.add_argument("--year",type=int,default=60)
# parser.add_argument("--time",type=float,default=1/12.)
args = parser.parse_args()


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



Xminarr = args.Xminarr
Xmaxarr = args.Xmaxarr
hXarr = args.hXarr

K_min = Xminarr[0]
K_max = Xmaxarr[0]
hK    = hXarr[0]
K     = np.arange(K_min, K_max + hK, hK)
nK    = len(K)
Y_min = Xminarr[1]
Y_max = Xmaxarr[1]
hY    = hXarr[1] # make sure it is float instead of int
Y     = np.arange(Y_min, Y_max + hY, hY)
nY    = len(Y)
L_min = Xminarr[2]
L_max = Xmaxarr[2]
hL    = hXarr[2]
L     = np.arange(L_min, L_max+hL,  hL)
nL    = len(L)


id_2 = np.abs(Y - y_bar).argmin()
Y_min_short = Xminarr[3]
Y_max_short = Xmaxarr[3]
Y_short     = np.arange(Y_min_short, Y_max_short + hY, hY)
nY_short    = len(Y_short)

# print("bY_short={:d}".format(nY_short))
(K_mat, Y_mat, L_mat) = np.meshgrid(K, Y_short, L, indexing="ij")

stateSpace = np.hstack([K_mat.reshape(-1,1,order = 'F'), Y_mat.reshape(-1,1,order = 'F'), L_mat.reshape(-1, 1, order='F')])







IntPeriod = 60
timespan = 1/12

# psi_0_grid = np.array([0.006,0.009])
# # # psi_0_grid = np.array([0.009])
# # # psi_1_grid = np.array([.5,.7,.9])
# psi_1_grid = np.array([.3,.4])

psi_0_grid = args.psi0arr
psi_1_grid = args.psi1arr

psi_0_meshgrid,psi_1_meshgrid = np.meshgrid(psi_0_grid,psi_1_grid)

psi_0_meshgrid_1d =psi_0_meshgrid.ravel(order='F')
psi_1_meshgrid_1d = psi_1_meshgrid.ravel(order='F')


mpl.rcParams["lines.linewidth"] = 4
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["figure.figsize"] = (12,7)
mpl.rcParams["font.size"] = 20
mpl.rcParams["legend.frameon"] = False


def simulate_pre(
    grid = (), model_args = (), controls = (), initial=(np.log(85/0.115), 1.1, -3.7), 
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


def graph_fixpsi0(psi_0,psi_1_grid,Ig_initial = 1/120):

    "Only work for xi_a=1000"



    def model_solution_extraction(xi_a,xi_g,psi_0,psi_1):
    
        # Data_Dir = "./abatement/data_2tech/"+args.dataname+"/"
        Output_Dir = "/scratch/bincheng/"
        Data_Dir = Output_Dir+"abatement/data_2tech/"+args.dataname+"/"


        File_Dir = "xi_a_{}_xi_g_{}_psi_0_{}_psi_1_{}_" .format(xi_a,xi_g,psi_0,psi_1)
        model_dir_post = Data_Dir + File_Dir+"model_tech1_pre_damage"

        model_simul_dir_post = Data_Dir + File_Dir+"model_tech1_pre_damage_simul"


        if os.path.exists(model_simul_dir_post):
            print("which passed 1")
            res = pickle.load(open(model_simul_dir_post, "rb"))

        else:
            print("which passed 2")

            with open(model_dir_post, "rb") as f:
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

            with open(model_simul_dir_post, "wb") as f:
                pickle.dump(res,f)

            res = pickle.load(open(model_simul_dir_post, "rb"))

        # print("which passed 3")

        # with open(model_dir_post, "rb") as f:
        #     tech1 = pickle.load(f)
        
        # model_args = (delta, mu_k, kappa,sigma_k, beta_f, zeta, psi_0, psi_1, sigma_g, theta, lambda_bar, vartheta_bar)
        # i = tech1["i_star"]
        # e = tech1["e_star"]
        # x = tech1["x_star"]
        # pi_c = tech1["pi_c"]
        # g_tech = tech1["g_tech"]
        # g_damage =  tech1["g_damage"]
        
        
        # # g_damage = np.ones((1, nK, nY, nL))
        # res = simulate_pre(grid = (K, Y_short, L), model_args = model_args, 
        #                             controls = (i,e,x, g_tech, g_damage, pi_c), 
        #                             T0=0, T=IntPeriod, dt=timespan,printing=False)

        # with open(model_simul_dir_post, "wb") as f:
        #     pickle.dump(res,f)

        # res = pickle.load(open(model_simul_dir_post, "rb"))

        return res




    def graph_solution_extraction(res):

        PDF_Dir = "./abatement/pdf_2tech/"+args.dataname+"/"

        if not os.path.exists(PDF_Dir):
            os.mkdir(PDF_Dir)

        File_Dir = args.pdfname+"_psi0={:.3f}_" .format(psi_0) 
        
        pdf_pages = PdfPages(PDF_Dir+File_Dir+'Years_'+str(IntPeriod)+'.pdf')

        figwidth = 10

        # fig1, axs1 = plt.subplots(3, 1, sharex=True, figsize=(2  * figwidth, 2 *figwidth))  
        fig1, axs1 = plt.subplots(3, 1, sharex=False, figsize=(12, 3 *figwidth))  



        for k in range(len(psi_1_grid)):
            axs1[0].plot(res[k]["years"], (res[k]["x"]/(alpha*np.exp(res[k]["states"][:,0])))*100,label=r'$\psi_0=$'+str(psi_0)+'$\psi_1=$'+str(psi_1_grid[k]),linewidth=7.0)
            # axs1[0].plot(res[k]["years"], (res[k]["x"]),label=r'$\psi_0=$'+str(psi_0_meshgrid_1d[k])+'$\psi_1=$'+str(psi_1_meshgrid_1d[k]),linewidth=7.0)
            axs1[0].set_xlabel('Years')
            axs1[0].set_ylabel('$\%$ of GDP')
            axs1[0].set_ylim(0,0.45)
            axs1[0].set_title('R&D investment as percentage of  GDP')            
            # axs1[0].set_ylabel('unit of capital')
            # axs1[0].set_title('R&D investment in unit of capital')
            axs1[0].grid(linestyle=':')
            axs1[0].legend(loc='upper right')


            axs1[1].plot(res[k]["years"], res[k]["i"],label=r'$\psi_0=$'+str(psi_0)+'$\psi_1=$'+str(psi_1_grid[k]),linewidth=7.0)
            axs1[1].set_xlabel('Years')
            axs1[1].set_ylabel("Capital investment")
            axs1[1].set_title("Capital investment")
            axs1[1].grid(linestyle=':')
            axs1[1].legend(loc='upper left')


            axs1[2].plot(res[k]["years"], res[k]["e"],label=r'$\psi_0=$'+str(psi_0)+'$\psi_1=$'+str(psi_1_grid[k]),linewidth=7.0)
            # axs1[2].plot(res2["years"][res2["states"][:, 1]<1.5], res2["e"][res2["states"][:, 1]<1.5],label=r'$\xi_p=\xi_g=0.050$',linewidth=7.0)
            # axs1[2].plot(res3["years"][res3["states"][:, 1]<1.5], res3["e"][res3["states"][:, 1]<1.5],label='baseline',linewidth=7.0)
            axs1[2].set_xlabel('Years')
            axs1[2].set_ylabel("Emission")
            axs1[2].set_title("Emission")
            axs1[2].grid(linestyle=':')
            axs1[2].legend(loc='upper left')

        # plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/whole.png")
        pdf_pages.savefig(fig1)
        plt.close()

        # fig2, axs2 = plt.subplots(2, 1, sharex=True, figsize=(2  * figwidth, 2 *figwidth))  
        fig2, axs2 = plt.subplots(2, 1, sharex=False, figsize=(12, 2 *figwidth))  

        for k in range(len(psi_1_grid)):




            axs2[0].plot(res[k]["years"], res[k]["states"][:, 1],label=r'$\psi_0=$'+str(psi_0)+'$\psi_1=$'+str(psi_1_grid[k]),linewidth=7.0)
            # axs2[0].plot(res2["years"][res2["states"][:, 1]<1.5], res2["states"][:, 1][res2["states"][:, 1]<1.5],label=r'$\xi_p=\xi_g=0.050$',linewidth=7.0)
            # axs2[0].plot(res3["years"][res3["states"][:, 1]<1.5], res3["states"][:, 1][res3["states"][:, 1]<1.5],label='baseline',linewidth=7.0)
            axs2[0].set_xlabel('Years')
            axs2[0].set_ylabel("Temperature anomaly")
            axs2[0].set_title("Temperature anomaly")
            axs2[0].grid(linestyle=':')
            axs2[0].legend(loc='upper left')

            axs2[1].plot(res[k]["years"], np.exp(res[k]["states"][:, 2]),label=r'$\psi_0=$'+str(psi_0)+'$\psi_1=$'+str(psi_1_grid[k]),linewidth=7.0)
            # axs2[1].plot(res2["years"][res2["states"][:, 1]<1.5], np.exp(res2["states"][:, 2])[res2["states"][:, 1]<1.5],label=r'$\xi_p=\xi_g=0.050$',linewidth=7.0)
            # axs2[1].plot(res3["years"][res3["states"][:, 1]<1.5], np.exp(res3["states"][:, 2])[res3["states"][:, 1]<1.5],label='baseline',linewidth=7.0)
            axs2[1].set_xlabel('Years')
            axs2[1].set_ylabel("Technology jump intensity")
            axs2[1].set_title("Technology jump intensity")
            axs2[1].grid(linestyle=':')
            axs2[1].legend(loc='upper left')


        pdf_pages.savefig(fig2)
        plt.close()


        fig3, axs3 = plt.subplots(2, 1, sharex=False, figsize=(12, 2 *figwidth))  

        for k in range(len(psi_1_grid)):



            axs3[0].plot(res[k]["years"], res[k]["distorted_tech_prob"],label=r'$\psi_0=$'+str(psi_0)+'$\psi_1=$'+str(psi_1_grid[k]),linewidth=7.0)
            # axs3[0].plot(res2["years"], res2["distorted_tech_prob"],label=r'$\xi_p=\xi_g=0.050$',linewidth=7.0)
            # axs3[0].plot(res3["years"], res3["distorted_tech_prob"],label='baseline',linewidth=7.0)
            axs3[0].set_xlabel('Years')
            axs3[0].set_ylabel("Distorted probability of first technology jump")
            axs3[0].set_title("Distorted probability of first technology jump")
            axs3[0].grid(linestyle=':')
            axs3[0].legend(loc='upper left')

            axs3[1].plot(res[k]["years"], res[k]["distorted_damage_prob"],label=r'$\psi_0=$'+str(psi_0)+'$\psi_1=$'+str(psi_1_grid[k]),linewidth=7.0)
            # axs3[1].plot(res2["years"], res2["distorted_damage_prob"],label=r'$\xi_p=\xi_g=0.050$',linewidth=7.0)
            # axs3[1].plot(res3["years"], res3["distorted_damage_prob"],label='baseline',linewidth=7.0)
            axs3[1].set_xlabel('Years')
            axs3[1].set_ylabel("Distorted probability of damage changes")
            axs3[1].set_title("Distorted probability of damage changes")
            axs3[1].grid(linestyle=':')
            axs3[1].legend(loc='upper left')

        pdf_pages.savefig(fig3)
        plt.close()

        fig5, axs5 = plt.subplots(2, 1, sharex=False, figsize=(12, 2 *figwidth))  

        for k in range(len(psi_1_grid)):



            axs5[0].plot(res[k]["years"], res[k]["true_tech_prob"],label=r'$\psi_0=$'+str(psi_0)+'$\psi_1=$'+str(psi_1_grid[k]),linewidth=7.0)
            # axs5[0].plot(res2["years"], res2["true_tech_prob"],label=r'$\xi_p=\xi_g=0.050$',linewidth=7.0)
            # axs5[0].plot(res3["years"], res3["true_tech_prob"],label='baseline',linewidth=7.0)
            axs5[0].set_xlabel("Years")
            axs5[0].set_ylabel("True probability of first technology jump")
            axs5[0].set_title("True probability of first technology jump")
            axs5[0].grid(linestyle=':')
            axs5[0].legend(loc='upper left')


            axs5[1].plot(res[k]["years"], res[k]["true_damage_prob"],label=r'$\psi_0=$'+str(psi_0)+'$\psi_1=$'+str(psi_1_grid[k]),linewidth=7.0)
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

        # R&D Investment

        # plt.plot(res[k]["years"], (res[k]["x"]/(alpha*np.exp(res[k]["states"][:,0])))*100,label=r'$\psi_0=$'+str(psi_0_meshgrid_1d[k])+'$\psi_1=$'+str(psi_1_meshgrid_1d[k]),linewidth=7.0)
        # # axs1[0].plot(res[k]["years"], (res[k]["x"]),label=r'$\psi_0=$'+str(psi_0_meshgrid_1d[k])+'$\psi_1=$'+str(psi_1_meshgrid_1d[k]),linewidth=7.0)
        # plt.set_xlabel('Years')
        # plt.set_ylabel('$\%$ of GDP')
        # plt.set_title('R&D investment as percentage of  GDP')            
        # # plt.set_ylabel('unit of capital')
        # # plt.set_title('R&D investment in unit of capital')
        # plt.grid(linestyle=':')
        # for k in range(len(psi_1_grid)):

        #     # plt.plot(res[k]["years"], (res[k]["x"]/(alpha*np.exp(res[k]["states"][:,0])))*100,label=r'$\psi_0=$'+str(psi_0_meshgrid_1d[k])+'$\psi_1=$'+str(psi_1_meshgrid_1d[k]),color="C3")
        #     plt.plot(res[k]["years"], (res[k]["x"]/(alpha*np.exp(res[k]["states"][:,0])))*100,label='$\\psi_0=$'+str(psi_0_meshgrid_1d[k])+'$, \\psi_1=$'+str(psi_1_meshgrid_1d[k]))
        #     plt.grid(linestyle=':')
        #     plt.xlabel('Years')
        #     plt.ylabel('$\%$ of GDP')
        #     plt.title('R&D investment as percentage of  GDP')   
        #     plt.ylim(0,0.4)
        #     plt.legend(loc='upper left')        

        # plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/RDInvestment.pdf")
        # plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/RDInvestment.png")
        # plt.close()

        for k in range(len(psi_1_grid)):

            # plt.plot(res[k]["years"], (res[k]["x"]/(alpha*np.exp(res[k]["states"][:,0])))*100,label=r'$\psi_0=$'+str(psi_0_meshgrid_1d[k])+'$\psi_1=$'+str(psi_1_meshgrid_1d[k]),color="C3")
            plt.plot(res[k]["years"], (res[k]["x"]/(alpha*np.exp(res[k]["states"][:,0])))*100,label='$\psi_0=${:.3f},$\psi_1=${:.1f}' .format(psi_0,psi_1_grid[k])  )
            plt.grid(linestyle=':')
            plt.xlabel('Years')
            plt.ylabel('$\%$ of GDP')
            plt.title('R&D investment as percentage of  GDP')   
            plt.ylim(0,0.45)
            plt.legend(loc='upper left')        

        plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/RDInvestment_psi0={:.3f}.pdf".format(psi_0))
        plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/RDInvestment_psi0={:.3f}.png".format(psi_0))
        plt.close()
    
    res = []
    for psi_1 in psi_1_grid:
        res.append(model_solution_extraction(1000.,1000.,psi_0,psi_1))


    graph_solution_extraction(res)


def graph_fixpsi1(psi_0_grid,psi_1,Ig_initial = 1/120):

    "Only work for xi_a=1000"



    def model_solution_extraction(xi_a,xi_g,psi_0,psi_1):
    
        # Data_Dir = "./abatement/data_2tech/"+args.dataname+"/"
        Output_Dir = "/scratch/bincheng/"
        Data_Dir = Output_Dir+"abatement/data_2tech/"+args.dataname+"/"


        File_Dir = "xi_a_{}_xi_g_{}_psi_0_{}_psi_1_{}_" .format(xi_a,xi_g,psi_0,psi_1)
        model_dir_post = Data_Dir + File_Dir+"model_tech1_pre_damage"

        model_simul_dir_post = Data_Dir + File_Dir+"model_tech1_pre_damage_simul"


        if os.path.exists(model_simul_dir_post):
            print("which passed 1")
            res = pickle.load(open(model_simul_dir_post, "rb"))

        else:
            print("which passed 2")

            with open(model_dir_post, "rb") as f:
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

            with open(model_simul_dir_post, "wb") as f:
                pickle.dump(res,f)

            res = pickle.load(open(model_simul_dir_post, "rb"))

        
        return res




    def graph_solution_extraction(res):

        PDF_Dir = "./abatement/pdf_2tech/"+args.dataname+"/"

        if not os.path.exists(PDF_Dir):
            os.mkdir(PDF_Dir)

        File_Dir = args.pdfname+"_psi1={:.1f}_" .format(psi_1) 
        
        pdf_pages = PdfPages(PDF_Dir+File_Dir+'Years_'+str(IntPeriod)+'.pdf')

        figwidth = 10

        # fig1, axs1 = plt.subplots(3, 1, sharex=True, figsize=(2  * figwidth, 2 *figwidth))  
        fig1, axs1 = plt.subplots(3, 1, sharex=False, figsize=(12, 3 *figwidth))  



        for k in range(len(psi_0_grid)):
            axs1[0].plot(res[k]["years"], (res[k]["x"]/(alpha*np.exp(res[k]["states"][:,0])))*100,label=r'$\psi_0=$'+str(psi_0_meshgrid_1d[k])+'$\psi_1=$'+str(psi_1_meshgrid_1d[k]),linewidth=7.0)
            # axs1[0].plot(res[k]["years"], (res[k]["x"]),label=r'$\psi_0=$'+str(psi_0_meshgrid_1d[k])+'$\psi_1=$'+str(psi_1_meshgrid_1d[k]),linewidth=7.0)
            axs1[0].set_xlabel('Years')
            axs1[0].set_ylabel('$\%$ of GDP')
            axs1[0].set_ylim(0,0.45)
            axs1[0].set_title('R&D investment as percentage of  GDP')            
            # axs1[0].set_ylabel('unit of capital')
            # axs1[0].set_title('R&D investment in unit of capital')
            axs1[0].grid(linestyle=':')
            axs1[0].legend(loc='upper right')


            axs1[1].plot(res[k]["years"], res[k]["i"],label=r'$\psi_0=$'+str(psi_0_meshgrid_1d[k])+'$\psi_1=$'+str(psi_1_meshgrid_1d[k]),linewidth=7.0)
            axs1[1].set_xlabel('Years')
            axs1[1].set_ylabel("Capital investment")
            axs1[1].set_title("Capital investment")
            axs1[1].grid(linestyle=':')
            axs1[1].legend(loc='upper left')


            axs1[2].plot(res[k]["years"], res[k]["e"],label=r'$\psi_0=$'+str(psi_0_meshgrid_1d[k])+'$\psi_1=$'+str(psi_1_meshgrid_1d[k]),linewidth=7.0)
            # axs1[2].plot(res2["years"][res2["states"][:, 1]<1.5], res2["e"][res2["states"][:, 1]<1.5],label=r'$\xi_p=\xi_g=0.050$',linewidth=7.0)
            # axs1[2].plot(res3["years"][res3["states"][:, 1]<1.5], res3["e"][res3["states"][:, 1]<1.5],label='baseline',linewidth=7.0)
            axs1[2].set_xlabel('Years')
            axs1[2].set_ylabel("Emission")
            axs1[2].set_title("Emission")
            axs1[2].grid(linestyle=':')
            axs1[2].legend(loc='upper left')

        # plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/whole.png")
        pdf_pages.savefig(fig1)
        plt.close()

        # fig2, axs2 = plt.subplots(2, 1, sharex=True, figsize=(2  * figwidth, 2 *figwidth))  
        fig2, axs2 = plt.subplots(2, 1, sharex=False, figsize=(12, 2 *figwidth))  

        for k in range(len(psi_0_grid)):




            axs2[0].plot(res[k]["years"], res[k]["states"][:, 1],label=r'$\psi_0=$'+str(psi_0_meshgrid_1d[k])+'$\psi_1=$'+str(psi_1_meshgrid_1d[k]),linewidth=7.0)
            # axs2[0].plot(res2["years"][res2["states"][:, 1]<1.5], res2["states"][:, 1][res2["states"][:, 1]<1.5],label=r'$\xi_p=\xi_g=0.050$',linewidth=7.0)
            # axs2[0].plot(res3["years"][res3["states"][:, 1]<1.5], res3["states"][:, 1][res3["states"][:, 1]<1.5],label='baseline',linewidth=7.0)
            axs2[0].set_xlabel('Years')
            axs2[0].set_ylabel("Temperature anomaly")
            axs2[0].set_title("Temperature anomaly")
            axs2[0].grid(linestyle=':')
            axs2[0].legend(loc='upper left')

            axs2[1].plot(res[k]["years"], np.exp(res[k]["states"][:, 2]),label=r'$\psi_0=$'+str(psi_0_meshgrid_1d[k])+'$\psi_1=$'+str(psi_1_meshgrid_1d[k]),linewidth=7.0)
            # axs2[1].plot(res2["years"][res2["states"][:, 1]<1.5], np.exp(res2["states"][:, 2])[res2["states"][:, 1]<1.5],label=r'$\xi_p=\xi_g=0.050$',linewidth=7.0)
            # axs2[1].plot(res3["years"][res3["states"][:, 1]<1.5], np.exp(res3["states"][:, 2])[res3["states"][:, 1]<1.5],label='baseline',linewidth=7.0)
            axs2[1].set_xlabel('Years')
            axs2[1].set_ylabel("Technology jump intensity")
            axs2[1].set_title("Technology jump intensity")
            axs2[1].grid(linestyle=':')
            axs2[1].legend(loc='upper left')


        pdf_pages.savefig(fig2)
        plt.close()


        fig3, axs3 = plt.subplots(2, 1, sharex=False, figsize=(12, 2 *figwidth))  

        for k in range(len(psi_0_grid)):



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

        fig5, axs5 = plt.subplots(2, 1, sharex=False, figsize=(12, 2 *figwidth))  

        for k in range(len(psi_0_grid)):



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

        # R&D Investment

        # plt.plot(res[k]["years"], (res[k]["x"]/(alpha*np.exp(res[k]["states"][:,0])))*100,label=r'$\psi_0=$'+str(psi_0_meshgrid_1d[k])+'$\psi_1=$'+str(psi_1_meshgrid_1d[k]),linewidth=7.0)
        # # axs1[0].plot(res[k]["years"], (res[k]["x"]),label=r'$\psi_0=$'+str(psi_0_meshgrid_1d[k])+'$\psi_1=$'+str(psi_1_meshgrid_1d[k]),linewidth=7.0)
        # plt.set_xlabel('Years')
        # plt.set_ylabel('$\%$ of GDP')
        # plt.set_title('R&D investment as percentage of  GDP')            
        # # plt.set_ylabel('unit of capital')
        # # plt.set_title('R&D investment in unit of capital')
        # plt.grid(linestyle=':')
        # for k in range(len(psi_1_grid)):

        #     # plt.plot(res[k]["years"], (res[k]["x"]/(alpha*np.exp(res[k]["states"][:,0])))*100,label=r'$\psi_0=$'+str(psi_0_meshgrid_1d[k])+'$\psi_1=$'+str(psi_1_meshgrid_1d[k]),color="C3")
        #     plt.plot(res[k]["years"], (res[k]["x"]/(alpha*np.exp(res[k]["states"][:,0])))*100,label='$\\psi_0=$'+str(psi_0_meshgrid_1d[k])+'$, \\psi_1=$'+str(psi_1_meshgrid_1d[k]))
        #     plt.grid(linestyle=':')
        #     plt.xlabel('Years')
        #     plt.ylabel('$\%$ of GDP')
        #     plt.title('R&D investment as percentage of  GDP')   
        #     plt.ylim(0,0.4)
        #     plt.legend(loc='upper left')        

        # plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/RDInvestment.pdf")
        # plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/RDInvestment.png")
        # plt.close()

        for k in range(len(psi_0_grid)):

            # plt.plot(res[k]["years"], (res[k]["x"]/(alpha*np.exp(res[k]["states"][:,0])))*100,label=r'$\psi_0=$'+str(psi_0_meshgrid_1d[k])+'$\psi_1=$'+str(psi_1_meshgrid_1d[k]),color="C3")
            plt.plot(res[k]["years"], (res[k]["x"]/(alpha*np.exp(res[k]["states"][:,0])))*100,label='$\psi_0=${:.3f},$\psi_1=${:.1f}' .format(psi_0_grid[k],psi_1)  )
            plt.grid(linestyle=':')
            plt.xlabel('Years')
            plt.ylabel('$\%$ of GDP')
            plt.title('R&D investment as percentage of  GDP')   
            plt.ylim(0,0.45)
            plt.legend(loc='upper left')        

        plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/RDInvestment_psi1={:.1f}.pdf".format(psi_1))
        plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/RDInvestment_psi1={:.1f}.png".format(psi_1))
        plt.close()
    
    res = []
    for psi_0 in psi_0_grid:
        res.append(model_solution_extraction(1000.,1000.,psi_0,psi_1))


    graph_solution_extraction(res)



for psi_0 in psi_0_grid:
    graph_fixpsi0(psi_0,psi_1_grid)

for psi_1 in psi_1_grid:
    graph_fixpsi1(psi_0_grid,psi_1)
