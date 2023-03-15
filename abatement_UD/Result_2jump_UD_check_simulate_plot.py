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
from src.Utility import finiteDiff_3D
import src.PreSolver_CRS
import src.ResultSolver_CRS
import os
import argparse


parser = argparse.ArgumentParser(description="xi_r values")
parser.add_argument("--dataname",type=str)
parser.add_argument("--pdfname",type=str)

parser.add_argument("--xiaarr",nargs='+', type=float)
parser.add_argument("--xigarr",nargs='+', type=float)

parser.add_argument("--psi0arr",nargs='+',type=float)
parser.add_argument("--psi1arr",nargs='+',type=float)


parser.add_argument("--hXarr",nargs='+',type=float)
parser.add_argument("--Xminarr",nargs='+',type=float)
parser.add_argument("--Xmaxarr",nargs='+',type=float)

parser.add_argument("--epsilonarr",nargs='+',type=float)
parser.add_argument("--fractionarr",nargs='+',type=float)
parser.add_argument("--maxiterarr",nargs='+',type=int)

parser.add_argument("--scheme",type=str)
parser.add_argument("--HJB_solution",type=str)



parser.add_argument("--auto",type=int)
parser.add_argument("--IntPeriod",type=int)

args = parser.parse_args()

epsilonarr = args.epsilonarr
fractionarr = args.fractionarr
maxiterarr = args.maxiterarr


# Update = args.Update
IntPeriod = args.IntPeriod
timespan = 1/12

# psi0arr = np.array([0.006,0.009])
# # # psi0arr = np.array([0.009])
# # # psi1arr = np.array([.5,.7,.9])
# psi1arr = np.array([.3,.4])

psi0arr = args.psi0arr
psi1arr = args.psi1arr
psi2arr = args.psi2arr
xiaarr = args.xiaarr
xigarr = args.xigarr 


Xminarr = args.Xminarr
Xmaxarr = args.Xmaxarr
hXarr = args.hXarr

auto = args.auto

scheme = args.scheme
HJB_solution = args.HJB_solution


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


num_gamma = args.num_gamma
gamma_3_list = np.linspace(0,1./3.,num_gamma)

y_bar = 2.
y_bar_lower = 1.5

theta_ell = pd.read_csv('./data/model144_p.csv', header=None).to_numpy()[:, 0]/1000.

# Tech
theta = 3
lambda_bar = 0.1206
vartheta_bar = 0.0453

lambda_bar_first = lambda_bar / 2.
vartheta_bar_first = vartheta_bar / 2.

lambda_bar_second = 1e-3
vartheta_bar_second = 0.





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

n_bar1 = len(Y_short)-1
n_bar2 = np.abs(Y_short - y_bar).argmin()


# print("bY_short={:d}".format(nY_short))
(K_mat, Y_mat, L_mat) = np.meshgrid(K, Y_short, L, indexing="ij")

theta_ell = np.array([temp * np.ones(K_mat.shape) for temp in theta_ell])






mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["figure.figsize"] = (16,10)
mpl.rcParams["font.size"] = 15
mpl.rcParams["legend.frameon"] = False
mpl.style.use('classic')
mpl.rcParams["lines.linewidth"] = 5


print("After, figure default size is: ", plt.rcParams["savefig.bbox"])
print("After, figure default size is: ", plt.rcParams["figure.figsize"])
print("After, figure default dpi is: ", plt.rcParams["figure.dpi"])
print("After, figure default size is: ", plt.rcParams["font.size"])
print("After, legend.frameon is: ", plt.rcParams["legend.frameon"])
print("After, lines.linewidth is: ", plt.rcParams["lines.linewidth"])


def simulate_pre(
    grid = (), 
    model_args = (), 
    controls = (),
    ME = (),
    n_bar = (),  
    initial=(np.log(85/0.115), 1.1, np.log(448/20)), 
    T0=0, T=40, dt=1/12,
    printing=True):

    K, Y, L = grid

    if printing==True:
        print("K_min={},K_max={},Y_min={},Y_max={},L_min={},L_max={}" .format(K.min(),K.max(),Y.min(),Y.max(),L.min(),L.max()))

    K_min, K_max, Y_min, Y_max, L_min, L_max = min(K), max(K), min(Y), max(Y), min(L), max(L)
    hK, hY, hL = K[1] - K[0], Y[1] - Y[0], L[1]-L[0]

    delta, mu_k, kappa, sigma_k, beta_f, zeta, psi_0, psi_1, psi_2, sigma_g, theta, lambda_bar, vartheta_bar = model_args
    ii, ee, xx, g_tech, g_damage, pi_c, v = controls
    ME_base = ME
    n_bar = n_bar
    K_0, Y_0, L_0 = initial

    
    # #### Temporary Checks
    # print("---------------Temporary Checks Start-----------")
    # ee_modified = ee/(alpha*lambda_bar* np.exp(K_mat))
    # print("ee_modified in [{},{}]".format(ee_modified.min(), ee_modified.max()))
    # # plt.close()
    # # plt.plot(Y, ee_modified[:,:,-1].T)
    # # plt.ylim(0,1)
    # # plt.savefig("./abatement/pdf_2tech/2jump_step_4.00,9.00_0.0,4.0_1.0,6.0_SS_0.2_LR_0.1/AA_ee_modified.png")
    # # plt.close()
    # print("---------------Temporary Checks End-----------")

    Y = Y[:n_bar+1]
    
    ii = ii[:,:n_bar+1,:]
    ee = ee[:,:n_bar+1,:]
    xx = xx[:,:n_bar+1,:]
    g_tech = g_tech[:,:n_bar+1,:]
    g_damage = g_damage[:,:,:n_bar+1,:]
    pi_c = pi_c[:,:,:n_bar+1,:]
    v = v[:,:n_bar+1,:]
    


    (K_mat, Y_mat, L_mat) = np.meshgrid(K, Y, L, indexing = 'ij')

    jj = alpha * vartheta_bar * (1 - ee / (alpha * lambda_bar * np.exp(K_mat)))**theta
    
    jj[jj <= 1e-16] = 1e-16
    consumption = alpha - ii - jj - xx
    ME_total = delta/ consumption  * alpha * vartheta_bar * theta * (1 - ee / ( alpha * lambda_bar * np.exp(K_mat)))**(theta - 1) /( alpha * lambda_bar * np.exp(K_mat) )


    years  = np.arange(T0, T0 + T + dt, dt)
    pers   = len(years)
       

    # some parameters remaiend unchanged across runs
    gamma_1  = 0.00017675
    gamma_2  = 2. * 0.0022
    beta_f   = 1.86 / 1000
    sigma_y  = 1.2 * 1.86 / 1000
    
    theta_ell = pd.read_csv("./data/model144_p.csv", header=None).to_numpy()[:, 0]/1000.
    pi_c_o = np.ones(len(theta_ell)) / len(theta_ell)
    pi_c_o = np.array([temp * np.ones(K_mat.shape) for temp in pi_c_o])
    # theta_ell = np.array([temp * np.ones(K_mat.shape) for temp in theta_ell])

    dL = finiteDiff_3D(v, 2,1,hL )

    gridpoints = (K, Y, L)

    i_func = RegularGridInterpolator(gridpoints, ii)
    e_func = RegularGridInterpolator(gridpoints, ee)
    x_func = RegularGridInterpolator(gridpoints, xx)
    tech_func = RegularGridInterpolator(gridpoints, g_tech)
    dL_func   = RegularGridInterpolator(gridpoints, dL)
    ME_total_func = RegularGridInterpolator(gridpoints, ME_total)
    ME_base_func = RegularGridInterpolator(gridpoints, ME_base)
    
    n_damage = len(g_damage)

    damage_func_list = []
    for i in range(n_damage):
        func_i = RegularGridInterpolator(gridpoints, g_damage[i])
        damage_func_list.append(func_i)
        
    n_climate = len(pi_c)
    
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

    def get_dL(x):
        return dL_func(x)


    def mu_K(i_x):
        return mu_k + i_x - 0.5 * kappa * i_x ** 2  - 0.5 * sigma_k ** 2
    
    def mu_L(Xt, state):
        # return -zeta + psi_0 * Xt **psi_1 * (np.exp( psi_1 * state[0]) )  * np.exp( (psi_2-1) * (state[2] - np.log(448)) ) - 0.5 * sigma_g**2
        return -zeta + psi_0 * Xt **psi_1 * (np.exp( psi_1 * state[0]) )  * np.exp( (psi_2-1) * (state[2] ) ) - 0.5 * sigma_g**2
    
    
    hist      = np.zeros([pers, 3])
    i_hist    = np.zeros([pers])
    e_hist    = np.zeros([pers])
    x_hist    = np.zeros([pers])
    scc_hist  = np.zeros([pers])
    gt_tech   = np.zeros([pers])
    dL_hist    = np.zeros([pers])

    gt_dmg    = np.zeros([n_damage, pers])
    pi_c_t = np.zeros([n_climate, pers])
    Ambiguity_mean_undis = np.zeros([pers])
    Ambiguity_mean_dis = np.zeros([pers])
    Ambiguity_mean_dis_h = np.zeros([pers])

    ME_base_hist = np.zeros([pers])
    ME_total_hist = np.zeros([pers])

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
            dL_hist[tm] = dL_func(hist[0,:])

            for i in range(n_damage):
                damage_func = damage_func_list[i]
                gt_dmg[i, 0] = damage_func(hist[0, :])
            
            for i in range(n_climate):
                climate_func = climate_func_list[i]
                pi_c_t[i, 0] = climate_func(hist[0, :])
            Ambiguity_mean_undis[tm] = np.mean(theta_ell)
            Ambiguity_mean_dis[tm] = np.average(theta_ell,weights=pi_c_t[:,tm])
            
            ME_total_hist[0] = ME_total_func(hist[0,:])
            ME_base_hist[0] = ME_base_func(hist[0,:])

        else:
            # other periods
            # print(hist[tm-1,:])
            i_hist[tm] = get_i(hist[tm-1,:])
            e_hist[tm] = get_e(hist[tm-1,:])
            x_hist[tm] = get_x(hist[tm-1,:])
            gt_tech[tm] = tech_func(hist[tm-1,:])
            dL_hist[tm] = dL_func(hist[tm-1,:])

            for i in range(n_damage):
                damage_func = damage_func_list[i]
                gt_dmg[i, tm] = damage_func(hist[tm-1, :])

            for i in range(n_climate):
                climate_func = climate_func_list[i]
                pi_c_t[i, tm] = climate_func(hist[tm -1, :])
                


            mu_K_hist[tm] = mu_K(i_hist[tm])
            mu_L_hist[tm] = mu_L(x_hist[tm], hist[tm-1, :])

            hist[tm,0] = hist[tm-1,0] + mu_K_hist[tm] * dt #logK
            hist[tm,1] = hist[tm-1,1] + beta_f * e_hist[tm] * dt
            hist[tm,2] = hist[tm-1,2] + mu_L_hist[tm] * dt # logλ
            Ambiguity_mean_undis[tm] = np.mean(theta_ell)
            Ambiguity_mean_dis[tm] = np.average(theta_ell,weights=pi_c_t[:,tm])
            # Ambiguity_mean_dis_h[tm] = np.average(theta_ell + sigma_y*gt_mean[tm],weights=pi_c_t[:,tm])
            ME_total_hist[tm] = ME_total_func(hist[tm,:])
            ME_base_hist[tm] = ME_base_func(hist[tm,:])
            
        if printing==True:
            print("time={}, K={},Y={},L={},mu_K={},mu_Y={},mu_L={},ii={},ee={},xx={},ME_total_base={:.3}" .format(tm, hist[tm,0],hist[tm,1],hist[tm,2],mu_K_hist[tm],beta_f * e_hist[tm],mu_L_hist[tm],i_hist[tm],e_hist[tm],x_hist[tm],np.log(ME_total_hist[tm]/ME_base_hist[tm])*100), flush=True)
        
    
    
        # using Kt instead of K0
    jt = 1 - e_hist/ (alpha * lambda_bar * np.exp(hist[:, 0]))
    jt[jt <= 1e-16] = 1e-16
    LHS = theta * vartheta_bar / lambda_bar * jt**(theta -1)
    MC = delta / (alpha  - i_hist - alpha * vartheta_bar * jt**theta - x_hist)

    
    scc_hist = LHS * 1000


    MU_RD = dL_hist * psi_0* psi_1 * x_hist**(psi_1-1) * np.exp(psi_1*hist[:,0]-(1-psi_2)*hist[:,2])

    scrd_hist = MU_RD/MC*1000


    distorted_tech_intensity = np.exp(hist[:, 2]) * gt_tech/448

    distorted_tech_prob = 1 - np.exp(- np.cumsum(np.insert(distorted_tech_intensity * dt, 0, 0) ))[:-1]

    true_tech_intensity = np.exp(hist[:, 2]) /448
    true_tech_prob = 1 - np.exp(- np.cumsum(np.insert(true_tech_intensity * dt, 0, 0) ))[:-1]
        
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
        scrd = scrd_hist,
        gt_tech = gt_tech,
        gt_dmg = gt_dmg,
        distorted_damage_prob=distorted_damage_prob,
        distorted_tech_prob=distorted_tech_prob,
        pic_t = pi_c_t,
        ME_total = ME_total_hist,
        ME_base = ME_base_hist,
        ME_total_base = np.log(ME_total_hist / ME_base_hist ) * 100,
        jt = jt,
        LHS = LHS,
        years=years,
        true_tech_prob = true_tech_prob,
        true_damage_prob = true_damage_prob,
        Ambiguity_mean_undis = Ambiguity_mean_undis,
        Ambiguity_mean_dis = Ambiguity_mean_dis,
        )
    

    return res

def Damage_Intensity(Yt, y_bar_lower=1.5):
    r_1 = 1.5
    r_2 = 2.5
    Intensity = r_1 * (np.exp(r_2 / 2 * (Yt - y_bar_lower)**2) -1) * (Yt > y_bar_lower)
    return Intensity



def model_simulation_generate(xi_a,xi_g,psi_0,psi_1):

    Output_Dir = "/scratch/bincheng/"
    Data_Dir = Output_Dir+"abatement/data_2tech/"+args.dataname+"/"
    File_Dir = "xi_a_{}_xi_g_{}_psi_0_{}_psi_1_{}_" .format(xi_a,xi_g,psi_0,psi_1)
    


    model_tech1_post_damage = pickle.load(open(Data_Dir + File_Dir + "model_tech1_post_damage_gamma_{:.4f}".format(gamma_3_i), "rb"))
    ii = model_tech1_post_damage['i_star']
    ee = model_tech1_post_damage['e_star']
    xx = model_tech1_post_damage['x_star']
    v0 = model_tech1_post_damage["v0"]

    print("-------------------------------------------")
    print("------------Check: Post damage, Tech I:xi_a={}, xi_g={}-----------".format(xi_a,xi_g))
    print("-------------------------------------------")

    for gamma_3_i in gamma_3_list:

        print("-------------------------------------------")
        print("------------gamma_3={}-----------".format(gamma_3_i))
        print("-------------------------------------------")
        
        xi_a_post = xi_a
        xi_g_post = xi_g
        xi_p_post = xi_g
        
        n_bar = len(Y)-1

        File_Name_Suffix = "_xiapost_{}_xig_post_{}_xippost_{}".format(xi_a_post, xi_g_post, xi_p_post) + "_full_" + scheme + "_" +HJB_solution
        
        Guess = None
        
        res_post = src.ResultSolver.hjb_pre_tech_partialupdate(
                state_grid=(K, Y, L), 
                model_args=(delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell, sigma_y, zeta, psi_0, psi_1, psi_2, sigma_g, V_post_tech2, gamma_1, gamma_2, gamma_3_i, y_bar, xi_a_post, xi_g_post, xi_p_post),
                control_fixed=(ii, ee, xx, v0),
                n_bar = n_bar,
                V_post_damage=None,
                tol=1e-7, epsilon=epsilonarr[1], fraction=fractionarr[1], 
                smart_guess=Guess, 
                max_iter=maxiterarr[1],
                )


        with open(Data_Dir+ File_Dir  + "model_tech1_post_damage_gamma_{:.4f}".format(gamma_3_i) + File_Name_Suffix, "wb") as f:
            pickle.dump(res_post, f)
        with open(Data_Dir+ File_Dir  + "model_tech1_post_damage_gamma_{:.4f}".format(gamma_3_i) + File_Name_Suffix, "rb") as f:
            res_post = pickle.load(f)

    

        print("-------------------------------------------")
        print("------------Check: Pre damage, Tech I:xi_a={}, xi_g={}-----------".format(xi_a,xi_g))
        print("-------------------------------------------")

        # Post damage, tech I
        print("-------------------------------------------")
        print("------------Load: Post damage, Tech I-----------")
        print("-------------------------------------------")
        model_tech1_post_damage = []
        for i in range(len(gamma_3_list)):
            gamma_3_i = gamma_3_list[i]
            model_i = pickle.load(open(Data_Dir+ File_Dir + "model_tech1_post_damage_gamma_{:.4f}".format(gamma_3_i) + File_Name_Suffix, "rb"))
            model_tech1_post_damage.append(model_i)

        with open(Data_Dir+ File_Dir + "model_tech1_post_damage" + File_Name_Suffix, "wb") as f:
            pickle.dump(model_tech1_post_damage, f)

        print("Compiled.")
        
        print("-------------------------------------------")
        print("------------Load: Pre damage, Tech II-----------")
        print("-------------------------------------------")
        model_tech2_pre_damage = pickle.load(open(Data_Dir+ File_Dir + "model_tech2_pre_damage", "rb"))
        
        print("Compiled.")

        theta_ell = pd.read_csv('./data/model144_p.csv', header=None).to_numpy()[:, 0]/1000.


        v_i = []
        for model in model_tech1_post_damage:
            v_post_damage_i = model["v0"]
            v_post_damage_temp = np.zeros((nK, nY_short, nL))
            for j in range(nY_short):
                v_post_damage_temp[:, j, :] = v_post_damage_i[:, id_2, :]
            v_i.append(v_post_damage_temp)
        v_i = np.array(v_i)
        
        v_post = model_tech2_pre_damage["v"][:, :nY_short]
        v_tech2 = np.zeros((nK, nY_short, nL))
        for i in range(nL):
            v_tech2[:, :, i] = v_post

        xi_a_pre = xi_a
        xi_g_pre = xi_g
        xi_p_pre = xi_g
        File_Name_Suffix_pre = "_xiapre_{}_xig_pre_{}_xippre_{}".format(xi_a_pre, xi_g_pre, xi_p_pre) + "_full_" + scheme + "_" +HJB_solution
        
        
        model_args =(delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell, sigma_y, zeta, psi_0, psi_1, sigma_g, v_tech2, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a_pre, xi_g_pre, xi_p_pre)

        #########################################
        ######### Start of Compute###############
        #########################################

        
        model_tech1_pre_damage = pickle.load(open(Data_Dir + File_Dir + "model_tech1_pre_damage", "rb"))
        ii = model_tech1_pre_damage['i_star']
        ee = model_tech1_pre_damage['e_star']
        xx = model_tech1_pre_damage['x_star']
        v0 = model_tech1_pre_damage["v0"]

        # Guess = pickle.load(open(Data_Dir+ File_Name + "model_tech1_pre_damage"+File_Name_Suffix_pre, "rb"))
        
        Guess = None
        model_tech1_pre_damage = src.ResultSolver.hjb_pre_tech_partialupdate(
                state_grid=(K, Y_short, L), 
                model_args=model_args, 
                control_fixed=(ii, ee, xx, v0),
                n_bar = n_bar1,
                V_post_damage=v_i, 
                tol=1e-6, epsilon=epsilonarr[1], fraction=fractionarr[1], max_iter=maxiterarr[1],
                smart_guess=Guess,
                )

        with open(Data_Dir+ File_Dir + "model_tech1_pre_damage"+File_Name_Suffix_pre, "wb") as f:
            pickle.dump(model_tech1_pre_damage, f)
        model_tech1_pre_damage_ME_base = pickle.load(open(Data_Dir+ File_Dir + "model_tech1_pre_damage"+File_Name_Suffix_pre, "rb"))

    ME_base = model_tech1_pre_damage_ME_base["ME"]


    v = model_tech1_pre_damage_ME_base["v0"]
    i = model_tech1_pre_damage_ME_base["i_star"]
    e = model_tech1_pre_damage_ME_base["e_star"]
    x = model_tech1_pre_damage_ME_base["x_star"]
    pi_c = model_tech1_pre_damage_ME_base["pi_c"]
    g_tech = model_tech1_pre_damage_ME_base["g_tech"]
    g_damage =  model_tech1_pre_damage_ME_base["g_damage"]



    with open(Data_Dir + File_Dir+"model_tech1_pre_damage", "rb") as f:
        tech1 = pickle.load(f)
    
    
    v_orig = tech1["v0"][:,:n_bar+1,:]
    i_orig = tech1["i_star"][:,:n_bar+1,:]
    e_orig = tech1["e_star"][:,:n_bar+1,:]
    x_orig = tech1["x_star"][:,:n_bar+1,:]
    pi_c_orig = tech1["pi_c"][:,:,:n_bar+1,:]
    g_tech_orig = tech1["g_tech"][:,:n_bar+1,:]
    g_damage_orig =  tech1["g_damage"][:,:,:n_bar+1,:]



    print("--------------Control Check Start--------------")
    print("Diff_i={}".format(np.max(abs(i-i_orig))))
    print("Diff_e={}".format(np.max(abs(e-e_orig))))
    print("Diff_x={}".format(np.max(abs(x-x_orig))))
    print("--------------Control Check End--------------")
    
    ME_family = ME_base

    
    model_args = (delta, mu_k, kappa,sigma_k, beta_f, zeta, psi_0, psi_1, sigma_g, theta, lambda_bar, vartheta_bar)

    res = simulate_pre(grid = (K, Y_short, L), 
                       model_args = model_args, 
                       controls = (i,e,x, g_tech, g_damage, pi_c, v),
                       ME = ME_family,
                       n_bar = n_bar,  
                       T0=0, 
                       T=IntPeriod, 
                       dt=timespan,printing=True)

    with open(Data_Dir + File_Dir+"model_tech1_pre_damage"+"_UD_simul_{}".format(IntPeriod)+ scheme + "_" +HJB_solution, "wb") as f:
        pickle.dump(res,f)


    
    return res


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            

                    res = model_simulation_generate(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])



os.makedirs("./abatement_UD/pdf_2tech/"+args.dataname+"/"+scheme+"_"+HJB_solution+"/", exist_ok=True)

Plot_Dir = "./abatement_UD/pdf_2tech/"+args.dataname+"/"+scheme+"_"+HJB_solution+"/"

def model_simulation_generate(xi_a,xi_g,psi_0,psi_1,psi_2):

    Output_Dir = "/scratch/bincheng/"
    Data_Dir = Output_Dir+"abatement/data_2tech/"+args.dataname+"/"
    File_Dir = "xi_a_{}_xi_g_{}_psi_0_{}_psi_1_{}_" .format(xi_a,xi_g,psi_0,psi_1)


    with open(Data_Dir + File_Dir+"model_tech1_pre_damage"+"_UD_simul_{}".format(IntPeriod)+ scheme + "_" +HJB_solution, "rb") as f:
        res = pickle.load(f)


    
    return res

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            

            res = model_simulation_generate(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])

            if xiaarr[id_xiag]>10:
                plt.plot(res["years"], (res["x"]/(alpha*np.exp(res["states"][:,0])))*100,label='baseline',linewidth=5.0)
            else:
                plt.plot(res["years"], (res["x"]/(alpha*np.exp(res["states"][:,0])))*100,label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag]) ,linewidth=5.0)
            plt.xlabel('Years')
            plt.ylabel('$\%$ of GDP')
            plt.title('R&D investment as percentage of  GDP')
            # if auto==0:   
            plt.ylim(0,0.5)
            plt.xlim(0,IntPeriod)

            plt.legend(loc='upper left')        
# print(res.keys())
plt.savefig(Plot_Dir+"/RD,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig(Plot_Dir+"/RD,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            

            res = model_simulation_generate(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])

            if xiaarr[id_xiag]>10:

                plt.plot(res["years"], res["i"],label='baseline',linewidth=5.0)
            else:
                plt.plot(res["years"], res["i"],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag]) ,linewidth=5.0)
            plt.xlabel('Years')
            plt.title("Capital investment")
            if auto==0:   
                plt.ylim(65,110)
            plt.xlim(0,IntPeriod)
            plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/CapI,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig(Plot_Dir+"/CapI,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            

            res = model_simulation_generate(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])

            if xiaarr[id_xiag]>10:
                plt.plot(res["years"], res["e"],label='baseline',linewidth=5.0)
            else:
                plt.plot(res["years"], res["e"],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag]) ,linewidth=5.0)
            # plt.plot(res2["years"][res2["states"][:, 1]<1.5], res2["e"][res2["states"][:, 1]<1.5],label=r'$\xi_p=\\xi_g=0.050$',linewidth=7.0)
            # plt.plot(res3["years"][res3["states"][:, 1]<1.5], res3["e"][res3["states"][:, 1]<1.5],label='baseline',linewidth=7.0)
            plt.xlabel('Years')
            plt.title("Carbon Emissions")
            if auto==0:   
                plt.ylim(6.0,12.0)
            plt.xlim(0,IntPeriod)
            plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/E,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig(Plot_Dir+"/E,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            

            res = model_simulation_generate(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])

            if xiaarr[id_xiag]>10:

                plt.plot(res["years"], res["states"][:, 1],label='baseline',linewidth=5.0)
            else:
                plt.plot(res["years"], res["states"][:, 1],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag]) ,linewidth=5.0)
            # plt.plot(res2["years"][res2["states"][:, 1]<1.5], res2["states"][:, 1][res2["states"][:, 1]<1.5],label=r'$\xi_p=\\xi_g=0.050$',linewidth=7.0)
            # plt.plot(res3["years"][res3["states"][:, 1]<1.5], res3["states"][:, 1][res3["states"][:, 1]<1.5],label='baseline',linewidth=7.0)
            plt.xlabel('Years')
            plt.title("Temperature anomaly")
            if auto==0:   
                plt.ylim(1.1,1.5)
            plt.xlim(0,IntPeriod)
            plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/TA,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig(Plot_Dir+"/TA,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            

            res = model_simulation_generate(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])

            if xiaarr[id_xiag]>10:

                plt.plot(res["years"], np.exp(res["states"][:, 2]),label='baseline',linewidth=5.0)
            else:
                plt.plot(res["years"], np.exp(res["states"][:, 2]),label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag]) ,linewidth=5.0)
            # plt.plot(res2["years"][res2["states"][:, 1]<1.5], np.exp(res2["states"][:, 2])[res2["states"][:, 1]<1.5],label=r'$\xi_p=\\xi_g=0.050$',linewidth=7.0)
            # plt.plot(res3["years"][res3["states"][:, 1]<1.5], np.exp(res3["states"][:, 2])[res3["states"][:, 1]<1.5],label='baseline',linewidth=7.0)
            plt.xlabel('Years')
            plt.title("Technology jump intensity $J_g$")
            # if auto==0:   
            plt.ylim(10.0,25.0)
            plt.xlim(0,IntPeriod)
            plt.legend(loc='upper left')


plt.savefig(Plot_Dir+"/Ig,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig(Plot_Dir+"/Ig,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            

            res = model_simulation_generate(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])

            if xiaarr[id_xiag]>10:

                plt.plot(res["years"], res["distorted_tech_prob"],label='baseline',linewidth=5.0)
            else:
                plt.plot(res["years"], res["distorted_tech_prob"],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag]) ,linewidth=5.0)
            # plt.plot(res2["years"], res2["distorted_tech_prob"],label=r'$\xi_p=\\xi_g=0.050$',linewidth=7.0)
            # plt.plot(res3["years"], res3["distorted_tech_prob"],label='baseline',linewidth=7.0)
            plt.xlabel('Years')
            plt.title("Distorted probability of a technology jump")
            plt.ylim(0,1)
            plt.xlim(0,IntPeriod)
            plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/PIgd,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig(Plot_Dir+"/PIgd,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            

            res = model_simulation_generate(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])

            if xiaarr[id_xiag]>10:

                plt.plot(res["years"], res["distorted_damage_prob"],label='baseline',linewidth=5.0)
            else:
                plt.plot(res["years"], res["distorted_damage_prob"],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag]) ,linewidth=5.0)
            # plt.plot(res2["years"], res2["distorted_damage_prob"],label=r'$\xi_p=\\xi_g=0.050$',linewidth=7.0)
            # plt.plot(res3["years"], res3["distorted_damage_prob"],label='baseline',linewidth=7.0)
            plt.xlabel('Years')
            plt.title("Distorted probability of damage changes")
            plt.ylim(0,1)
            plt.xlim(0,IntPeriod)
            plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/PIdd,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig(Plot_Dir+"/PIdd,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            

            res = model_simulation_generate(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])

            if xiaarr[id_xiag]>10:

                plt.plot(res["years"], res["true_tech_prob"],label='baseline',linewidth=5.0)
            else:
                plt.plot(res["years"], res["true_tech_prob"],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag]) ,linewidth=5.0)

            plt.xlabel("Years")
            plt.title("True probability of a technology jump")
            plt.ylim(0.0,1.0)
            plt.xlim(0,IntPeriod)
            plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/TPIg,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig(Plot_Dir+"/TPIg,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            

            res = model_simulation_generate(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])

            if xiaarr[id_xiag]>10:

                plt.plot(res["years"], res["true_damage_prob"],label='baseline',linewidth=5.0)
            else:
                plt.plot(res["years"], res["true_damage_prob"],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag]) ,linewidth=5.0)

            plt.xlabel("Years")
            plt.title("True probability of damage changes")
            plt.ylim(0,1)
            plt.xlim(0,IntPeriod)
            plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/TPId,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig(Plot_Dir+"/TPId,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            

            res = model_simulation_generate(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])

            if xiaarr[id_xiag]>10:

                plt.plot(res["years"], np.log(res["scc"]),label='baseline',linewidth=5.0)
            else:
                plt.plot(res["years"], np.log(res["scc"]),label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag]) ,linewidth=5.0)

            plt.xlabel("Years")
            plt.title("Log of Social Cost of Carbon")
            if auto==0:   
                plt.ylim(3.0,6.5)
            plt.xlim(0,IntPeriod)
            plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/logSCC,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig(Plot_Dir+"/logSCC,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            

            res = model_simulation_generate(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])

            if xiaarr[id_xiag]>10:

                plt.plot(res["years"], np.log(res["scrd"]),label='baseline',linewidth=5.0)
            else:
                plt.plot(res["years"], np.log(res["scrd"]),label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag]) ,linewidth=5.0)

            plt.xlabel("Years")
            plt.ticklabel_format(useOffset=False)

            plt.title("Log of Social Cost of R&D")
            if auto==0:   
                plt.ylim(6.5,8.0)
            plt.xlim(0,IntPeriod)
            plt.legend(loc='upper right')

plt.savefig(Plot_Dir+"/logSCRD,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig(Plot_Dir+"/logSCRD,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            

            res = model_simulation_generate(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])

            if xigarr[id_xiag]>10:

                plt.plot(res["years"], (res["Ambiguity_mean_dis"]-res["Ambiguity_mean_undis"])*1000,label='baseline')
            else:
                plt.plot(res["years"], (res["Ambiguity_mean_dis"]-res["Ambiguity_mean_undis"])*1000,label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag]))

            plt.xlabel("Years")
            plt.title("Mean Difference")
            plt.ylim(0,0.8)   
            plt.legend()


plt.savefig(Plot_Dir+"/MeanDiff,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig(Plot_Dir+"/MeanDiff,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            
                
            res = model_simulation_generate(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])

            if xigarr[id_xiag]>10:

                plt.plot(res["years"], res["ME_total"],label='baseline')
            else:
                plt.plot(res["years"], res["ME_total"],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag]))
                
            plt.xlabel("Years")

            plt.title("ME_total")
            plt.xlim(0,IntPeriod)
            if auto==0:   
                plt.ylim(0,0.000110)   
            plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/ME_total,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()

# for id_xiag in range(len(xiaarr)): 
#     for id_psi0 in range(len(psi0arr)):
#         for id_psi1 in range(len(psi1arr)):
#             
#                     grid_info = (Xminarr, Xmaxarr, hXarr)
#                     data_info = (dataname)
#                     varying_argument_extraction = (xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], IntPeriod, timespan)
#                     constant_argument_extraction = (delta, alpha, kappa, mu_k, sigma_k, beta_f, sigma_y, zeta, sigma_g, gamma_1, gamma_2, y_bar, y_bar_lower, theta, lambda_bar, vartheta_bar, lambda_bar_first, vartheta_bar_first, lambda_bar_second, vartheta_bar_second, num_gamma, gamma_3_list)
#                     res = model_simulation_graph(grid_info, data_info, varying_argument_extraction, constant_argument_extraction)

#                     if xigarr[id_xiag]>10:

#                         plt.plot(res["years"], res["ME_total2"],label='baseline')
#                     else:
#                         plt.plot(res["years"], res["ME_total2"],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag]))
                        
#                     plt.xlabel("Years")

#                     plt.title("ME_total2")
#                     plt.xlim(0,IntPeriod)
#                     plt.legend(loc='upper left')

# plt.savefig(Plot_Dir+"/ME_total2,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
# plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            
                res = model_simulation_generate(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])

                if xigarr[id_xiag]>10:

                    plt.plot(res["years"], res["ME_base"],label='baseline')
                else:
                    plt.plot(res["years"], res["ME_base"],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag]))
                    
                plt.xlabel("Years")

                plt.title("ME_base")
                plt.xlim(0,IntPeriod)
                if auto==0:   
                    plt.ylim(0,0.000050)   
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/ME_base,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()


# for id_xiag in range(len(xiaarr)): 
#     for id_psi0 in range(len(psi0arr)):
#         for id_psi1 in range(len(psi1arr)):
#             
#                     grid_info = (Xminarr, Xmaxarr, hXarr)
#                     data_info = (dataname)
#                     varying_argument_extraction = (xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], IntPeriod, timespan)
#                     constant_argument_extraction = (delta, alpha, kappa, mu_k, sigma_k, beta_f, sigma_y, zeta, sigma_g, gamma_1, gamma_2, y_bar, y_bar_lower, theta, lambda_bar, vartheta_bar, lambda_bar_first, vartheta_bar_first, lambda_bar_second, vartheta_bar_second, num_gamma, gamma_3_list)
#                     res = model_simulation_graph(grid_info, data_info, varying_argument_extraction, constant_argument_extraction)

#                     if xigarr[id_xiag]>10:

#                         plt.plot(res["years"], res["ME_SCC"],label='baseline')
#                     else:
#                         plt.plot(res["years"], res["ME_SCC"],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag]))
                        
#                     plt.xlabel("Years")

#                     plt.title("ME_SCC")
#                     plt.xlim(0,IntPeriod)
#                     plt.legend(loc='upper left')

# plt.savefig(Plot_Dir+"/ME_SCC,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
# plt.close()


# for id_xiag in range(len(xiaarr)): 
#     for id_psi0 in range(len(psi0arr)):
#         for id_psi1 in range(len(psi1arr)):
#             
#                     grid_info = (Xminarr, Xmaxarr, hXarr)
#                     data_info = (dataname)
#                     varying_argument_extraction = (xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1], IntPeriod, timespan)
#                     constant_argument_extraction = (delta, alpha, kappa, mu_k, sigma_k, beta_f, sigma_y, zeta, sigma_g, gamma_1, gamma_2, y_bar, y_bar_lower, theta, lambda_bar, vartheta_bar, lambda_bar_first, vartheta_bar_first, lambda_bar_second, vartheta_bar_second, num_gamma, gamma_3_list)
#                     res = model_simulation_graph(grid_info, data_info, varying_argument_extraction, constant_argument_extraction)

#                     if xigarr[id_xiag]>10:

#                         plt.plot(res["years"], res["ME_consumption"],label='baseline')
#                     else:
#                         plt.plot(res["years"], res["ME_consumption"],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag]))
                        
#                     plt.xlabel("Years")

#                     plt.title("ME_consumption")
#                     plt.xlim(0,IntPeriod)
#                     plt.legend(loc='upper left')

# plt.savefig(Plot_Dir+"/ME_consumption,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
# plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            
                res = model_simulation_generate(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])

                if xigarr[id_xiag]>10:

                    plt.plot(res["years"], res["ME_total_base"],label='baseline')
                else:
                    plt.plot(res["years"], res["ME_total_base"],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag]))
                    
                plt.xlabel("Years")

                plt.title("ME_total_base")
                plt.xlim(0,IntPeriod)
                # if auto==0:   
                #     plt.ylim(0,150)   
                plt.legend()

plt.savefig(Plot_Dir+"/ME_total_base,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()





plt.style.use('default')
plt.rcParams["lines.linewidth"] = 20
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["figure.figsize"] = (16,10)
plt.rcParams["font.size"] = 25
plt.rcParams["legend.frameon"] = True


print("After, figure default size is: ", plt.rcParams["savefig.bbox"])
print("After, figure default size is: ", plt.rcParams["figure.figsize"])
print("After, figure default dpi is: ", plt.rcParams["figure.dpi"])
print("After, figure default size is: ", plt.rcParams["font.size"])
print("After, legend.frameon is: ", plt.rcParams["legend.frameon"])
print("After, lines.linewidth is: ", plt.rcParams["lines.linewidth"])

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            
                
            res = model_simulation_generate(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])

            # theta_ell_new = res["theta_ell_new"][:,-1]
            # histogram of beta_f
            theta_ell = pd.read_csv("./data/model144_p.csv", header=None).to_numpy()[:, 0]
            # print("theta_ell")
            # print(theta_ell)
            # print("theta_ell_new")
            # print(theta_ell_new)
            pi_c_o = np.ones(len(theta_ell)) / len(theta_ell)
            # pi_c = np.load("πc_5.npy")
            time = 1/timespan
            pi_c = res["pic_t"][:, int(time)]


            # plt.figure(figsize=(16,10))

            print("mean of uncondition = {}" .format(np.average(theta_ell,weights = pi_c_o)))
            print("mean of condition = {}" .format(np.average(theta_ell,weights = pi_c)))
                
            plt.hist(theta_ell, weights=pi_c_o, bins=np.linspace(0.8, 3., 16), density=True, 
                    alpha=0.5, ec="darkgrey", color="C3",label='baseline')
            plt.hist(theta_ell, weights=pi_c, bins=np.linspace(0.8, 3., 16), density=True, 
                    alpha=0.5, ec="darkgrey", color="C0",label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag]))
            plt.legend(loc='upper left')
            plt.title("Distorted probability of Climate Models")

            plt.ylim(0, 3)
            plt.xlabel("Climate Sensitivity")
            
            plt.savefig(Plot_Dir+"/ClimateSensitivity_0,xia={:.5f},xig={:.3f},psi0={:.3f},psi1={:.3f}.png".format(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1]))
            plt.close()


# for id_xiag in range(len(xiaarr)): 
#     for id_psi0 in range(len(psi0arr)):
#         for id_psi1 in range(len(psi1arr)):
            
                
#                 res = model_simulation_generate(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])

#                 # theta_ell_new = res["theta_ell_new"][:,-1]
#                 # histogram of beta_f
#                 psi_2 = pd.read_csv("./data/psi2value_p.csv", header=None).to_numpy()[:, 0]
#                 # print("theta_ell")
#                 # print(theta_ell)
#                 # print("theta_ell_new")
#                 # print(theta_ell_new)
#                 pi_c_o = np.ones(len(psi_2)) / len(psi_2)
#                 # pi_c = np.load("πc_5.npy")
#                 time = 1/timespan
#                 pi_c = res["pic_t"][:, int(time)]


#                 # plt.figure(figsize=(16,10))

#                 print("mean of uncondition = {}" .format(np.average(psi_2,weights = pi_c_o)))
#                 print("mean of condition = {}" .format(np.average(psi_2,weights = pi_c)))
                    
#                 # plt.hist(psi_2, weights=pi_c_o, bins=np.linspace(0.8, 3., 16), density=True, 
#                 plt.hist(psi_2, weights=pi_c_o, density=True, 
#                         alpha=0.5, ec="darkgrey", color="C3",label='baseline')
#                 # plt.hist(psi_2, weights=pi_c, bins=np.linspace(0.8, 3., 16), density=True, 
#                 plt.hist(psi_2, weights=pi_c, density=True, 
#                         alpha=0.5, ec="darkgrey", color="C0",label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag]))
#                 plt.legend(loc='upper left')
#                 plt.title("Distorted probability of R&D Parameters")

#                 plt.ylim(0, 24)
#                 plt.xlabel("R&D Parameter Sensitivity")
                
#                 plt.savefig(Plot_Dir+"/DRSSensitivity_0,xia={:.5f},xig={:.3f},psi0={:.3f},psi1={:.3f}.png".format(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1]))
#                 plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            
            
            res = model_simulation_generate(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])

            # theta_ell_new = res["theta_ell_new"][:,-1]
            # histogram of beta_f
            theta_ell = pd.read_csv("./data/model144_p.csv", header=None).to_numpy()[:, 0]
            # print("theta_ell")
            # print(theta_ell)
            # print("theta_ell_new")
            # print(theta_ell_new)
            pi_c_o = np.ones(len(theta_ell)) / len(theta_ell)
            # pi_c = np.load("πc_5.npy")
            time = 1/timespan
            pi_c = res["pic_t"][:, -1]

            # plt.figure(figsize=(16,10))

            print("mean of uncondition = {}" .format(np.average(theta_ell,weights = pi_c_o)))
            print("mean of condition = {}" .format(np.average(theta_ell,weights = pi_c)))
                
            plt.hist(theta_ell, weights=pi_c_o, bins=np.linspace(0.8, 3., 16), density=True, 
                    alpha=0.5, ec="darkgrey", color="C3",label='baseline')
            plt.hist(theta_ell, weights=pi_c, bins=np.linspace(0.8, 3., 16), density=True, 
                    alpha=0.5, ec="darkgrey", color="C0",label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag]))
            plt.legend(loc='upper left')
            plt.title("Distorted probability of Climate Models")

            plt.ylim(0, 3)
            plt.xlabel("Climate Sensitivity")
            
            plt.savefig(Plot_Dir+"/ClimateSensitivity_25,xia={:.5f},xig={:.3f},psi0={:.3f},psi1={:.3f}.png".format(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1]))
            plt.close()


# for id_xiag in range(len(xiaarr)): 
#     for id_psi0 in range(len(psi0arr)):
#         for id_psi1 in range(len(psi1arr)):
            
                
#                 res = model_simulation_generate(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])

#                 # theta_ell_new = res["theta_ell_new"][:,-1]
#                 # histogram of beta_f
#                 psi_2 = pd.read_csv("./data/psi2value_p.csv", header=None).to_numpy()[:, 0]
#                 # print("theta_ell")
#                 # print(theta_ell)
#                 # print("theta_ell_new")
#                 # print(theta_ell_new)
#                 pi_c_o = np.ones(len(psi_2)) / len(psi_2)
#                 # pi_c = np.load("πc_5.npy")
#                 time = 1/timespan
#                 pi_c = res["pic_t"][:, -1]

#                 # plt.figure(figsize=(16,10))

#                 print("mean of uncondition = {}" .format(np.average(psi_2,weights = pi_c_o)))
#                 print("mean of condition = {}" .format(np.average(psi_2,weights = pi_c)))
                    
#                 # plt.hist(psi_2, weights=pi_c_o, bins=np.linspace(0.8, 3., 16), density=True, 
#                 plt.hist(psi_2, weights=pi_c_o, density=True, 
#                         alpha=0.5, ec="darkgrey", color="C3",label='baseline')
#                 # plt.hist(psi_2, weights=pi_c, bins=np.linspace(0.8, 3., 16), density=True, 
#                 plt.hist(psi_2, weights=pi_c, density=True, 
#                         alpha=0.5, ec="darkgrey", color="C0",label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag]))
#                 plt.legend(loc='upper left')
#                 plt.title("Distorted probability of R&D Parameters")

#                 plt.ylim(0, 24)
#                 plt.xlabel("R&D Parameter Sensitivity")
                
#                 plt.savefig(Plot_Dir+"/DRSSensitivity_25,xia={:.5f},xig={:.3f},psi0={:.3f},psi1={:.3f}.png".format(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1]))
#                 plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            
                
            res = model_simulation_generate(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])

            NUM_DAMAGE = res["gt_dmg"].shape[0]
            gamma_3_list = np.linspace(0., 1./3., NUM_DAMAGE)

            # γ3_distort = np.load("γ3_5.npy")

            γ3_distort = res["gt_dmg"][:, -1] 
            # plt.figure(figsize=(16,10))
            plt.hist(gamma_3_list, weights=np.ones(len(gamma_3_list)) / len(gamma_3_list), 
                    alpha=0.5, color="C3", ec="darkgray",label='baseline')
            plt.hist(gamma_3_list, weights= γ3_distort / np.sum(γ3_distort), 
                    alpha=0.5, color="C0", ec="darkgray",label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag]))
            plt.ylim(0, 1)
            plt.title("Distorted probability of Damage Models")
            plt.xlabel("Damage Curvature")
            plt.legend(loc='upper left')

                
            plt.savefig(Plot_Dir+"/Gamma3,xia={:.5f},xig={:.3f},psi0={:.3f},psi1={:.3f}.png".format(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1]))
            plt.close()

