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
parser.add_argument("--dataname",type=str)
parser.add_argument("--pdfname",type=str)

parser.add_argument("--xiaarr",nargs='+', type=float)
parser.add_argument("--xigarr",nargs='+', type=float)

parser.add_argument("--psi0arr",nargs='+',type=float)
parser.add_argument("--psi1arr",nargs='+',type=float)

parser.add_argument("--hXarr",nargs='+',type=float)
parser.add_argument("--Xminarr",nargs='+',type=float)
parser.add_argument("--Xmaxarr",nargs='+',type=float)

# parser.add_argument("--Update",type=int)

# parser.add_argument("--year",type=int,default=60)
# parser.add_argument("--time",type=float,default=1/12.)
args = parser.parse_args()


# Update = args.Update
IntPeriod = 55
timespan = 1/12

# psi0arr = np.array([0.006,0.009])
# # # psi0arr = np.array([0.009])
# # # psi1arr = np.array([.5,.7,.9])
# psi1arr = np.array([.3,.4])

psi0arr = args.psi0arr
psi1arr = args.psi1arr
xiaarr = args.xiaarr
xigarr = args.xigarr 


Xminarr = args.Xminarr
Xmaxarr = args.Xmaxarr
hXarr = args.hXarr

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







mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["figure.figsize"] = (8,5)
mpl.rcParams["font.size"] = 12
mpl.rcParams["legend.frameon"] = False
mpl.style.use('classic')
mpl.rcParams["lines.linewidth"] = 7.0

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
    ii, ee, xx, g_tech, g_damage, g_mean, pi_c = controls
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

    # theta_ell = np.array([temp * np.ones(K_mat.shape) for temp in theta_ell])
    # args = (delta, alpha, kappa, mu_k, sigma_k, gamma_1, gamma_2, theta_ell, pi_c_o, sigma_y,  theta, vartheta_bar, lambda_bar)

#     v, ME_base, diff = decompose(v0, stateSpace, (K_mat, Y_mat, L_mat), (ii, ee, xx), args=args)

    gridpoints = (K, Y, L)

    i_func = RegularGridInterpolator(gridpoints, ii)
    e_func = RegularGridInterpolator(gridpoints, ee)
    x_func = RegularGridInterpolator(gridpoints, xx)
    tech_func = RegularGridInterpolator(gridpoints, g_tech)
    mean_func = RegularGridInterpolator(gridpoints, g_mean)
    # print()
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
    gt_mean   = np.zeros([pers])
#     if pre_damage:
    gt_dmg    = np.zeros([n_damage, pers])
    pi_c_t = np.zeros([n_climate, pers])
    theta_ell_hist = np.zeros([len(theta_ell),pers])
#     ME_base_t = np.zeros([pers])

    mu_K_hist = np.zeros([pers])
    mu_L_hist = np.zeros([pers])

    Ambiguity_mean_undis = np.zeros([pers])
    Ambiguity_mean_dis = np.zeros([pers])
    Ambiguity_mean_dis_h = np.zeros([pers])
    

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
            gt_mean[0] = mean_func(hist[0, :])
#             if pre_damage:
            for i in range(n_damage):
                damage_func = damage_func_list[i]
                gt_dmg[i, 0] = damage_func(hist[0, :])
            
            for i in range(n_climate):
                climate_func = climate_func_list[i]
                pi_c_t[i, 0] = climate_func(hist[0, :])

            Ambiguity_mean_undis[tm] = np.mean(theta_ell)
            Ambiguity_mean_dis[tm] = np.average(theta_ell,weights=pi_c_t[:,tm])
            Ambiguity_mean_dis_h[tm] = np.average(theta_ell + sigma_y*gt_mean[tm],weights=pi_c_t[:,tm])

            theta_ell_hist[:,0] = theta_ell + sigma_y*gt_mean[0]
            

        else:
            # other periods
            # print(hist[tm-1,:])
            i_hist[tm] = get_i(hist[tm-1,:])
            e_hist[tm] = get_e(hist[tm-1,:])
            x_hist[tm] = get_x(hist[tm-1,:])
            gt_tech[tm] = tech_func(hist[tm-1,:])
            gt_mean[tm] = mean_func(hist[tm-1,:])
#             if pre_damage:
            for i in range(n_damage):
                damage_func = damage_func_list[i]
                gt_dmg[i, tm] = damage_func(hist[tm-1, :])

            for i in range(n_climate):
                climate_func = climate_func_list[i]
                pi_c_t[i, tm] = climate_func(hist[tm -1, :])
                
#             ME_base_t[tm] = ME_base_func(hist[tm-1, :])
            theta_ell_hist[:,tm] = theta_ell + sigma_y*gt_mean[tm]
            gt_mean_mul = sigma_y*gt_mean*1000
            # print("iteration={}".format(tm))
            # print(gt_mean[tm])

            mu_K_hist[tm] = mu_K(i_hist[tm])
            mu_L_hist[tm] = mu_L(x_hist[tm], hist[tm-1, :])

            hist[tm,0] = hist[tm-1,0] + mu_K_hist[tm] * dt #logK
            hist[tm,1] = hist[tm-1,1] + beta_f * e_hist[tm] * dt 
            hist[tm,2] = hist[tm-1,2] + mu_L_hist[tm] * dt # logλ

            Ambiguity_mean_undis[tm] = np.mean(theta_ell)
            Ambiguity_mean_dis[tm] = np.average(theta_ell,weights=pi_c_t[:,tm])
            Ambiguity_mean_dis_h[tm] = np.average(theta_ell + sigma_y*gt_mean[tm],weights=pi_c_t[:,tm])

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
        gt_mean = gt_mean,
        gt_mean_mul = gt_mean_mul,
        distorted_damage_prob=distorted_damage_prob,
        distorted_tech_prob=distorted_tech_prob,
        pic_t = pi_c_t,
#         ME_base = ME_base_t,
        jt = jt,
        LHS = LHS,
        years=years,
        true_tech_prob = true_tech_prob,
        true_damage_prob = true_damage_prob,
        theta_ell_new = theta_ell_hist,
        Ambiguity_mean_undis = Ambiguity_mean_undis,
        Ambiguity_mean_dis = Ambiguity_mean_dis,
        Ambiguity_mean_dis_h = Ambiguity_mean_dis_h,
    )
    
#     if pre_damage:
#         res["gt_dmg"] = gt_dmg
    
    return res

def Damage_Intensity(Yt, y_bar_lower=1.5):
    r_1 = 1.5
    r_2 = 2.5
    Intensity = r_1 * (np.exp(r_2 / 2 * (Yt - y_bar_lower)**2) -1) * (Yt > y_bar_lower)
    return Intensity



def model_solution_extraction(xi_a,xi_g,psi_0,psi_1):
    
        # Data_Dir = "./abatement/data_2tech/"+args.dataname+"/"
        Output_Dir = "/scratch/bincheng/"
        Data_Dir = Output_Dir+"abatement/data_2tech/"+args.dataname+"/"


        File_Dir = "xi_a_{}_xi_g_{}_psi_0_{}_psi_1_{}_" .format(xi_a,xi_g,psi_0,psi_1)
        model_dir_post = Data_Dir + File_Dir+"model_tech1_pre_damage"

        model_simul_dir_post = Data_Dir + File_Dir+"model_tech1_pre_damage_simul_{}" .format(IntPeriod)


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
            g_mean = tech1["gg_mean"]
            
            # g_damage = np.ones((1, nK, nY, nL))
            res = simulate_pre(grid = (K, Y_short, L), model_args = model_args, 
                                        controls = (i,e,x, g_tech, g_damage, g_mean, pi_c), 
                                        T0=0, T=IntPeriod, dt=timespan,printing=False)

            with open(model_simul_dir_post, "wb") as f:
                pickle.dump(res,f)

            res = pickle.load(open(model_simul_dir_post, "rb"))

        
        return res


# os.makedirs("./abatement/pdf_2tech/"+args.dataname+"/")

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):

            res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])

            if xigarr[id_xiag]>10:
                plt.plot(res["years"], (res["x"]/(alpha*np.exp(res["states"][:,0])))*100,label='baseline'  )
            else:
                plt.plot(res["years"], (res["x"]/(alpha*np.exp(res["states"][:,0])))*100,label='$\\xi_a={:.4f}$,$\\xi_g=\\xi_d=\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag])  )
            plt.xlabel('Years')
            plt.ylabel('$\%$ of GDP')
            plt.title('R&D investment as percentage of  GDP')   
            plt.ylim(0,0.8)
            plt.xlim(0,IntPeriod)

            plt.legend(loc='upper left')        

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/RD,xia={},xig={},psi0={},psi1={}_v2.pdf".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/RD,xia={},xig={},psi0={},psi1={}_v2.png".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):

            res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])

            if xigarr[id_xiag]>10:

                plt.plot(res["years"], res["i"],label='baseline' )
            else:
                plt.plot(res["years"], res["i"],label='$\\xi_a={:.4f}$,$\\xi_g=\\xi_d=\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag])  )
            plt.xlabel('Years')
            plt.title("Capital investment")
            plt.ylim(60,220)
            plt.xlim(0,IntPeriod)
            plt.legend(loc='upper left')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/CapI,xia={},xig={},psi0={},psi1={}_v2.pdf".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/CapI,xia={},xig={},psi0={},psi1={}_v2.png".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):

            res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])

            if xigarr[id_xiag]>10:
                plt.plot(res["years"], res["e"],label='baseline'  )
            else:
                plt.plot(res["years"], res["e"],label='$\\xi_a={:.4f}$,$\\xi_g=\\xi_d=\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag])  )
            # plt.plot(res2["years"][res2["states"][:, 1]<1.5], res2["e"][res2["states"][:, 1]<1.5],label=r'$\xi_p=\\xi_g=0.050$',linewidth=7.0)
            # plt.plot(res3["years"][res3["states"][:, 1]<1.5], res3["e"][res3["states"][:, 1]<1.5],label='baseline',linewidth=7.0)
            plt.xlabel('Years')
            plt.title("Carbon Emissions")
            plt.ylim(6,22)
            plt.xlim(0,IntPeriod)
            plt.legend(loc='upper left')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/E,xia={},xig={},psi0={},psi1={}_v2.pdf".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/E,xia={},xig={},psi0={},psi1={}_v2.png".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):

            res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])

            if xigarr[id_xiag]>10:

                plt.plot(res["years"], res["states"][:, 1],label='baseline'  )
            else:
                plt.plot(res["years"], res["states"][:, 1],label='$\\xi_a={:.4f}$,$\\xi_g=\\xi_d=\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag])  )
            # plt.plot(res2["years"][res2["states"][:, 1]<1.5], res2["states"][:, 1][res2["states"][:, 1]<1.5],label=r'$\xi_p=\\xi_g=0.050$',linewidth=7.0)
            # plt.plot(res3["years"][res3["states"][:, 1]<1.5], res3["states"][:, 1][res3["states"][:, 1]<1.5],label='baseline',linewidth=7.0)
            plt.xlabel('Years')
            plt.title("Temperature anomaly")
            plt.ylim(1,2.8)
            plt.xlim(0,IntPeriod)
            plt.legend(loc='upper left')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/TA,xia={},xig={},psi0={},psi1={}_v2.pdf".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/TA,xia={},xig={},psi0={},psi1={}_v2.png".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):

            res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])


            if xigarr[id_xiag]>10:

                plt.plot(res["years"], np.exp(res["states"][:, 2]),label='baseline'  )
            else:
                plt.plot(res["years"], np.exp(res["states"][:, 2]),label='$\\xi_a={:.4f}$,$\\xi_g=\\xi_d=\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag])  )
            # plt.plot(res2["years"][res2["states"][:, 1]<1.5], np.exp(res2["states"][:, 2])[res2["states"][:, 1]<1.5],label=r'$\xi_p=\\xi_g=0.050$',linewidth=7.0)
            # plt.plot(res3["years"][res3["states"][:, 1]<1.5], np.exp(res3["states"][:, 2])[res3["states"][:, 1]<1.5],label='baseline',linewidth=7.0)
            plt.xlabel('Years')
            plt.title("Technology jump intensity")
            plt.ylim(0,0.23)
            plt.xlim(0,IntPeriod)
            plt.legend(loc='upper left')


plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/Ig,xia={},xig={},psi0={},psi1={}_v2.pdf".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/Ig,xia={},xig={},psi0={},psi1={}_v2.png".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):

            res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])


            if xigarr[id_xiag]>10:

                plt.plot(res["years"], res["distorted_tech_prob"],label='baseline'  )
            else:
                plt.plot(res["years"], res["distorted_tech_prob"],label='$\\xi_a={:.4f}$,$\\xi_g=\\xi_d=\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag])  )
            # plt.plot(res2["years"], res2["distorted_tech_prob"],label=r'$\xi_p=\\xi_g=0.050$',linewidth=7.0)
            # plt.plot(res3["years"], res3["distorted_tech_prob"],label='baseline',linewidth=7.0)
            plt.xlabel('Years')
            plt.title("Distorted probability of a technology jump")
            plt.ylim(0,1)
            plt.xlim(0,IntPeriod)
            plt.legend(loc='upper left')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/PIgd,xia={},xig={},psi0={},psi1={}_v2.pdf".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/PIgd,xia={},xig={},psi0={},psi1={}_v2.png".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):

            res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])

            if xigarr[id_xiag]>10:

                plt.plot(res["years"], res["distorted_damage_prob"],label='baseline'  )
            else:
                plt.plot(res["years"], res["distorted_damage_prob"],label='$\\xi_a={:.4f}$,$\\xi_g=\\xi_d=\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag])  )
            # plt.plot(res2["years"], res2["distorted_damage_prob"],label=r'$\xi_p=\\xi_g=0.050$',linewidth=7.0)
            # plt.plot(res3["years"], res3["distorted_damage_prob"],label='baseline',linewidth=7.0)
            plt.xlabel('Years')
            plt.title("Distorted probability of damage changes")
            plt.ylim(0,1)
            plt.xlim(0,IntPeriod)
            plt.legend(loc='upper left')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/PIdd,xia={},xig={},psi0={},psi1={}_v2.pdf".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/PIdd,xia={},xig={},psi0={},psi1={}_v2.png".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):

            res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])

            if xigarr[id_xiag]>10:

                plt.plot(res["years"], res["true_tech_prob"],label='baseline' )
            else:
                plt.plot(res["years"], res["true_tech_prob"],label='$\\xi_a={:.4f}$,$\\xi_g=\\xi_d=\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag])  )

            plt.xlabel("Years")
            plt.title("True probability of first technology jump")
            plt.ylim(0,1)
            plt.xlim(0,IntPeriod)
            plt.legend(loc='upper left')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/TPIg,xia={},xig={},psi0={},psi1={}_v2.pdf".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/TPIg,xia={},xig={},psi0={},psi1={}_v2.png".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):

            res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])

            if xigarr[id_xiag]>10:

                plt.plot(res["years"], res["true_damage_prob"],label='baseline' )
            else:
                plt.plot(res["years"], res["true_damage_prob"],label='$\\xi_a={:.4f}$,$\\xi_g=\\xi_d=\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag])  )

            plt.xlabel("Years")
            plt.title("True probability of damage changes")
            plt.ylim(0,1)
            plt.xlim(0,IntPeriod)
            plt.legend(loc='upper left')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/TPId,xia={},xig={},psi0={},psi1={}_v2.pdf".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/TPId,xia={},xig={},psi0={},psi1={}_v2.png".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):

            res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])

            if xigarr[id_xiag]>10:

                plt.plot(res["years"], res["scc"],label='baseline' )
            else:
                plt.plot(res["years"], res["scc"],label='$\\xi_a={:.4f}$,$\\xi_g=\\xi_d=\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag])  )

            plt.xlabel("Years")
            plt.title("Social Cost of Carbon")
            plt.ylim(0,250)
            plt.xlim(0,IntPeriod)
            plt.legend(loc='upper left')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/SCC,xia={},xig={},psi0={},psi1={}_v2.pdf".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/SCC,xia={},xig={},psi0={},psi1={}_v2.png".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):

            res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])
            if xigarr[id_xiag]>10:

                plt.plot(res["years"], res["gt_mean_mul"],label='baseline'  )
            else:
                plt.plot(res["years"], res["gt_mean_mul"],label='$\\xi_a={:.1f}$,$\\xi_g=\\xi_d=\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag])  )

            plt.xlabel("Years")
            plt.title("$1000\sigma_yh$")
            plt.ylim(0,0.08)
            plt.legend(loc='upper left')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/h,xia={},xig={},psi0={},psi1={}_v2.pdf".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/h,xia={},xig={},psi0={},psi1={}_v2.png".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):

            res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])
            if xigarr[id_xiag]>10:

                plt.plot(res["years"][res["states"][:, 1]<1.5], ((res["x"]/(alpha*np.exp(res["states"][:,0])))*100)[res["states"][:, 1]<1.5],label='baseline' )
            else:
                plt.plot(res["years"][res["states"][:, 1]<1.5], ((res["x"]/(alpha*np.exp(res["states"][:,0])))*100)[res["states"][:, 1]<1.5],label='$\\xi_a={:.4f}$,$\\xi_g=\\xi_d=\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag])  )
            plt.xlabel('Years')
            plt.ylabel('$\%$ of GDP')
            plt.title('R&D investment as percentage of  GDP')   
            plt.ylim(0,0.65)
            plt.legend(loc='upper left')        

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/RD,xia={},xig={},psi0={},psi1={},BC_v2.pdf".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/RD,xia={},xig={},psi0={},psi1={},BC_v2.png".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):

            res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])
            if xigarr[id_xiag]>10:

                plt.plot(res["years"][res["states"][:, 1]<1.5], res["i"][res["states"][:, 1]<1.5],label='baseline' )
            else:
                plt.plot(res["years"][res["states"][:, 1]<1.5], res["i"][res["states"][:, 1]<1.5],label='$\\xi_a={:.4f}$,$\\xi_g=\\xi_d=\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag])  )
            plt.xlabel('Years')
            plt.title("Capital investment")
            plt.ylim(60,220)
            plt.legend(loc='upper left')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/CapI,xia={},xig={},psi0={},psi1={},BC_v2.pdf".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/CapI,xia={},xig={},psi0={},psi1={},BC_v2.png".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):

            res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])

            if xigarr[id_xiag]>10:

                plt.plot(res["years"][res["states"][:, 1]<1.5], res["e"][res["states"][:, 1]<1.5],label='baseline'  )
            else:
                plt.plot(res["years"][res["states"][:, 1]<1.5], res["e"][res["states"][:, 1]<1.5],label='$\\xi_a={:.4f}$,$\\xi_g=\\xi_d=\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag])  )
            # plt.plot(res2["years"][res2["states"][:, 1]<1.5], res2["e"][res2["states"][:, 1]<1.5],label=r'$\xi_p=\\xi_g=0.050$',linewidth=7.0)
            # plt.plot(res3["years"][res3["states"][:, 1]<1.5], res3["e"][res3["states"][:, 1]<1.5],label='baseline',linewidth=7.0)
            plt.xlabel('Years')
            plt.title("Carbon Emissions")
            plt.ylim(4,14)
            plt.legend(loc='upper left')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/E,xia={},xig={},psi0={},psi1={},BC_v2.pdf".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/E,xia={},xig={},psi0={},psi1={},BC_v2.png".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):

            res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])
            if xigarr[id_xiag]>10:

                plt.plot(res["years"][res["states"][:, 1]<1.5], res["states"][:, 1][res["states"][:, 1]<1.5],label='baseline'  )
            else:
                plt.plot(res["years"][res["states"][:, 1]<1.5], res["states"][:, 1][res["states"][:, 1]<1.5],label='$\\xi_a={:.4f}$,$\\xi_g=\\xi_d=\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag])  )
            # plt.plot(res2["years"][res2["states"][:, 1]<1.5], res2["states"][:, 1][res2["states"][:, 1]<1.5],label=r'$\xi_p=\\xi_g=0.050$',linewidth=7.0)
            # plt.plot(res3["years"][res3["states"][:, 1]<1.5], res3["states"][:, 1][res3["states"][:, 1]<1.5],label='baseline',linewidth=7.0)
            plt.xlabel('Years')
            plt.title("Temperature anomaly")
            plt.ylim(1,2.8)
            plt.legend(loc='upper left')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/TA,xia={},xig={},psi0={},psi1={},BC_v2.pdf".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/TA,xia={},xig={},psi0={},psi1={},BC_v2.png".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):

            res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])
            if xigarr[id_xiag]>10:

                plt.plot(res["years"][res["states"][:, 1]<1.5], np.exp(res["states"][:, 2])[res["states"][:, 1]<1.5],label='baseline' )
            else:
                plt.plot(res["years"][res["states"][:, 1]<1.5], np.exp(res["states"][:, 2])[res["states"][:, 1]<1.5],label='$\\xi_a={:.4f}$,$\\xi_g=\\xi_d=\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag])  )
            # plt.plot(res2["years"][res2["states"][:, 1]<1.5], np.exp(res2["states"][:, 2])[res2["states"][:, 1]<1.5],label=r'$\xi_p=\\xi_g=0.050$',linewidth=7.0)
            # plt.plot(res3["years"][res3["states"][:, 1]<1.5], np.exp(res3["states"][:, 2])[res3["states"][:, 1]<1.5],label='baseline',linewidth=7.0)
            plt.xlabel('Years')
            plt.title("Technology jump intensity")
            plt.ylim(0,0.23)
            plt.legend(loc='upper left')


plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/Ig,xia={},xig={},psi0={},psi1={},BC_v2.pdf".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/Ig,xia={},xig={},psi0={},psi1={},BC_v2.png".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):

            res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])

            if xigarr[id_xiag]>10:

                plt.plot(res["years"][res["states"][:, 1]<1.5], res["distorted_tech_prob"][res["states"][:, 1]<1.5],label='baseline' )
            else:
                plt.plot(res["years"][res["states"][:, 1]<1.5], res["distorted_tech_prob"][res["states"][:, 1]<1.5],label='$\\xi_a={:.4f}$,$\\xi_g=\\xi_d=\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag])  )
            # plt.plot(res2["years"], res2["distorted_tech_prob"],label=r'$\xi_p=\\xi_g=0.050$',linewidth=7.0)
            # plt.plot(res3["years"], res3["distorted_tech_prob"],label='baseline',linewidth=7.0)
            plt.xlabel('Years')
            plt.title("Distorted probability of a technology jump")
            plt.ylim(0,1)
            plt.legend(loc='upper left')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/PIgd,xia={},xig={},psi0={},psi1={},BC_v2.pdf".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/PIgd,xia={},xig={},psi0={},psi1={},BC_v2.png".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):

            res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])
            if xigarr[id_xiag]>10:

                plt.plot(res["years"][res["states"][:, 1]<1.5], res["distorted_damage_prob"][res["states"][:, 1]<1.5],label='baseline'  )
            else:
                plt.plot(res["years"][res["states"][:, 1]<1.5], res["distorted_damage_prob"][res["states"][:, 1]<1.5],label='$\\xi_a={:.4f}$,$\\xi_g=\\xi_d=\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag])  )
            # plt.plot(res2["years"], res2["distorted_damage_prob"],label=r'$\xi_p=\\xi_g=0.050$',linewidth=7.0)
            # plt.plot(res3["years"], res3["distorted_damage_prob"],label='baseline',linewidth=7.0)
            plt.xlabel('Years')
            plt.title("Distorted probability of damage changes")
            plt.ylim(0,1)
            plt.legend(loc='upper left')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/PIdd,xia={},xig={},psi0={},psi1={},BC_v2.pdf".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/PIdd,xia={},xig={},psi0={},psi1={},BC_v2.png".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):

            res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])

            if xigarr[id_xiag]>10:

                plt.plot(res["years"][res["states"][:, 1]<1.5], res["true_tech_prob"][res["states"][:, 1]<1.5],label='baseline'  )
            else:

                plt.plot(res["years"][res["states"][:, 1]<1.5], res["true_tech_prob"][res["states"][:, 1]<1.5],label='$\\xi_a={:.4f}$,$\\xi_g=\\xi_d=\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag])  )

            plt.xlabel("Years")
            plt.title("True probability of a technology jump")
            plt.ylim(0,1)
            plt.legend(loc='upper left')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/TPIg,xia={},xig={},psi0={},psi1={},BC_v2.pdf".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/TPIg,xia={},xig={},psi0={},psi1={},BC_v2.png".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):

            res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])
            if xigarr[id_xiag]>10:

                plt.plot(res["years"][res["states"][:, 1]<1.5], res["true_damage_prob"][res["states"][:, 1]<1.5],label='baseline' )
            else:
 
                plt.plot(res["years"][res["states"][:, 1]<1.5], res["true_damage_prob"][res["states"][:, 1]<1.5],label='$\\xi_a={:.4f}$,$\\xi_g=\\xi_d=\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag])  )

            plt.xlabel("Years")
            plt.title("True probability of damage changes")
            plt.ylim(0,1)
            plt.legend(loc='upper left')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/TPId,xia={},xig={},psi0={},psi1={},BC_v2.pdf".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/TPId,xia={},xig={},psi0={},psi1={},BC_v2.png".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):

            res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])
            if xigarr[id_xiag]>10:

                plt.plot(res["years"][res["states"][:, 1]<1.5], res["scc"][res["states"][:, 1]<1.5],label='baseline'  )
            else:
                plt.plot(res["years"][res["states"][:, 1]<1.5], res["scc"][res["states"][:, 1]<1.5],label='$\\xi_a={:.4f}$,$\\xi_g=\\xi_d=\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag])  )

            plt.xlabel("Years")
            plt.title("Social Cost of Carbon")
            plt.ylim(0,250)
            plt.legend(loc='upper left')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/SCC,xia={},xig={},psi0={},psi1={},BC_v2.pdf".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/SCC,xia={},xig={},psi0={},psi1={},BC_v2.png".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):

            res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])
            if xigarr[id_xiag]>10:

                plt.plot(res["years"][res["states"][:, 1]<1.5], res["gt_mean_mul"][res["states"][:, 1]<1.5],label='baseline'  )
            else:
                plt.plot(res["years"][res["states"][:, 1]<1.5], res["gt_mean_mul"][res["states"][:, 1]<1.5],label='$\\xi_a={:.1f}$,$\\xi_g=\\xi_d=\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag])  )

            plt.xlabel("Years")
            plt.title("$1000\sigma_yh$")
            plt.ylim(0,0.06)
            plt.legend(loc='upper left')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/h,xia={},xig={},psi0={},psi1={},BC_v2.pdf".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/h,xia={},xig={},psi0={},psi1={},BC_v2.png".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):

     
            res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])

            # histogram of gamma_3
            NUM_DAMAGE = res["gt_dmg"].shape[0]
            gamma_3_list = np.linspace(0., 1./3., NUM_DAMAGE)

            # γ3_distort = np.load("γ3_5.npy")

            γ3_distort = res["gt_dmg"][:, -1] 
            plt.figure(figsize=(16,10))
            plt.hist(gamma_3_list, weights=np.ones(len(gamma_3_list)) / len(gamma_3_list), 
                    alpha=0.5, color="C3", ec="darkgray",label='baseline')
            plt.hist(gamma_3_list, weights= γ3_distort / np.sum(γ3_distort), 
                    alpha=0.5, color="C0", ec="darkgray",label='$\\xi_a={:.4f}$,$\\xi_g=\\xi_d=\\xi_m={:.3f}$'.format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag])  )
            plt.ylim(0, 0.3)
            plt.title("Distorted probability of Damage Models")
            plt.xlabel("$\\gamma_3$")
            plt.legend(loc='upper left')

            plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/Gamma3,xia={:.4f},xig={:.3f},psi0={:.3f},psi1={:.3f},BC.pdf".format(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1]))
            plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/Gamma3,xia={:.4f},xig={:.3f},psi0={:.3f},psi1={:.3f},BC.png".format(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1]))
            plt.close()




for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):

     
            res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])


            theta_ell_new = res["theta_ell_new"][:,-1]
            # histogram of beta_f
            theta_ell = pd.read_csv("./data/model144.csv", header=None).to_numpy()[:, 0]
            # print("theta_ell")
            # print(theta_ell)
            # print("theta_ell_new")
            # print(theta_ell_new)
            pi_c_o = np.ones(len(theta_ell)) / len(theta_ell)
            # pi_c = np.load("πc_5.npy")
            pi_c = res["pic_t"][:, -1]
            plt.figure(figsize=(16,10))

            plt.hist(theta_ell, weights=pi_c_o, bins=np.linspace(0.8, 3., 16), density=True, 
                    alpha=0.5, ec="darkgrey", color="C3",label='baseline')
            plt.hist(theta_ell, weights=pi_c, bins=np.linspace(0.8, 3., 16), density=True, 
                    alpha=0.5, ec="darkgrey", color="C0",label='$\\xi_a={:.4f}$,$\\xi_g=\\xi_d=\\xi_m={:.3f}$'.format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag])  )
            plt.legend(loc='upper left')
            plt.title("Distorted probability of Climate Models")

            plt.ylim(0, 1.4)
            plt.xlabel("Climate Sensitivity")
            plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ClimateSensitivity,xia={:.4f},xig={:.3f},psi0={:.3f},psi1={:.3f},BC.pdf".format(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1]))
            plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ClimateSensitivity,xia={:.4f},xig={:.3f},psi0={:.3f},psi1={:.3f},BC.png".format(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1]))
            plt.close()

            plt.figure(figsize=(16,10))

            plt.hist(theta_ell, weights=pi_c_o, bins=np.linspace(0.8, 3., 16), density=True, 
                    alpha=0.5, ec="darkgrey", color="C3",label='baseline')
            plt.hist(theta_ell_new*1000, weights=pi_c, bins=np.linspace(0.8, 3., 16), density=True, 
                    alpha=0.5, ec="darkgrey", color="C0",label='$\\xi_a={:.4f}$,$\\xi_g=\\xi_d=\\xi_m={:.3f}$'.format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag])  )
            plt.legend(loc='upper left')
            plt.title("Distorted probability of Climate Models")


            print("mean of uncondition = {}" .format(np.average(theta_ell,weights = pi_c_o)))
            print("mean of condition = {}" .format(np.average(theta_ell,weights = pi_c)))
                

            plt.ylim(0, 1.4)
            plt.xlabel("Climate Sensitivity")
            plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ClimateSensitivity_pmean,xia={:.4f},xig={:.3f},psi0={:.3f},psi1={:.3f},BC.pdf".format(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1]))
            plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ClimateSensitivity_pmean,xia={:.4f},xig={:.3f},psi0={:.3f},psi1={:.3f},BC.png".format(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1]))
            plt.close()




for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):

            res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])
            
            
            if xigarr[id_xiag]>10:

                plt.plot(res["years"][res["states"][:, 1]<1.5], (res["Ambiguity_mean_dis"][res["states"][:, 1]<1.5]-res["Ambiguity_mean_undis"][res["states"][:, 1]<1.5])*1000,label='baseline'  )
            else:
                plt.plot(res["years"][res["states"][:, 1]<1.5], (res["Ambiguity_mean_dis"][res["states"][:, 1]<1.5]-res["Ambiguity_mean_undis"][res["states"][:, 1]<1.5])*1000,label='$\\xi_a={:.1f}$,$\\xi_g=\\xi_d=\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag])  )

            plt.xlabel("Years")
            plt.title("Mean Difference")
            # plt.ylim(0,250)
            plt.legend(loc='upper left')


plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/MeanDiff,xia={},xig={},psi0={},psi1={},BC_v2.pdf".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/MeanDiff,xia={},xig={},psi0={},psi1={},BC_v2.png".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):

            res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])
            
            
            if xigarr[id_xiag]>10:

                plt.plot(res["years"], (res["Ambiguity_mean_dis"]-res["Ambiguity_mean_undis"])*1000,label='baseline'  )
            else:
                plt.plot(res["years"], (res["Ambiguity_mean_dis"]-res["Ambiguity_mean_undis"])*1000,label='$\\xi_a={:.1f}$,$\\xi_g=\\xi_d=\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag])  )

            plt.xlabel("Years")
            plt.title("Mean Difference")
            # plt.ylim(0,250)
            plt.legend(loc='lower right')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/MeanDiff,xia={},xig={},psi0={},psi1={}_v2.pdf".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/MeanDiff,xia={},xig={},psi0={},psi1={}_v2.png".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):

            res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])
            
            
            if xigarr[id_xiag]>10:

                plt.plot(res["years"][res["states"][:, 1]<1.5], (res["Ambiguity_mean_dis"][res["states"][:, 1]<1.5]-res["Ambiguity_mean_undis"][res["states"][:, 1]<1.5])*1000,label='baseline'  )
            else:
                plt.plot(res["years"][res["states"][:, 1]<1.5], (res["Ambiguity_mean_dis_h"][res["states"][:, 1]<1.5]-res["Ambiguity_mean_undis"][res["states"][:, 1]<1.5])*1000,label='$\\xi_a={:.1f}$,$\\xi_g=\\xi_d=\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag])  )

            plt.xlabel("Years")
            plt.title("Mean Difference")
            plt.ylim(0,0.06)
            plt.legend(loc='upper left')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/MeanDiff_h,xia={},xig={},psi0={},psi1={},BC_v2.pdf".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/MeanDiff_h,xia={},xig={},psi0={},psi1={},BC_v2.png".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):

            res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1])
            
            
            if xigarr[id_xiag]>10:

                plt.plot(res["years"], (res["Ambiguity_mean_dis"]-res["Ambiguity_mean_undis"])*1000,label='baseline'  )
            else:
                plt.plot(res["years"], (res["Ambiguity_mean_dis_h"]-res["Ambiguity_mean_undis"])*1000,label='$\\xi_a={:.1f}$,$\\xi_g=\\xi_d=\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag])  )

            plt.xlabel("Years")
            plt.title("Mean Difference")
            # plt.ylim(0,250)
            plt.legend(loc='lower right')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/MeanDiff_h,xia={},xig={},psi0={},psi1={}_v2.pdf".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/MeanDiff_h,xia={},xig={},psi0={},psi1={}_v2.png".format(xiaarr,xigarr,psi0arr,psi1arr))
plt.close()
