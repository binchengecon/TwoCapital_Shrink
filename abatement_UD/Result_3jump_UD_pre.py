"""
pre_damage.py
=================
Solver for pre damage HJBs, tech III, tech I
"""
# Optimization of post jump HJB
#Required packages
import os
import sys
sys.path.append('./src')
import csv
from src.Utility import *
from src.Utility import finiteDiff_3D
sys.stdout.flush()
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import petsclinearsystem
from scipy.sparse import spdiags
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from datetime import datetime
# from solver import solver_3d
import src.PreSolver
import src.ResultSolver
import time
import numpy as np
import argparse
import pandas as pd
import pickle

reporterror = True
# Linear solver choices
# Chosse among petsc, petsc4py, eigen, both
# petsc: matrix assembled in C
# petsc4py: matrix assembled in Python
# eigen: matrix assembled in C++
# both: petsc+petsc4py
#
now = datetime.now()
current_time = now.strftime("%d-%H:%M")

parser = argparse.ArgumentParser(description="xi_r values")
parser.add_argument("--xi_a", type=float, default=1000.)
parser.add_argument("--xi_g", type=float, default=1000.)
parser.add_argument("--psi_0", type=float, default=0.003)
parser.add_argument("--psi_1", type=float, default=0.5)
parser.add_argument("--num_gamma",type=int,default=6)
parser.add_argument("--name",type=str,default="ReplicateSuri")
parser.add_argument("--hXarr",nargs='+',type=float)
parser.add_argument("--Xminarr",nargs='+',type=float)
parser.add_argument("--Xmaxarr",nargs='+',type=float)
parser.add_argument("--epsilonarr",nargs='+',type=float)
parser.add_argument("--fractionarr",nargs='+',type=float)
parser.add_argument("--maxiterarr",nargs='+',type=int)
parser.add_argument("--scheme",type=str)
parser.add_argument("--HJB_solution",type=str)


args = parser.parse_args()


epsilonarr = args.epsilonarr
fractionarr = args.fractionarr
maxiterarr = args.maxiterarr


start_time = time.time()
# Parameters as defined in the paper
xi_a = args.xi_a # Smooth ambiguity
xi_p = args.xi_g  # Damage poisson
xi_b = 1000. # Brownian misspecification
xi_g = args.xi_g  # Technology jump

# DataDir = "./res_data/6damage/xi_a_" + str(xi_a) + "_xi_g_" + str(xi_g) +  "/"
# if not os.path.exists(DataDir):
    # os.mkdir(DataDir)
    
scheme = args.scheme
HJB_solution = args.HJB_solution


# Model parameters
delta   = 0.010
alpha   = 0.115
kappa   = 6.667
mu_k    = -0.043
sigma_k = np.sqrt(0.0087**2 + 0.0038**2)
# Technology
theta        = 3
lambda_bar   = 0.1206
vartheta_bar = 0.0453
# Damage function
gamma_1 = 1.7675/10000
gamma_2 = 0.0022 * 2
# gamma_3 = 0.3853 * 2

num_gamma = args.num_gamma
gamma_3_list = np.linspace(0,1./3.,num_gamma)


y_bar = 2.
y_bar_lower = 1.5


theta_ell = pd.read_csv('./data/model144_p.csv', header=None).to_numpy()[:, 0]/1000.
psi_2 = pd.read_csv('./data/psi2value_p.csv', header=None).to_numpy()[:, 0]
pi_c_o    = np.ones_like(theta_ell)/len(theta_ell)
sigma_y   = 1.2 * np.mean(theta_ell)
beta_f    = 1.86 / 1000
# Jump intensity
zeta      = 0.00
psi_0     = args.psi_0
psi_1     = args.psi_1
sigma_g   = 0.016
# Tech jump
lambda_bar_first = lambda_bar / 2
vartheta_bar_first = vartheta_bar / 2
lambda_bar_second = 1e-9
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

X1     = K
nX1    = len(X1)
hX1    = X1[1] - X1[0]
X1_min = X1.min()
X1_max = X1.max()
X2     = Y
nX2    = len(X2)
hX2    = X2[1] - X2[0]
X2_min = X2.min()
X2_max = X2.max()
X3     = L
nX3    = len(X3)
hX3    = X3[1] - X3[0]
X3_min = X3.min()
X3_max = X3.max()


Output_Dir = "/scratch/bincheng/"
Data_Dir = Output_Dir+"abatement/data_2tech/"+args.name+"/"

File_Name = "xi_a_{}_xi_g_{}_psi_0_{}_psi_1_{}_" .format(xi_a,xi_g,psi_0,psi_1)

os.makedirs(Data_Dir, exist_ok=True)

# if not os.path.exists(DataDir):
print("Grid dimension: [{}, {}, {}]\n".format(nX1, nX2, nX3))
print("Grid step: [{}, {}, {}]\n".format(hX1, hX2, hX3))

# Discretization of the state space for numerical PDE solution.
######## post jump, 3 states
(X1_mat, X2_mat, X3_mat) = np.meshgrid(X1, X2, X3, indexing = 'ij')
stateSpace = np.hstack([X1_mat.reshape(-1,1,order = 'F'), X2_mat.reshape(-1,1,order = 'F'), X3_mat.reshape(-1, 1, order='F')])
K_mat = X1_mat
Y_mat = X2_mat
L_mat = X3_mat
# For PETSc
X1_mat_1d = X1_mat.ravel(order='F')
X2_mat_1d = X2_mat.ravel(order='F')
X3_mat_1d = X3_mat.ravel(order='F')
lowerLims = np.array([X1_min, X2_min, X3_min], dtype=np.float64)
upperLims = np.array([X1_max, X2_max, X3_max], dtype=np.float64)



print("-------------------------------------------")
print("---------Pre damage, Tech II--------------")
print("-------------------------------------------")
id_2 = np.abs(Y - y_bar).argmin()
Y_min_short = Xminarr[3]
Y_max_short = Xmaxarr[3]
Y_short     = np.arange(Y_min_short, Y_max_short + hY, hY)
nY_short    = len(Y_short)

n_bar1 = len(Y_short)-1
n_bar2 = np.abs(Y_short - y_bar).argmin()


model_tech2_pre_damage = pickle.load(open(Data_Dir+ File_Name + "model_tech2_pre_damage", "rb"))





if scheme == "check":
    
    if HJB_solution=="iterative_partial":
        xi_a_post = xi_a
        xi_g_post = xi_g
        xi_p_post = xi_g
        File_Name_Suffix = "_xiapost_{}_xig_post_{}_xippost_{}".format(xi_a_post, xi_g_post, xi_p_post) + "_full_" + scheme + "_" +HJB_solution
        
        
        # Post damage, tech I
        print("-------------------------------------------")
        print("------------Post damage, Tech I-----------")
        print("-------------------------------------------")
        model_tech1_post_damage = []
        for i in range(len(gamma_3_list)):
            gamma_3_i = gamma_3_list[i]
            model_i = pickle.load(open(Data_Dir+ File_Name + "model_tech1_post_damage_gamma_{:.4f}".format(gamma_3_i) + File_Name_Suffix, "rb"))
            model_tech1_post_damage.append(model_i)

        with open(Data_Dir+ File_Name + "model_tech1_post_damage" + File_Name_Suffix, "wb") as f:
            pickle.dump(model_tech1_post_damage, f)

        # model_tech1_post_damage = pickle.load(open(Data_Dir+ File_Name + "model_tech1_post_damage", "rb"))
        print("Compiled.")

        # Pre damage, tech I
        # pi_d_o = np.ones(len(gamma_3_list)) / len(gamma_3_list)
        # pi_d_o = np.array([temp * np.ones((nK, nY_short)) for temp in pi_d_o])

        theta_ell = pd.read_csv('./data/model144_p.csv', header=None).to_numpy()[:, 0]/1000.
        psi_2 = pd.read_csv('./data/psi2value_p.csv', header=None).to_numpy()[:, 0]


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
        
        
        model_args =(delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell, sigma_y, zeta, psi_0, psi_1, psi_2, sigma_g, v_tech2, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a_pre, xi_g_pre, xi_p_pre)

        #########################################
        ######### Start of Compute###############
        #########################################

        
        model_tech1_pre_damage = pickle.load(open(Data_Dir + File_Name + "model_tech1_pre_damage", "rb"))
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

        with open(Data_Dir+ File_Name + "model_tech1_pre_damage"+File_Name_Suffix_pre, "wb") as f:
            pickle.dump(model_tech1_pre_damage, f)
        model_tech1_pre_damage = pickle.load(open(Data_Dir+ File_Name + "model_tech1_pre_damage"+File_Name_Suffix_pre, "rb"))
