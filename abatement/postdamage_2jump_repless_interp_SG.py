"""
post_damage.py
======================
Solver for solving post damage HJBs, with different values of gamma_3 
"""
# Optimization of post jump HJB
#Required packages
import os
import sys
sys.path.append('./src')
import csv
from supportfunctions import *
from supportfunctions import finiteDiff_3D
sys.stdout.flush()
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import petsclinearsystem
from scipy.sparse import spdiags
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from datetime import datetime
from solver import solver_3d
from PostSolver_repless import hjb_post_damage_post_tech, hjb_pre_damage_post_tech
from src.solver_repless import pde_one_interation
from src.solver_repless import hjb_pre_tech
import argparse
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import NearestNDInterpolator
import matplotlib.pyplot as plt
# from interpolate import interpolate_grid_sparse2dense

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
parser.add_argument("--id", type=int, default=0)
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

parser.add_argument("--hXarr_SG",nargs='+',type=float, default=(0.2, 0.2, 0.2))
parser.add_argument("--Xminarr_SG",nargs='+',type=float, default=(4.0, 0.0, -5.5, 0.0))
parser.add_argument("--Xmaxarr_SG",nargs='+',type=float, default=(9.0, 4.0, 0.0, 3.0))
parser.add_argument("--fstr_SG",type=str,default="LinearNDInterpolator")
parser.add_argument("--interp_action_name",type=str,default="2jump_step02verify_new")
args = parser.parse_args()

epsilonarr = args.epsilonarr
fractionarr = args.fractionarr
maxiterarr = args.maxiterarr


start_time = time.time()
# Parameters as defined in the paper
xi_a = args.xi_a  # Smooth ambiguity
xi_b = 1000. # Brownian misspecification
xi_g = args.xi_g  # Technology jump
xi_p = args.xi_g # Hold place for arguments, no real effects 


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

# num_gamma = args.num_gamma
# gamma_3_list = np.linspace(0,1./3.,num_gamma)

num_gamma = args.num_gamma
gamma_3_list = np.linspace(0,1./3.,num_gamma)

id_damage = args.id
gamma_3_i = gamma_3_list[id_damage]
# gamma_3_list = np.array([0.])
y_bar = 2.
y_bar_lower = 1.5


theta_ell = pd.read_csv('./data/model144.csv', header=None).to_numpy()[:, 0]/1000.
pi_c_o    = np.ones_like(theta_ell)/len(theta_ell)
sigma_y   = 1.2 * np.mean(theta_ell)
beta_f    = np.mean(theta_ell)
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

#if not os.path.exists(DataDir):
#    os.mkdir(DataDir)

os.makedirs(Data_Dir, exist_ok=True)

# filename =  "post_damage_" + str(gamma_3)  + '_{}'.format(current_time)
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


# Post damage, tech II
print("-------------------------------------------")
print("------------Post damage, Tech II----------")
print("-------------------------------------------")

model_args = (delta, alpha, kappa, mu_k, sigma_k, theta_ell, pi_c_o, sigma_y, xi_a, xi_b, gamma_1, gamma_2, gamma_3_i, y_bar, theta, lambda_bar_second, vartheta_bar_second)

model_tech2_post_damage = hjb_post_damage_post_tech(
        K, Y, model_args, v0=None,
       epsilon=epsilonarr[0], fraction=fractionarr[0] ,tol=1e-8, max_iter=maxiterarr[0], print_iteration=False)


# model_tech2_post_damage = pickle.load(open(Data_Dir + File_Name + "model_tech2_post_damage_gamma_{:.4f}".format(gamma_3_i), "rb"))

v_post = model_tech2_post_damage["v"]
V_post_3D = np.zeros_like(K_mat)
for j in range(nL):
    V_post_3D[:,:,j] = v_post

with open(Data_Dir+ File_Name + "model_tech2_post_damage_gamma_{:.4f}".format(gamma_3_i), "wb") as f:
   pickle.dump(model_tech2_post_damage, f)

model_tech2_post_damage = pickle.load(open(Data_Dir + File_Name + "model_tech2_post_damage_gamma_{:.4f}".format(gamma_3_i), "rb"))


# Post damage, tech II
pi_c = np.array([temp * np.ones(K_mat.shape) for temp in pi_c_o])
pi_c_o = pi_c.copy()
theta_ell = np.array([temp * np.ones(K_mat.shape) for temp in theta_ell])
print("-------------------------------------------")
print("------------Post damage, Tech I-----------")
print("-------------------------------------------")

V_post_tech2 = V_post_3D

# with open(Data_Dir+ File_Name + + "model_tech2_post_damage_gamma_{:.4f}".format(gamma_3_i), "rb") as f:
#     Guess = pickle.load(f)

SOutput_Dir = "/scratch/bincheng/"
SData_Dir = Output_Dir+"abatement/data_2tech/"+ args.interp_action_name  +"/"

SFile_Name = "xi_a_{}_xi_g_{}_psi_0_{}_psi_1_{}_" .format(xi_a,xi_g,psi_0,psi_1)

with open(SData_Dir+ SFile_Name  + "model_tech1_post_damage_gamma_{:.4f}".format(gamma_3_i), "rb") as f:
    Guess = pickle.load(f)

Xminarr_SG = args.Xminarr_SG
Xmaxarr_SG = args.Xmaxarr_SG
hXarr_SG = args.hXarr_SG
fstr = args.fstr_SG


def interpolate_grid_sparse2dense(Guess,fstr):


    v0_temp = Guess["v0"]
    i_star_temp = Guess["i_star"]
    e_star_temp = Guess["e_star"]
    x_star_temp = Guess["x_star"]

    print("information of Guess\n")
    print(v0_temp.shape)
    print(i_star_temp.shape)
    print(e_star_temp.shape)
    print(x_star_temp.shape)


    # defintion of Sparse Grid

    K_min_SG = Xminarr_SG[0]
    K_max_SG = Xmaxarr_SG[0]
    hK_SG    = hXarr_SG[0]
    K_SG     = np.arange(K_min_SG, K_max_SG + hK_SG, hK_SG)
    nK_SG    = len(K_SG)
    Y_min_SG = Xminarr_SG[1]
    Y_max_SG = Xmaxarr_SG[1]
    hY_SG    = hXarr_SG[1] # make sure it is float instead of int
    Y_SG     = np.arange(Y_min_SG, Y_max_SG + hY_SG, hY_SG)
    nY_SG    = len(Y_SG)
    L_min_SG = Xminarr_SG[2]
    L_max_SG = Xmaxarr_SG[2]
    hL_SG    = hXarr_SG[2]
    L_SG     = np.arange(L_min_SG, L_max_SG+hL_SG,  hL_SG)
    nL_SG    = len(L_SG)

    K_SG_mat, Y_SG_mat, L_SG_mat = np.meshgrid(K_SG, Y_SG, L_SG, indexing='ij')
    K_SG_mat_1d = K_SG_mat.ravel(order='F')
    Y_SG_mat_1d = Y_SG_mat.ravel(order='F')
    L_SG_mat_1d = L_SG_mat.ravel(order='F')

    print("information of Grid Point 1d\n")
    print(K_SG_mat_1d.shape)
    print(Y_SG_mat_1d.shape)
    print(L_SG_mat_1d.shape)

    v0_temp_1d = v0_temp.ravel(order='F')
    i_star_temp_1d = i_star_temp.ravel(order='F')
    e_star_temp_1d = e_star_temp.ravel(order='F')
    x_star_temp_1d = x_star_temp.ravel(order='F')

    print("information of guess 1d\n")
    print(v0_temp_1d.shape)
    print(i_star_temp_1d.shape)
    print(e_star_temp_1d.shape)
    print(x_star_temp_1d.shape)

    # print("original grid information in the function\n")
    # print(Xminarr)
    # print(Xmaxarr)
    # print(hXarr)

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
    L     = np.arange(L_min, L_max + hL,  hL)
    nL    = len(L)

    
    K_mat, Y_mat, L_mat = np.meshgrid(K, Y, L, indexing='ij') # ???????????????

    # print("original grid in the function\n")
    # print(K.shape)
    # print(Y.shape)
    # print(L.shape)
    # print(K_mat.shape)
    # print(Y_mat.shape)
    # print(L_mat.shape)    
    # print(np.mean(K))
    # print(np.mean(Y))
    # print(np.mean(L))
    # print("\n")
    # print(np.mean(K_mat))
    # print(np.mean(Y_mat))
    # print(np.mean(L_mat))

    print("Key Information 1d\n")
    print(K_SG_mat_1d.shape)
    print(Y_SG_mat_1d.shape)
    print(L_SG_mat_1d.shape)
    print(v0_temp_1d.shape)
    
    if fstr =="LinearNDInterpolator":
        v0_temp_interp = LinearNDInterpolator(list(zip(K_SG_mat_1d, Y_SG_mat_1d, L_SG_mat_1d)), v0_temp_1d)
        i_star_temp_interp = LinearNDInterpolator(list(zip(K_SG_mat_1d, Y_SG_mat_1d, L_SG_mat_1d)), i_star_temp_1d)
        e_star_temp_interp = LinearNDInterpolator(list(zip(K_SG_mat_1d, Y_SG_mat_1d, L_SG_mat_1d)), e_star_temp_1d)
        x_star_temp_interp = LinearNDInterpolator(list(zip(K_SG_mat_1d, Y_SG_mat_1d, L_SG_mat_1d)), x_star_temp_1d)

    elif fstr =="NearestNDInterpolator":
        v0_temp_interp = NearestNDInterpolator(list(zip(K_SG_mat_1d, Y_SG_mat_1d, L_SG_mat_1d)), v0_temp_1d)
        i_star_temp_interp = NearestNDInterpolator(list(zip(K_SG_mat_1d, Y_SG_mat_1d, L_SG_mat_1d)), i_star_temp_1d)
        e_star_temp_interp = NearestNDInterpolator(list(zip(K_SG_mat_1d, Y_SG_mat_1d, L_SG_mat_1d)), e_star_temp_1d)
        x_star_temp_interp = NearestNDInterpolator(list(zip(K_SG_mat_1d, Y_SG_mat_1d, L_SG_mat_1d)), x_star_temp_1d)

    # print("pass function name")
    # v0_temp_interp = fname(list(zip(K_SG_mat_1d, Y_SG_mat_1d, L_SG_mat_1d)), v0_temp_1d)
    # i_star_temp_interp = fname(list(zip(K_SG_mat_1d, Y_SG_mat_1d, L_SG_mat_1d)), i_star_temp_1d)
    # e_star_temp_interp = fname(list(zip(K_SG_mat_1d, Y_SG_mat_1d, L_SG_mat_1d)), e_star_temp_1d)
    # x_star_temp_interp = fname(list(zip(K_SG_mat_1d, Y_SG_mat_1d, L_SG_mat_1d)), x_star_temp_1d)

    # print("dense interval\n")
    # print(K_mat.shape)
    # print(Y_mat.shape)
    # print(L_mat.shape)
    # print("sparse interval\n")

    # print(K_SG_mat.shape)
    # print(Y_SG_mat.shape)
    # print(L_SG_mat.shape)

    v0_temp_new = v0_temp_interp(K_mat, Y_mat, L_mat)
    i_star_temp_new = i_star_temp_interp(K_mat, Y_mat, L_mat)
    e_star_temp_new = e_star_temp_interp(K_mat, Y_mat, L_mat)
    x_star_temp_new = x_star_temp_interp(K_mat, Y_mat, L_mat)

    # print("value shape\n")
    # print(v0_temp_new.shape)
    # print(i_star_temp_new.shape)
    # print(e_star_temp_new.shape)
    # print(x_star_temp_new.shape)

    res = {
            "v0"    : v0_temp_new,
            "i_star": i_star_temp_new,
            "e_star": e_star_temp_new,
            "x_star": x_star_temp_new,
            }
    
    plt.pcolormesh(K_mat[:,:,0], Y_mat[:,:,0], v0_temp_new[:,:,0], shading='auto')
    # plt.plot(x_mat[:,:,0], y_mat[:,:,0], "ok", label="input point")
    plt.plot(K_SG_mat_1d, Y_SG_mat_1d, "ok", label="input point")
    plt.legend()
    plt.colorbar()
    plt.axis("equal")
    plt.savefig("./abatement/pdf_2tech/interpolate/"+SFile_Name+args.interp_action_name+"_"+args.fstr_SG[:7]+".pdf")
    plt.savefig("./abatement/pdf_2tech/interpolate/"+SFile_Name+args.interp_action_name+"_"+args.fstr_SG[:7]+".png")
    
    with open(Data_Dir+ File_Name  +"SG_model_tech1_post_damage_gamma_{:.4f}".format(gamma_3_i), "wb") as f:
        pickle.dump(res,f)

    return res



Guess_interpolated = interpolate_grid_sparse2dense(Guess,fstr)


# Guess = None

res = hjb_pre_tech(
        state_grid=(K, Y, L), 
        model_args=(delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell, pi_c_o, pi_c, sigma_y, zeta, psi_0, psi_1, sigma_g, V_post_tech2, gamma_1, gamma_2, gamma_3_i, y_bar, xi_a, xi_g, xi_p),
        V_post_damage=None,
        tol=1e-7, epsilon=epsilonarr[1], fraction=fractionarr[1], 
        smart_guess=Guess_interpolated, 
        max_iter=maxiterarr[1],
        )


with open(Data_Dir+ File_Name  + "model_tech1_post_damage_gamma_{:.4f}".format(gamma_3_i), "wb") as f:
    pickle.dump(res, f)

