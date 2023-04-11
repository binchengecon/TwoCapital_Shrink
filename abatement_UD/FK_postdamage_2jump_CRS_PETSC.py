"""
post_damage.py
======================
Solver for solving post damage HJBs, with different values of gamma_3 

python3 -u /home/bcheng4/TwoCapital_Shrink/abatement_UD/FK_postdamage_2jump_CRS_PETSC.py --num_gamma 3 --xi_a 0.0002 --xi_g 0.025  --epsilonarr 0.1 0.1  --fractionarr 0.1 0.1   --maxiterarr 80000 200000  --id 5 --psi_0 0.105830 --psi_1 0.5 --name 2jump_step_4.00,9.00_0.0,4.0_1.0,6.0_SS_0.2,0.2,0.2_LR_0.1_CRS_PETSCFK --hXarr 0.2 0.2 0.2 --Xminarr 4.00 0.0 1.0 0.0 --Xmaxarr 9.00 4.0 6.0 3.0
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
from solver import solver_3d
from src.FK_PreSolver_CRS import fk_pre_tech
from src.FK_PreSolver_CRS import fk_pre_tech_petsc, hjb_pre_tech_check, fk_y_pre_tech_petsc, fk_yshort_pre_tech_petsc, fk_y_pre_tech, fk_y_pre_tech_plot
from src.ResultSolver_CRS import *
import argparse
import pickle
import matplotlib.pyplot as plt

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

# args = parser.parse_args()
args, unknown = parser.parse_known_args()

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
# vartheta_bar = 0.0453
vartheta_bar = 0.05
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

with open(Data_Dir+ File_Name + "model_tech2_post_damage_gamma_{:.4f}".format(gamma_3_i), "rb") as f:
    model_tech2_post_damage = pickle.load(f)
    
# Value function Extraction

Phi_m_II_2D = model_tech2_post_damage['v']
Phi_m_II_2D_dvdy = finiteDiff_3D(Phi_m_II_2D,1,1,hY)
print("dvdY=",Phi_m_II_2D_dvdy.min(),Phi_m_II_2D_dvdy.max())

# v_post = model_tech2_post_damage["v"]
Phi_m_II_3D = np.zeros_like(K_mat)
for j in range(nL):
    Phi_m_II_3D[:,:,j] = Phi_m_II_2D


F_post_3D = np.zeros_like(K_mat)
F_m_II = F_post_3D

theta_ell = pd.read_csv('./data/model144.csv', header=None).to_numpy()[:, 0]/1000.
pi_c_o    = np.ones_like(theta_ell)/len(theta_ell)
pi_c_o = np.array([temp * np.ones(K_mat.shape) for temp in pi_c_o])
theta_ell = np.array([temp * np.ones(K_mat.shape) for temp in theta_ell])


# n_bar1 = len(Y_short)-1
# n_bar2 = np.abs(Y_short - y_bar).argmin()

print("-------------------------------------------")
print("------------Y FK Distorted: Post damage, Tech I: -----------")
print("-------------------------------------------")



with open(Data_Dir+ File_Name + "model_tech1_post_damage_gamma_{:.4f}".format(gamma_3_i), "rb") as f:
    model_tech1_post_damage = pickle.load(f)

Phi_m = model_tech1_post_damage['v0']


print("Phi_m_II-Phi_m={},{}".format((Phi_m_II_3D-Phi_m).min(),(Phi_m_II_3D-Phi_m).max()))

# Control Extraction
 
i = model_tech1_post_damage['i_star']
e = model_tech1_post_damage['e_star']
# e = model_tech1_post_damage['e_orig']
x = model_tech1_post_damage['x_star']
pi_c = model_tech1_post_damage['pi_c']
g_tech = model_tech1_post_damage['g_tech']
h  = model_tech1_post_damage['h']

# for i_temp in range(len(Y)):
#     print(i_temp,Y[i_temp],e[:,i_temp,:].min()) # Y[i_temp] goes to 2.2 before minimum emission become 0
    
# res = fk_yshort_pre_tech(
# res = fk_yshort_pre_tech_petsc(
# res = fk_y_pre_tech_petsc(
    
res = fk_y_pre_tech(
        state_grid=(K, Y, L), 
        model_args=(delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell, pi_c_o, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3_i, y_bar, xi_a, xi_g, xi_p),
        controls=(i,e,x,pi_c,g_tech, h),
        VF = (Phi_m_II_3D, Phi_m),
        FFK = (F_m_II),
        # n_bar = 50,
        V_post_damage=None,
        tol=1e-7, epsilon=epsilonarr[1], fraction=fractionarr[1], 
        max_iter=maxiterarr[1],
        )


with open(Data_Dir+ File_Name  + "FK_Y_Distorted_model_tech1_post_damage_gamma_{:.4f}".format(gamma_3_i), "wb") as f:
    pickle.dump(res, f)

with open(Data_Dir+ File_Name  + "FK_Y_Distorted_model_tech1_post_damage_gamma_{:.4f}".format(gamma_3_i), "rb") as f:
    res = pickle.load(f)

# res = fk_y_pre_tech_plot(
#         state_grid=(K, Y, L), 
#         model_args=(delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell, pi_c_o, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3_i, y_bar, xi_a, xi_g, xi_p),
#         controls=(i,e,x,pi_c,g_tech),
#         VF = (Phi_m_II_3D, Phi_m),
#         FFK = (F_m_II),
#         # n_bar = 50,
#         V_post_damage=None,
#         tol=1e-7, epsilon=epsilonarr[1], fraction=fractionarr[1], 
#         max_iter=maxiterarr[1],
#         )


print("-------------------------------------------")
print("------------Y FK Undistorted: Post damage, Tech I: -----------")
print("-------------------------------------------")



with open(Data_Dir+ File_Name + "model_tech1_post_damage_gamma_{:.4f}".format(gamma_3_i), "rb") as f:
    model_tech1_post_damage = pickle.load(f)

Phi_m = model_tech1_post_damage['v0']


print("Phi_m_II-Phi_m={},{}".format((Phi_m_II_3D-Phi_m).min(),(Phi_m_II_3D-Phi_m).max()))

# Control Extraction
 
i = model_tech1_post_damage['i_star']
e = model_tech1_post_damage['e_star']
x = model_tech1_post_damage['x_star']
pi_c = model_tech1_post_damage['pi_c']
pi_c = np.ones(pi_c.shape)/len(theta_ell)
# g_tech = model_tech1_post_damage['g_tech']
g_tech = np.ones(g_tech.shape)
h  = model_tech1_post_damage['h']
h = np.zeros(h.shape)

# res = fk_yshort_pre_tech_petsc(
# res = fk_y_pre_tech_petsc(
res = fk_y_pre_tech(
        state_grid=(K, Y, L), 
        model_args=(delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell, pi_c_o, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3_i, y_bar, xi_a, xi_g, xi_p),
        controls=(i,e,x,pi_c,g_tech, h),
        VF = (Phi_m_II_3D, Phi_m),
        FFK = (F_m_II),
        # n_bar = 50,
        V_post_damage=None,
        tol=1e-7, epsilon=epsilonarr[1], fraction=fractionarr[1], 
        max_iter=maxiterarr[1],
        )



with open(Data_Dir+ File_Name  + "FK_Y_Undistorted_model_tech1_post_damage_gamma_{:.4f}".format(gamma_3_i), "wb") as f:
    pickle.dump(res, f)

with open(Data_Dir+ File_Name  + "FK_Y_Undistorted_model_tech1_post_damage_gamma_{:.4f}".format(gamma_3_i), "rb") as f:
    res = pickle.load(f)



print("-------------------------------------------")
print("------------FK Distorted: Post damage, Tech I: -----------")
print("-------------------------------------------")



with open(Data_Dir+ File_Name + "model_tech1_post_damage_gamma_{:.4f}".format(gamma_3_i), "rb") as f:
    model_tech1_post_damage = pickle.load(f)

Phi_m = model_tech1_post_damage['v0']


print("Phi_m_II-Phi_m={},{}".format((Phi_m_II_3D-Phi_m).min(),(Phi_m_II_3D-Phi_m).max()))

# Control Extraction
 
i = model_tech1_post_damage['i_star']
e = model_tech1_post_damage['e_star']
x = model_tech1_post_damage['x_star']
pi_c = model_tech1_post_damage['pi_c']
g_tech = model_tech1_post_damage['g_tech']
h  = model_tech1_post_damage['h']


res = fk_pre_tech(
# res = fk_pre_tech_petsc(
        state_grid=(K, Y, L), 
        model_args=(delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell, pi_c_o, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3_i, y_bar, xi_a, xi_g, xi_p),
        controls=(i,e,x,pi_c,g_tech, h),
        VF = (Phi_m_II_3D, Phi_m),
        FFK = (F_m_II),
        V_post_damage=None,
        tol=1e-7, epsilon=epsilonarr[1], fraction=fractionarr[1], 
        max_iter=maxiterarr[1],
        )


with open(Data_Dir+ File_Name  + "FK_Distorted_model_tech1_post_damage_gamma_{:.4f}".format(gamma_3_i), "wb") as f:
    pickle.dump(res, f)

with open(Data_Dir+ File_Name  + "FK_Distorted_model_tech1_post_damage_gamma_{:.4f}".format(gamma_3_i), "rb") as f:
    res = pickle.load(f)


print("-------------------------------------------")
print("------------FK Undistorted: Post damage, Tech I: -----------")
print("-------------------------------------------")

with open(Data_Dir+ File_Name + "model_tech1_post_damage_gamma_{:.4f}".format(gamma_3_i), "rb") as f:
    model_tech1_post_damage = pickle.load(f)

with open(Data_Dir+ File_Name + "model_tech2_post_damage_gamma_{:.4f}".format(gamma_3_i), "rb") as f:
    model_tech2_post_damage = pickle.load(f)

# Value function Extraction

Phi_m_II_2D = model_tech2_post_damage['v']

# v_post = model_tech2_post_damage["v"]
Phi_m_II_3D = np.zeros_like(K_mat)
for j in range(nL):
    Phi_m_II_3D[:,:,j] = Phi_m_II_2D


F_post_3D = np.zeros_like(K_mat)
F_m_II = F_post_3D


Phi_m = model_tech1_post_damage['v0']

# Control Extraction
theta_ell = pd.read_csv('./data/model144.csv', header=None).to_numpy()[:, 0]/1000.


i = model_tech1_post_damage['i_star']
e = model_tech1_post_damage['e_star']
x = model_tech1_post_damage['x_star']
pi_c = model_tech1_post_damage['pi_c']
g_tech = model_tech1_post_damage['g_tech']
g_tech = np.ones(Phi_m.shape)
h  = model_tech1_post_damage['h']
h  = np.zeros(h.shape)

pi_c_o    = np.ones_like(theta_ell)/len(theta_ell)
pi_c_o = np.array([temp * np.ones(K_mat.shape) for temp in pi_c_o])
pi_c = pi_c
theta_ell = np.array([temp * np.ones(K_mat.shape) for temp in theta_ell])


# Guess = None

res = fk_pre_tech(
# res = fk_pre_tech_petsc(
        state_grid=(K, Y, L), 
        model_args=(delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell, pi_c_o, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3_i, y_bar, xi_a, xi_g, xi_p),
        controls=(i,e,x,pi_c,g_tech,h),
        VF = (Phi_m_II_3D, Phi_m),
        FFK = (F_m_II),
        V_post_damage=None,
        tol=1e-7, epsilon=epsilonarr[1], fraction=fractionarr[1], 
        max_iter=maxiterarr[1],
        )


with open(Data_Dir+ File_Name  + "FK_Undistorted_model_tech1_post_damage_gamma_{:.4f}".format(gamma_3_i), "wb") as f:
    pickle.dump(res, f)

with open(Data_Dir+ File_Name  + "FK_Undistorted_model_tech1_post_damage_gamma_{:.4f}".format(gamma_3_i), "rb") as f:
    res = pickle.load(f)





print("-------------------------------------------")
print("------------HJB Distorted: Post damage, Tech I: -----------")
print("-------------------------------------------")

with open(Data_Dir+ File_Name + "model_tech1_post_damage_gamma_{:.4f}".format(gamma_3_i), "rb") as f:
    model_tech1_post_damage = pickle.load(f)

with open(Data_Dir+ File_Name + "model_tech2_post_damage_gamma_{:.4f}".format(gamma_3_i), "rb") as f:
    model_tech2_post_damage = pickle.load(f)

# Value function Extraction

Phi_m_II_2D = model_tech2_post_damage['v']

# v_post = model_tech2_post_damage["v"]
Phi_m_II_3D = np.zeros_like(K_mat)
for j in range(nL):
    Phi_m_II_3D[:,:,j] = Phi_m_II_2D


F_post_3D = np.zeros_like(K_mat)
F_m_II = F_post_3D



Phi_m = model_tech1_post_damage['v0']

# Control Extraction

i = model_tech1_post_damage['i_star']
e = model_tech1_post_damage['e_star']
x = model_tech1_post_damage['x_star']
pi_c = model_tech1_post_damage['pi_c']
g_tech = model_tech1_post_damage['g_tech']
h  = model_tech1_post_damage['h']

theta_ell = pd.read_csv('./data/model144.csv', header=None).to_numpy()[:, 0]/1000.
pi_c_o    = np.ones_like(theta_ell)/len(theta_ell)
pi_c_o = np.array([temp * np.ones(K_mat.shape) for temp in pi_c_o])
theta_ell = np.array([temp * np.ones(K_mat.shape) for temp in theta_ell])


# Guess = None

res = hjb_pre_tech_check(
        state_grid=(K, Y, L), 
        model_args=(delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell, pi_c_o, sigma_y, zeta, psi_0, psi_1, sigma_g, Phi_m_II_3D, gamma_1, gamma_2, gamma_3_i, y_bar, xi_a, xi_g, xi_p),
        controls=(i,e,x, pi_c, g_tech, h, Phi_m),
        V_post_damage=None,
        tol=1e-7, epsilon=epsilonarr[1], fraction=fractionarr[1], 
        max_iter=maxiterarr[1],
        )


with open(Data_Dir+ File_Name  + "HJB_Distorted_model_tech1_post_damage_gamma_{:.4f}".format(gamma_3_i), "wb") as f:
    pickle.dump(res, f)

with open(Data_Dir+ File_Name  + "HJB_Distorted_model_tech1_post_damage_gamma_{:.4f}".format(gamma_3_i), "rb") as f:
    res = pickle.load(f)



print("-------------------------------------------")
print("------------HJB Undistorted g_tech: Post damage, Tech I: -----------")
print("-------------------------------------------")

with open(Data_Dir+ File_Name + "model_tech1_post_damage_gamma_{:.4f}".format(gamma_3_i), "rb") as f:
    model_tech1_post_damage = pickle.load(f)

with open(Data_Dir+ File_Name + "model_tech2_post_damage_gamma_{:.4f}".format(gamma_3_i), "rb") as f:
    model_tech2_post_damage = pickle.load(f)

# Value function Extraction

Phi_m_II_2D = model_tech2_post_damage['v']

# v_post = model_tech2_post_damage["v"]
Phi_m_II_3D = np.zeros_like(K_mat)
for j in range(nL):
    Phi_m_II_3D[:,:,j] = Phi_m_II_2D

Phi_m = model_tech1_post_damage['v0']

# Control Extraction

i = model_tech1_post_damage['i_star']
e = model_tech1_post_damage['e_star']
x = model_tech1_post_damage['x_star']

pi_c = model_tech1_post_damage['pi_c']
# pi_c = np.ones(pi_c.shape)
g_tech = model_tech1_post_damage['g_tech']
g_tech = np.ones(g_tech.shape)
h  = model_tech1_post_damage['h']


theta_ell = pd.read_csv('./data/model144.csv', header=None).to_numpy()[:, 0]/1000.
pi_c_o    = np.ones_like(theta_ell)/len(theta_ell)
# pi_c_o = np.array([temp * np.ones(K_mat.shape) for temp in pi_c_o])
# theta_ell = np.array([temp * np.ones(K_mat.shape) for temp in theta_ell])


# Guess = None

res = hjb_pre_tech_check(
        state_grid=(K, Y, L), 
        model_args=(delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell, pi_c_o, sigma_y, zeta, psi_0, psi_1, sigma_g, Phi_m_II_3D, gamma_1, gamma_2, gamma_3_i, y_bar, xi_a, xi_g, xi_p),
        controls=(i,e,x,pi_c,g_tech, h, Phi_m),
        V_post_damage=None,
        tol=1e-7, epsilon=epsilonarr[1], fraction=fractionarr[1], 
        max_iter=maxiterarr[1],
        )


with open(Data_Dir+ File_Name  + "HJB_Undistorted_model_tech1_post_damage_gamma_{:.4f}".format(gamma_3_i), "wb") as f:
    pickle.dump(res, f)

with open(Data_Dir+ File_Name  + "HJB_Undistorted_model_tech1_post_damage_gamma_{:.4f}".format(gamma_3_i), "rb") as f:
    res = pickle.load(f)




print("-------------------------------------------")
print("------------HJB Undistorted pi_c and g_tech: Post damage, Tech I: -----------")
print("-------------------------------------------")

with open(Data_Dir+ File_Name + "model_tech1_post_damage_gamma_{:.4f}".format(gamma_3_i), "rb") as f:
    model_tech1_post_damage = pickle.load(f)

with open(Data_Dir+ File_Name + "model_tech2_post_damage_gamma_{:.4f}".format(gamma_3_i), "rb") as f:
    model_tech2_post_damage = pickle.load(f)

# Value function Extraction

Phi_m_II_2D = model_tech2_post_damage['v']

# v_post = model_tech2_post_damage["v"]
Phi_m_II_3D = np.zeros_like(K_mat)
for j in range(nL):
    Phi_m_II_3D[:,:,j] = Phi_m_II_2D

Phi_m = model_tech1_post_damage['v0']

# Control Extraction

i = model_tech1_post_damage['i_star']
e = model_tech1_post_damage['e_star']
x = model_tech1_post_damage['x_star']

pi_c = model_tech1_post_damage['pi_c']
# pi_c = np.ones(pi_c.shape)
g_tech = model_tech1_post_damage['g_tech']
g_tech = np.ones(g_tech.shape)
h  = model_tech1_post_damage['h']


theta_ell = pd.read_csv('./data/model144.csv', header=None).to_numpy()[:, 0]/1000.
pi_c_o    = np.ones_like(theta_ell)/len(theta_ell)
pi_c_o = np.array([temp * np.ones(K_mat.shape) for temp in pi_c_o])
pi_c = pi_c_o
theta_ell = np.array([temp * np.ones(K_mat.shape) for temp in theta_ell])


# Guess = None

res = hjb_pre_tech_check(
        state_grid=(K, Y, L), 
        model_args=(delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell, pi_c_o, sigma_y, zeta, psi_0, psi_1, sigma_g, Phi_m_II_3D, gamma_1, gamma_2, gamma_3_i, y_bar, xi_a, xi_g, xi_p),
        controls=(i,e,x,pi_c,g_tech,h, Phi_m),
        V_post_damage=None,
        tol=1e-7, epsilon=epsilonarr[1], fraction=fractionarr[1], 
        max_iter=maxiterarr[1],
        )


with open(Data_Dir+ File_Name  + "HJB_UndistortedFull_model_tech1_post_damage_gamma_{:.4f}".format(gamma_3_i), "wb") as f:
    pickle.dump(res, f)

with open(Data_Dir+ File_Name  + "HJB_UndistortedFull_model_tech1_post_damage_gamma_{:.4f}".format(gamma_3_i), "rb") as f:
    res = pickle.load(f)




# print("-------------------------------------------")
# print("------------New function: HJB Undistorted pi_c and g_tech: Post damage, Tech I: -----------")
# print("-------------------------------------------")

# with open(Data_Dir+ File_Name + "model_tech1_post_damage_gamma_{:.4f}".format(gamma_3_i), "rb") as f:
#     model_tech1_post_damage = pickle.load(f)

# with open(Data_Dir+ File_Name + "model_tech2_post_damage_gamma_{:.4f}".format(gamma_3_i), "rb") as f:
#     model_tech2_post_damage = pickle.load(f)

# # Value function Extraction

# Phi_m_II_2D = model_tech2_post_damage['v']

# # v_post = model_tech2_post_damage["v"]
# Phi_m_II_3D = np.zeros_like(K_mat)
# for j in range(nL):
#     Phi_m_II_3D[:,:,j] = Phi_m_II_2D

# Phi_m = model_tech1_post_damage['v0']

# # Control Extraction

# i = model_tech1_post_damage['i_star']
# e = model_tech1_post_damage['e_star']
# x = model_tech1_post_damage['x_star']

# pi_c = model_tech1_post_damage['pi_c']
# pi_c = np.ones(pi_c.shape)
# g_tech = model_tech1_post_damage['g_tech']
# g_tech = np.ones(g_tech.shape)
# h  = model_tech1_post_damage['h']


# theta_ell = pd.read_csv('./data/model144.csv', header=None).to_numpy()[:, 0]/1000.
# pi_c_o    = np.ones_like(theta_ell)/len(theta_ell)
# # pi_c_o = np.array([temp * np.ones(K_mat.shape) for temp in pi_c_o])
# # theta_ell = np.array([temp * np.ones(K_mat.shape) for temp in theta_ell])


# # Guess = None
# xi_a_post = 100000.
# xi_g_post = 100000.
# xi_p_post = 100000.

# n_bar = len(Y)-1
# res = hjb_pre_tech_noupdate_noFT(
#         state_grid=(K, Y, L), 
#         model_args=(delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell, sigma_y, zeta, psi_0, psi_1, sigma_g, Phi_m_II_3D, gamma_1, gamma_2, gamma_3_i, y_bar, xi_a_post, xi_g_post, xi_p_post),
#         control_fixed=(i, e, x, h, Phi_m),
#         n_bar = n_bar,
#         V_post_damage=None,
#         tol=1e-7, epsilon=epsilonarr[1], fraction=fractionarr[1], 
#         smart_guess=None, 
#         max_iter=maxiterarr[1],
#         )



# with open(Data_Dir+ File_Name  + "HJB_NewUndistortedFull_model_tech1_post_damage_gamma_{:.4f}".format(gamma_3_i), "wb") as f:
#     pickle.dump(res, f)

# with open(Data_Dir+ File_Name  + "HJB_NewUndistortedFull_model_tech1_post_damage_gamma_{:.4f}".format(gamma_3_i), "rb") as f:
#     res = pickle.load(f)


