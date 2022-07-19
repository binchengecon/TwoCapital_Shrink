# Optimization of post jump HJB
#Required packages
import os
import sys

from numpy import meshgrid
sys.path.append('./src')
print(sys.path)
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
from PostSolver import hjb_post_damage_post_tech, hjb_pre_damage_post_tech
from src.solver import pde_one_interation
from src.solver import hjb_pre_tech
import argparse

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
parser.add_argument("--xi_r", type=float, default=1000.)
args = parser.parse_args()


start_time = time.time()
# Parameters as defined in the paper
# xi_a = 2/10000.  # Smooth ambiguity
# xi_p = 0.025 # Damage poisson
xi_b = 1000. # Brownian misspecification
# xi_g = 0.025  # Technology jump

# psi_0_grid = np.array([0.003,0.006,0.009])
psi_0_grid = np.array([0.008, 0.010, 0.012])
# psi_1_grid = np.array([.5,.7,.9])
psi_1_grid = np.array([.8])

psi_0_meshgrid,psi_1_meshgrid = np.meshgrid(psi_0_grid,psi_1_grid)

psi_0_meshgrid_1d =psi_0_meshgrid.ravel(order='F')
psi_1_meshgrid_1d = psi_1_meshgrid.ravel(order='F')

maxiter = 15000

def model(xi_a,xi_g,xi_p,psi_0,psi_1):

    # Data_Dir = "./res_data/xi_p_" + str(xi_p) + "_xi_g_" + str(xi_g) +  "/"
# Data_Dir = "./abatement/data/"+"xi_a_"+str(xi_a*10000.) +"_xi_p_" + str(xi_p) + "_xi_g_" + str(xi_g)+"_"
    Data_Dir = "./abatement/data_2tech/"

    File_Name = "Compare_Psi0_Psi1_xi_a_{}_xi_g_{}_psi_0_{}_psi_1_{}_" .format(xi_a,xi_g,psi_0,psi_1)


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
    gamma_3_list = np.linspace(0., 1./3., 6)
    # gamma_3_list = np.array([0.])
    y_bar = 2.
    y_bar_lower = 1.5


    theta_ell = pd.read_csv('./data/model144.csv', header=None).to_numpy()[:, 0]/1000.
    pi_c_o    = np.ones_like(theta_ell)/len(theta_ell)
    sigma_y   = 1.2 * np.mean(theta_ell)
    beta_f    = 1.86 / 1000
    # Jump intensity
    zeta      = 0.00
    # psi_0     = 0.00025
    # psi_1     = 1/2
    sigma_g   = 0.016
    # Tech jump
    lambda_bar_first = lambda_bar / 2
    vartheta_bar_first = vartheta_bar / 2
    lambda_bar_second = 1e-9
    vartheta_bar_second = 0.

    # Grids Specification
    # Coarse Grids
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


    # Post damage, tech III
    print("-------------------------------------------")
    print("------------Post damage, Tech III----------")
    print("-------------------------------------------")
    # model_tech3_post_damage =  []
    # # for gamma_3_i in gamma_3_list:

    model_args = (delta, alpha, kappa, mu_k, sigma_k, theta_ell, pi_c_o, sigma_y, xi_a, xi_b, gamma_1, gamma_2, 0., y_bar, theta, lambda_bar_second, vartheta_bar_second)

    model_tech3_post_damage = hjb_post_damage_post_tech(
            K, Y, model_args, v0=None,
            epsilon=.005, fraction=.005,tol=1e-8, max_iter=2000, print_iteration=False)



    v_post = model_tech3_post_damage["v"]
    V_post_3D = np.zeros_like(K_mat)
    for j in range(nL):
        V_post_3D[:,:,j] = v_post

    # model_tech3_post_damage.append(v_post_i)
    with open(Data_Dir + File_Name+ "model_tech3_post_damage", "wb") as f:
        pickle.dump(model_tech3_post_damage, f)

    model_tech3_post_damage = pickle.load(open(Data_Dir + File_Name+ "model_tech3_post_damage", "rb"))


    # Post damage, tech II

    print("-------------------------------------------")
    print("------------Post damage, Tech I-----------")
    print("-------------------------------------------")

    pi_c = np.array([temp * np.ones(K_mat.shape) for temp in pi_c_o])
    pi_c_o = pi_c.copy()
    theta_ell = np.array([temp * np.ones(K_mat.shape) for temp in theta_ell])

    # Guess = pickle.load(open(Data_Dir + File_Name+ "model_tech1_post_damage", "rb"))
    model_tech1_post_damage = []

    for i in range(len(gamma_3_list)):
        gamma_3_i = gamma_3_list[i]
        V_post_tech1 = V_post_3D

        res = hjb_pre_tech(
                state_grid=(K, Y, L), 
                model_args=(delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell, pi_c_o, pi_c, sigma_y, zeta, psi_0, psi_1, sigma_g, V_post_tech1, gamma_1, gamma_2, gamma_3_i, y_bar, xi_a, xi_g, xi_p),
                V_post_damage=None,
                # v0=Guess[i]["v0"],
                v0=None,
                tol=1e-7, epsilon=.005, fraction=.005, 
                smart_guess=None,
                # smart_guess=Guess[i],
                max_iter=maxiter
                )

        model_tech1_post_damage.append(res)

    with open(Data_Dir + File_Name+ "model_tech1_post_damage", "wb") as f:
        pickle.dump(model_tech1_post_damage, f)

    model_tech1_post_damage = pickle.load(open(Data_Dir + File_Name+ "model_tech1_post_damage", "rb"))


    # PRE DAMAGE
    print("-------------------------------------------")
    print("---------Pre damage, Tech III--------------")
    print("-------------------------------------------")
    id_2 = np.abs(Y - y_bar).argmin()
    Y_min_short = 0.
    Y_max_short = 3.
    Y_short     = np.arange(Y_min_short, Y_max_short + hY, hY)
    nY_short    = len(Y_short)
    # Pre damage, tech III
    # # Compile v_i
    v_i = []
    for i in range(len(gamma_3_list)):
        v_post_damage_i = np.zeros((nK, nY_short))
        for j in range(nY_short):
            v_post_damage_i[:, j] = model_tech3_post_damage["v"][:, id_2]

        v_i.append(v_post_damage_i)

    v_i = np.array(v_i)
    pi_d_o = np.ones(len(gamma_3_list)) / len(gamma_3_list)
    pi_d_o = np.array([temp * np.ones((nK, nY_short)) for temp in pi_d_o])

    theta_ell = pd.read_csv('./data/model144.csv', header=None).to_numpy()[:, 0]/1000.
    pi_c_o = np.ones(len(theta_ell)) / len(theta_ell)
    pi_c_o = np.array([temp * np.ones((nK, nY_short)) for temp in pi_c_o])
    theta_ell = np.array([temp * np.ones((nK, nY_short)) for temp in theta_ell])

    model_tech3_pre_damage = hjb_pre_damage_post_tech(
            K, Y_short, 
            model_args=(delta, alpha, kappa, mu_k, sigma_k, theta_ell, pi_c_o, sigma_y, xi_a, xi_b, xi_p, pi_d_o, v_i, gamma_1, gamma_2, theta, lambda_bar_second, vartheta_bar_second, y_bar_lower),
            v0=np.mean(v_i, axis=0), epsilon=.005, fraction=.005,
            tol=1e-8, max_iter=maxiter, print_iteration=True
            )

    with open(Data_Dir + File_Name+ "model_tech3_pre_damage", "wb") as f:
        pickle.dump(model_tech3_pre_damage, f)

    # model_tech3_pre_damage = pickle.load(open(Data_Dir + File_Name+ "model_tech3_pre_damage", "rb"))

    print("-------------------------------------------")
    print("---------Pre damage, Tech I---------------")
    print("-------------------------------------------")
    pi_d_o = np.ones(len(gamma_3_list)) / len(gamma_3_list)
    pi_d_o = np.array([temp * np.ones((nK, nY_short)) for temp in pi_d_o])

    theta_ell = pd.read_csv('./data/model144.csv', header=None).to_numpy()[:, 0]/1000.
    pi_c_o = np.ones(len(theta_ell)) / len(theta_ell)
    pi_c_o = np.array([temp * np.ones((nK, nY_short, nL)) for temp in pi_c_o])
    pi_c = pi_c_o.copy()
    theta_ell = np.array([temp * np.ones((nK, nY_short, nL)) for temp in theta_ell])
    v_i = []
    for model in model_tech1_post_damage:
        v_post_damage_i = model["v0"]
        v_post_damage_temp = np.zeros((nK, nY_short, nL))
        for j in range(nY_short):
            v_post_damage_temp[:, j, :] = v_post_damage_i[:, id_2, :]
        v_i.append(v_post_damage_temp)
    v_i = np.array(v_i)
    v_post = model_tech3_pre_damage["v"][:, :nY_short]
    v_tech3 = np.zeros((nK, nY_short, nL))
    for i in range(nL):
        v_tech3[:, :, i] = v_post

    model_args =(delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell, pi_c_o, pi_c, sigma_y, zeta, psi_0, psi_1, sigma_g, v_tech3, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a, xi_g, xi_p)

    # Guess = pickle.load(open(Data_Dir + File_Name+ "model_tech2_pre_damage", "rb"))
    model_tech1_pre_damage = hjb_pre_tech(
            state_grid=(K, Y_short, L), 
            model_args=model_args, V_post_damage=v_i, 
            tol=1e-8, epsilon=.005, fraction=.005, max_iter=maxiter,
            v0=np.mean(v_i, axis=0),
            smart_guess=None,
            )

    with open(Data_Dir + File_Name+ "model_tech1_pre_damage", "wb") as f:
        pickle.dump(model_tech1_pre_damage, f)



# model(1000.,1000.,1000.)
# model(2/10000.,0.050,0.050)
# model(2/10000.,0.025,0.025)

for psi_0,psi_1 in zip(psi_0_meshgrid_1d,psi_1_meshgrid_1d):
    # print(psi_0,psi_1)
    model(1000.,1000.,1000.,psi_0,psi_1)
    # model(2/10000.,0.050,0.050,psi_0,psi_1)
    # model(2/10000.,0.025,0.025,psi_0,psi_1)

