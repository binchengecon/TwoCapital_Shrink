#########################################
# Optimization of post jump HJB
#########################################


#########################################
# Library Loading
#########################################

import os
import sys
sys.path.append('../src')
import csv
from supportfunctions import *
sys.stdout.flush()
import petsc4py
#petsc4py.init(sys.argv)
from petsc4py import PETSc
import petsclinearsystem
from scipy.sparse import spdiags
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from datetime import datetime
import argparse
import numpy as np
import joblib
from joblib import Parallel, delayed
import pickle

from joblib import load, dump
Attemp = 7

# Attemp = 1: add more output of variable like multi1,2 to diagnose where the problem is.  change starting multi* as 1.
# Attemp = 2: delete multi2[<1e-8]=1e-8 to see the explosion. This is for presentation to Lars.
# Attemp = 3: Maybe epsilon=0.1 worked, I need to check.
# Attemp = 4: delete multi2[<1e-8] and see the effect, previous results didnt get stored.
# Attemp = 5: freeze id=0 and see what is the effect.
# Attemp = 6: add pickle to save ultimate convergent result
# Attemp = 7: solve pickle saving bugs with parallel computation, and only graph with i_g still.

tol = 1e-7
max_iter = 30000
path_name = "./tech4D/data/PostJump/"

# test_code = "_epsfrac_"+ str(epsilon)[0:len(str(epsilon))]+"_Paral"+"_Attemp_" +str(Attemp)

gamma_3_list = np.linspace(0., 1./3., 10)
# gamma_3_list = np.linspace(0., 1./3., 1)
eta_list     = np.array([0.1,0.05,0.01,0.001])
# eta_list     = np.array([0.001])
epsilon_list = np.array([0.1])



#########################################
# Method Loading
#########################################
 


reporterror = True
# Linear solver choices
# Chosse among petsc, petsc4py, eigen, both
# petsc: matrix assembled in C
# petsc4py: matrix assembled in Python
# eigen: matrix assembled in C++
# both: petsc+petsc4py
#
linearsolver = 'petsc'


start_time = time.time()

#########################################
# Global PArameters Loading
#########################################


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

#########################################
# Computation Setup
#########################################


now = datetime.now()
current_time = now.strftime("%d-%H:%M")


print("Grid dimension: [{}, {}, {}]\n".format(nK, nR, nY))

# Discretization of the state space for numerical PDE solution.
######## post jump, 3 states

(K_mat, R_mat, Y_mat) = np.meshgrid(K, R, Y, indexing = 'ij')
stateSpace = np.hstack([K_mat.reshape(-1,1,order = 'F'), R_mat.reshape(-1,1,order = 'F'), Y_mat.reshape(-1, 1, order='F')])

# For PETSc
K_mat_1d =K_mat.ravel(order='F')
R_mat_1d = R_mat.ravel(order='F')
Y_mat_1d = Y_mat.ravel(order='F')
lowerLims = np.array([K_min, R_min, Y_min], dtype=np.float64)
upperLims = np.array([K_max, R_max, Y_max], dtype=np.float64)




############# step up of optimization
# FC_Err = 1
# epoch = 0
# tol = 1e-7
# epsilon  = 0.005
# fraction = 0.005
# max_iter = 20000

# id_star = np.zeros_like(K_mat)
# ig_star = np.zeros_like(K_mat)

# #########################################
# # Result Storage Setup
# #########################################
(gamma_3_list,eta_list,epsilon_list) = np.meshgrid(gamma_3_list,eta_list,epsilon_list,indexing='ij')

gamma_3_list = gamma_3_list.ravel(order='F')
eta_list = eta_list.ravel(order='F')
epsilon_list = epsilon_list.ravel(order='F')

param_list = zip(gamma_3_list,eta_list,epsilon_list)
#########################################
epoch_list = list(range(1,max_iter+1,1))

def model(gamma_3, eta, epsilon ):
    file_name = "gamma_" + str(gamma_3)+"_"+"eta_" +str(eta)
    fraction = epsilon
    # file_header = ['maxid','minid','maxig','minig','iterationtime','PDEError','FCError','PetscTotal','PetscNormRes']

    file_header = ['maxid','minid','maxig','minig','maxconsumption','minconsumption','maxmulti1','minmulti1','PDEError','FCError']
    test_code = "_epsfrac_"+ str(epsilon)[0:len(str(epsilon))]+"_Paral"+"_Attemp_" +str(Attemp)

    min_id_list = epoch_list.copy()
    min_ig_list = epoch_list.copy()

    max_id_list = epoch_list.copy()
    max_ig_list = epoch_list.copy()

    min_consumption_list = epoch_list.copy()
    max_consumption_list = epoch_list.copy()

    min_multi1_list = epoch_list.copy()
    max_multi1_list = epoch_list.copy()

    min_multi2_list = epoch_list.copy()
    max_multi2_list = epoch_list.copy()

    PDE_Err_list = epoch_list.copy()
    FC_Err_list = epoch_list.copy()

    total_list = epoch_list.copy()
    normres_list = epoch_list.copy()
    iteration_time_list = epoch_list.copy()


    epoch = 0
    FC_Err = 1
    
    v0 = K_mat - (gamma_1 + gamma_2 * Y_mat)

    # id_star = np.zeros_like(K_mat)
    ig_star = np.zeros_like(K_mat)    

    while FC_Err > tol and epoch < max_iter:

        start_ep = time.time()
        # Applying finite difference scheme to the value function
        ######## first order
        dK = finiteDiff(v0,0,1,hK)
        # dK[dK < 1e-16] = 1e-16
        dR = finiteDiff(v0,1,1,hR)
        # dR[dR < 1e-8] = 1e-8
        dY = finiteDiff(v0,2,1,hY)
        ######## second order
        ddK = finiteDiff(v0,0,2,hK)
        ddR = finiteDiff(v0,1,2,hR)
        ddY = finiteDiff(v0,2,2,hY)


        # update control
        if epoch == 0:

            i_d = np.zeros(K_mat.shape)
            i_g = np.zeros(R_mat.shape)
            consumption_0 = A_d * (1 - R_mat) + A_g * R_mat
            consumption = consumption_0
            mc = delta / consumption
            # i_d = 1 -  mc / (dK - R_mat *  dR)
            # i_d /= phi_d
            # i_d[i_d < 0] = 0

            i_g = 1 - mc / (dK + (1 - R_mat) * dR)
            i_g /= phi_g


            q = delta * ((A_g * R_mat - i_g * R_mat) + (A_d * (1 - R_mat) - i_d * (1 - R_mat))) ** (-1)
            
            min_multi1_list[epoch]=1
            max_multi1_list[epoch]=1

            min_multi2_list[epoch]=1
            max_multi2_list[epoch]=1
        else:

            multi_1 = dK + (1 - R_mat) * dR
            # multi_2 = dK - R_mat * dR
            # multi_2[multi_2 <= 0.01] = 0.01

            # aa = (1 - multi_1 / multi_2) / phi_d
            # bb = phi_g / phi_d * multi_1 / multi_2

            AA = phi_g * (R_mat)
            BB = phi_g * ((1 - R_mat) * A_d + R_mat * A_g )  + R_mat
            CC = (1 - R_mat) * A_d + R_mat * A_g  - delta / multi_1

            DELTA = BB**2 - 4 * AA * CC
            DELTA[DELTA <= 0] = 0

            i_g_new = (BB - np.sqrt(DELTA)) / (2 * AA)
            # i_d_new = aa + bb * i_g_new

            min_multi1_list[epoch]=multi_1.min()
            max_multi1_list[epoch]=multi_1.max()

            # min_multi2_list[epoch]=multi_2.min()
            # max_multi2_list[epoch]=multi_2.max()
        


            # i_d = i_d_new * fraction + id_star * (1 - fraction)
            i_g = i_g_new * fraction + ig_star * (1 - fraction)
            
        # i_d[i_d <= 1e-14] = 1e-14
        i_g[i_g <= 1e-14] = 1e-14
        
        i_d[i_d >= 1 / phi_d - 1e-14] = 1 / phi_d - 1e-14
        i_g[i_g >= 1 / phi_g - 1e-14] = 1 / phi_g - 1e-14
        # Result Storage: id, ig
        
        min_id_list[epoch] = np.min(i_d)
        max_id_list[epoch] = np.max(i_d)
        min_ig_list[epoch] = np.min(i_g)
        max_ig_list[epoch] = np.max(i_g)
        
        consumption = (A_d -i_d) * (1 - R_mat) + (A_g - i_g) * R_mat
        consumption[consumption < 1e-18] = 1e-18
        
        # Result Storage: consumption

        min_consumption_list[epoch] = np.min(consumption)
        max_consumption_list[epoch] = np.max(consumption)


        start_time2 = time.time()

        if epoch == 0:
            dVec = np.array([hK, hR, hY])
            increVec = np.array([1, nK, nK * nR],dtype=np.int32)
            # These are constant
            A = - delta * np.ones(K_mat.shape)
            C_dd = 0.5 * ( sigma_d * (1 - R_mat) + sigma_g * R_mat )**2
            C_gg = 0.5 * (1- R_mat)**2 * R_mat**2 * (sigma_d + sigma_g)**2
            C_yy = 0.5 * (eta * varsigma * A_d * np.exp(K_mat) * (1 - R_mat))** 2

            if linearsolver == 'petsc4py' or linearsolver == 'petsc' or linearsolver == 'both':
                petsc_mat = PETSc.Mat().create()
                petsc_mat.setType('aij')
                petsc_mat.setSizes([nK*nR*nY, nK*nR*nY])
                petsc_mat.setPreallocationNNZ(13)
                petsc_mat.setUp()
                ksp = PETSc.KSP()
                ksp.create(PETSc.COMM_WORLD)
                ksp.setType('bcgs')
                ksp.getPC().setType('ilu')
                ksp.setFromOptions()

                A_1d = A.ravel(order = 'F')
                C_dd_1d = C_dd.ravel(order = 'F')
                C_gg_1d = C_gg.ravel(order = 'F')
                C_yy_1d = C_yy.ravel(order = 'F')

                if linearsolver == 'petsc4py':
                    I_LB_d = (stateSpace[:,0] == K_min)
                    I_UB_d = (stateSpace[:,0] == K_max)
                    I_LB_g = (stateSpace[:,1] == R_min)
                    I_UB_g = (stateSpace[:,1] == R_max)
                    I_LB_y = (stateSpace[:,2] == Y_min)
                    I_UB_y = (stateSpace[:,2] == Y_max)
                    diag_0_base = A_1d[:]
                    diag_0_base += (I_LB_d * C_dd_1d[:] + I_UB_d * C_dd_1d[:] - 2 * (1 - I_LB_d - I_UB_d) * C_dd_1d[:]) / dVec[0] ** 2
                    diag_0_base += (I_LB_g * C_gg_1d[:] + I_UB_g * C_gg_1d[:] - 2 * (1 - I_LB_g - I_UB_g) * C_gg_1d[:]) / dVec[1] ** 2
                    diag_0_base += (I_LB_K * C_kk_1d[:] + I_UB_K * C_kk_1d[:] - 2 * (1 - I_LB_K - I_UB_K) * C_kk_1d[:]) / dVec[2] ** 2
                    diag_d_base = - 2 * I_LB_d * C_dd_1d[:] / dVec[0] ** 2 + (1 - I_LB_d - I_UB_d) * C_dd_1d[:] / dVec[0] ** 2
                    diag_dm_base = - 2 * I_UB_d * C_dd_1d[:] / dVec[0] ** 2 + (1 - I_LB_d - I_UB_d) * C_dd_1d[:] / dVec[0] ** 2
                    diag_g_base = - 2 * I_LB_g * C_gg_1d[:] / dVec[1] ** 2 + (1 - I_LB_g - I_UB_g) * C_gg_1d[:] / dVec[1] ** 2
                    diag_gm_base = - 2 * I_UB_g * C_gg_1d[:] / dVec[1] ** 2 + (1 - I_LB_g - I_UB_g) * C_gg_1d[:] / dVec[1] ** 2
                    diag_y_base = - 2 * I_LB_y * C_yy_1d[:] / dVec[2] ** 2 + (1 - I_LB_y - I_UB_y) * C_yy_1d[:] / dVec[2] ** 2
                    diag_ym_base = - 2 * I_UB_y * C_yy_1d[:] / dVec[2] ** 2 + (1 - I_LB_y - I_UB_y) * C_yy_1d[:] / dVec[2] ** 2
                    diag_dd = I_LB_d * C_dd_1d[:] / dVec[0] ** 2
                    diag_ddm = I_UB_d * C_dd_1d[:] / dVec[0] ** 2
                    diag_gg = I_LB_g * C_gg_1d[:] / dVec[1] ** 2
                    diag_ggm = I_UB_g * C_gg_1d[:] / dVec[1] ** 2
                    diag_yy = I_LB_y * C_yy_1d[:] / dVec[2] ** 2
                    diag_yym = I_UB_y * C_yy_1d[:] / dVec[2] ** 2




        B_d = (alpha_d + i_d - 0.5* phi_d * i_d**2) * (1 - R_mat) +  (alpha_g + i_g - 0.5 * phi_g * i_g**2) * R_mat - C_dd
        B_g = ((alpha_g + i_g - 0.5 * phi_g * i_g**2) -  (alpha_d + i_d - 0.5* phi_d * i_d**2) - R_mat * sigma_g**2 + (1 - R_mat)* sigma_d**2) * R_mat * (1 - R_mat)
        B_y = beta_f * eta * A_d * np.exp(K_mat) * (1 - R_mat)

        D = delta * np.log(consumption) + delta * K_mat  - (gamma_1 + gamma_2 * Y_mat + gamma_3 * (Y_mat -2) * (Y_mat > 2))* beta_f * eta * A_d * np.exp(K_mat) * (1 - R_mat)  - 0.5 * (gamma_2 + gamma_3 * (Y_mat > 2)) * (varsigma * eta * A_d * np.exp(K_mat) * (1 - R_mat) )**2

        if linearsolver == 'eigen' or linearsolver == 'both':
            start_eigen = time.time()
            out_eigen = PDESolver(stateSpace, A, B_d, B_g, B_y, C_dd, C_gg, C_yy, D, v0, epsilon, solverType = 'False Transient')
            out_comp = out_eigen[2].reshape(v0.shape,order = "F")
            iteration_time_list[epoch] = time.time() - start_eigen

            if epoch % 1 == 0 and reporterror:
                v = np.array(out_eigen[2])
                res = np.linalg.norm(out_eigen[3].dot(v) - out_eigen[4])
                total_list[epoch]=0
                normres_list[epoch] = res
                PDE_rhs = A * v0 + B_d * dK + B_g * dR + B_y * dY + C_dd * ddK + C_gg * ddR + C_yy * ddY + D
                PDE_Err = np.max(abs(PDE_rhs))
                FC_Err = np.max(abs((out_comp - v0)))

                PDE_Err_list[epoch] = PDE_Err
                FC_Err_list[epoch] = FC_Err

        if linearsolver == 'petsc4py':
            bpoint1 = time.time()
            # ==== original impl ====
            B_d_1d = B_d.ravel(order = 'F')
            B_g_1d = B_g.ravel(order = 'F')
            B_y_1d = B_y.ravel(order = 'F')
            D_1d = D.ravel(order = 'F')
            v0_1d = v0.ravel(order = 'F')
            # profiling
            # bpoint2 = time.time()
            # print("reshape: {:.3f}s".format(bpoint2 - bpoint1))
            diag_0 = diag_0_base - 1 / epsilon + I_LB_R * B_r_1d[:] / -dVec[0] + I_UB_R * B_r_1d[:] / dVec[0] - (1 - I_LB_R - I_UB_R) * np.abs(B_r_1d[:]) / dVec[0] + I_LB_F * B_f_1d[:] / -dVec[1] + I_UB_F * B_f_1d[:] / dVec[1] - (1 - I_LB_F - I_UB_F) * np.abs(B_f_1d[:]) / dVec[1] + I_LB_K * B_k_1d[:] / -dVec[2] + I_UB_K * B_k_1d[:] / dVec[2] - (1 - I_LB_K - I_UB_K) * np.abs(B_k_1d[:]) / dVec[2]
            diag_R = I_LB_R * B_r_1d[:] / dVec[0] + (1 - I_LB_R - I_UB_R) * B_r_1d.clip(min=0.0) / dVec[0] + diag_R_base
            diag_Rm = I_UB_R * B_r_1d[:] / -dVec[0] - (1 - I_LB_R - I_UB_R) * B_r_1d.clip(max=0.0) / dVec[0] + diag_Rm_base
            diag_F = I_LB_F * B_f_1d[:] / dVec[1] + (1 - I_LB_F - I_UB_F) * B_f_1d.clip(min=0.0) / dVec[1] + diag_F_base
            diag_Fm = I_UB_F * B_f_1d[:] / -dVec[1] - (1 - I_LB_F - I_UB_F) * B_f_1d.clip(max=0.0) / dVec[1] + diag_Fm_base
            diag_K = I_LB_K * B_k_1d[:] / dVec[2] + (1 - I_LB_K - I_UB_K) * B_k_1d.clip(min=0.0) / dVec[2] + diag_K_base
            diag_Km = I_UB_K * B_k_1d[:] / -dVec[2] - (1 - I_LB_K - I_UB_K) * B_k_1d.clip(max=0.0) / dVec[2] + diag_Km_base
            # profiling
            # bpoint3 = time.time()
            # print("prepare: {:.3f}s".format(bpoint3 - bpoint2))

            data = [diag_0, diag_R, diag_Rm, diag_RR, diag_RRm, diag_F, diag_Fm, diag_FF, diag_FFm, diag_K, diag_Km, diag_KK, diag_KKm]
            diags = np.array([0,-increVec[0],increVec[0],-2*increVec[0],2*increVec[0],
                            -increVec[1],increVec[1],-2*increVec[1],2*increVec[1],
                            -increVec[2],increVec[2],-2*increVec[2],2*increVec[2]])
            # The transpose of matrix A_sp is the desired. Create the csc matrix so that it can be used directly as the transpose of the corresponding csr matrix.
            A_sp = spdiags(data, diags, len(diag_0), len(diag_0), format='csc')
            b = -v0_1d/epsilon - D_1d
            # A_sp = spdiags(data, diags, len(diag_0), len(diag_0))
            # A_sp = csr_matrix(A_sp.T)
            # b = -v0/ε - D
            # profiling
            # bpoint4 = time.time()
            # print("create matrix and rhs: {:.3f}s".format(bpoint4 - bpoint3))
            petsc_mat = PETSc.Mat().createAIJ(size=A_sp.shape, csr=(A_sp.indptr, A_sp.indices, A_sp.data))
            petsc_rhs = PETSc.Vec().createWithArray(b)
            x = petsc_mat.createVecRight()
            # profiling
            # bpoint5 = time.time()
            # print("assemble: {:.3f}s".format(bpoint5 - bpoint4))

            # dump to files
            #x.set(0)
            #viewer = PETSc.Viewer().createBinary('TCRE_MacDougallEtAl2017_A.dat', 'w')
            #petsc_mat.view(viewer)
            #viewer = PETSc.Viewer().createBinary('TCRE_MacDougallEtAl2017_b.dat', 'w')
            #petsc_rhs.view(viewer)

            # create linear solver
            start_ksp = time.time()
            ksp.setOperators(petsc_mat)
            ksp.setTolerances(rtol=1e-14)
            ksp.solve(petsc_rhs, x)
            petsc_mat.destroy()
            petsc_rhs.destroy()
            x.destroy()
            out_comp = np.array(ksp.getSolution()).reshape(R_mat.shape,order = "F")
            end_ksp = time.time()

            print("petsc4py total: {:.3f}s".format(end_ksp - bpoint1))
            print("PETSc preconditioned residual norm is {:g}; iterations: {}".format(ksp.getResidualNorm(), ksp.getIterationNumber()))
            total_list[epoch] = end_ksp - bpoint1
            normres_list[epoch] = ksp.getResidualNorm()
            if epoch % 1 == 0 and reporterror:
                # Calculating PDE error and False Transient error
                PDE_rhs = A * v0 + B_d * dK + B_g * dR + B_y * dY + C_dd * ddK + C_gg * ddR + C_yy * ddY + D
                PDE_Err = np.max(abs(PDE_rhs))
                FC_Err = np.max(abs((out_comp - v0) / epsilon))

                PDE_Err_list[epoch] = PDE_Err
                FC_Err_list[epoch] = FC_Err

        if linearsolver == 'petsc' or linearsolver == 'both':
            bpoint1 = time.time()
            B_d_1d = B_d.ravel(order = 'F')
            B_g_1d = B_g.ravel(order = 'F')
            B_y_1d = B_y.ravel(order = 'F')
            D_1d = D.ravel(order = 'F')
            v0_1d = v0.ravel(order = 'F')
            petsclinearsystem.formLinearSystem(K_mat_1d, R_mat_1d, Y_mat_1d, A_1d, B_d_1d, B_g_1d, B_y_1d, C_dd_1d, C_gg_1d, C_yy_1d, epsilon, lowerLims, upperLims, dVec, increVec, petsc_mat)
            # profiling
            # bpoint2 = time.time()
            # print("form petsc mat: {:.3f}s".format(bpoint2 - bpoint1))
            b = v0_1d + D_1d*epsilon
            # petsc4py setting
            # petsc_mat.scale(-1./ε)
            # b = -v0_1d/ε - D_1d
            petsc_rhs = PETSc.Vec().createWithArray(b)
            x = petsc_mat.createVecRight()
            # profiling
            # bpoint3 = time.time()
            # print("form rhs and workvector: {:.3f}s".format(bpoint3 - bpoint2))


            # create linear solver
            start_ksp = time.time()
            ksp.setOperators(petsc_mat)
            ksp.setTolerances(rtol=1e-16)
            ksp.solve(petsc_rhs, x)
            # petsc_mat.destroy()
            petsc_rhs.destroy()
            x.destroy()
            out_comp = np.array(ksp.getSolution()).reshape(K_mat.shape,order = "F")
            end_ksp = time.time()
            # profiling
            # print("ksp solve: {:.3f}s".format(end_ksp - start_ksp))
            num_iter = ksp.getIterationNumber()
            # file_iter.write("%s \n" % num_iter)
            total_list[epoch] = end_ksp - bpoint1
            normres_list[epoch] = ksp.getResidualNorm()
            if epoch % 1 == 0 and reporterror:
                # Calculating PDE error and False Transient error
                PDE_rhs = A * v0 + B_d * dK + B_g * dR + B_y * dY + C_dd * ddK + C_gg * ddR + C_yy * ddY + D
                PDE_Err = np.max(abs(PDE_rhs))
                FC_Err = np.max(abs((out_comp - v0)/ epsilon))
                PDE_Err_list[epoch] = PDE_Err
                FC_Err_list[epoch] = FC_Err
        iteration_time_list[epoch] = time.time() - start_ep

        # step 9: keep iterating until convergence
        # rowcontent = {
            # "epoch": epoch,
            # "iterations": num_iter,
            # "residual norm": ksp.getResidualNorm(),
            # "PDE_Err": PDE_Err,
            # "FC_Err": FC_Err
        # }
        # writer.writerow(rowcontent)
        id_star = i_d
        ig_star = i_g
        v0 = out_comp
        epoch += 1

    if reporterror:
        print("===============================================")
        print("Fianal epoch {:d}: PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch -1, PDE_Err, FC_Err))
    print("--- Total running time: %s seconds ---" % (time.time() - start_time))

    Data_List = list(zip(max_id_list,min_id_list,max_ig_list,min_ig_list,max_consumption_list,min_consumption_list,max_multi1_list,min_multi1_list,PDE_Err_list,FC_Err_list))
    Data_List = Data_List[0:epoch]

    with open(path_name+file_name+test_code +'.csv','w+') as f:
        writer  = csv.writer(f)
        writer.writerow(file_header)
        writer.writerows(Data_List)
    
    # filename = filename
    my_shelf = {}
    for key in dir():
        if isinstance(locals()[key], (int, float, str, bool, np.ndarray,list)):
            try:
                my_shelf[key] = locals()[key]
            except TypeError:
                #
                # __builtins__, my_shelf, and imported modules can not be shelved.
                #
                print('ERROR shelving: {0}'.format(key))
        else:
            pass
    for key in globals():
        if isinstance(globals()[key], (int, float, bool, np.ndarray)):
            try:
                my_shelf[key] = globals()[key]
            except TypeError:
                #
                # __builtins__, my_shelf, and imported modules can not be shelved.
                #
                print('ERROR shelving: {0}'.format(key))
        else:
            pass


    file = open(path_name+file_name+test_code, 'wb')
    pickle.dump(my_shelf, file)
    file.close()






number_of_cpu = joblib.cpu_count()
delayed_funcs = [delayed(model)(gamma_3, eta, epsilon) for gamma_3, eta, epsilon in param_list]
parallel_pool = Parallel(n_jobs=number_of_cpu,require='sharedmem')
res = parallel_pool(delayed_funcs)


# tol = 1e-7
# max_iter = 30000
# path_name = "./tech4D/data/PostJump/"

# test_code = "_epsfrac_"+ str(epsilon)[0:len(str(epsilon))]+"_Paral"+"_Attemp_" +str(Attemp)

gamma_3_list = np.linspace(0., 1./3., 10)
# gamma_3_list = np.linspace(0., 1./3., 1)
eta_list     = np.array([0.1,0.05,0.01,0.001])
# eta_list     = np.array([0.001])
epsilon_list = np.array([0.1])

from cProfile import label
from math import ceil
import matplotlib as mpl

from mpl_toolkits.mplot3d.axes3d import Axes3D
from numpy import array, loadtxt
import numpy as np

import sympy
import csv
import pandas as pd
import joblib
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages



#########################################


(gamma_3_list,eta_list) = np.meshgrid(gamma_3_list,eta_list,indexing='ij')

gamma_3_list = gamma_3_list.ravel(order='F')
eta_list = eta_list.ravel(order='F')

param_list = zip(gamma_3_list,eta_list)
#########################################





def print2pdf(epsilon):

    test_code = "_epsfrac_"+ str(epsilon)[0:len(str(epsilon))]+"_Paral"+"_Attemp_" +str(Attemp)

    pdf_pages = PdfPages(path_name+'gamma_eta_plot'+test_code+'.pdf')

    for gamma,eta in param_list:

        # gamma = param_list[num][0]
        # eta=param_list[num][1]

        file_name = "gamma_" + str(gamma)+"_"+"eta_"+str(eta)
        file = open(path_name+file_name+test_code+'.csv','r')
        reader = csv.reader(file,delimiter=',')
        file_header= next(reader)
        file_varnum = len(file_header)
        data = np.array(list(reader)).astype(float)
        file_length = len(data[:,1])

        file_pickle = open(path_name+file_name+test_code,'rb')
        res = pickle.load(file_pickle)
        i_g = res['i_g']
        indices_max =np.where(i_g==i_g.max())
        indices_min = np.where(i_g==i_g.min())



        figwidth = 10
        fig, axs = plt.subplots(int(np.ceil(file_varnum/2)), 2, sharex=True, figsize=(2  * figwidth, 2 *figwidth))  

        plt.text(0.05,0.95,"ig is maximal at grid point:"+" K=" +str(indices_max[0]) +","" R=" +str(indices_max[1])+","+" Y=" +str(indices_max[2]), transform=fig.transFigure, size=12)
        plt.text(0.05,0.93,"ig is minimal at grid point:"+" K=" +str(indices_min[0]) +","" R=" +str(indices_min[1])+","+" Y=" +str(indices_min[2]), transform=fig.transFigure, size=12)

        if file_length < max_iter:
            plt.suptitle("$\gamma$="+str(gamma)[0:5]+" , $\eta$=" + str(eta)+"_Success",fontsize=16)
        else :
            plt.suptitle("$\gamma$="+str(gamma)[0:5]+" , $\eta$=" + str(eta)+"_Failure",fontsize=16)
        for varnum in np.array(range(file_varnum)):
            axs[varnum//2,varnum%2].plot(np.array(range(file_length)),data[:,varnum],label='')
            axs[varnum//2,varnum%2].set_xlabel('epoch')
            axs[varnum//2,varnum%2].set_ylabel('%s' %file_header[varnum])
            axs[varnum//2,varnum%2].set_title('%s' %file_header[varnum])
            axs[varnum//2,varnum%2].grid(linestyle=':')
            axs[varnum//2,varnum%2].legend()
            # plt.savefig(f"tech4D/data/PostJump/gamma_{str(gamma)[:5]}_eta_{eta}.pdf")
        pdf_pages.savefig(fig)
        plt.close()



    pdf_pages.close()          

number_of_cpu = joblib.cpu_count()
delayed_funcs = [delayed(print2pdf)(epsilon) for epsilon in epsilon_list]
parallel_pool = Parallel(n_jobs=number_of_cpu)
res = parallel_pool(delayed_funcs)


