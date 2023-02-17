import numpy as np
import pandas as pd
import sys
print(sys.path)

sys.path.append('./src')

import pickle
import plotly.graph_objects as go
import plotly.offline as pyo
import matplotlib.pyplot as plt
import SolveLinSys
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import CubicSpline
from matplotlib.backends.backend_pdf import PdfPages
from src.supportfunctions import finiteDiff_3D
import os
import argparse
import time
import petsc4py
from petsc4py import PETSc
import petsclinearsystem
sys.stdout.flush()


parser = argparse.ArgumentParser(description="xi_r values")
parser.add_argument("--dataname",type=str)
parser.add_argument("--pdfname",type=str)

parser.add_argument("--xiaarr",nargs='+', type=float)
parser.add_argument("--xigarr",nargs='+', type=float)

parser.add_argument("--psi0arr",nargs='+',type=float)
parser.add_argument("--psi1arr",nargs='+',type=float)
parser.add_argument("--psi2arr",nargs='+',type=float)
parser.add_argument("--num_gamma",type=int)

parser.add_argument("--hXarr",nargs='+',type=float)
parser.add_argument("--Xminarr",nargs='+',type=float)
parser.add_argument("--Xmaxarr",nargs='+',type=float)

parser.add_argument("--auto",type=int)
parser.add_argument("--IntPeriod",type=int)

# parser.add_argument("--Update",type=int)

# parser.add_argument("--year",type=int,default=60)
# parser.add_argument("--time",type=float,default=1/12.)
args = parser.parse_args()


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


# Tech
theta = 3
lambda_bar = 0.1206
vartheta_bar = 0.0453

lambda_bar_first = lambda_bar / 2.
vartheta_bar_first = vartheta_bar / 2.

lambda_bar_second = 1e-3
vartheta_bar_second = 0.








plt.style.use('classic')
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["figure.figsize"] = (16,10)
plt.rcParams["figure.dpi"] = 500
plt.rcParams["font.size"] = 15
plt.rcParams["legend.frameon"] = False
plt.rcParams["lines.linewidth"] = 5

print("After, figure default size is: ", plt.rcParams["savefig.bbox"])
print("After, figure default size is: ", plt.rcParams["figure.figsize"])
print("After, figure default dpi is: ", plt.rcParams["figure.dpi"])
print("After, figure default size is: ", plt.rcParams["font.size"])
print("After, legend.frameon is: ", plt.rcParams["legend.frameon"])
print("After, lines.linewidth is: ", plt.rcParams["lines.linewidth"])

def Damage_Intensity(Yt, y_bar_lower=1.5):
    r_1 = 1.5
    r_2 = 2.5
    Intensity = r_1 * (np.exp(r_2 / 2 * (Yt - y_bar_lower)**2) -1) * (Yt > y_bar_lower)
    return Intensity



def minimize_void(grid=(),  args = (), n_bar=(), variable_args= (), controls = (), tol=1e-6, epsilon=0.3, max_iter=10000):
    """
    compute jump model with ambiguity over climate models
    """

    K, Y, L = grid

    

    delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a, xi_g, xi_p = args
    # n_bar, = n_bar
    # n_bar = len(Y)-1
    # n_bar = np.abs(Y - y_bar).argmin()

    Y_cap = Y[:n_bar+1]
    
    K_min, K_max, Y_min, Y_max, L_min, L_max = min(K), max(K), min(Y_cap), max(Y_cap), min(L), max(L)
    hK, hY, hL = K[1] - K[0], Y_cap[1] - Y_cap[0], L[1]-L[0]
    nK, nY, nL = len(K), len(Y_cap), len(L)
    

    print("K_min={},K_max={},Y_min={},Y_max={},L_min={},L_max={}" .format(K_min, K_max, Y_min, Y_max, L_min, L_max))
    print("hK={},hY={},hL={}" .format(hK, hY, hL))
    print("nK={},nY={},nL={}" .format(nK, nY, nL))


    (K_mat, Y_mat, L_mat) = np.meshgrid(K, Y_cap, L, indexing = 'ij')


    Intensity=Damage_Intensity(Y_mat)
    

    theta_ell_orig, psi_2_orig = variable_args
    
    theta_ell = theta_ell_orig[:, :, :n_bar+1, :]
    psi_2 = psi_2_orig[:, :, :n_bar+1, :]
    


    # args = (delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a, xi_g, xi_p)
    # variable_args = (theta_ell_wakeup, psi_2_wakeup)
    # controls=(i,e,x, g_tech, g_damage,  pi_c_o_wakeup, pi_c_wakeup, pi_d_o_wakeup, v0, v_i, v_tech2)

    # ME_base = minimize_void(grid, args, variable_args, controls, tol=1e-6, epsilon=0.3, max_iter=10000)



    ii_orig, ee_orig, xx_orig, g_tech_orig, g_damage_orig, pi_c_o_orig, pi_c_orig, pi_d_o_orig, v0_orig, v_i_orig, V_post_tech_orig = controls


    pi_c_o = pi_c_o_orig[:, :, :n_bar+1, :]
    pi_c = pi_c_orig[:, :, :n_bar+1, :]
    pi_d_o = pi_d_o_orig[:, :, :n_bar+1, :]
    
    # n_dmg = len(gamma_3_list)
    # n_climate = len(pi_c_orig)
    
    ii = ii_orig[:, :n_bar+1, :]
    ee = ee_orig[:, :n_bar+1, :]
    xx = xx_orig[:, :n_bar+1, :]
    
    g_tech = g_tech_orig[:, :n_bar+1, :]
    V_post_tech = V_post_tech_orig[:, :n_bar+1, :]

    v0 = v0_orig[:, :n_bar+1, :]
    
    v_i = v_i_orig[:, :, :n_bar+1, :]
    g_damage = g_damage_orig[:, :, :n_bar+1, :]
    pi_c = pi_c_orig[:, :, :n_bar+1, :]

    FC_Err   = 1
    epoch    = 0
        # For PETSc
    K_mat_1d = K_mat.ravel(order='F')
    Y_mat_1d = Y_mat.ravel(order='F')
    L_mat_1d = L_mat.ravel(order='F')
    
    lowerLims = np.array([K_min, Y_min, L_min], dtype=np.float64)
    upperLims = np.array([K_max, Y_max, L_max], dtype=np.float64)

    dVec = np.array([hK, hY, hL])
    increVec = np.array([1, nK, nK * nY],dtype=np.int32)


    petsc_mat = PETSc.Mat().create()
    petsc_mat.setType('aij')
    petsc_mat.setSizes([nK * nY * nL, nK * nY * nL])
    petsc_mat.setPreallocationNNZ(13)
    petsc_mat.setUp()
    ksp = PETSc.KSP()
    ksp.create(PETSc.COMM_WORLD)
    ksp.setType('bcgs')
    ksp.getPC().setType('ilu')
    ksp.setFromOptions()

    while FC_Err > tol and epoch < max_iter:
        
        start_ep = time.time()

        
        dVdK  = finiteDiff_3D(v0,0,1,hK)
        dVdK[dVdK <= 1e-16] = 1e-16
        dK = dVdK
        dVdY  = finiteDiff_3D(v0,1,1,hY)
        dY = dVdY
        dVdL  = finiteDiff_3D(v0,2,1,hL)
        dVdL[dVdL <= 1e-16] = 1e-16
        dL = dVdL
        ######## second order
        ddVdK = finiteDiff_3D(v0,0,2,hK)
        ddVdY = finiteDiff_3D(v0,1,2,hY)
        ddY = ddVdY
        ddVdL = finiteDiff_3D(v0,2,2,hL)
        
        dG  = gamma_1 + gamma_2 * Y_mat
        ddG = gamma_2 
        G = dY -  dG

        ################## Control Updating######################
        # update pi_c
        log_pi_c_ratio = - G * ee * theta_ell/ xi_a
        log_pi_c_ratio += -dL * psi_0 * xx**psi_1 * np.exp( psi_1 * K_mat - (1-psi_2) * L_mat) / xi_a
        pi_c_ratio = log_pi_c_ratio - np.max(log_pi_c_ratio)
        pi_c = np.exp(pi_c_ratio) * pi_c_o
        pi_c = (pi_c <= 0) * 1e-16 + (pi_c > 0) * pi_c
        pi_c = pi_c / np.sum(pi_c, axis=0)
        entropy = np.sum(pi_c * (np.log(pi_c) - np.log(pi_c_o)), axis=0)

        # update g_dmg

        g_damage = np.exp(- (v_i-v0)/xi_p)

        # update g_tech

        g_tech = np.exp(1 / xi_g * (v0 - V_post_tech))
        g_tech[g_tech <=1e-16] = 1e-16
        ################## Control Updating######################


        jj = alpha * vartheta_bar * (1 - ee / (alpha * lambda_bar * np.exp(K_mat)))**theta
        jj[jj <= 1e-16] = 1e-16
        consumption = alpha - ii - jj - xx
        
        
        A   = - delta * np.ones(K_mat.shape) - np.exp(  L_mat - np.log(448) ) * g_tech
        B_1 = mu_k + ii - 0.5 * kappa * ii**2 - 0.5 * sigma_k**2
        B_2 = np.sum(theta_ell* pi_c, axis=0) * ee
        # B_3 = - zeta + psi_0 * (xx * np.exp(K_mat - L_mat))**psi_1 - 0.5 * sigma_g**2
        B_3 = - zeta + psi_0 * xx** psi_1 * np.exp( psi_1 * K_mat ) * np.sum(pi_c * np.exp( -( 1-psi_2) * L_mat  ), axis=0 )- 0.5 * sigma_g**2

        C_1 = 0.5 * sigma_k**2 * np.ones(K_mat.shape)
        C_2 = 0.5 * sigma_y**2 * ee**2
        C_3 = 0.5 * sigma_g**2 * np.ones(K_mat.shape)
        D = delta * np.log(consumption) + delta * K_mat  - dG * np.sum(theta_ell* pi_c, axis=0) * ee  - 0.5 * ddG * sigma_y**2 * ee**2  + xi_a * entropy + xi_g * np.exp((L_mat - np.log(448))) * (1 - g_tech + g_tech * np.log(g_tech)) + np.exp( (L_mat - np.log(448)) ) * g_tech * V_post_tech

        D += xi_p * Intensity * np.sum( pi_d_o*(1-g_damage+g_damage*np.log(g_damage)),axis=0) +Intensity*np.sum(pi_d_o*g_damage*v_i,axis=0)
        A -=  Intensity*np.sum(pi_d_o*g_damage,axis=0)

        bpoint1 = time.time()

        A_1d   = A.ravel(order = 'F')
        C_1_1d = C_1.ravel(order = 'F')
        C_2_1d = C_2.ravel(order = 'F')
        C_3_1d = C_3.ravel(order = 'F')
        B_1_1d = B_1.ravel(order = 'F')
        B_2_1d = B_2.ravel(order = 'F')
        B_3_1d = B_3.ravel(order = 'F')
        D_1d   = D.ravel(order = 'F')
        v0_1d  = v0.ravel(order = 'F')
        


        petsclinearsystem.formLinearSystem(K_mat_1d, Y_mat_1d, L_mat_1d, A_1d, B_1_1d, B_2_1d, B_3_1d, C_1_1d, C_2_1d, C_3_1d, epsilon, lowerLims, upperLims, dVec, increVec, petsc_mat)
        b = v0_1d + D_1d * epsilon
        petsc_rhs = PETSc.Vec().createWithArray(b)
        x = petsc_mat.createVecRight()


        # create linear solver
        start_ksp = time.time()
        ksp.setOperators(petsc_mat)
        ksp.setTolerances(rtol=tol)
        ksp.solve(petsc_rhs, x)
        petsc_rhs.destroy()
        x.destroy()
        out_comp = np.array(ksp.getSolution()).reshape(A.shape,order = "F")
        end_ksp = time.time()
        num_iter = ksp.getIterationNumber()
        
        # print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
        # print("PETSc preconditioned residual norm is {:g}; iterations: {}".format(ksp.getResidualNorm(), ksp.getIterationNumber()))
        PDE_rhs = A * v0 + B_1 * dVdK + B_2 * dVdY + B_3 * dVdL + C_1 * ddVdK + C_2 * ddVdY + C_3 * ddVdL + D
        PDE_Err = np.max(abs(PDE_rhs))
        FC_Err = np.max(abs((out_comp - v0)/ epsilon))
        
        # if FC_Err < 2*tol:
            
        #     print("-----------------------------------")
        #     print("---------Epoch {}---------------".format(epoch))
        #     # print("-----------------------------------")
        #     # print("min i: {},\t max i: {}\t".format(ii.min(), ii.max()))
        #     # print("min e: {},\t max e: {}\t".format(ee.min(), ee.max()))
        #     # print("min x: {},\t max x: {}\t".format(xx.min(), xx.max()))
        #     print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
        #     print("Epoch {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))
        #     print("Epoch time: {:.4f}".format(time.time() - start_ep))
        # elif epoch%100==0:
            
        #     print("-----------------------------------")
        #     print("---------Epoch {}---------------".format(epoch))
        #     print("-----------------------------------")
        #     print("min i: {},\t max i: {}\t".format(ii.min(), ii.max()))
        #     print("min e: {},\t max e: {}\t".format(ee.min(), ee.max()))
        #     print("min x: {},\t max x: {}\t".format(xx.min(), xx.max()))
        #     print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
        #     print("Epoch {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))
        #     print("Epoch time: {:.4f}".format(time.time() - start_ep))
          
            
        v0     = out_comp
        epoch += 1
        
    print("-----------------------------------")
    print("---------Converged, Epoch {}---------------".format(epoch))
    print("-----------------------------------")
    print("min i: {},\t max i: {}\t".format(ii.min(), ii.max()))
    print("min e: {},\t max e: {}\t".format(ee.min(), ee.max()))
    print("min x: {},\t max x: {}\t".format(xx.min(), xx.max()))
    print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
    print("Epoch {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))
    print("Epoch time: {:.4f}".format(time.time() - start_ep))
        
        
    ME = - dVdY * np.sum(pi_c * theta_ell, axis=0) - ddVdY * sigma_y**2 * ee + dG * np.sum(theta_ell * pi_c, axis=0) +  ddG * sigma_y**2 * ee

    
    return ME



def minimize_pi_c(grid=(),  args = (), n_bar=(), variable_args= (), controls = (), tol=1e-6, epsilon=0.3, max_iter=10000):
    """
    compute jump model with ambiguity over climate models
    """

    K, Y, L = grid

    

    delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a, xi_g, xi_p = args
    # n_bar, = n_bar
    # n_bar = len(Y)-1
    # n_bar = np.abs(Y - y_bar).argmin()

    Y_cap = Y[:n_bar+1]
    
    K_min, K_max, Y_min, Y_max, L_min, L_max = min(K), max(K), min(Y_cap), max(Y_cap), min(L), max(L)
    hK, hY, hL = K[1] - K[0], Y_cap[1] - Y_cap[0], L[1]-L[0]
    nK, nY, nL = len(K), len(Y_cap), len(L)
    

    print("K_min={},K_max={},Y_min={},Y_max={},L_min={},L_max={}" .format(K_min, K_max, Y_min, Y_max, L_min, L_max))
    print("hK={},hY={},hL={}" .format(hK, hY, hL))
    print("nK={},nY={},nL={}" .format(nK, nY, nL))


    (K_mat, Y_mat, L_mat) = np.meshgrid(K, Y_cap, L, indexing = 'ij')


    Intensity=Damage_Intensity(Y_mat)
    

    theta_ell_orig, psi_2_orig = variable_args
    
    theta_ell = theta_ell_orig[:, :, :n_bar+1, :]
    psi_2 = psi_2_orig[:, :, :n_bar+1, :]
    


    # args = (delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a, xi_g, xi_p)
    # variable_args = (theta_ell_wakeup, psi_2_wakeup)
    # controls=(i,e,x, g_tech, g_damage,  pi_c_o_wakeup, pi_c_wakeup, pi_d_o_wakeup, v0, v_i, v_tech2)

    # ME_base = minimize_void(grid, args, variable_args, controls, tol=1e-6, epsilon=0.3, max_iter=10000)



    ii_orig, ee_orig, xx_orig, g_tech_orig, g_damage_orig, pi_c_o_orig, pi_c_orig, pi_d_o_orig, v0_orig, v_i_orig, V_post_tech_orig = controls


    pi_c_o = pi_c_o_orig[:, :, :n_bar+1, :]
    pi_c = pi_c_orig[:, :, :n_bar+1, :]
    pi_d_o = pi_d_o_orig[:, :, :n_bar+1, :]
    
    # n_dmg = len(gamma_3_list)
    # n_climate = len(pi_c_orig)
    
    ii = ii_orig[:, :n_bar+1, :]
    ee = ee_orig[:, :n_bar+1, :]
    xx = xx_orig[:, :n_bar+1, :]
    
    g_tech = g_tech_orig[:, :n_bar+1, :]
    V_post_tech = V_post_tech_orig[:, :n_bar+1, :]

    v0 = v0_orig[:, :n_bar+1, :]
    
    v_i = v_i_orig[:, :, :n_bar+1, :]
    g_damage = g_damage_orig[:, :, :n_bar+1, :]
    pi_c = pi_c_orig[:, :, :n_bar+1, :]

    FC_Err   = 1
    epoch    = 0
        # For PETSc
    K_mat_1d = K_mat.ravel(order='F')
    Y_mat_1d = Y_mat.ravel(order='F')
    L_mat_1d = L_mat.ravel(order='F')
    
    lowerLims = np.array([K_min, Y_min, L_min], dtype=np.float64)
    upperLims = np.array([K_max, Y_max, L_max], dtype=np.float64)

    dVec = np.array([hK, hY, hL])
    increVec = np.array([1, nK, nK * nY],dtype=np.int32)


    petsc_mat = PETSc.Mat().create()
    petsc_mat.setType('aij')
    petsc_mat.setSizes([nK * nY * nL, nK * nY * nL])
    petsc_mat.setPreallocationNNZ(13)
    petsc_mat.setUp()
    ksp = PETSc.KSP()
    ksp.create(PETSc.COMM_WORLD)
    ksp.setType('bcgs')
    ksp.getPC().setType('ilu')
    ksp.setFromOptions()

    while FC_Err > tol and epoch < max_iter:
        
        start_ep = time.time()

        
        dVdK  = finiteDiff_3D(v0,0,1,hK)
        dVdK[dVdK <= 1e-16] = 1e-16
        dK = dVdK
        dVdY  = finiteDiff_3D(v0,1,1,hY)
        dY = dVdY
        dVdL  = finiteDiff_3D(v0,2,1,hL)
        dVdL[dVdL <= 1e-16] = 1e-16
        dL = dVdL
        ######## second order
        ddVdK = finiteDiff_3D(v0,0,2,hK)
        ddVdY = finiteDiff_3D(v0,1,2,hY)
        ddY = ddVdY
        ddVdL = finiteDiff_3D(v0,2,2,hL)
        
        dG  = gamma_1 + gamma_2 * Y_mat
        ddG = gamma_2 
        G = dY -  dG

        ################## Control Updating######################
        # update pi_c
        log_pi_c_ratio = - G * ee * theta_ell/ xi_a
        log_pi_c_ratio += -dL * psi_0 * xx**psi_1 * np.exp( psi_1 * K_mat - (1-psi_2) * L_mat) / xi_a
        pi_c_ratio = log_pi_c_ratio - np.max(log_pi_c_ratio)
        pi_c = np.exp(pi_c_ratio) * pi_c_o
        pi_c = (pi_c <= 0) * 1e-16 + (pi_c > 0) * pi_c
        pi_c = pi_c / np.sum(pi_c, axis=0)
        entropy = np.sum(pi_c * (np.log(pi_c) - np.log(pi_c_o)), axis=0)

        # update g_dmg

        # g_damage = np.exp(- (v_i-v0)/xi_p)

        # # update g_tech

        # g_tech = np.exp(1 / xi_g * (v0 - V_post_tech))
        # g_tech[g_tech <=1e-16] = 1e-16
        ################## Control Updating######################


        jj = alpha * vartheta_bar * (1 - ee / (alpha * lambda_bar * np.exp(K_mat)))**theta
        jj[jj <= 1e-16] = 1e-16
        consumption = alpha - ii - jj - xx
        
        
        A   = - delta * np.ones(K_mat.shape) - np.exp(  L_mat - np.log(448) ) * g_tech
        B_1 = mu_k + ii - 0.5 * kappa * ii**2 - 0.5 * sigma_k**2
        B_2 = np.sum(theta_ell* pi_c, axis=0) * ee
        # B_3 = - zeta + psi_0 * (xx * np.exp(K_mat - L_mat))**psi_1 - 0.5 * sigma_g**2
        B_3 = - zeta + psi_0 * xx** psi_1 * np.exp( psi_1 * K_mat ) * np.sum(pi_c * np.exp( -( 1-psi_2) * L_mat  ), axis=0 )- 0.5 * sigma_g**2

        C_1 = 0.5 * sigma_k**2 * np.ones(K_mat.shape)
        C_2 = 0.5 * sigma_y**2 * ee**2
        C_3 = 0.5 * sigma_g**2 * np.ones(K_mat.shape)
        D = delta * np.log(consumption) + delta * K_mat  - dG * np.sum(theta_ell* pi_c, axis=0) * ee  - 0.5 * ddG * sigma_y**2 * ee**2  + xi_a * entropy + xi_g * np.exp((L_mat - np.log(448))) * (1 - g_tech + g_tech * np.log(g_tech)) + np.exp( (L_mat - np.log(448)) ) * g_tech * V_post_tech

        D += xi_p * Intensity * np.sum( pi_d_o*(1-g_damage+g_damage*np.log(g_damage)),axis=0) +Intensity*np.sum(pi_d_o*g_damage*v_i,axis=0)
        A -=  Intensity*np.sum(pi_d_o*g_damage,axis=0)

        bpoint1 = time.time()

        A_1d   = A.ravel(order = 'F')
        C_1_1d = C_1.ravel(order = 'F')
        C_2_1d = C_2.ravel(order = 'F')
        C_3_1d = C_3.ravel(order = 'F')
        B_1_1d = B_1.ravel(order = 'F')
        B_2_1d = B_2.ravel(order = 'F')
        B_3_1d = B_3.ravel(order = 'F')
        D_1d   = D.ravel(order = 'F')
        v0_1d  = v0.ravel(order = 'F')
        


        petsclinearsystem.formLinearSystem(K_mat_1d, Y_mat_1d, L_mat_1d, A_1d, B_1_1d, B_2_1d, B_3_1d, C_1_1d, C_2_1d, C_3_1d, epsilon, lowerLims, upperLims, dVec, increVec, petsc_mat)
        b = v0_1d + D_1d * epsilon
        petsc_rhs = PETSc.Vec().createWithArray(b)
        x = petsc_mat.createVecRight()


        # create linear solver
        start_ksp = time.time()
        ksp.setOperators(petsc_mat)
        ksp.setTolerances(rtol=tol)
        ksp.solve(petsc_rhs, x)
        petsc_rhs.destroy()
        x.destroy()
        out_comp = np.array(ksp.getSolution()).reshape(A.shape,order = "F")
        end_ksp = time.time()
        num_iter = ksp.getIterationNumber()
        
        # print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
        # print("PETSc preconditioned residual norm is {:g}; iterations: {}".format(ksp.getResidualNorm(), ksp.getIterationNumber()))
        PDE_rhs = A * v0 + B_1 * dVdK + B_2 * dVdY + B_3 * dVdL + C_1 * ddVdK + C_2 * ddVdY + C_3 * ddVdL + D
        PDE_Err = np.max(abs(PDE_rhs))
        FC_Err = np.max(abs((out_comp - v0)/ epsilon))
        
        # if FC_Err < 2*tol:
            
        #     print("-----------------------------------")
        #     print("---------Epoch {}---------------".format(epoch))
        #     print("-----------------------------------")
        #     print("min i: {},\t max i: {}\t".format(ii.min(), ii.max()))
        #     print("min e: {},\t max e: {}\t".format(ee.min(), ee.max()))
        #     print("min x: {},\t max x: {}\t".format(xx.min(), xx.max()))
        #     print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
        #     print("Epoch {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))
        #     print("Epoch time: {:.4f}".format(time.time() - start_ep))
        # elif epoch%100==0:
            
        #     print("-----------------------------------")
        #     print("---------Epoch {}---------------".format(epoch))
        #     print("-----------------------------------")
        #     print("min i: {},\t max i: {}\t".format(ii.min(), ii.max()))
        #     print("min e: {},\t max e: {}\t".format(ee.min(), ee.max()))
        #     print("min x: {},\t max x: {}\t".format(xx.min(), xx.max()))
        #     print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
        #     print("Epoch {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))
        #     print("Epoch time: {:.4f}".format(time.time() - start_ep))
          
          
        v0     = out_comp
        epoch += 1
        
    print("-----------------------------------")
    print("---------Converged, Epoch {}---------------".format(epoch))
    print("-----------------------------------")
    print("min i: {},\t max i: {}\t".format(ii.min(), ii.max()))
    print("min e: {},\t max e: {}\t".format(ee.min(), ee.max()))
    print("min x: {},\t max x: {}\t".format(xx.min(), xx.max()))
    print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
    print("Epoch {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))
    print("Epoch time: {:.4f}".format(time.time() - start_ep))
        
        
    ME = - dVdY * np.sum(pi_c * theta_ell, axis=0) - ddVdY * sigma_y**2 * ee + dG * np.sum(theta_ell * pi_c, axis=0) +  ddG * sigma_y**2 * ee

    
    return ME


def minimize_dmg(grid=(),  args = (), n_bar=(),variable_args= (), controls = (), tol=1e-6, epsilon=0.3, max_iter=10000):
    """
    compute jump model with ambiguity over climate models
    """

    K, Y, L = grid

    

    delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a, xi_g, xi_p = args
    # n_bar,  = n_bar  
    # n_bar = len(Y)-1
    # n_bar = np.abs(Y - y_bar).argmin()

    Y_cap = Y[:n_bar+1]
    
    K_min, K_max, Y_min, Y_max, L_min, L_max = min(K), max(K), min(Y_cap), max(Y_cap), min(L), max(L)
    hK, hY, hL = K[1] - K[0], Y_cap[1] - Y_cap[0], L[1]-L[0]
    nK, nY, nL = len(K), len(Y_cap), len(L)
    

    print("K_min={},K_max={},Y_min={},Y_max={},L_min={},L_max={}" .format(K_min, K_max, Y_min, Y_max, L_min, L_max))
    print("hK={},hY={},hL={}" .format(hK, hY, hL))
    print("nK={},nY={},nL={}" .format(nK, nY, nL))


    (K_mat, Y_mat, L_mat) = np.meshgrid(K, Y_cap, L, indexing = 'ij')


    Intensity=Damage_Intensity(Y_mat)
    

    theta_ell_orig, psi_2_orig = variable_args
    
    theta_ell = theta_ell_orig[:, :, :n_bar+1, :]
    psi_2 = psi_2_orig[:, :, :n_bar+1, :]
    


    # args = (delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a, xi_g, xi_p)
    # variable_args = (theta_ell_wakeup, psi_2_wakeup)
    # controls=(i,e,x, g_tech, g_damage,  pi_c_o_wakeup, pi_c_wakeup, pi_d_o_wakeup, v0, v_i, v_tech2)

    # ME_base = minimize_void(grid, args, variable_args, controls, tol=1e-6, epsilon=0.3, max_iter=10000)



    ii_orig, ee_orig, xx_orig, g_tech_orig, g_damage_orig, pi_c_o_orig, pi_c_orig, pi_d_o_orig, v0_orig, v_i_orig, V_post_tech_orig = controls


    pi_c_o = pi_c_o_orig[:, :, :n_bar+1, :]
    pi_c = pi_c_orig[:, :, :n_bar+1, :]
    pi_d_o = pi_d_o_orig[:, :, :n_bar+1, :]
    
    # n_dmg = len(gamma_3_list)
    # n_climate = len(pi_c_orig)
    
    ii = ii_orig[:, :n_bar+1, :]
    ee = ee_orig[:, :n_bar+1, :]
    xx = xx_orig[:, :n_bar+1, :]
    
    g_tech = g_tech_orig[:, :n_bar+1, :]
    V_post_tech = V_post_tech_orig[:, :n_bar+1, :]

    v0 = v0_orig[:, :n_bar+1, :]
    
    v_i = v_i_orig[:, :, :n_bar+1, :]
    g_damage = g_damage_orig[:, :, :n_bar+1, :]
    pi_c = pi_c_orig[:, :, :n_bar+1, :]

    FC_Err   = 1
    epoch    = 0
        # For PETSc
    K_mat_1d = K_mat.ravel(order='F')
    Y_mat_1d = Y_mat.ravel(order='F')
    L_mat_1d = L_mat.ravel(order='F')
    
    lowerLims = np.array([K_min, Y_min, L_min], dtype=np.float64)
    upperLims = np.array([K_max, Y_max, L_max], dtype=np.float64)

    dVec = np.array([hK, hY, hL])
    increVec = np.array([1, nK, nK * nY],dtype=np.int32)


    petsc_mat = PETSc.Mat().create()
    petsc_mat.setType('aij')
    petsc_mat.setSizes([nK * nY * nL, nK * nY * nL])
    petsc_mat.setPreallocationNNZ(13)
    petsc_mat.setUp()
    ksp = PETSc.KSP()
    ksp.create(PETSc.COMM_WORLD)
    ksp.setType('bcgs')
    ksp.getPC().setType('ilu')
    ksp.setFromOptions()

    while FC_Err > tol and epoch < max_iter:
        
        start_ep = time.time()

        
        dVdK  = finiteDiff_3D(v0,0,1,hK)
        dVdK[dVdK <= 1e-16] = 1e-16
        dK = dVdK
        dVdY  = finiteDiff_3D(v0,1,1,hY)
        dY = dVdY
        dVdL  = finiteDiff_3D(v0,2,1,hL)
        dVdL[dVdL <= 1e-16] = 1e-16
        dL = dVdL
        ######## second order
        ddVdK = finiteDiff_3D(v0,0,2,hK)
        ddVdY = finiteDiff_3D(v0,1,2,hY)
        ddY = ddVdY
        ddVdL = finiteDiff_3D(v0,2,2,hL)
        
        dG  = gamma_1 + gamma_2 * Y_mat
        ddG = gamma_2 
        G = dY -  dG

        ################## Control Updating######################
        # update pi_c
        # log_pi_c_ratio = - G * ee * theta_ell/ xi_a
        # log_pi_c_ratio += -dL * psi_0 * xx**psi_1 * np.exp( psi_1 * K_mat - (1-psi_2) * L_mat) / xi_a
        # pi_c_ratio = log_pi_c_ratio - np.max(log_pi_c_ratio)
        # pi_c = np.exp(pi_c_ratio) * pi_c_o
        # pi_c = (pi_c <= 0) * 1e-16 + (pi_c > 0) * pi_c
        # pi_c = pi_c / np.sum(pi_c, axis=0)
        entropy = np.sum(pi_c * (np.log(pi_c) - np.log(pi_c_o)), axis=0)

        # update g_dmg

        g_damage = np.exp(- (v_i-v0)/xi_p)

        # update g_tech

        # g_tech = np.exp(1 / xi_g * (v0 - V_post_tech))
        # g_tech[g_tech <=1e-16] = 1e-16
        ################## Control Updating######################


        jj = alpha * vartheta_bar * (1 - ee / (alpha * lambda_bar * np.exp(K_mat)))**theta
        jj[jj <= 1e-16] = 1e-16
        consumption = alpha - ii - jj - xx
        
        
        A   = - delta * np.ones(K_mat.shape) - np.exp(  L_mat - np.log(448) ) * g_tech
        B_1 = mu_k + ii - 0.5 * kappa * ii**2 - 0.5 * sigma_k**2
        B_2 = np.sum(theta_ell* pi_c, axis=0) * ee
        # B_3 = - zeta + psi_0 * (xx * np.exp(K_mat - L_mat))**psi_1 - 0.5 * sigma_g**2
        B_3 = - zeta + psi_0 * xx** psi_1 * np.exp( psi_1 * K_mat ) * np.sum(pi_c * np.exp( -( 1-psi_2) * L_mat  ), axis=0 )- 0.5 * sigma_g**2

        C_1 = 0.5 * sigma_k**2 * np.ones(K_mat.shape)
        C_2 = 0.5 * sigma_y**2 * ee**2
        C_3 = 0.5 * sigma_g**2 * np.ones(K_mat.shape)
        D = delta * np.log(consumption) + delta * K_mat  - dG * np.sum(theta_ell* pi_c, axis=0) * ee  - 0.5 * ddG * sigma_y**2 * ee**2  + xi_a * entropy + xi_g * np.exp((L_mat - np.log(448))) * (1 - g_tech + g_tech * np.log(g_tech)) + np.exp( (L_mat - np.log(448)) ) * g_tech * V_post_tech

        D += xi_p * Intensity * np.sum( pi_d_o*(1-g_damage+g_damage*np.log(g_damage)),axis=0) +Intensity*np.sum(pi_d_o*g_damage*v_i,axis=0)
        A -=  Intensity*np.sum(pi_d_o*g_damage,axis=0)

        bpoint1 = time.time()

        A_1d   = A.ravel(order = 'F')
        C_1_1d = C_1.ravel(order = 'F')
        C_2_1d = C_2.ravel(order = 'F')
        C_3_1d = C_3.ravel(order = 'F')
        B_1_1d = B_1.ravel(order = 'F')
        B_2_1d = B_2.ravel(order = 'F')
        B_3_1d = B_3.ravel(order = 'F')
        D_1d   = D.ravel(order = 'F')
        v0_1d  = v0.ravel(order = 'F')
        


        petsclinearsystem.formLinearSystem(K_mat_1d, Y_mat_1d, L_mat_1d, A_1d, B_1_1d, B_2_1d, B_3_1d, C_1_1d, C_2_1d, C_3_1d, epsilon, lowerLims, upperLims, dVec, increVec, petsc_mat)
        b = v0_1d + D_1d * epsilon
        petsc_rhs = PETSc.Vec().createWithArray(b)
        x = petsc_mat.createVecRight()


        # create linear solver
        start_ksp = time.time()
        ksp.setOperators(petsc_mat)
        ksp.setTolerances(rtol=tol)
        ksp.solve(petsc_rhs, x)
        petsc_rhs.destroy()
        x.destroy()
        out_comp = np.array(ksp.getSolution()).reshape(A.shape,order = "F")
        end_ksp = time.time()
        num_iter = ksp.getIterationNumber()
        
        # print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
        # print("PETSc preconditioned residual norm is {:g}; iterations: {}".format(ksp.getResidualNorm(), ksp.getIterationNumber()))
        PDE_rhs = A * v0 + B_1 * dVdK + B_2 * dVdY + B_3 * dVdL + C_1 * ddVdK + C_2 * ddVdY + C_3 * ddVdL + D
        PDE_Err = np.max(abs(PDE_rhs))
        FC_Err = np.max(abs((out_comp - v0)/ epsilon))
        
        # if FC_Err < 2*tol:
            
        #     print("-----------------------------------")
        #     print("---------Epoch {}---------------".format(epoch))
        #     print("-----------------------------------")
        #     print("min i: {},\t max i: {}\t".format(ii.min(), ii.max()))
        #     print("min e: {},\t max e: {}\t".format(ee.min(), ee.max()))
        #     print("min x: {},\t max x: {}\t".format(xx.min(), xx.max()))
        #     print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
        #     print("Epoch {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))
        #     print("Epoch time: {:.4f}".format(time.time() - start_ep))
        # elif epoch%100==0:
            
        #     print("-----------------------------------")
        #     print("---------Epoch {}---------------".format(epoch))
        #     print("-----------------------------------")
        #     print("min i: {},\t max i: {}\t".format(ii.min(), ii.max()))
        #     print("min e: {},\t max e: {}\t".format(ee.min(), ee.max()))
        #     print("min x: {},\t max x: {}\t".format(xx.min(), xx.max()))
        #     print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
        #     print("Epoch {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))
        #     print("Epoch time: {:.4f}".format(time.time() - start_ep))
          
          
        v0     = out_comp
        epoch += 1
        
    print("-----------------------------------")
    print("---------Converged, Epoch {}---------------".format(epoch))
    print("-----------------------------------")
    print("min i: {},\t max i: {}\t".format(ii.min(), ii.max()))
    print("min e: {},\t max e: {}\t".format(ee.min(), ee.max()))
    print("min x: {},\t max x: {}\t".format(xx.min(), xx.max()))
    print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
    print("Epoch {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))
    print("Epoch time: {:.4f}".format(time.time() - start_ep))
        
        
    ME = - dVdY * np.sum(pi_c * theta_ell, axis=0) - ddVdY * sigma_y**2 * ee + dG * np.sum(theta_ell * pi_c, axis=0) +  ddG * sigma_y**2 * ee

    
    return ME

def minimize_tech(grid=(),  args = (), n_bar=(),variable_args= (), controls = (), tol=1e-6, epsilon=0.3, max_iter=10000):
    """
    compute jump model with ambiguity over climate models
    """

    K, Y, L = grid

    

    delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a, xi_g, xi_p = args
    # n_bar,  = n_bar  

    # n_bar = len(Y)-1
    # n_bar = np.abs(Y - y_bar).argmin()

    Y_cap = Y[:n_bar+1]
    
    K_min, K_max, Y_min, Y_max, L_min, L_max = min(K), max(K), min(Y_cap), max(Y_cap), min(L), max(L)
    hK, hY, hL = K[1] - K[0], Y_cap[1] - Y_cap[0], L[1]-L[0]
    nK, nY, nL = len(K), len(Y_cap), len(L)
    

    print("K_min={},K_max={},Y_min={},Y_max={},L_min={},L_max={}" .format(K_min, K_max, Y_min, Y_max, L_min, L_max))
    print("hK={},hY={},hL={}" .format(hK, hY, hL))
    print("nK={},nY={},nL={}" .format(nK, nY, nL))


    (K_mat, Y_mat, L_mat) = np.meshgrid(K, Y_cap, L, indexing = 'ij')


    Intensity=Damage_Intensity(Y_mat)
    

    theta_ell_orig, psi_2_orig = variable_args
    
    theta_ell = theta_ell_orig[:, :, :n_bar+1, :]
    psi_2 = psi_2_orig[:, :, :n_bar+1, :]
    


    # args = (delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a, xi_g, xi_p)
    # variable_args = (theta_ell_wakeup, psi_2_wakeup)
    # controls=(i,e,x, g_tech, g_damage,  pi_c_o_wakeup, pi_c_wakeup, pi_d_o_wakeup, v0, v_i, v_tech2)

    # ME_base = minimize_void(grid, args, variable_args, controls, tol=1e-6, epsilon=0.3, max_iter=10000)



    ii_orig, ee_orig, xx_orig, g_tech_orig, g_damage_orig, pi_c_o_orig, pi_c_orig, pi_d_o_orig, v0_orig, v_i_orig, V_post_tech_orig = controls


    pi_c_o = pi_c_o_orig[:, :, :n_bar+1, :]
    pi_c = pi_c_orig[:, :, :n_bar+1, :]
    pi_d_o = pi_d_o_orig[:, :, :n_bar+1, :]
    
    # n_dmg = len(gamma_3_list)
    # n_climate = len(pi_c_orig)
    
    ii = ii_orig[:, :n_bar+1, :]
    ee = ee_orig[:, :n_bar+1, :]
    xx = xx_orig[:, :n_bar+1, :]
    
    g_tech = g_tech_orig[:, :n_bar+1, :]
    V_post_tech = V_post_tech_orig[:, :n_bar+1, :]

    v0 = v0_orig[:, :n_bar+1, :]
    
    v_i = v_i_orig[:, :, :n_bar+1, :]
    g_damage = g_damage_orig[:, :, :n_bar+1, :]
    pi_c = pi_c_orig[:, :, :n_bar+1, :]

    FC_Err   = 1
    epoch    = 0
        # For PETSc
    K_mat_1d = K_mat.ravel(order='F')
    Y_mat_1d = Y_mat.ravel(order='F')
    L_mat_1d = L_mat.ravel(order='F')
    
    lowerLims = np.array([K_min, Y_min, L_min], dtype=np.float64)
    upperLims = np.array([K_max, Y_max, L_max], dtype=np.float64)

    dVec = np.array([hK, hY, hL])
    increVec = np.array([1, nK, nK * nY],dtype=np.int32)


    petsc_mat = PETSc.Mat().create()
    petsc_mat.setType('aij')
    petsc_mat.setSizes([nK * nY * nL, nK * nY * nL])
    petsc_mat.setPreallocationNNZ(13)
    petsc_mat.setUp()
    ksp = PETSc.KSP()
    ksp.create(PETSc.COMM_WORLD)
    ksp.setType('bcgs')
    ksp.getPC().setType('ilu')
    ksp.setFromOptions()

    while FC_Err > tol and epoch < max_iter:
        
        start_ep = time.time()

        
        dVdK  = finiteDiff_3D(v0,0,1,hK)
        dVdK[dVdK <= 1e-16] = 1e-16
        dK = dVdK
        dVdY  = finiteDiff_3D(v0,1,1,hY)
        dY = dVdY
        dVdL  = finiteDiff_3D(v0,2,1,hL)
        dVdL[dVdL <= 1e-16] = 1e-16
        dL = dVdL
        ######## second order
        ddVdK = finiteDiff_3D(v0,0,2,hK)
        ddVdY = finiteDiff_3D(v0,1,2,hY)
        ddY = ddVdY
        ddVdL = finiteDiff_3D(v0,2,2,hL)
        
        dG  = gamma_1 + gamma_2 * Y_mat
        ddG = gamma_2 
        G = dY -  dG

        ################## Control Updating######################
        # update pi_c
        # log_pi_c_ratio = - G * ee * theta_ell/ xi_a
        # log_pi_c_ratio += -dL * psi_0 * xx**psi_1 * np.exp( psi_1 * K_mat - (1-psi_2) * L_mat) / xi_a
        # pi_c_ratio = log_pi_c_ratio - np.max(log_pi_c_ratio)
        # pi_c = np.exp(pi_c_ratio) * pi_c_o
        # pi_c = (pi_c <= 0) * 1e-16 + (pi_c > 0) * pi_c
        # pi_c = pi_c / np.sum(pi_c, axis=0)
        entropy = np.sum(pi_c * (np.log(pi_c) - np.log(pi_c_o)), axis=0)

        # update g_dmg

        # g_damage = np.exp(- (v_i-v0)/xi_p)

        # update g_tech

        g_tech = np.exp(1 / xi_g * (v0 - V_post_tech))
        g_tech[g_tech <=1e-16] = 1e-16
        ################## Control Updating######################


        jj = alpha * vartheta_bar * (1 - ee / (alpha * lambda_bar * np.exp(K_mat)))**theta
        jj[jj <= 1e-16] = 1e-16
        consumption = alpha - ii - jj - xx
        
        
        A   = - delta * np.ones(K_mat.shape) - np.exp(  L_mat - np.log(448) ) * g_tech
        B_1 = mu_k + ii - 0.5 * kappa * ii**2 - 0.5 * sigma_k**2
        B_2 = np.sum(theta_ell* pi_c, axis=0) * ee
        # B_3 = - zeta + psi_0 * (xx * np.exp(K_mat - L_mat))**psi_1 - 0.5 * sigma_g**2
        B_3 = - zeta + psi_0 * xx** psi_1 * np.exp( psi_1 * K_mat ) * np.sum(pi_c * np.exp( -( 1-psi_2) * L_mat  ), axis=0 )- 0.5 * sigma_g**2

        C_1 = 0.5 * sigma_k**2 * np.ones(K_mat.shape)
        C_2 = 0.5 * sigma_y**2 * ee**2
        C_3 = 0.5 * sigma_g**2 * np.ones(K_mat.shape)
        D = delta * np.log(consumption) + delta * K_mat  - dG * np.sum(theta_ell* pi_c, axis=0) * ee  - 0.5 * ddG * sigma_y**2 * ee**2  + xi_a * entropy + xi_g * np.exp((L_mat - np.log(448))) * (1 - g_tech + g_tech * np.log(g_tech)) + np.exp( (L_mat - np.log(448)) ) * g_tech * V_post_tech

        D += xi_p * Intensity * np.sum( pi_d_o*(1-g_damage+g_damage*np.log(g_damage)),axis=0) +Intensity*np.sum(pi_d_o*g_damage*v_i,axis=0)
        A -=  Intensity*np.sum(pi_d_o*g_damage,axis=0)

        bpoint1 = time.time()

        A_1d   = A.ravel(order = 'F')
        C_1_1d = C_1.ravel(order = 'F')
        C_2_1d = C_2.ravel(order = 'F')
        C_3_1d = C_3.ravel(order = 'F')
        B_1_1d = B_1.ravel(order = 'F')
        B_2_1d = B_2.ravel(order = 'F')
        B_3_1d = B_3.ravel(order = 'F')
        D_1d   = D.ravel(order = 'F')
        v0_1d  = v0.ravel(order = 'F')
        


        petsclinearsystem.formLinearSystem(K_mat_1d, Y_mat_1d, L_mat_1d, A_1d, B_1_1d, B_2_1d, B_3_1d, C_1_1d, C_2_1d, C_3_1d, epsilon, lowerLims, upperLims, dVec, increVec, petsc_mat)
        b = v0_1d + D_1d * epsilon
        petsc_rhs = PETSc.Vec().createWithArray(b)
        x = petsc_mat.createVecRight()


        # create linear solver
        start_ksp = time.time()
        ksp.setOperators(petsc_mat)
        ksp.setTolerances(rtol=tol)
        ksp.solve(petsc_rhs, x)
        petsc_rhs.destroy()
        x.destroy()
        out_comp = np.array(ksp.getSolution()).reshape(A.shape,order = "F")
        end_ksp = time.time()
        num_iter = ksp.getIterationNumber()
        
        # print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
        # print("PETSc preconditioned residual norm is {:g}; iterations: {}".format(ksp.getResidualNorm(), ksp.getIterationNumber()))
        PDE_rhs = A * v0 + B_1 * dVdK + B_2 * dVdY + B_3 * dVdL + C_1 * ddVdK + C_2 * ddVdY + C_3 * ddVdL + D
        PDE_Err = np.max(abs(PDE_rhs))
        FC_Err = np.max(abs((out_comp - v0)/ epsilon))
        
        # if FC_Err < 2*tol:
            
        #     print("-----------------------------------")
        #     print("---------Epoch {}---------------".format(epoch))
        #     print("-----------------------------------")
        #     print("min i: {},\t max i: {}\t".format(ii.min(), ii.max()))
        #     print("min e: {},\t max e: {}\t".format(ee.min(), ee.max()))
        #     print("min x: {},\t max x: {}\t".format(xx.min(), xx.max()))
        #     print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
        #     print("Epoch {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))
        #     print("Epoch time: {:.4f}".format(time.time() - start_ep))
        # elif epoch%100==0:
            
        #     print("-----------------------------------")
        #     print("---------Epoch {}---------------".format(epoch))
        #     print("-----------------------------------")
        #     print("min i: {},\t max i: {}\t".format(ii.min(), ii.max()))
        #     print("min e: {},\t max e: {}\t".format(ee.min(), ee.max()))
        #     print("min x: {},\t max x: {}\t".format(xx.min(), xx.max()))
        #     print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
        #     print("Epoch {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))
        #     print("Epoch time: {:.4f}".format(time.time() - start_ep))
          
          
        v0     = out_comp
        epoch += 1
        
    print("-----------------------------------")
    print("---------Converged, Epoch {}---------------".format(epoch))
    print("-----------------------------------")
    print("min i: {},\t max i: {}\t".format(ii.min(), ii.max()))
    print("min e: {},\t max e: {}\t".format(ee.min(), ee.max()))
    print("min x: {},\t max x: {}\t".format(xx.min(), xx.max()))
    print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
    print("Epoch {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))
    print("Epoch time: {:.4f}".format(time.time() - start_ep))
        
        
    ME = - dVdY * np.sum(pi_c * theta_ell, axis=0) - ddVdY * sigma_y**2 * ee + dG * np.sum(theta_ell * pi_c, axis=0) +  ddG * sigma_y**2 * ee

    
    return ME




def minimize_pi_c_dmg_tech(grid=(),  args = (), n_bar=(), variable_args= (), controls = (), tol=1e-6, epsilon=0.3, max_iter=10000):
    """
    compute jump model with ambiguity over climate models
    """

    K, Y, L = grid

    

    delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a, xi_g, xi_p = args
    # n_bar, = n_bar
    # n_bar = len(Y)-1
    # n_bar = np.abs(Y - y_bar).argmin()

    Y_cap = Y[:n_bar+1]
    
    K_min, K_max, Y_min, Y_max, L_min, L_max = min(K), max(K), min(Y_cap), max(Y_cap), min(L), max(L)
    hK, hY, hL = K[1] - K[0], Y_cap[1] - Y_cap[0], L[1]-L[0]
    nK, nY, nL = len(K), len(Y_cap), len(L)
    

    print("K_min={},K_max={},Y_min={},Y_max={},L_min={},L_max={}" .format(K_min, K_max, Y_min, Y_max, L_min, L_max))
    print("hK={},hY={},hL={}" .format(hK, hY, hL))
    print("nK={},nY={},nL={}" .format(nK, nY, nL))


    (K_mat, Y_mat, L_mat) = np.meshgrid(K, Y_cap, L, indexing = 'ij')


    Intensity=Damage_Intensity(Y_mat)
    

    theta_ell_orig, psi_2_orig = variable_args
    
    theta_ell = theta_ell_orig[:, :, :n_bar+1, :]
    psi_2 = psi_2_orig[:, :, :n_bar+1, :]
    


    # args = (delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a, xi_g, xi_p)
    # variable_args = (theta_ell_wakeup, psi_2_wakeup)
    # controls=(i,e,x, g_tech, g_damage,  pi_c_o_wakeup, pi_c_wakeup, pi_d_o_wakeup, v0, v_i, v_tech2)

    # ME_base = minimize_void(grid, args, variable_args, controls, tol=1e-6, epsilon=0.3, max_iter=10000)



    ii_orig, ee_orig, xx_orig, g_tech_orig, g_damage_orig, pi_c_o_orig, pi_c_orig, pi_d_o_orig, v0_orig, v_i_orig, V_post_tech_orig = controls


    pi_c_o = pi_c_o_orig[:, :, :n_bar+1, :]
    pi_c = pi_c_orig[:, :, :n_bar+1, :]
    pi_d_o = pi_d_o_orig[:, :, :n_bar+1, :]
    
    # n_dmg = len(gamma_3_list)
    # n_climate = len(pi_c_orig)
    
    ii = ii_orig[:, :n_bar+1, :]
    ee = ee_orig[:, :n_bar+1, :]
    xx = xx_orig[:, :n_bar+1, :]
    
    g_tech = g_tech_orig[:, :n_bar+1, :]
    V_post_tech = V_post_tech_orig[:, :n_bar+1, :]

    v0 = v0_orig[:, :n_bar+1, :]
    
    v_i = v_i_orig[:, :, :n_bar+1, :]
    g_damage = g_damage_orig[:, :, :n_bar+1, :]
    pi_c = pi_c_orig[:, :, :n_bar+1, :]

    FC_Err   = 1
    epoch    = 0
        # For PETSc
    K_mat_1d = K_mat.ravel(order='F')
    Y_mat_1d = Y_mat.ravel(order='F')
    L_mat_1d = L_mat.ravel(order='F')
    
    lowerLims = np.array([K_min, Y_min, L_min], dtype=np.float64)
    upperLims = np.array([K_max, Y_max, L_max], dtype=np.float64)

    dVec = np.array([hK, hY, hL])
    increVec = np.array([1, nK, nK * nY],dtype=np.int32)


    petsc_mat = PETSc.Mat().create()
    petsc_mat.setType('aij')
    petsc_mat.setSizes([nK * nY * nL, nK * nY * nL])
    petsc_mat.setPreallocationNNZ(13)
    petsc_mat.setUp()
    ksp = PETSc.KSP()
    ksp.create(PETSc.COMM_WORLD)
    ksp.setType('bcgs')
    ksp.getPC().setType('ilu')
    ksp.setFromOptions()

    while FC_Err > tol and epoch < max_iter:
        
        start_ep = time.time()

        
        dVdK  = finiteDiff_3D(v0,0,1,hK)
        dVdK[dVdK <= 1e-16] = 1e-16
        dK = dVdK
        dVdY  = finiteDiff_3D(v0,1,1,hY)
        dY = dVdY
        dVdL  = finiteDiff_3D(v0,2,1,hL)
        dVdL[dVdL <= 1e-16] = 1e-16
        dL = dVdL
        ######## second order
        ddVdK = finiteDiff_3D(v0,0,2,hK)
        ddVdY = finiteDiff_3D(v0,1,2,hY)
        ddY = ddVdY
        ddVdL = finiteDiff_3D(v0,2,2,hL)
        
        dG  = gamma_1 + gamma_2 * Y_mat
        ddG = gamma_2 
        G = dY -  dG

        ################## Control Updating######################
        # update pi_c
        log_pi_c_ratio = - G * ee * theta_ell/ xi_a
        log_pi_c_ratio += -dL * psi_0 * xx**psi_1 * np.exp( psi_1 * K_mat - (1-psi_2) * L_mat) / xi_a
        pi_c_ratio = log_pi_c_ratio - np.max(log_pi_c_ratio)
        pi_c = np.exp(pi_c_ratio) * pi_c_o
        pi_c = (pi_c <= 0) * 1e-16 + (pi_c > 0) * pi_c
        pi_c = pi_c / np.sum(pi_c, axis=0)
        entropy = np.sum(pi_c * (np.log(pi_c) - np.log(pi_c_o)), axis=0)

        # update g_dmg

        g_damage = np.exp(- (v_i-v0)/xi_p)

        # # update g_tech

        g_tech = np.exp(1 / xi_g * (v0 - V_post_tech))
        g_tech[g_tech <=1e-16] = 1e-16
        ################## Control Updating######################


        jj = alpha * vartheta_bar * (1 - ee / (alpha * lambda_bar * np.exp(K_mat)))**theta
        jj[jj <= 1e-16] = 1e-16
        consumption = alpha - ii - jj - xx
        
        
        A   = - delta * np.ones(K_mat.shape) - np.exp(  L_mat - np.log(448) ) * g_tech
        B_1 = mu_k + ii - 0.5 * kappa * ii**2 - 0.5 * sigma_k**2
        B_2 = np.sum(theta_ell* pi_c, axis=0) * ee
        # B_3 = - zeta + psi_0 * (xx * np.exp(K_mat - L_mat))**psi_1 - 0.5 * sigma_g**2
        B_3 = - zeta + psi_0 * xx** psi_1 * np.exp( psi_1 * K_mat ) * np.sum(pi_c * np.exp( -( 1-psi_2) * L_mat  ), axis=0 )- 0.5 * sigma_g**2

        C_1 = 0.5 * sigma_k**2 * np.ones(K_mat.shape)
        C_2 = 0.5 * sigma_y**2 * ee**2
        C_3 = 0.5 * sigma_g**2 * np.ones(K_mat.shape)
        D = delta * np.log(consumption) + delta * K_mat  - dG * np.sum(theta_ell* pi_c, axis=0) * ee  - 0.5 * ddG * sigma_y**2 * ee**2  + xi_a * entropy + xi_g * np.exp((L_mat - np.log(448))) * (1 - g_tech + g_tech * np.log(g_tech)) + np.exp( (L_mat - np.log(448)) ) * g_tech * V_post_tech

        D += xi_p * Intensity * np.sum( pi_d_o*(1-g_damage+g_damage*np.log(g_damage)),axis=0) +Intensity*np.sum(pi_d_o*g_damage*v_i,axis=0)
        A -=  Intensity*np.sum(pi_d_o*g_damage,axis=0)

        bpoint1 = time.time()

        A_1d   = A.ravel(order = 'F')
        C_1_1d = C_1.ravel(order = 'F')
        C_2_1d = C_2.ravel(order = 'F')
        C_3_1d = C_3.ravel(order = 'F')
        B_1_1d = B_1.ravel(order = 'F')
        B_2_1d = B_2.ravel(order = 'F')
        B_3_1d = B_3.ravel(order = 'F')
        D_1d   = D.ravel(order = 'F')
        v0_1d  = v0.ravel(order = 'F')
        


        petsclinearsystem.formLinearSystem(K_mat_1d, Y_mat_1d, L_mat_1d, A_1d, B_1_1d, B_2_1d, B_3_1d, C_1_1d, C_2_1d, C_3_1d, epsilon, lowerLims, upperLims, dVec, increVec, petsc_mat)
        b = v0_1d + D_1d * epsilon
        petsc_rhs = PETSc.Vec().createWithArray(b)
        x = petsc_mat.createVecRight()


        # create linear solver
        start_ksp = time.time()
        ksp.setOperators(petsc_mat)
        ksp.setTolerances(rtol=tol)
        ksp.solve(petsc_rhs, x)
        petsc_rhs.destroy()
        x.destroy()
        out_comp = np.array(ksp.getSolution()).reshape(A.shape,order = "F")
        end_ksp = time.time()
        num_iter = ksp.getIterationNumber()
        
        # print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
        # print("PETSc preconditioned residual norm is {:g}; iterations: {}".format(ksp.getResidualNorm(), ksp.getIterationNumber()))
        PDE_rhs = A * v0 + B_1 * dVdK + B_2 * dVdY + B_3 * dVdL + C_1 * ddVdK + C_2 * ddVdY + C_3 * ddVdL + D
        PDE_Err = np.max(abs(PDE_rhs))
        FC_Err = np.max(abs((out_comp - v0)/ epsilon))
        
        # if FC_Err < 2*tol:
            
        #     print("-----------------------------------")
        #     print("---------Epoch {}---------------".format(epoch))
        #     print("-----------------------------------")
        #     print("min i: {},\t max i: {}\t".format(ii.min(), ii.max()))
        #     print("min e: {},\t max e: {}\t".format(ee.min(), ee.max()))
        #     print("min x: {},\t max x: {}\t".format(xx.min(), xx.max()))
        #     print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
        #     print("Epoch {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))
        #     print("Epoch time: {:.4f}".format(time.time() - start_ep))
        # elif epoch%100==0:
            
        #     print("-----------------------------------")
        #     print("---------Epoch {}---------------".format(epoch))
        #     print("-----------------------------------")
        #     print("min i: {},\t max i: {}\t".format(ii.min(), ii.max()))
        #     print("min e: {},\t max e: {}\t".format(ee.min(), ee.max()))
        #     print("min x: {},\t max x: {}\t".format(xx.min(), xx.max()))
        #     print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
        #     print("Epoch {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))
        #     print("Epoch time: {:.4f}".format(time.time() - start_ep))
          
          
        v0     = out_comp
        epoch += 1
        
    print("-----------------------------------")
    print("---------Converged, Epoch {}---------------".format(epoch))
    print("-----------------------------------")
    print("min i: {},\t max i: {}\t".format(ii.min(), ii.max()))
    print("min e: {},\t max e: {}\t".format(ee.min(), ee.max()))
    print("min x: {},\t max x: {}\t".format(xx.min(), xx.max()))
    print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
    print("Epoch {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))
    print("Epoch time: {:.4f}".format(time.time() - start_ep))
        
        
    ME = - dVdY * np.sum(pi_c * theta_ell, axis=0) - ddVdY * sigma_y**2 * ee + dG * np.sum(theta_ell * pi_c, axis=0) +  ddG * sigma_y**2 * ee

    
    return ME



def minimize_pi_c_dmg(grid=(),  args = (), n_bar=(), variable_args= (), controls = (), tol=1e-6, epsilon=0.3, max_iter=10000):
    """
    compute jump model with ambiguity over climate models
    """

    K, Y, L = grid

    

    delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a, xi_g, xi_p = args
    # n_bar, = n_bar
    # n_bar = len(Y)-1
    # n_bar = np.abs(Y - y_bar).argmin()

    Y_cap = Y[:n_bar+1]
    
    K_min, K_max, Y_min, Y_max, L_min, L_max = min(K), max(K), min(Y_cap), max(Y_cap), min(L), max(L)
    hK, hY, hL = K[1] - K[0], Y_cap[1] - Y_cap[0], L[1]-L[0]
    nK, nY, nL = len(K), len(Y_cap), len(L)
    

    print("K_min={},K_max={},Y_min={},Y_max={},L_min={},L_max={}" .format(K_min, K_max, Y_min, Y_max, L_min, L_max))
    print("hK={},hY={},hL={}" .format(hK, hY, hL))
    print("nK={},nY={},nL={}" .format(nK, nY, nL))


    (K_mat, Y_mat, L_mat) = np.meshgrid(K, Y_cap, L, indexing = 'ij')


    Intensity=Damage_Intensity(Y_mat)
    

    theta_ell_orig, psi_2_orig = variable_args
    
    theta_ell = theta_ell_orig[:, :, :n_bar+1, :]
    psi_2 = psi_2_orig[:, :, :n_bar+1, :]
    


    # args = (delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a, xi_g, xi_p)
    # variable_args = (theta_ell_wakeup, psi_2_wakeup)
    # controls=(i,e,x, g_tech, g_damage,  pi_c_o_wakeup, pi_c_wakeup, pi_d_o_wakeup, v0, v_i, v_tech2)

    # ME_base = minimize_void(grid, args, variable_args, controls, tol=1e-6, epsilon=0.3, max_iter=10000)



    ii_orig, ee_orig, xx_orig, g_tech_orig, g_damage_orig, pi_c_o_orig, pi_c_orig, pi_d_o_orig, v0_orig, v_i_orig, V_post_tech_orig = controls


    pi_c_o = pi_c_o_orig[:, :, :n_bar+1, :]
    pi_c = pi_c_orig[:, :, :n_bar+1, :]
    pi_d_o = pi_d_o_orig[:, :, :n_bar+1, :]
    
    # n_dmg = len(gamma_3_list)
    # n_climate = len(pi_c_orig)
    
    ii = ii_orig[:, :n_bar+1, :]
    ee = ee_orig[:, :n_bar+1, :]
    xx = xx_orig[:, :n_bar+1, :]
    
    g_tech = g_tech_orig[:, :n_bar+1, :]
    V_post_tech = V_post_tech_orig[:, :n_bar+1, :]

    v0 = v0_orig[:, :n_bar+1, :]
    
    v_i = v_i_orig[:, :, :n_bar+1, :]
    g_damage = g_damage_orig[:, :, :n_bar+1, :]
    pi_c = pi_c_orig[:, :, :n_bar+1, :]

    FC_Err   = 1
    epoch    = 0
        # For PETSc
    K_mat_1d = K_mat.ravel(order='F')
    Y_mat_1d = Y_mat.ravel(order='F')
    L_mat_1d = L_mat.ravel(order='F')
    
    lowerLims = np.array([K_min, Y_min, L_min], dtype=np.float64)
    upperLims = np.array([K_max, Y_max, L_max], dtype=np.float64)

    dVec = np.array([hK, hY, hL])
    increVec = np.array([1, nK, nK * nY],dtype=np.int32)


    petsc_mat = PETSc.Mat().create()
    petsc_mat.setType('aij')
    petsc_mat.setSizes([nK * nY * nL, nK * nY * nL])
    petsc_mat.setPreallocationNNZ(13)
    petsc_mat.setUp()
    ksp = PETSc.KSP()
    ksp.create(PETSc.COMM_WORLD)
    ksp.setType('bcgs')
    ksp.getPC().setType('ilu')
    ksp.setFromOptions()

    while FC_Err > tol and epoch < max_iter:
        
        start_ep = time.time()

        
        dVdK  = finiteDiff_3D(v0,0,1,hK)
        dVdK[dVdK <= 1e-16] = 1e-16
        dK = dVdK
        dVdY  = finiteDiff_3D(v0,1,1,hY)
        dY = dVdY
        dVdL  = finiteDiff_3D(v0,2,1,hL)
        dVdL[dVdL <= 1e-16] = 1e-16
        dL = dVdL
        ######## second order
        ddVdK = finiteDiff_3D(v0,0,2,hK)
        ddVdY = finiteDiff_3D(v0,1,2,hY)
        ddY = ddVdY
        ddVdL = finiteDiff_3D(v0,2,2,hL)
        
        dG  = gamma_1 + gamma_2 * Y_mat
        ddG = gamma_2 
        G = dY -  dG

        ################## Control Updating######################
        # update pi_c
        log_pi_c_ratio = - G * ee * theta_ell/ xi_a
        log_pi_c_ratio += -dL * psi_0 * xx**psi_1 * np.exp( psi_1 * K_mat - (1-psi_2) * L_mat) / xi_a
        pi_c_ratio = log_pi_c_ratio - np.max(log_pi_c_ratio)
        pi_c = np.exp(pi_c_ratio) * pi_c_o
        pi_c = (pi_c <= 0) * 1e-16 + (pi_c > 0) * pi_c
        pi_c = pi_c / np.sum(pi_c, axis=0)
        entropy = np.sum(pi_c * (np.log(pi_c) - np.log(pi_c_o)), axis=0)

        # update g_dmg

        g_damage = np.exp(- (v_i-v0)/xi_p)

        # # update g_tech

        # g_tech = np.exp(1 / xi_g * (v0 - V_post_tech))
        # g_tech[g_tech <=1e-16] = 1e-16
        ################## Control Updating######################


        jj = alpha * vartheta_bar * (1 - ee / (alpha * lambda_bar * np.exp(K_mat)))**theta
        jj[jj <= 1e-16] = 1e-16
        consumption = alpha - ii - jj - xx
        
        
        A   = - delta * np.ones(K_mat.shape) - np.exp(  L_mat - np.log(448) ) * g_tech
        B_1 = mu_k + ii - 0.5 * kappa * ii**2 - 0.5 * sigma_k**2
        B_2 = np.sum(theta_ell* pi_c, axis=0) * ee
        # B_3 = - zeta + psi_0 * (xx * np.exp(K_mat - L_mat))**psi_1 - 0.5 * sigma_g**2
        B_3 = - zeta + psi_0 * xx** psi_1 * np.exp( psi_1 * K_mat ) * np.sum(pi_c * np.exp( -( 1-psi_2) * L_mat  ), axis=0 )- 0.5 * sigma_g**2

        C_1 = 0.5 * sigma_k**2 * np.ones(K_mat.shape)
        C_2 = 0.5 * sigma_y**2 * ee**2
        C_3 = 0.5 * sigma_g**2 * np.ones(K_mat.shape)
        D = delta * np.log(consumption) + delta * K_mat  - dG * np.sum(theta_ell* pi_c, axis=0) * ee  - 0.5 * ddG * sigma_y**2 * ee**2  + xi_a * entropy + xi_g * np.exp((L_mat - np.log(448))) * (1 - g_tech + g_tech * np.log(g_tech)) + np.exp( (L_mat - np.log(448)) ) * g_tech * V_post_tech

        D += xi_p * Intensity * np.sum( pi_d_o*(1-g_damage+g_damage*np.log(g_damage)),axis=0) +Intensity*np.sum(pi_d_o*g_damage*v_i,axis=0)
        A -=  Intensity*np.sum(pi_d_o*g_damage,axis=0)

        bpoint1 = time.time()

        A_1d   = A.ravel(order = 'F')
        C_1_1d = C_1.ravel(order = 'F')
        C_2_1d = C_2.ravel(order = 'F')
        C_3_1d = C_3.ravel(order = 'F')
        B_1_1d = B_1.ravel(order = 'F')
        B_2_1d = B_2.ravel(order = 'F')
        B_3_1d = B_3.ravel(order = 'F')
        D_1d   = D.ravel(order = 'F')
        v0_1d  = v0.ravel(order = 'F')
        


        petsclinearsystem.formLinearSystem(K_mat_1d, Y_mat_1d, L_mat_1d, A_1d, B_1_1d, B_2_1d, B_3_1d, C_1_1d, C_2_1d, C_3_1d, epsilon, lowerLims, upperLims, dVec, increVec, petsc_mat)
        b = v0_1d + D_1d * epsilon
        petsc_rhs = PETSc.Vec().createWithArray(b)
        x = petsc_mat.createVecRight()


        # create linear solver
        start_ksp = time.time()
        ksp.setOperators(petsc_mat)
        ksp.setTolerances(rtol=tol)
        ksp.solve(petsc_rhs, x)
        petsc_rhs.destroy()
        x.destroy()
        out_comp = np.array(ksp.getSolution()).reshape(A.shape,order = "F")
        end_ksp = time.time()
        num_iter = ksp.getIterationNumber()
        
        # print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
        # print("PETSc preconditioned residual norm is {:g}; iterations: {}".format(ksp.getResidualNorm(), ksp.getIterationNumber()))
        PDE_rhs = A * v0 + B_1 * dVdK + B_2 * dVdY + B_3 * dVdL + C_1 * ddVdK + C_2 * ddVdY + C_3 * ddVdL + D
        PDE_Err = np.max(abs(PDE_rhs))
        FC_Err = np.max(abs((out_comp - v0)/ epsilon))
        
        # if FC_Err < 2*tol:
            
        #     print("-----------------------------------")
        #     print("---------Epoch {}---------------".format(epoch))
        #     print("-----------------------------------")
        #     print("min i: {},\t max i: {}\t".format(ii.min(), ii.max()))
        #     print("min e: {},\t max e: {}\t".format(ee.min(), ee.max()))
        #     print("min x: {},\t max x: {}\t".format(xx.min(), xx.max()))
        #     print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
        #     print("Epoch {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))
        #     print("Epoch time: {:.4f}".format(time.time() - start_ep))
        # elif epoch%100==0:
            
        #     print("-----------------------------------")
        #     print("---------Epoch {}---------------".format(epoch))
        #     print("-----------------------------------")
        #     print("min i: {},\t max i: {}\t".format(ii.min(), ii.max()))
        #     print("min e: {},\t max e: {}\t".format(ee.min(), ee.max()))
        #     print("min x: {},\t max x: {}\t".format(xx.min(), xx.max()))
        #     print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
        #     print("Epoch {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))
        #     print("Epoch time: {:.4f}".format(time.time() - start_ep))
          
          
        v0     = out_comp
        epoch += 1
        
    print("-----------------------------------")
    print("---------Converged, Epoch {}---------------".format(epoch))
    print("-----------------------------------")
    print("min i: {},\t max i: {}\t".format(ii.min(), ii.max()))
    print("min e: {},\t max e: {}\t".format(ee.min(), ee.max()))
    print("min x: {},\t max x: {}\t".format(xx.min(), xx.max()))
    print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
    print("Epoch {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))
    print("Epoch time: {:.4f}".format(time.time() - start_ep))
        
        
    ME = - dVdY * np.sum(pi_c * theta_ell, axis=0) - ddVdY * sigma_y**2 * ee + dG * np.sum(theta_ell * pi_c, axis=0) +  ddG * sigma_y**2 * ee

    
    return ME



def minimize_pi_c_tech(grid=(),  args = (), n_bar=(), variable_args= (), controls = (), tol=1e-6, epsilon=0.3, max_iter=10000):
    """
    compute jump model with ambiguity over climate models
    """

    K, Y, L = grid

    

    delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a, xi_g, xi_p = args
    # n_bar, = n_bar
    # n_bar = len(Y)-1
    # n_bar = np.abs(Y - y_bar).argmin()

    Y_cap = Y[:n_bar+1]
    
    K_min, K_max, Y_min, Y_max, L_min, L_max = min(K), max(K), min(Y_cap), max(Y_cap), min(L), max(L)
    hK, hY, hL = K[1] - K[0], Y_cap[1] - Y_cap[0], L[1]-L[0]
    nK, nY, nL = len(K), len(Y_cap), len(L)
    

    print("K_min={},K_max={},Y_min={},Y_max={},L_min={},L_max={}" .format(K_min, K_max, Y_min, Y_max, L_min, L_max))
    print("hK={},hY={},hL={}" .format(hK, hY, hL))
    print("nK={},nY={},nL={}" .format(nK, nY, nL))


    (K_mat, Y_mat, L_mat) = np.meshgrid(K, Y_cap, L, indexing = 'ij')


    Intensity=Damage_Intensity(Y_mat)
    

    theta_ell_orig, psi_2_orig = variable_args
    
    theta_ell = theta_ell_orig[:, :, :n_bar+1, :]
    psi_2 = psi_2_orig[:, :, :n_bar+1, :]
    


    # args = (delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a, xi_g, xi_p)
    # variable_args = (theta_ell_wakeup, psi_2_wakeup)
    # controls=(i,e,x, g_tech, g_damage,  pi_c_o_wakeup, pi_c_wakeup, pi_d_o_wakeup, v0, v_i, v_tech2)

    # ME_base = minimize_void(grid, args, variable_args, controls, tol=1e-6, epsilon=0.3, max_iter=10000)



    ii_orig, ee_orig, xx_orig, g_tech_orig, g_damage_orig, pi_c_o_orig, pi_c_orig, pi_d_o_orig, v0_orig, v_i_orig, V_post_tech_orig = controls


    pi_c_o = pi_c_o_orig[:, :, :n_bar+1, :]
    pi_c = pi_c_orig[:, :, :n_bar+1, :]
    pi_d_o = pi_d_o_orig[:, :, :n_bar+1, :]
    
    # n_dmg = len(gamma_3_list)
    # n_climate = len(pi_c_orig)
    
    ii = ii_orig[:, :n_bar+1, :]
    ee = ee_orig[:, :n_bar+1, :]
    xx = xx_orig[:, :n_bar+1, :]
    
    g_tech = g_tech_orig[:, :n_bar+1, :]
    V_post_tech = V_post_tech_orig[:, :n_bar+1, :]

    v0 = v0_orig[:, :n_bar+1, :]
    
    v_i = v_i_orig[:, :, :n_bar+1, :]
    g_damage = g_damage_orig[:, :, :n_bar+1, :]
    pi_c = pi_c_orig[:, :, :n_bar+1, :]

    FC_Err   = 1
    epoch    = 0
        # For PETSc
    K_mat_1d = K_mat.ravel(order='F')
    Y_mat_1d = Y_mat.ravel(order='F')
    L_mat_1d = L_mat.ravel(order='F')
    
    lowerLims = np.array([K_min, Y_min, L_min], dtype=np.float64)
    upperLims = np.array([K_max, Y_max, L_max], dtype=np.float64)

    dVec = np.array([hK, hY, hL])
    increVec = np.array([1, nK, nK * nY],dtype=np.int32)


    petsc_mat = PETSc.Mat().create()
    petsc_mat.setType('aij')
    petsc_mat.setSizes([nK * nY * nL, nK * nY * nL])
    petsc_mat.setPreallocationNNZ(13)
    petsc_mat.setUp()
    ksp = PETSc.KSP()
    ksp.create(PETSc.COMM_WORLD)
    ksp.setType('bcgs')
    ksp.getPC().setType('ilu')
    ksp.setFromOptions()

    while FC_Err > tol and epoch < max_iter:
        
        start_ep = time.time()

        
        dVdK  = finiteDiff_3D(v0,0,1,hK)
        dVdK[dVdK <= 1e-16] = 1e-16
        dK = dVdK
        dVdY  = finiteDiff_3D(v0,1,1,hY)
        dY = dVdY
        dVdL  = finiteDiff_3D(v0,2,1,hL)
        dVdL[dVdL <= 1e-16] = 1e-16
        dL = dVdL
        ######## second order
        ddVdK = finiteDiff_3D(v0,0,2,hK)
        ddVdY = finiteDiff_3D(v0,1,2,hY)
        ddY = ddVdY
        ddVdL = finiteDiff_3D(v0,2,2,hL)
        
        dG  = gamma_1 + gamma_2 * Y_mat
        ddG = gamma_2 
        G = dY -  dG

        ################## Control Updating######################
        # update pi_c
        log_pi_c_ratio = - G * ee * theta_ell/ xi_a
        log_pi_c_ratio += -dL * psi_0 * xx**psi_1 * np.exp( psi_1 * K_mat - (1-psi_2) * L_mat) / xi_a
        pi_c_ratio = log_pi_c_ratio - np.max(log_pi_c_ratio)
        pi_c = np.exp(pi_c_ratio) * pi_c_o
        pi_c = (pi_c <= 0) * 1e-16 + (pi_c > 0) * pi_c
        pi_c = pi_c / np.sum(pi_c, axis=0)
        entropy = np.sum(pi_c * (np.log(pi_c) - np.log(pi_c_o)), axis=0)

        # update g_dmg

        # g_damage = np.exp(- (v_i-v0)/xi_p)

        # # update g_tech

        g_tech = np.exp(1 / xi_g * (v0 - V_post_tech))
        g_tech[g_tech <=1e-16] = 1e-16
        ################## Control Updating######################


        jj = alpha * vartheta_bar * (1 - ee / (alpha * lambda_bar * np.exp(K_mat)))**theta
        jj[jj <= 1e-16] = 1e-16
        consumption = alpha - ii - jj - xx
        
        
        A   = - delta * np.ones(K_mat.shape) - np.exp(  L_mat - np.log(448) ) * g_tech
        B_1 = mu_k + ii - 0.5 * kappa * ii**2 - 0.5 * sigma_k**2
        B_2 = np.sum(theta_ell* pi_c, axis=0) * ee
        # B_3 = - zeta + psi_0 * (xx * np.exp(K_mat - L_mat))**psi_1 - 0.5 * sigma_g**2
        B_3 = - zeta + psi_0 * xx** psi_1 * np.exp( psi_1 * K_mat ) * np.sum(pi_c * np.exp( -( 1-psi_2) * L_mat  ), axis=0 )- 0.5 * sigma_g**2

        C_1 = 0.5 * sigma_k**2 * np.ones(K_mat.shape)
        C_2 = 0.5 * sigma_y**2 * ee**2
        C_3 = 0.5 * sigma_g**2 * np.ones(K_mat.shape)
        D = delta * np.log(consumption) + delta * K_mat  - dG * np.sum(theta_ell* pi_c, axis=0) * ee  - 0.5 * ddG * sigma_y**2 * ee**2  + xi_a * entropy + xi_g * np.exp((L_mat - np.log(448))) * (1 - g_tech + g_tech * np.log(g_tech)) + np.exp( (L_mat - np.log(448)) ) * g_tech * V_post_tech

        D += xi_p * Intensity * np.sum( pi_d_o*(1-g_damage+g_damage*np.log(g_damage)),axis=0) +Intensity*np.sum(pi_d_o*g_damage*v_i,axis=0)
        A -=  Intensity*np.sum(pi_d_o*g_damage,axis=0)

        bpoint1 = time.time()

        A_1d   = A.ravel(order = 'F')
        C_1_1d = C_1.ravel(order = 'F')
        C_2_1d = C_2.ravel(order = 'F')
        C_3_1d = C_3.ravel(order = 'F')
        B_1_1d = B_1.ravel(order = 'F')
        B_2_1d = B_2.ravel(order = 'F')
        B_3_1d = B_3.ravel(order = 'F')
        D_1d   = D.ravel(order = 'F')
        v0_1d  = v0.ravel(order = 'F')
        


        petsclinearsystem.formLinearSystem(K_mat_1d, Y_mat_1d, L_mat_1d, A_1d, B_1_1d, B_2_1d, B_3_1d, C_1_1d, C_2_1d, C_3_1d, epsilon, lowerLims, upperLims, dVec, increVec, petsc_mat)
        b = v0_1d + D_1d * epsilon
        petsc_rhs = PETSc.Vec().createWithArray(b)
        x = petsc_mat.createVecRight()


        # create linear solver
        start_ksp = time.time()
        ksp.setOperators(petsc_mat)
        ksp.setTolerances(rtol=tol)
        ksp.solve(petsc_rhs, x)
        petsc_rhs.destroy()
        x.destroy()
        out_comp = np.array(ksp.getSolution()).reshape(A.shape,order = "F")
        end_ksp = time.time()
        num_iter = ksp.getIterationNumber()
        
        # print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
        # print("PETSc preconditioned residual norm is {:g}; iterations: {}".format(ksp.getResidualNorm(), ksp.getIterationNumber()))
        PDE_rhs = A * v0 + B_1 * dVdK + B_2 * dVdY + B_3 * dVdL + C_1 * ddVdK + C_2 * ddVdY + C_3 * ddVdL + D
        PDE_Err = np.max(abs(PDE_rhs))
        FC_Err = np.max(abs((out_comp - v0)/ epsilon))
        
        # if FC_Err < 2*tol:
            
        #     print("-----------------------------------")
        #     print("---------Epoch {}---------------".format(epoch))
        #     print("-----------------------------------")
        #     print("min i: {},\t max i: {}\t".format(ii.min(), ii.max()))
        #     print("min e: {},\t max e: {}\t".format(ee.min(), ee.max()))
        #     print("min x: {},\t max x: {}\t".format(xx.min(), xx.max()))
        #     print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
        #     print("Epoch {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))
        #     print("Epoch time: {:.4f}".format(time.time() - start_ep))
        # elif epoch%100==0:
            
        #     print("-----------------------------------")
        #     print("---------Epoch {}---------------".format(epoch))
        #     print("-----------------------------------")
        #     print("min i: {},\t max i: {}\t".format(ii.min(), ii.max()))
        #     print("min e: {},\t max e: {}\t".format(ee.min(), ee.max()))
        #     print("min x: {},\t max x: {}\t".format(xx.min(), xx.max()))
        #     print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
        #     print("Epoch {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))
        #     print("Epoch time: {:.4f}".format(time.time() - start_ep))
          
          
        v0     = out_comp
        epoch += 1
        
    print("-----------------------------------")
    print("---------Converged, Epoch {}---------------".format(epoch))
    print("-----------------------------------")
    print("min i: {},\t max i: {}\t".format(ii.min(), ii.max()))
    print("min e: {},\t max e: {}\t".format(ee.min(), ee.max()))
    print("min x: {},\t max x: {}\t".format(xx.min(), xx.max()))
    print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
    print("Epoch {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))
    print("Epoch time: {:.4f}".format(time.time() - start_ep))
        
        
    ME = - dVdY * np.sum(pi_c * theta_ell, axis=0) - ddVdY * sigma_y**2 * ee + dG * np.sum(theta_ell * pi_c, axis=0) +  ddG * sigma_y**2 * ee

    
    return ME


def minimize_dmg_tech(grid=(),  args = (), n_bar=(), variable_args= (), controls = (), tol=1e-6, epsilon=0.3, max_iter=10000):
    """
    compute jump model with ambiguity over climate models
    """

    K, Y, L = grid

    

    delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a, xi_g, xi_p = args
    # n_bar, = n_bar
    # n_bar = len(Y)-1
    # n_bar = np.abs(Y - y_bar).argmin()

    Y_cap = Y[:n_bar+1]
    
    K_min, K_max, Y_min, Y_max, L_min, L_max = min(K), max(K), min(Y_cap), max(Y_cap), min(L), max(L)
    hK, hY, hL = K[1] - K[0], Y_cap[1] - Y_cap[0], L[1]-L[0]
    nK, nY, nL = len(K), len(Y_cap), len(L)
    

    print("K_min={},K_max={},Y_min={},Y_max={},L_min={},L_max={}" .format(K_min, K_max, Y_min, Y_max, L_min, L_max))
    print("hK={},hY={},hL={}" .format(hK, hY, hL))
    print("nK={},nY={},nL={}" .format(nK, nY, nL))


    (K_mat, Y_mat, L_mat) = np.meshgrid(K, Y_cap, L, indexing = 'ij')


    Intensity=Damage_Intensity(Y_mat)
    

    theta_ell_orig, psi_2_orig = variable_args
    
    theta_ell = theta_ell_orig[:, :, :n_bar+1, :]
    psi_2 = psi_2_orig[:, :, :n_bar+1, :]
    


    # args = (delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a, xi_g, xi_p)
    # variable_args = (theta_ell_wakeup, psi_2_wakeup)
    # controls=(i,e,x, g_tech, g_damage,  pi_c_o_wakeup, pi_c_wakeup, pi_d_o_wakeup, v0, v_i, v_tech2)

    # ME_base = minimize_void(grid, args, variable_args, controls, tol=1e-6, epsilon=0.3, max_iter=10000)



    ii_orig, ee_orig, xx_orig, g_tech_orig, g_damage_orig, pi_c_o_orig, pi_c_orig, pi_d_o_orig, v0_orig, v_i_orig, V_post_tech_orig = controls


    pi_c_o = pi_c_o_orig[:, :, :n_bar+1, :]
    pi_c = pi_c_orig[:, :, :n_bar+1, :]
    pi_d_o = pi_d_o_orig[:, :, :n_bar+1, :]
    
    # n_dmg = len(gamma_3_list)
    # n_climate = len(pi_c_orig)
    
    ii = ii_orig[:, :n_bar+1, :]
    ee = ee_orig[:, :n_bar+1, :]
    xx = xx_orig[:, :n_bar+1, :]
    
    g_tech = g_tech_orig[:, :n_bar+1, :]
    V_post_tech = V_post_tech_orig[:, :n_bar+1, :]

    v0 = v0_orig[:, :n_bar+1, :]
    
    v_i = v_i_orig[:, :, :n_bar+1, :]
    g_damage = g_damage_orig[:, :, :n_bar+1, :]
    pi_c = pi_c_orig[:, :, :n_bar+1, :]

    FC_Err   = 1
    epoch    = 0
        # For PETSc
    K_mat_1d = K_mat.ravel(order='F')
    Y_mat_1d = Y_mat.ravel(order='F')
    L_mat_1d = L_mat.ravel(order='F')
    
    lowerLims = np.array([K_min, Y_min, L_min], dtype=np.float64)
    upperLims = np.array([K_max, Y_max, L_max], dtype=np.float64)

    dVec = np.array([hK, hY, hL])
    increVec = np.array([1, nK, nK * nY],dtype=np.int32)


    petsc_mat = PETSc.Mat().create()
    petsc_mat.setType('aij')
    petsc_mat.setSizes([nK * nY * nL, nK * nY * nL])
    petsc_mat.setPreallocationNNZ(13)
    petsc_mat.setUp()
    ksp = PETSc.KSP()
    ksp.create(PETSc.COMM_WORLD)
    ksp.setType('bcgs')
    ksp.getPC().setType('ilu')
    ksp.setFromOptions()

    while FC_Err > tol and epoch < max_iter:
        
        start_ep = time.time()

        
        dVdK  = finiteDiff_3D(v0,0,1,hK)
        dVdK[dVdK <= 1e-16] = 1e-16
        dK = dVdK
        dVdY  = finiteDiff_3D(v0,1,1,hY)
        dY = dVdY
        dVdL  = finiteDiff_3D(v0,2,1,hL)
        dVdL[dVdL <= 1e-16] = 1e-16
        dL = dVdL
        ######## second order
        ddVdK = finiteDiff_3D(v0,0,2,hK)
        ddVdY = finiteDiff_3D(v0,1,2,hY)
        ddY = ddVdY
        ddVdL = finiteDiff_3D(v0,2,2,hL)
        
        dG  = gamma_1 + gamma_2 * Y_mat
        ddG = gamma_2 
        G = dY -  dG

        ################## Control Updating######################
        # update pi_c
        # log_pi_c_ratio = - G * ee * theta_ell/ xi_a
        # log_pi_c_ratio += -dL * psi_0 * xx**psi_1 * np.exp( psi_1 * K_mat - (1-psi_2) * L_mat) / xi_a
        # pi_c_ratio = log_pi_c_ratio - np.max(log_pi_c_ratio)
        # pi_c = np.exp(pi_c_ratio) * pi_c_o
        # pi_c = (pi_c <= 0) * 1e-16 + (pi_c > 0) * pi_c
        # pi_c = pi_c / np.sum(pi_c, axis=0)
        entropy = np.sum(pi_c * (np.log(pi_c) - np.log(pi_c_o)), axis=0)

        # update g_dmg

        g_damage = np.exp(- (v_i-v0)/xi_p)

        # # update g_tech

        g_tech = np.exp(1 / xi_g * (v0 - V_post_tech))
        g_tech[g_tech <=1e-16] = 1e-16
        ################## Control Updating######################


        jj = alpha * vartheta_bar * (1 - ee / (alpha * lambda_bar * np.exp(K_mat)))**theta
        jj[jj <= 1e-16] = 1e-16
        consumption = alpha - ii - jj - xx
        
        
        A   = - delta * np.ones(K_mat.shape) - np.exp(  L_mat - np.log(448) ) * g_tech
        B_1 = mu_k + ii - 0.5 * kappa * ii**2 - 0.5 * sigma_k**2
        B_2 = np.sum(theta_ell* pi_c, axis=0) * ee
        # B_3 = - zeta + psi_0 * (xx * np.exp(K_mat - L_mat))**psi_1 - 0.5 * sigma_g**2
        B_3 = - zeta + psi_0 * xx** psi_1 * np.exp( psi_1 * K_mat ) * np.sum(pi_c * np.exp( -( 1-psi_2) * L_mat  ), axis=0 )- 0.5 * sigma_g**2

        C_1 = 0.5 * sigma_k**2 * np.ones(K_mat.shape)
        C_2 = 0.5 * sigma_y**2 * ee**2
        C_3 = 0.5 * sigma_g**2 * np.ones(K_mat.shape)
        D = delta * np.log(consumption) + delta * K_mat  - dG * np.sum(theta_ell* pi_c, axis=0) * ee  - 0.5 * ddG * sigma_y**2 * ee**2  + xi_a * entropy + xi_g * np.exp((L_mat - np.log(448))) * (1 - g_tech + g_tech * np.log(g_tech)) + np.exp( (L_mat - np.log(448)) ) * g_tech * V_post_tech

        D += xi_p * Intensity * np.sum( pi_d_o*(1-g_damage+g_damage*np.log(g_damage)),axis=0) +Intensity*np.sum(pi_d_o*g_damage*v_i,axis=0)
        A -=  Intensity*np.sum(pi_d_o*g_damage,axis=0)

        bpoint1 = time.time()

        A_1d   = A.ravel(order = 'F')
        C_1_1d = C_1.ravel(order = 'F')
        C_2_1d = C_2.ravel(order = 'F')
        C_3_1d = C_3.ravel(order = 'F')
        B_1_1d = B_1.ravel(order = 'F')
        B_2_1d = B_2.ravel(order = 'F')
        B_3_1d = B_3.ravel(order = 'F')
        D_1d   = D.ravel(order = 'F')
        v0_1d  = v0.ravel(order = 'F')
        


        petsclinearsystem.formLinearSystem(K_mat_1d, Y_mat_1d, L_mat_1d, A_1d, B_1_1d, B_2_1d, B_3_1d, C_1_1d, C_2_1d, C_3_1d, epsilon, lowerLims, upperLims, dVec, increVec, petsc_mat)
        b = v0_1d + D_1d * epsilon
        petsc_rhs = PETSc.Vec().createWithArray(b)
        x = petsc_mat.createVecRight()


        # create linear solver
        start_ksp = time.time()
        ksp.setOperators(petsc_mat)
        ksp.setTolerances(rtol=tol)
        ksp.solve(petsc_rhs, x)
        petsc_rhs.destroy()
        x.destroy()
        out_comp = np.array(ksp.getSolution()).reshape(A.shape,order = "F")
        end_ksp = time.time()
        num_iter = ksp.getIterationNumber()
        
        # print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
        # print("PETSc preconditioned residual norm is {:g}; iterations: {}".format(ksp.getResidualNorm(), ksp.getIterationNumber()))
        PDE_rhs = A * v0 + B_1 * dVdK + B_2 * dVdY + B_3 * dVdL + C_1 * ddVdK + C_2 * ddVdY + C_3 * ddVdL + D
        PDE_Err = np.max(abs(PDE_rhs))
        FC_Err = np.max(abs((out_comp - v0)/ epsilon))
        
        # if FC_Err < 2*tol:
            
        #     print("-----------------------------------")
        #     print("---------Epoch {}---------------".format(epoch))
        #     print("-----------------------------------")
        #     print("min i: {},\t max i: {}\t".format(ii.min(), ii.max()))
        #     print("min e: {},\t max e: {}\t".format(ee.min(), ee.max()))
        #     print("min x: {},\t max x: {}\t".format(xx.min(), xx.max()))
        #     print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
        #     print("Epoch {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))
        #     print("Epoch time: {:.4f}".format(time.time() - start_ep))
        # elif epoch%100==0:
            
        #     print("-----------------------------------")
        #     print("---------Epoch {}---------------".format(epoch))
        #     print("-----------------------------------")
        #     print("min i: {},\t max i: {}\t".format(ii.min(), ii.max()))
        #     print("min e: {},\t max e: {}\t".format(ee.min(), ee.max()))
        #     print("min x: {},\t max x: {}\t".format(xx.min(), xx.max()))
        #     print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
        #     print("Epoch {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))
        #     print("Epoch time: {:.4f}".format(time.time() - start_ep))
          
          
        v0     = out_comp
        epoch += 1
        
    print("-----------------------------------")
    print("---------Converged, Epoch {}---------------".format(epoch))
    print("-----------------------------------")
    print("min i: {},\t max i: {}\t".format(ii.min(), ii.max()))
    print("min e: {},\t max e: {}\t".format(ee.min(), ee.max()))
    print("min x: {},\t max x: {}\t".format(xx.min(), xx.max()))
    print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
    print("Epoch {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))
    print("Epoch time: {:.4f}".format(time.time() - start_ep))
        
        
    ME = - dVdY * np.sum(pi_c * theta_ell, axis=0) - ddVdY * sigma_y**2 * ee + dG * np.sum(theta_ell * pi_c, axis=0) +  ddG * sigma_y**2 * ee

    
    return ME




def simulate_pre_UD(
    grid = (), model_args = (), controls = (),  initial=(np.log(85/0.115), 1.1, 2.4), 
    T0=0, T=40, dt=1/12,
    printing=True):

    K, Y, L = grid
    
    K_min, K_max, Y_min, Y_max, L_min, L_max = min(K), max(K), min(Y), max(Y), min(L), max(L)
    hK, hY, hL = K[1] - K[0], Y[1] - Y[0], L[1]-L[0]
    nK, nY, nL = len(K), len(Y), len(L)
    

    (K_mat, Y_mat, L_mat) = np.meshgrid(K, Y, L, indexing = 'ij')



    delta, mu_k, kappa, sigma_k, beta_f, zeta, psi_0, psi_1, psi_2, sigma_g, theta, lambda_bar, vartheta_bar, xi_a, xi_g, xi_p = model_args


    ii, ee, xx, g_tech, g_damage, pi_c, v0, v_i, v_tech2 = controls
    print("-------------import parameters--------------")
    print("---------------------------")
    print("---------------------------")
    print("---------------------------")
    print("---------------------------")
    print("---------------------------")
    print("psi_2={},xi_a={},xi_g={}" .format(psi_2, xi_a, xi_g))
    print("---------------------------")
    print("---------------------------")
    print("---------------------------")
    print("---------------------------")
    
    print("K_min={},K_max={},Y_min={},Y_max={},L_min={},L_max={}" .format(K.min(),K.max(),Y.min(),Y.max(),L.min(),L.max()))
    print("hK={},hY={},hL={}" .format(hK, hY, hL))
    print("nK={},nY={},nL={}" .format(nK, nY, nL))

    n_climate = len(pi_c)

    method = 'linear'
    years  = np.arange(T0, T0 + T + dt, dt)
    pers   = len(years)
       
    # some parameters remaiend unchanged across runs
    gamma_1  = 0.00017675
    gamma_2  = 2. * 0.0022
    beta_f   = 1.86 / 1000
    sigma_y  = 1.2 * 1.86 / 1000
    
    theta_ell_array = pd.read_csv("./data/model144_p.csv", header=None).to_numpy()[:, 0]/1000.
    psi_2_array = pd.read_csv('./data/psi2value_p.csv', header=None).to_numpy()[:, 0]
    pi_d_o = np.ones(len(gamma_3_list)) / len(gamma_3_list)
    pi_c_o = np.ones(len(theta_ell_array)) / len(theta_ell_array)

    theta_ell_wakeup = np.array([temp * np.ones(K_mat.shape) for temp in theta_ell_array])
    psi_2_wakeup = np.array([temp * np.ones(K_mat.shape) for temp in psi_2_array])
    pi_d_o_wakeup = np.array([temp * np.ones(K_mat.shape) for temp in pi_d_o ])
    pi_c_o_wakeup = np.array([temp * np.ones((nK, nY, nL)) for temp in pi_c_o])


    # Uncertainty decomposition
    n_temp = 16
    n_carb = 9
    n_RD = 3
    theta_ell_reshape = theta_ell_wakeup.reshape(n_temp, n_carb,n_RD,nK, nY, nL)
    theta_ell_temp = np.mean(theta_ell_reshape, axis=(1,2))
    theta_ell_carb = np.mean(theta_ell_reshape, axis=(0,2))
    theta_ell_RD = np.mean(theta_ell_reshape, axis=(0,1))

    theta_ell_tempcarb = np.mean(theta_ell_reshape, axis=2)
    theta_ell_carbRD = np.mean(theta_ell_reshape, axis=0)
    theta_ell_RDtemp = np.mean(theta_ell_reshape, axis=1)



    theta_ell_temp_wakeup = theta_ell_temp.reshape(n_temp,nK, nY, nL)
    theta_ell_carb_wakeup = theta_ell_carb.reshape(n_carb, nK, nY, nL)
    theta_ell_RD_wakeup = theta_ell_RD.reshape(n_RD, nK, nY, nL)


    theta_ell_tempcarb_wakeup = theta_ell_tempcarb.reshape(n_temp*n_carb,nK, nY, nL)
    theta_ell_carbRD_wakeup = theta_ell_carbRD.reshape(n_carb*n_RD, nK, nY, nL)
    theta_ell_RDtemp_wakeup = theta_ell_RDtemp.reshape(n_RD*n_temp, nK, nY, nL)



    psi_2_reshape = psi_2_wakeup.reshape(n_temp, n_carb,n_RD,nK, nY, nL)
    psi_2_temp = np.mean(psi_2_reshape, axis=(1,2))
    psi_2_carb = np.mean(psi_2_reshape, axis=(0,2))
    psi_2_RD = np.mean(psi_2_reshape, axis=(0,1))


    psi_2_tempcarb = np.mean(psi_2_reshape, axis=2)
    psi_2_carbRD = np.mean(psi_2_reshape, axis=0)
    psi_2_RDtemp = np.mean(psi_2_reshape, axis=1)


    psi_2_temp_wakeup = psi_2_temp.reshape(n_temp,nK, nY, nL)
    psi_2_carb_wakeup = psi_2_carb.reshape(n_carb, nK, nY, nL)
    psi_2_RD_wakeup = psi_2_RD.reshape(n_RD, nK, nY, nL)

    psi_2_tempcarb_wakeup = psi_2_tempcarb.reshape(n_temp*n_carb,nK, nY, nL)
    psi_2_carbRD_wakeup = psi_2_carbRD.reshape(n_carb*n_RD, nK, nY, nL)
    psi_2_RDtemp_wakeup = psi_2_RDtemp.reshape(n_RD*n_temp, nK, nY, nL)



    pi_c_wakeup = pi_c
    
    pi_c_reshape = pi_c.reshape(n_temp, n_carb,n_RD, nK, nY, nL)
    pi_c_temp = np.mean(pi_c_reshape, axis=(1,2))
    pi_c_carb = np.mean(pi_c_reshape, axis=(0,2))
    pi_c_RD = np.mean(pi_c_reshape, axis=(0,1))

    pi_c_tempcarb = np.mean(pi_c_reshape, axis=2)
    pi_c_carbRD = np.mean(pi_c_reshape, axis=0)
    pi_c_RDtemp = np.mean(pi_c_reshape, axis=1)
    
    
    pi_c_temp_wakeup = pi_c_temp.reshape(n_temp,nK, nY, nL)
    pi_c_carb_wakeup = pi_c_carb.reshape(n_carb, nK, nY, nL)
    pi_c_RD_wakeup = pi_c_RD.reshape(n_RD, nK, nY, nL)

    pi_c_tempcarb_wakeup = pi_c_tempcarb.reshape(n_temp*n_carb,nK, nY, nL)
    pi_c_carbRD_wakeup = pi_c_carbRD.reshape(n_carb*n_RD, nK, nY, nL)
    pi_c_RDtemp_wakeup = pi_c_RDtemp.reshape(n_RD*n_temp, nK, nY, nL)


    pi_c_o_temp = np.sum(pi_c_reshape, axis=(1,2))
    pi_c_o_carb = np.sum(pi_c_reshape, axis=(0,2))
    pi_c_o_RD = np.sum(pi_c_reshape, axis=(0,1))
    
    pi_c_o_tempcarb = np.sum(pi_c_reshape, axis=2)
    pi_c_o_carbRD = np.sum(pi_c_reshape, axis=0)
    pi_c_o_RDtemp = np.sum(pi_c_reshape, axis=1)
    
    
    
    pi_c_o_temp_wakeup =  pi_c_o_temp.reshape(n_temp,nK, nY, nL)
    pi_c_o_carb_wakeup = pi_c_o_carb.reshape(n_carb, nK, nY, nL)
    pi_c_o_RD_wakeup = pi_c_o_RD.reshape(n_RD, nK, nY, nL)
    
    pi_c_o_tempcarb_wakeup = pi_c_o_tempcarb.reshape(n_temp*n_carb,nK, nY, nL)
    pi_c_o_carbRD_wakeup = pi_c_o_carbRD.reshape(n_carb*n_RD, nK, nY, nL)
    pi_c_o_RDtemp_wakeup = pi_c_o_RDtemp.reshape(n_RD*n_temp, nK, nY, nL)

 
    
    
    
    n_bar1 = len(Y)-1
    n_bar2 = np.abs(Y - y_bar).argmin()





    jj = alpha * vartheta_bar * (1 - ee / (alpha * lambda_bar * np.exp(K_mat)))**theta
    jj[jj <= 1e-16] = 1e-16
    consumption = alpha - ii - jj - xx
    ME_total = delta/ consumption  * alpha * vartheta_bar * theta * (1 - ee / ( alpha * lambda_bar * np.exp(K_mat)))**(theta - 1) /( alpha * lambda_bar * np.exp(K_mat) )
    
    # base
    print("-------------Base Start--------------")

    args = (delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a, xi_g, xi_p)
    variable_args = (theta_ell_wakeup, psi_2_wakeup)
    controls=(ii,ee,xx, g_tech, g_damage,  pi_c_o_wakeup, pi_c_wakeup, pi_d_o_wakeup, v0, v_i, v_tech2)
    ME_base = minimize_void(grid, args, n_bar1, variable_args, controls, tol=1e-6, epsilon=0.3, max_iter=10000) # n_bar free
    ME_base2 = minimize_void(grid, args, n_bar2, variable_args, controls, tol=1e-6, epsilon=0.3, max_iter=10000) #n_bar hit 2

    

    print("Look at differences")
    print(ME_base.shape)
    print(ME_base2.shape)
    print(n_bar1,n_bar2)
    print(np.max(abs(ME_base[:,:n_bar2+1,:]-ME_base2)))
    print("Look at differences")
    print("-------------Base Done--------------")

    print("---------------------------")
    print("---------------------------")
    print("---------------------------")


    # Temp
    print("-------------Temp Start--------------")

    args = (delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a, xi_g, xi_p)
    variable_args = (theta_ell_temp_wakeup, psi_2_temp_wakeup)
    controls=(ii,ee,xx, g_tech, g_damage,  pi_c_o_temp_wakeup, pi_c_temp_wakeup, pi_d_o_wakeup, v0, v_i, v_tech2)
    ME_temp = minimize_pi_c(grid, args, n_bar1, variable_args, controls, tol=1e-6, epsilon=0.3, max_iter=10000) # n_bar free
    ME_temp2 = minimize_pi_c(grid, args, n_bar2, variable_args, controls, tol=1e-6, epsilon=0.3, max_iter=10000) #n_bar hit 2

    


    print("Look at differences")
    print(ME_temp.shape)
    print(ME_temp2.shape)
    print(n_bar1,n_bar2)
    print(np.max(abs(ME_temp[:,:n_bar2+1,:]-ME_temp2)))
    print("Look at differences")
    print("-------------Temp Done--------------")
    print("---------------------------")
    print("---------------------------")
    print("---------------------------")

    # Carb
    print("-------------Carb Start--------------")

    args = (delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a, xi_g, xi_p)
    variable_args = (theta_ell_carb_wakeup, psi_2_carb_wakeup)
    controls=(ii,ee,xx, g_tech, g_damage,  pi_c_o_carb_wakeup, pi_c_carb_wakeup, pi_d_o_wakeup, v0, v_i, v_tech2)
    ME_carb = minimize_pi_c(grid, args, n_bar1, variable_args, controls, tol=1e-6, epsilon=0.3, max_iter=10000) # n_bar free
    ME_carb2 = minimize_pi_c(grid, args, n_bar2, variable_args, controls, tol=1e-6, epsilon=0.3, max_iter=10000) #n_bar hit 2

    
    n_bar1 = len(Y)-1
    n_bar2 = np.abs(Y - y_bar).argmin()

    print("Look at differences")
    print(ME_carb.shape)
    print(ME_carb2.shape)
    print(n_bar1,n_bar2)
    print(np.max(abs(ME_carb[:,:n_bar2+1,:]-ME_carb2)))
    print("Look at differences")
    print("-------------Carb Done--------------")
    print("---------------------------")
    print("---------------------------")
    print("---------------------------")

    # RD
    print("-------------RD Start--------------")

    args = (delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a, xi_g, xi_p)
    variable_args = (theta_ell_RD_wakeup, psi_2_RD_wakeup)
    controls=(ii,ee,xx, g_tech, g_damage,  pi_c_o_RD_wakeup, pi_c_RD_wakeup, pi_d_o_wakeup, v0, v_i, v_tech2)
    ME_RD = minimize_pi_c(grid, args, n_bar1, variable_args, controls, tol=1e-6, epsilon=0.3, max_iter=10000) # n_bar free
    ME_RD2 = minimize_pi_c(grid, args, n_bar2, variable_args, controls, tol=1e-6, epsilon=0.3, max_iter=10000) #n_bar hit 2

    
    n_bar1 = len(Y)-1
    n_bar2 = np.abs(Y - y_bar).argmin()

    print("Look at differences")
    print(ME_RD.shape)
    print(ME_RD2.shape)
    print(n_bar1,n_bar2)
    print(np.max(abs(ME_RD[:,:n_bar2+1,:]-ME_RD2)))
    print("Look at differences")
    print("-------------RD Done--------------")

    print("---------------------------")
    print("---------------------------")
    print("---------------------------")

    # Damage
    print("-------------Damage Start--------------")

    args = (delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a, xi_g, xi_p)
    variable_args = (theta_ell_wakeup, psi_2_wakeup)
    controls=(ii,ee,xx, g_tech, g_damage,  pi_c_o_wakeup, pi_c_wakeup, pi_d_o_wakeup, v0, v_i, v_tech2)
    ME_dmg = minimize_dmg(grid, args, n_bar1, variable_args, controls, tol=1e-6, epsilon=0.3, max_iter=10000) # n_bar free
    ME_dmg2 = minimize_dmg(grid, args, n_bar2, variable_args, controls, tol=1e-6, epsilon=0.3, max_iter=10000) #n_bar hit 2

    
    n_bar1 = len(Y)-1
    n_bar2 = np.abs(Y - y_bar).argmin()

    print("Look at differences")
    print(ME_dmg.shape)
    print(ME_dmg2.shape)
    print(n_bar1,n_bar2)
    print(np.max(abs(ME_dmg[:,:n_bar2+1,:]-ME_dmg2)))
    print("Look at differences")
    print("-------------Damage Done--------------")
    print("---------------------------")
    print("---------------------------")
    print("---------------------------")

    # Tech
    print("-------------Tech Start--------------")

    args = (delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a, xi_g, xi_p)
    variable_args = (theta_ell_wakeup, psi_2_wakeup)
    controls=(ii,ee,xx, g_tech, g_damage,  pi_c_o_wakeup, pi_c_wakeup, pi_d_o_wakeup, v0, v_i, v_tech2)
    ME_tech = minimize_tech(grid, args, n_bar1, variable_args, controls, tol=1e-6, epsilon=0.3, max_iter=10000) # n_bar free
    ME_tech2 = minimize_tech(grid, args, n_bar2, variable_args, controls, tol=1e-6, epsilon=0.3, max_iter=10000) #n_bar hit 2

    
    n_bar1 = len(Y)-1
    n_bar2 = np.abs(Y - y_bar).argmin()

    print("Look at differences")
    print(ME_tech.shape)
    print(ME_tech2.shape)
    print(n_bar1,n_bar2)
    print(np.max(abs(ME_tech[:,:n_bar2+1,:]-ME_tech2)))
    print("Look at differences")
    print("-------------Tech Done--------------")

    print("---------------------------")
    print("---------------------------")
    print("---------------------------")


    # Omit Temp
    print("-------------Omit Temp Start--------------")

    args = (delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a, xi_g, xi_p)
    variable_args = (theta_ell_carbRD_wakeup, psi_2_carbRD_wakeup)
    controls=(ii,ee,xx, g_tech, g_damage,  pi_c_o_carbRD_wakeup, pi_c_carbRD_wakeup, pi_d_o_wakeup, v0, v_i, v_tech2)
    ME_notemp = minimize_pi_c_dmg_tech(grid, args, n_bar1, variable_args, controls, tol=1e-6, epsilon=0.3, max_iter=10000) # n_bar free
    ME_notemp2 = minimize_pi_c_dmg_tech(grid, args, n_bar2, variable_args, controls, tol=1e-6, epsilon=0.3, max_iter=10000) #n_bar hit 2

    


    print("Look at differences")
    print(ME_notemp.shape)
    print(ME_notemp2.shape)
    print(n_bar1,n_bar2)
    print(np.max(abs(ME_notemp[:,:n_bar2+1,:]-ME_notemp2)))
    print("Look at differences")
    print("-------------Omit Temp Done--------------")
    print("---------------------------")
    print("---------------------------")
    print("---------------------------")

    # Omit Carb
    print("-------------Omit Carb Start--------------")

    args = (delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a, xi_g, xi_p)
    variable_args = (theta_ell_RDtemp_wakeup, psi_2_RDtemp_wakeup)
    controls=(ii,ee,xx, g_tech, g_damage,  pi_c_o_RDtemp_wakeup, pi_c_RDtemp_wakeup, pi_d_o_wakeup, v0, v_i, v_tech2)
    ME_nocarb = minimize_pi_c_dmg_tech(grid, args, n_bar1, variable_args, controls, tol=1e-6, epsilon=0.3, max_iter=10000) # n_bar free
    ME_nocarb2 = minimize_pi_c_dmg_tech(grid, args, n_bar2, variable_args, controls, tol=1e-6, epsilon=0.3, max_iter=10000) #n_bar hit 2

    
    n_bar1 = len(Y)-1
    n_bar2 = np.abs(Y - y_bar).argmin()

    print("Look at differences")
    print(ME_nocarb.shape)
    print(ME_nocarb2.shape)
    print(n_bar1,n_bar2)
    print(np.max(abs(ME_nocarb[:,:n_bar2+1,:]-ME_nocarb2)))
    print("Look at differences")
    print("-------------Omit Carb Done--------------")
    print("---------------------------")
    print("---------------------------")
    print("---------------------------")

    # Omit RD
    print("-------------Omit RD Start--------------")

    args = (delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a, xi_g, xi_p)
    variable_args = (theta_ell_tempcarb_wakeup, psi_2_tempcarb_wakeup)
    controls=(ii,ee,xx, g_tech, g_damage,  pi_c_o_tempcarb_wakeup, pi_c_tempcarb_wakeup, pi_d_o_wakeup, v0, v_i, v_tech2)
    ME_noRD = minimize_pi_c(grid, args, n_bar1, variable_args, controls, tol=1e-6, epsilon=0.3, max_iter=10000) # n_bar free
    ME_noRD2 = minimize_pi_c(grid, args, n_bar2, variable_args, controls, tol=1e-6, epsilon=0.3, max_iter=10000) #n_bar hit 2

    
    n_bar1 = len(Y)-1
    n_bar2 = np.abs(Y - y_bar).argmin()

    print("Look at differences")
    print(ME_noRD.shape)
    print(ME_noRD2.shape)
    print(n_bar1,n_bar2)
    print(np.max(abs(ME_noRD[:,:n_bar2+1,:]-ME_noRD2)))
    print("Look at differences")
    print("-------------Omit RD Done--------------")

    print("---------------------------")
    print("---------------------------")
    print("---------------------------")

    # Omit Damage
    print("-------------Omit Damage Start--------------")

    args = (delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a, xi_g, xi_p)
    variable_args = (theta_ell_wakeup, psi_2_wakeup)
    controls=(ii,ee,xx, g_tech, g_damage,  pi_c_o_wakeup, pi_c_wakeup, pi_d_o_wakeup, v0, v_i, v_tech2)
    ME_nodmg = minimize_pi_c_tech(grid, args, n_bar1, variable_args, controls, tol=1e-6, epsilon=0.3, max_iter=10000) # n_bar free
    ME_nodmg2 = minimize_pi_c_tech(grid, args, n_bar2, variable_args, controls, tol=1e-6, epsilon=0.3, max_iter=10000) #n_bar hit 2

    
    n_bar1 = len(Y)-1
    n_bar2 = np.abs(Y - y_bar).argmin()

    print("Look at differences")
    print(ME_nodmg.shape)
    print(ME_nodmg2.shape)
    print(n_bar1,n_bar2)
    print(np.max(abs(ME_nodmg[:,:n_bar2+1,:]-ME_nodmg2)))
    print("Look at differences")
    print("-------------Omit Damage Done--------------")
    print("---------------------------")
    print("---------------------------")
    print("---------------------------")

    # Omit Tech
    print("-------------Omit Tech Start--------------")

    args = (delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a, xi_g, xi_p)
    variable_args = (theta_ell_wakeup, psi_2_wakeup)
    controls=(ii,ee,xx, g_tech, g_damage,  pi_c_o_wakeup, pi_c_wakeup, pi_d_o_wakeup, v0, v_i, v_tech2)
    ME_notech = minimize_pi_c_dmg(grid, args, n_bar1, variable_args, controls, tol=1e-6, epsilon=0.3, max_iter=10000) # n_bar free
    ME_notech2 = minimize_pi_c_dmg(grid, args, n_bar2, variable_args, controls, tol=1e-6, epsilon=0.3, max_iter=10000) #n_bar hit 2

    
    n_bar1 = len(Y)-1
    n_bar2 = np.abs(Y - y_bar).argmin()

    print("Look at differences")
    print(ME_notech.shape)
    print(ME_notech2.shape)
    print(n_bar1,n_bar2)
    print(np.max(abs(ME_notech[:,:n_bar2+1,:]-ME_notech2)))
    print("Look at differences")
    print("-------------Omit Tech Done--------------")

    print("---------------------------")
    print("---------------------------")
    print("---------------------------")



    dL = finiteDiff_3D(v0, 2,1,hL )

    gridpoints = (K, Y, L)

    i_func = RegularGridInterpolator(gridpoints, ii)
    e_func = RegularGridInterpolator(gridpoints, ee)
    x_func = RegularGridInterpolator(gridpoints, xx)
    tech_func = RegularGridInterpolator(gridpoints, g_tech)
    # mean_func = RegularGridInterpolator(gridpoints, gg_damageean)
    dL_func   = RegularGridInterpolator(gridpoints, dL)
    ME_total_func = RegularGridInterpolator(gridpoints, ME_total)
    ME_base_func = RegularGridInterpolator(gridpoints, ME_base)
    ME_temp_func = RegularGridInterpolator(gridpoints, ME_temp)
    ME_carb_func = RegularGridInterpolator(gridpoints, ME_carb)
    ME_RD_func = RegularGridInterpolator(gridpoints, ME_RD)
    ME_dmg_func = RegularGridInterpolator(gridpoints, ME_dmg)
    ME_tech_func = RegularGridInterpolator(gridpoints, ME_tech)
    ME_notemp_func = RegularGridInterpolator(gridpoints, ME_notemp)
    ME_nocarb_func = RegularGridInterpolator(gridpoints, ME_nocarb)
    ME_noRD_func = RegularGridInterpolator(gridpoints, ME_noRD)
    ME_nodmg_func = RegularGridInterpolator(gridpoints, ME_nodmg)
    ME_notech_func = RegularGridInterpolator(gridpoints, ME_notech)
     
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

    def get_dL(x):
        return dL_func(x)
#     K_0 = np.log(85 / 0.115)
#     Y_0 = 1.1
#     L_0 = -3.7
    
    K_0, Y_0, L_0 = initial

    print("K0={},Y0={},L0={}".format(K_0,Y_0,L_0))
    print(np.log(85/0.115))
    
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
    gt_mean   = np.zeros([pers])
    dL_hist    = np.zeros([pers])

#     if pre_damage:
    gt_dmg    = np.zeros([n_damage, pers])
    pi_c_t = np.zeros([n_climate, pers])
    Ambiguity_mean_undis = np.zeros([pers])
    Ambiguity_mean_dis = np.zeros([pers])
    Ambiguity_mean_dis_h = np.zeros([pers])
    
    ME_total_hist = np.zeros([pers])
    ME_base_hist = np.zeros([pers])
    ME_temp_hist = np.zeros([pers])
    ME_carb_hist = np.zeros([pers])
    ME_RD_hist = np.zeros([pers])
    ME_dmg_hist = np.zeros([pers])
    ME_tech_hist = np.zeros([pers])

    ME_notemp_hist = np.zeros([pers])
    ME_nocarb_hist = np.zeros([pers])
    ME_noRD_hist = np.zeros([pers])
    ME_nodmg_hist = np.zeros([pers])
    ME_notech_hist = np.zeros([pers])


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
            # gt_mean[0] = mean_func(hist[0, :])
            dL_hist[tm] = dL_func(hist[0,:])
            
            ME_total_hist[tm] = ME_total_func(hist[0,:])
            ME_base_hist[tm] = ME_base_func(hist[0,:])
            ME_temp_hist[tm] = ME_temp_func(hist[0,:])
            ME_carb_hist[tm] = ME_carb_func(hist[0,:])
            ME_RD_hist[tm] = ME_RD_func(hist[0,:])
            ME_dmg_hist[tm] = ME_dmg_func(hist[0,:])
            ME_tech_hist[tm] = ME_tech_func(hist[0,:])
            ME_notemp_hist[tm] = ME_notemp_func(hist[0,:])
            ME_nocarb_hist[tm] = ME_nocarb_func(hist[0,:])
            ME_noRD_hist[tm] = ME_noRD_func(hist[0,:])
            ME_nodmg_hist[tm] = ME_nodmg_func(hist[0,:])
            ME_notech_hist[tm] = ME_notech_func(hist[0,:])

#             if pre_damage:
            for i in range(n_damage):
                damage_func = damage_func_list[i]
                gt_dmg[i, 0] = damage_func(hist[0, :])
            
            for i in range(n_climate):
                climate_func = climate_func_list[i]
                pi_c_t[i, 0] = climate_func(hist[0, :])
            Ambiguity_mean_undis[tm] = np.mean(theta_ell_array)
            Ambiguity_mean_dis[tm] = np.average(theta_ell_array,weights=pi_c_t[:,tm])
            # Ambiguity_mean_dis_h[tm] = np.average(theta_ell + sigma_y*gt_mean[tm],weights=pi_c_t[:,tm])
            # print(hist[0,:])
        else:
            # other periods
            # print(hist[tm-1,:])
            i_hist[tm] = get_i(hist[tm-1,:])
            e_hist[tm] = get_e(hist[tm-1,:])
            x_hist[tm] = get_x(hist[tm-1,:])
            gt_tech[tm] = tech_func(hist[tm-1,:])
            # gt_mean[tm] = mean_func(hist[tm-1,:])
            dL_hist[tm] = dL_func(hist[tm-1,:])

#             if pre_damage:
            for i in range(n_damage):
                damage_func = damage_func_list[i]
                gt_dmg[i, tm] = damage_func(hist[tm-1, :])

            for i in range(n_climate):
                climate_func = climate_func_list[i]
                pi_c_t[i, tm] = climate_func(hist[tm -1, :])
                
#             ME_base_t[tm] = ME_base_func(hist[tm-1, :])

            # gt_mean_mul = sigma_y*gt_mean*1000
            
            ME_total_hist[tm] = ME_total_func(hist[tm-1,:])
            ME_base_hist[tm] = ME_base_func(hist[tm-1,:])
            ME_temp_hist[tm] = ME_temp_func(hist[tm-1,:])
            ME_carb_hist[tm] = ME_carb_func(hist[tm-1,:])
            ME_RD_hist[tm] = ME_RD_func(hist[tm-1,:])
            ME_dmg_hist[tm] = ME_dmg_func(hist[tm-1,:])
            ME_tech_hist[tm] = ME_tech_func(hist[tm-1,:])
            
            ME_notemp_hist[tm] = ME_notemp_func(hist[tm-1,:])
            ME_nocarb_hist[tm] = ME_nocarb_func(hist[tm-1,:])
            ME_noRD_hist[tm] = ME_noRD_func(hist[tm-1,:])
            ME_nodmg_hist[tm] = ME_nodmg_func(hist[tm-1,:])
            ME_notech_hist[tm] = ME_notech_func(hist[tm-1,:])


            mu_K_hist[tm] = mu_K(i_hist[tm])
            mu_L_hist[tm] = mu_L(x_hist[tm], hist[tm-1, :])

            hist[tm,0] = hist[tm-1,0] + mu_K_hist[tm] * dt #logK
            hist[tm,1] = hist[tm-1,1] + beta_f * e_hist[tm] * dt
            hist[tm,2] = hist[tm-1,2] + mu_L_hist[tm] * dt # logλ
            Ambiguity_mean_undis[tm] = np.mean(theta_ell_array)
            Ambiguity_mean_dis[tm] = np.average(theta_ell_array,weights=pi_c_t[:,tm])
            # Ambiguity_mean_dis_h[tm] = np.average(theta_ell + sigma_y*gt_mean[tm],weights=pi_c_t[:,tm])
            

        if printing==True:
            print("time={}, K={},Y={},L={},mu_K={},mu_Y={},mu_L={},ii={},ee={},xx={}" .format(tm, hist[tm,0],hist[tm,1],hist[tm,2],mu_K_hist[tm],beta_f * e_hist[tm],mu_L_hist[tm],ii.max(),ee.max(),xx.max()))
        
    
    
        # using Kt instead of K0
    jt = 1 - e_hist/ (alpha * lambda_bar * np.exp(hist[:, 0]))
    jt[jt <= 1e-16] = 1e-16
    LHS = theta * vartheta_bar / lambda_bar * jt**(theta -1)
    MC = delta / (alpha  - i_hist - alpha * vartheta_bar * jt**theta - x_hist)

    
    scc_hist = LHS * 1000


    MU_RD = dL_hist * psi_0* psi_1 * x_hist**(psi_1-1) * np.exp(psi_1*hist[:,0]-(1-psi_2)*hist[:,2])

    scrd_hist = MU_RD/MC*1000

    # scc_hist = LHS / MC * 1000
#     scc_0 = ME_base_t / MC * 1000 * np.exp(hist[:, 0])
    
    # distorted_tech_intensity = np.exp(hist[:, 2]) * gt_tech

    distorted_tech_intensity = np.exp(hist[:, 2]) * gt_tech/448

    distorted_tech_prob = 1 - np.exp(- np.cumsum(np.insert(distorted_tech_intensity * dt, 0, 0) ))[:-1]

    # true_tech_intensity = np.exp(hist[:, 2]) 
    true_tech_intensity = np.exp(hist[:, 2]) /448
    true_tech_prob = 1 - np.exp(- np.cumsum(np.insert(true_tech_intensity * dt, 0, 0) ))[:-1]
        
#     if pre_damage:
    damage_intensity = Damage_Intensity(hist[:, 1])
    distorted_damage_intensity = np.mean(gt_dmg, axis=0) * damage_intensity
    distorted_damage_prob = 1 - np.exp(- np.cumsum(np.insert(distorted_damage_intensity * dt, 0, 0) ))[:-1]
    
    true_damage_intensity =  damage_intensity
    true_damage_prob = 1 - np.exp(- np.cumsum(np.insert(true_damage_intensity * dt, 0, 0) ))[:-1]

    # ratio list

    ME_total_base_hist = np.log(ME_total_hist / ME_base_hist ) * 100
    ME_temp_base_hist =np.log(ME_temp_hist  / ME_base_hist ) * 100
    ME_carb_base_hist =np.log(ME_carb_hist  / ME_base_hist ) * 100
    ME_RD_base_hist =np.log(ME_RD_hist  / ME_base_hist ) * 100
    ME_dmg_base_hist =np.log(ME_dmg_hist  / ME_base_hist ) * 100
    ME_tech_base_hist =np.log(ME_tech_hist  / ME_base_hist ) * 100
    ME_notemp_base_hist =np.log(ME_notemp_hist  / ME_base_hist ) * 100
    ME_nocarb_base_hist =np.log(ME_nocarb_hist  / ME_base_hist ) * 100
    ME_noRD_base_hist =np.log(ME_noRD_hist  / ME_base_hist ) * 100
    ME_nodmg_base_hist =np.log(ME_nodmg_hist  / ME_base_hist ) * 100
    ME_notech_base_hist =np.log(ME_notech_hist  / ME_base_hist ) * 100

    print("----------------------------------------")
    print("----------------------------------------")
    print("----------------------------------------")
    print("ratio list shape------------------------")
    print(ME_total_hist.shape)
    print(ME_base_hist.shape)
    print((ME_total_hist / ME_base_hist ).shape)
    print((np.log(ME_total_hist / ME_base_hist ) * 100).shape)
    print(ME_total_base_hist.shape)
    
    
    print("----------------------------------------")
    print("----------------------------------------")
    print("----------------------------------------")
    
    res = dict(
        states= hist, 
        i = i_hist * np.exp(hist[:, 0]), 
        e = e_hist,
        # x = x_hist * np.exp(hist[:, 0]),
        x = x_hist * np.exp(hist[:, 0]),
        scc = scc_hist,
        scrd = scrd_hist,
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
        true_damage_prob = true_damage_prob,
        Ambiguity_mean_undis = Ambiguity_mean_undis,
        Ambiguity_mean_dis = Ambiguity_mean_dis,
        ME_total = ME_total_hist,
        ME_base = ME_base_hist,
        ME_temp = ME_temp_hist,
        ME_carb = ME_carb_hist,
        ME_RD = ME_RD_hist,
        ME_dmg = ME_dmg_hist,
        ME_tech = ME_tech_hist,
        ME_notemp = ME_notemp_hist,
        ME_nocarb = ME_nocarb_hist,
        ME_noRD = ME_noRD_hist,
        ME_nodmg = ME_nodmg_hist,
        ME_notech = ME_notech_hist,
        ME_total_base = ME_total_base_hist,
        ME_temp_base = ME_temp_base_hist,
        ME_carb_base = ME_carb_base_hist,
        ME_RD_base = ME_RD_base_hist,
        ME_dmg_base = ME_dmg_base_hist,
        ME_tech_base = ME_tech_base_hist,
        ME_notemp_base = ME_notemp_base_hist,
        ME_nocarb_base = ME_nocarb_base_hist,
        ME_noRD_base = ME_noRD_base_hist,
        ME_nodmg_base = ME_nodmg_base_hist,
        ME_notech_base = ME_notech_base_hist,
        # gt_mean_mul = gt_mean_mul,
        )
    
#     if pre_damage:
#         res["gt_dmg"] = gt_dmg
    
    return res




def model_solution_extraction(xi_a,xi_g,psi_0,psi_1,psi_2):
    
    # Data_Dir = "./abatement/data_2tech/"+args.dataname+"/"
    Output_Dir = "/scratch/bincheng/"
    Data_Dir = Output_Dir+"abatement/data_2tech/"+args.dataname+"/"



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
    
    Y_min_short = Xminarr[3]
    Y_max_short = Xmaxarr[3]
    Y_short     = np.arange(Y_min_short, Y_max_short + hY, hY)
    nY_short    = len(Y_short)
    



    File_Dir = "xi_a_{}_xi_g_{}_psi_0_{}_psi_1_{}_" .format(xi_a,xi_g,psi_0,psi_1)

    model_dir_post = Data_Dir + File_Dir+"model_tech1_pre_damage"
    
    File_Dir2 = "xi_a_{}_xi_g_{}_psi_0_{}_psi_1_{}_psi_2_{}_" .format(xi_a,xi_g,psi_0,psi_1,psi_2)

    model_simul_dir_post = Data_Dir + File_Dir2+"model_tech1_pre_damage_simul_{}" .format(IntPeriod)
    print("-------------------------------------------")
    print("------------Post damage, Tech II----------")
    print("-------------------------------------------")
    model_tech2_post_damage = pickle.load(open(Data_Dir+ File_Dir + "model_tech2_post_damage", "rb"))
    print("Load Success.")
    print("-------------------------------------------")
    print("------------Post damage, Tech I-----------")
    print("-------------------------------------------")
    model_tech1_post_damage = pickle.load(open(Data_Dir+ File_Dir + "model_tech1_post_damage", "rb"))
    print("Load Success.")

    print("-------------------------------------------")
    print("------------Pre damage, Tech II-----------")
    print("-------------------------------------------")
    model_tech2_pre_damage = pickle.load(open(Data_Dir+ File_Dir + "model_tech2_pre_damage", "rb"))
    print("Load Success.")


    
    id_2 = np.abs(Y - y_bar).argmin()
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



    if os.path.exists(model_simul_dir_post):
        print("which passed 1")
        res = pickle.load(open(model_simul_dir_post, "rb"))


    else:
        print("which passed 2")

        with open(model_dir_post, "rb") as f:
            tech1 = pickle.load(f)
        
        model_args = (delta, mu_k, kappa,sigma_k, beta_f, zeta, psi_0, psi_1, psi_2, sigma_g, theta, lambda_bar, vartheta_bar, xi_a, xi_g, xi_g)
        
        v = tech1["v0"]
        i = tech1["i_star"]
        e = tech1["e_star"]
        x = tech1["x_star"]
        pi_c = tech1["pi_c"]
        g_tech = tech1["g_tech"]
        g_damage =  tech1["g_damage"]

        res = simulate_pre_UD(grid = (K, Y_short, L), model_args = model_args, 
                                    controls = (i,e,x, g_tech, g_damage, pi_c, v, v_i, v_tech2),  
                                    T0=0, T=IntPeriod, dt=timespan,printing=True)

        with open(model_simul_dir_post, "wb") as f:
            pickle.dump(res,f)

        res = pickle.load(open(model_simul_dir_post, "rb"))

    
    return res


if os.path.exists("./abatement/pdf_2tech/"+args.dataname+"/")==False:
    os.makedirs("./abatement/pdf_2tech/"+args.dataname+"/")

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                    res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                    if xiaarr[id_xiag]>10:
                        plt.plot(res["years"], (res["x"]/(alpha*np.exp(res["states"][:,0])))*100,label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                    else:
                        plt.plot(res["years"], (res["x"]/(alpha*np.exp(res["states"][:,0])))*100,label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                    plt.xlabel('Years')
                    plt.ylabel('$\%$ of GDP')
                    plt.title('R&D investment as percentage of  GDP')
                    if auto==0:   
                        plt.ylim(0,1.0)
                    plt.xlim(0,IntPeriod)

                    plt.legend(loc='upper left')        

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/RD,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/RD,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], res["i"],label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"], res["i"],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                plt.xlabel('Years')
                plt.title("Capital investment")
                if auto==0:   
                    plt.ylim(65,110)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper left')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/CapI,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/CapI,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                if xiaarr[id_xiag]>10:
                    plt.plot(res["years"], res["e"],label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"], res["e"],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                # plt.plot(res2["years"][res2["states"][:, 1]<1.5], res2["e"][res2["states"][:, 1]<1.5],label=r'$\xi_p=\\xi_g=0.050$',linewidth=7.0)
                # plt.plot(res3["years"][res3["states"][:, 1]<1.5], res3["e"][res3["states"][:, 1]<1.5],label='baseline',linewidth=7.0)
                plt.xlabel('Years')
                plt.title("Carbon Emissions")
                if auto==0:   
                    plt.ylim(6.5,11.5)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper left')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/E,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/E,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], res["states"][:, 1],label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"], res["states"][:, 1],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                # plt.plot(res2["years"][res2["states"][:, 1]<1.5], res2["states"][:, 1][res2["states"][:, 1]<1.5],label=r'$\xi_p=\\xi_g=0.050$',linewidth=7.0)
                # plt.plot(res3["years"][res3["states"][:, 1]<1.5], res3["states"][:, 1][res3["states"][:, 1]<1.5],label='baseline',linewidth=7.0)
                plt.xlabel('Years')
                plt.title("Temperature anomaly")
                if auto==0:   
                    plt.ylim(1.1,1.6)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper left')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/TA,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/TA,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])


                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], np.exp(res["states"][:, 2]),label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"], np.exp(res["states"][:, 2]),label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                # plt.plot(res2["years"][res2["states"][:, 1]<1.5], np.exp(res2["states"][:, 2])[res2["states"][:, 1]<1.5],label=r'$\xi_p=\\xi_g=0.050$',linewidth=7.0)
                # plt.plot(res3["years"][res3["states"][:, 1]<1.5], np.exp(res3["states"][:, 2])[res3["states"][:, 1]<1.5],label='baseline',linewidth=7.0)
                plt.xlabel('Years')
                plt.title("Technology jump intensity $J_g$")
                if auto==0:   
                    plt.ylim(10.0,18.0)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper left')


plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/Ig,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/Ig,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])


                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], res["distorted_tech_prob"],label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"], res["distorted_tech_prob"],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                # plt.plot(res2["years"], res2["distorted_tech_prob"],label=r'$\xi_p=\\xi_g=0.050$',linewidth=7.0)
                # plt.plot(res3["years"], res3["distorted_tech_prob"],label='baseline',linewidth=7.0)
                plt.xlabel('Years')
                plt.title("Distorted probability of a technology jump")
                if auto==0:   
                    plt.ylim(0,1)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper left')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/PIgd,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/PIgd,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], res["distorted_damage_prob"],label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"], res["distorted_damage_prob"],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                # plt.plot(res2["years"], res2["distorted_damage_prob"],label=r'$\xi_p=\\xi_g=0.050$',linewidth=7.0)
                # plt.plot(res3["years"], res3["distorted_damage_prob"],label='baseline',linewidth=7.0)
                plt.xlabel('Years')
                plt.title("Distorted probability of damage changes")
                if auto==0:   
                    plt.ylim(0,1)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper left')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/PIdd,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/PIdd,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], res["true_tech_prob"],label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"], res["true_tech_prob"],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)

                plt.xlabel("Years")
                plt.title("True probability of a technology jump")
                if auto==0:   
                    plt.ylim(0.0,1.0)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper left')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/TPIg,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/TPIg,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], res["true_damage_prob"],label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"], res["true_damage_prob"],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)

                plt.xlabel("Years")
                plt.title("True probability of damage changes")
                if auto==0:   
                    plt.ylim(0,1)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper left')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/TPId,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/TPId,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], np.log(res["scc"]),label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"], np.log(res["scc"]),label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)

                plt.xlabel("Years")
                plt.title("Log of Social Cost of Carbon")
                if auto==0:   
                    plt.ylim(3.0,6.5)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper left')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/logSCC,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/logSCC,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], np.log(res["scrd"]),label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"], np.log(res["scrd"]),label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)

                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                plt.title("Log of Social Cost of R&D")
                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper right')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/logSCRD,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/logSCRD,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], np.log(res["ME_base"]),label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"], np.log(res["ME_base"]),label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)

                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                plt.title("ME_base")
                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper right')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_base,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_base,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], np.log(res["ME_temp"]),label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"], np.log(res["ME_temp"]),label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)

                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                plt.title("ME_temp")
                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper right')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_temp,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_temp,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], np.log(res["ME_carb"]),label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"], np.log(res["ME_carb"]),label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)

                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                plt.title("ME_carb")
                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper right')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_carb,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_carb,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], np.log(res["ME_RD"]),label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"], np.log(res["ME_RD"]),label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)

                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                plt.title("ME_RD")
                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper right')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_RD,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_RD,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], np.log(res["ME_dmg"]),label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"], np.log(res["ME_dmg"]),label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)

                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                plt.title("ME_dmg")
                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper right')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_dmg,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_dmg,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], np.log(res["ME_tech"]),label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"], np.log(res["ME_tech"]),label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)

                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                plt.title("ME_tech")
                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper right')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_tech,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_tech,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], np.log(res["ME_notemp"]),label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"], np.log(res["ME_notemp"]),label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)

                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                plt.title("ME_notemp")
                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper right')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_notemp,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_notemp,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], np.log(res["ME_nocarb"]),label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"], np.log(res["ME_nocarb"]),label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)

                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                plt.title("ME_nocarb")
                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper right')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_nocarb,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_nocarb,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], np.log(res["ME_noRD"]),label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"], np.log(res["ME_noRD"]),label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)

                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                plt.title("ME_noRD")
                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper right')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_noRD,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_noRD,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], np.log(res["ME_nodmg"]),label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"], np.log(res["ME_nodmg"]),label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)

                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                plt.title("ME_nodmg")
                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper right')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_nodmg,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_nodmg,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], np.log(res["ME_notech"]),label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"], np.log(res["ME_notech"]),label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)

                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                plt.title("ME_notech")
                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper right')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_notech,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_notech,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()





for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])
                # print(res["ME_total_base"].shape)
                # print(res["ME_total"].shape)
                # print(res["ME_base"].shape)
                # print(np.log(res["ME_total"]/res["ME_base"]).shape)
                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], res["ME_total_base"],label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"], res["ME_total_base"],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)

                
                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                plt.title("ME_total_base")
                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper right')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_total_base,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_total_base,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], res["ME_temp_base"],label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"],res["ME_temp_base"],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)

                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                plt.title("ME_temp_base")
                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper right')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_temp_base,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_temp_base,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], res["ME_carb_base"],label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"],res["ME_carb_base"],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)

                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                plt.title("ME_carb_base")
                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper right')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_carb_base,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_carb_base,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], res["ME_RD_base"],label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"],res["ME_RD_base"],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)

                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                plt.title("ME_RD_base")
                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper right')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_RD_base,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_RD_base,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], res["ME_dmg_base"],label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"],res["ME_dmg_base"],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)

                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                plt.title("ME_dmg_base")
                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper right')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_dmg_base,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_dmg_base,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], res["ME_tech_base"],label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"],res["ME_tech_base"],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)

                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                plt.title("ME_tech_base")
                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper right')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_tech_base,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_tech_base,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], res["ME_notemp_base"],label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"],res["ME_notemp_base"],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)

                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                plt.title("ME_notemp_base")
                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper right')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_notemp_base,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_notemp_base,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], res["ME_nocarb_base"],label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"],res["ME_nocarb_base"],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)

                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                plt.title("ME_nocarb_base")
                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper right')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_nocarb_base,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_nocarb_base,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], res["ME_noRD_base"],label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"],res["ME_noRD_base"],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)

                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                plt.title("ME_noRD_base")
                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper right')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_noRD_base,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_noRD_base,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], res["ME_nodmg_base"],label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"],res["ME_nodmg_base"],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)

                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                plt.title("ME_nodmg_base")
                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper right')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_nodmg_base,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_nodmg_base,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], res["ME_notech_base"],label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"],res["ME_notech_base"],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)

                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                plt.title("ME_notech_base")
                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper right')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_notech_base,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_notech_base,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])
                # print(res["ME_total_base"].shape)
                # print(res["ME_total"].shape)
                # print(res["ME_base"].shape)
                # print(np.log(res["ME_total"]/res["ME_base"]).shape)

                plt.plot(res["years"], res["ME_total_base"],label='Total Uncertainty' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                plt.plot(res["years"], res["ME_temp_base"],label='Temperature Uncertainty' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                plt.plot(res["years"], res["ME_carb_base"],label='Carbon Uncertainty' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                plt.plot(res["years"], res["ME_RD_base"],label='RD Uncertainty' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                plt.plot(res["years"], res["ME_dmg_base"],label='Damage Uncertainty' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                plt.plot(res["years"], res["ME_tech_base"],label='Technology Uncertainty' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)


                
                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                plt.title("ME Ratio: Single Case")
                
                plt.ylim(-40,20)

                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper right')

                plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_single_ratio,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2]))
                plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_single_ratio,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2]))
                plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                plt.plot(res["years"], res["ME_total_base"],label='Total Uncertainty' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                plt.plot(res["years"], res["ME_notemp_base"],label='No Temperature Uncertainty' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                plt.plot(res["years"], res["ME_nocarb_base"],label='No Carbon Uncertainty' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                plt.plot(res["years"], res["ME_noRD_base"],label='No RD Uncertainty' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                plt.plot(res["years"], res["ME_nodmg_base"],label='No Damage Uncertainty' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                plt.plot(res["years"], res["ME_notech_base"],label='No Technology Uncertainty' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)


                
                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                plt.title("ME Ratio: Multiple Case")
                
                plt.ylim(-40,20)

                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper right')

                plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_multiple_ratio,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2]))
                plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_multiple_ratio,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2]))
                plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])
                
                if xigarr[id_xiag]>10:

                    plt.plot(res["years"], (res["Ambiguity_mean_dis"]-res["Ambiguity_mean_undis"])*1000,label='baseline'.format(psi2arr[id_psi2]))
                else:
                    plt.plot(res["years"], (res["Ambiguity_mean_dis"]-res["Ambiguity_mean_undis"])*1000,label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag]))

                plt.xlabel("Years")
                plt.title("Mean Difference")
                if auto==0:   
                    plt.ylim(0,0.8)   
                # plt.legend(loc='upper left')

# if xigarr[0]==xigarr[1]:
#     gt_mean1 = pd.read_csv("./data/gg_damageean1,xig={},psi0={},psi1={},psi2={}.csv".format(xigarr[0],psi0arr[0],psi1arr[0],psi2arr[0]), header=None).to_numpy()[:, 0]

#     plt.plot(res["years"][res["states"][:, 1]<1.5], gt_mean1[res["states"][:, 1]<1.5],label='$\\xi_m={:.3f}$'.format(xigarr[id_xiag]))
plt.legend(loc='upper left')

plt.legend(loc='upper left')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/MeanDiff,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/MeanDiff,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()

# for id_xiag in range(len(xiaarr)): 
#     for id_psi0 in range(len(psi0arr)):
#         for id_psi1 in range(len(psi1arr)):
#             for id_psi2 in range(len(psi2arr)):

#                 res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])
                
                
#                 if xigarr[id_xiag]>10:

#                     plt.plot(res["years"], (np.zeros((res["Ambiguity_mean_dis"]-res["Ambiguity_mean_undis"]).shape))*1000,label='baseline'.format(psi2arr[id_psi2]))
#                 else:
#                     plt.plot(res["years"], (np.zeros((res["Ambiguity_mean_dis"]-res["Ambiguity_mean_undis"]).shape))*1000,label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag]))

#                 plt.xlabel("Years")
#                 plt.title("Mean Difference")
#                 if auto==0:   
#                     plt.ylim(0,0.8)
#                 plt.legend(loc='upper left')


# plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/MeanDiff2,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
# plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/MeanDiff2,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
# plt.close()


# for id_xiag in range(len(xiaarr)): 
#     for id_psi0 in range(len(psi0arr)):
#         for id_psi1 in range(len(psi1arr)):
#             for id_psi2 in range(len(psi2arr)):

#                 res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])
                
                
#                 if xigarr[id_xiag]>10:

#                     plt.plot(res["years"], res["gt_mean_mul"],label='baseline'.format(psi2arr[id_psi2]))
#                 else:
#                     plt.plot(res["years"], res["gt_mean_mul"],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag]))

#                 plt.xlabel("Years")
#                 plt.title("$1000\sigma_yh$")
#                 if auto==0:   
#                     plt.ylim(0,0.08)
#                 plt.legend(loc='upper left')


# plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/h,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
# plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/h,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
# plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])
                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], ((res["x"]/(alpha*np.exp(res["states"][:,0])))*100)[res["states"][:, 1]<1.5],label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], ((res["x"]/(alpha*np.exp(res["states"][:,0])))*100)[res["states"][:, 1]<1.5],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                plt.xlabel('Years')
                plt.ylabel('$\%$ of GDP')
                plt.title('R&D investment as percentage of  GDP')   
                if auto==0:   
                    plt.ylim(0,1.0)
                plt.legend(loc='upper left')        

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/RD,xia={},xig={},psi0={},psi1={},psi2={},BC.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/RD,xia={},xig={},psi0={},psi1={},psi2={},BC.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])
                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["i"][res["states"][:, 1]<1.5],label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["i"][res["states"][:, 1]<1.5],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                plt.xlabel('Years')
                if auto==0:   
                    plt.ylim(65,110)
                plt.legend(loc='upper left')
                plt.title("Capital investment")

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/CapI,xia={},xig={},psi0={},psi1={},psi2={},BC.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/CapI,xia={},xig={},psi0={},psi1={},psi2={},BC.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["e"][res["states"][:, 1]<1.5],label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["e"][res["states"][:, 1]<1.5],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                # plt.plot(res2["years"][res2["states"][:, 1]<1.5], res2["e"][res2["states"][:, 1]<1.5],label=r'$\xi_p=\\xi_g=0.050$',linewidth=7.0)
                # plt.plot(res3["years"][res3["states"][:, 1]<1.5], res3["e"][res3["states"][:, 1]<1.5],label='baseline',linewidth=7.0)
                plt.xlabel('Years')
                if auto==0:   
                    plt.ylim(6.5,11.5)
                plt.legend(loc='upper left')
                plt.title("Carbon Emissions")
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/E,xia={},xig={},psi0={},psi1={},psi2={},BC.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/E,xia={},xig={},psi0={},psi1={},psi2={},BC.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])
                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["states"][:, 1][res["states"][:, 1]<1.5],label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["states"][:, 1][res["states"][:, 1]<1.5],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                # plt.plot(res2["years"][res2["states"][:, 1]<1.5], res2["states"][:, 1][res2["states"][:, 1]<1.5],label=r'$\xi_p=\\xi_g=0.050$',linewidth=7.0)
                # plt.plot(res3["years"][res3["states"][:, 1]<1.5], res3["states"][:, 1][res3["states"][:, 1]<1.5],label='baseline',linewidth=7.0)
                plt.xlabel('Years')
                if auto==0:   
                    plt.ylim(1.1,1.6)
                plt.legend(loc='upper left')
                plt.title("Temperature anomaly")
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/TA,xia={},xig={},psi0={},psi1={},psi2={},BC.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/TA,xia={},xig={},psi0={},psi1={},psi2={},BC.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])
                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.exp(res["states"][:, 2])[res["states"][:, 1]<1.5],label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.exp(res["states"][:, 2])[res["states"][:, 1]<1.5],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                # plt.plot(res2["years"][res2["states"][:, 1]<1.5], np.exp(res2["states"][:, 2])[res2["states"][:, 1]<1.5],label=r'$\xi_p=\\xi_g=0.050$',linewidth=7.0)
                # plt.plot(res3["years"][res3["states"][:, 1]<1.5], np.exp(res3["states"][:, 2])[res3["states"][:, 1]<1.5],label='baseline',linewidth=7.0)
                plt.xlabel('Years')
                if auto==0:   
                    plt.ylim(10.0,18.0)
                plt.legend(loc='upper left')
                plt.title("Technology jump intensity $J_g$")

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/Ig,xia={},xig={},psi0={},psi1={},psi2={},BC.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/Ig,xia={},xig={},psi0={},psi1={},psi2={},BC.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["distorted_tech_prob"][res["states"][:, 1]<1.5],label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["distorted_tech_prob"][res["states"][:, 1]<1.5],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                # plt.plot(res2["years"], res2["distorted_tech_prob"],label=r'$\xi_p=\\xi_g=0.050$',linewidth=7.0)
                # plt.plot(res3["years"], res3["distorted_tech_prob"],label='baseline',linewidth=7.0)
                plt.xlabel('Years')
                if auto==0:   
                    plt.ylim(0,1)
                plt.legend(loc='upper left')
                plt.title("Distorted probability of a technology jump")
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/PIgd,xia={},xig={},psi0={},psi1={},psi2={},BC.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/PIgd,xia={},xig={},psi0={},psi1={},psi2={},BC.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])
                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["distorted_damage_prob"][res["states"][:, 1]<1.5],label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["distorted_damage_prob"][res["states"][:, 1]<1.5],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                # plt.plot(res2["years"], res2["distorted_damage_prob"],label=r'$\xi_p=\\xi_g=0.050$',linewidth=7.0)
                # plt.plot(res3["years"], res3["distorted_damage_prob"],label='baseline',linewidth=7.0)
                plt.xlabel('Years')
                if auto==0:   
                    plt.ylim(0,1)
                plt.legend(loc='upper left')
                plt.title("Distorted probability of damage changes")
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/PIdd,xia={},xig={},psi0={},psi1={},psi2={},BC.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/PIdd,xia={},xig={},psi0={},psi1={},psi2={},BC.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["true_tech_prob"][res["states"][:, 1]<1.5],label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["true_tech_prob"][res["states"][:, 1]<1.5],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)

                plt.xlabel("Years")
                if auto==0:   
                    plt.ylim(0,1)
                plt.legend(loc='upper left')
                plt.title("True probability of a technology jump")
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/TPIg,xia={},xig={},psi0={},psi1={},psi2={},BC.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/TPIg,xia={},xig={},psi0={},psi1={},psi2={},BC.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])
                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["true_damage_prob"][res["states"][:, 1]<1.5],label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
    
                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["true_damage_prob"][res["states"][:, 1]<1.5],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)

                plt.xlabel("Years")
                if auto==0:   
                    plt.ylim(0,1)
                plt.legend(loc='upper left')
                plt.title("True probability of damage changes")
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/TPId,xia={},xig={},psi0={},psi1={},psi2={},BC.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/TPId,xia={},xig={},psi0={},psi1={},psi2={},BC.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])
                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scc"][res["states"][:, 1]<1.5]),label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scc"][res["states"][:, 1]<1.5]),label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                plt.xlabel("Years")
                
                if auto==0:   
                    plt.ylim(3.0,6.5)
                plt.title("Log of Social Cost of Carbon")

                plt.legend(loc='upper left')


plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/logSCC,xia={},xig={},psi0={},psi1={},psi2={},BC.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/logSCC,xia={},xig={},psi0={},psi1={},psi2={},BC.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])
                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scrd"][res["states"][:, 1]<1.5]),label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["scrd"][res["states"][:, 1]<1.5]),label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.title("Log of Social Cost of R&D")

                plt.legend(loc='upper right')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/logSCRD,xia={},xig={},psi0={},psi1={},psi2={},BC.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/logSCRD,xia={},xig={},psi0={},psi1={},psi2={},BC.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])
                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["ME_base"][res["states"][:, 1]<1.5]),label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["ME_base"][res["states"][:, 1]<1.5]),label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.title("ME_base")

                plt.legend(loc='upper right')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_base,xia={},xig={},psi0={},psi1={},psi2={},BC.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_base,xia={},xig={},psi0={},psi1={},psi2={},BC.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])
                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["ME_temp"][res["states"][:, 1]<1.5]),label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["ME_temp"][res["states"][:, 1]<1.5]),label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.title("ME_temp")

                plt.legend(loc='upper right')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_temp,xia={},xig={},psi0={},psi1={},psi2={},BC.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_temp,xia={},xig={},psi0={},psi1={},psi2={},BC.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])
                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["ME_carb"][res["states"][:, 1]<1.5]),label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["ME_carb"][res["states"][:, 1]<1.5]),label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.title("ME_carb")

                plt.legend(loc='upper right')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_carb,xia={},xig={},psi0={},psi1={},psi2={},BC.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_carb,xia={},xig={},psi0={},psi1={},psi2={},BC.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])
                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["ME_RD"][res["states"][:, 1]<1.5]),label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["ME_RD"][res["states"][:, 1]<1.5]),label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.title("ME_RD")

                plt.legend(loc='upper right')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_RD,xia={},xig={},psi0={},psi1={},psi2={},BC.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_RD,xia={},xig={},psi0={},psi1={},psi2={},BC.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])
                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["ME_dmg"][res["states"][:, 1]<1.5]),label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["ME_dmg"][res["states"][:, 1]<1.5]),label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.title("ME_dmg")

                plt.legend(loc='upper right')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_dmg,xia={},xig={},psi0={},psi1={},psi2={},BC.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_dmg,xia={},xig={},psi0={},psi1={},psi2={},BC.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])
                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["ME_tech"][res["states"][:, 1]<1.5]),label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["ME_tech"][res["states"][:, 1]<1.5]),label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.title("ME_tech")

                plt.legend(loc='upper right')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_tech,xia={},xig={},psi0={},psi1={},psi2={},BC.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_tech,xia={},xig={},psi0={},psi1={},psi2={},BC.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])
                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["ME_notemp"][res["states"][:, 1]<1.5]),label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["ME_notemp"][res["states"][:, 1]<1.5]),label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.title("ME_notemp")

                plt.legend(loc='upper right')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_notemp,xia={},xig={},psi0={},psi1={},psi2={},BC.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_notemp,xia={},xig={},psi0={},psi1={},psi2={},BC.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])
                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["ME_nocarb"][res["states"][:, 1]<1.5]),label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["ME_nocarb"][res["states"][:, 1]<1.5]),label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.title("ME_nocarb")

                plt.legend(loc='upper right')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_nocarb,xia={},xig={},psi0={},psi1={},psi2={},BC.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_nocarb,xia={},xig={},psi0={},psi1={},psi2={},BC.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])
                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["ME_noRD"][res["states"][:, 1]<1.5]),label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["ME_noRD"][res["states"][:, 1]<1.5]),label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.title("ME_ME_noRDRD")

                plt.legend(loc='upper right')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_noRD,xia={},xig={},psi0={},psi1={},psi2={},BC.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_noRD,xia={},xig={},psi0={},psi1={},psi2={},BC.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])
                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["ME_nodmg"][res["states"][:, 1]<1.5]),label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["ME_nodmg"][res["states"][:, 1]<1.5]),label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.title("ME_nodmg")

                plt.legend(loc='upper right')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_nodmg,xia={},xig={},psi0={},psi1={},psi2={},BC.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_nodmg,xia={},xig={},psi0={},psi1={},psi2={},BC.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])
                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["ME_notech"][res["states"][:, 1]<1.5]),label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["ME_notech"][res["states"][:, 1]<1.5]),label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.title("ME_notech")

                plt.legend(loc='upper right')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_notech,xia={},xig={},psi0={},psi1={},psi2={},BC.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_notech,xia={},xig={},psi0={},psi1={},psi2={},BC.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])
                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["ME_temp_base"][res["states"][:, 1]<1.5]),label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], np.log(res["ME_temp_base"][res["states"][:, 1]<1.5]),label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.title("ME_temp_base")

                plt.legend(loc='upper right')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_temp_base,xia={},xig={},psi0={},psi1={},psi2={},BC.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_temp_base,xia={},xig={},psi0={},psi1={},psi2={},BC.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()




for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])
                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["ME_carb_base"][res["states"][:, 1]<1.5],label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["ME_carb_base"][res["states"][:, 1]<1.5],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.title("ME_carb_base")

                plt.legend(loc='upper right')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_carb_base,xia={},xig={},psi0={},psi1={},psi2={},BC.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_carb_base,xia={},xig={},psi0={},psi1={},psi2={},BC.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])
                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["ME_RD_base"][res["states"][:, 1]<1.5],label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["ME_RD_base"][res["states"][:, 1]<1.5],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.title("ME_RD_base")

                plt.legend(loc='upper right')
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_RD_base,xia={},xig={},psi0={},psi1={},psi2={},BC.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_RD_base,xia={},xig={},psi0={},psi1={},psi2={},BC.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])
                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["ME_dmg_base"][res["states"][:, 1]<1.5],label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["ME_dmg_base"][res["states"][:, 1]<1.5],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.title("ME_dmg_base")

                plt.legend(loc='upper right')
                
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_dmg_base,xia={},xig={},psi0={},psi1={},psi2={},BC.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_dmg_base,xia={},xig={},psi0={},psi1={},psi2={},BC.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])
                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["ME_tech_base"][res["states"][:, 1]<1.5],label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["ME_tech_base"][res["states"][:, 1]<1.5],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.title("ME_tech_base")

                plt.legend(loc='upper right')
                
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_tech_base,xia={},xig={},psi0={},psi1={},psi2={},BC.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_tech_base,xia={},xig={},psi0={},psi1={},psi2={},BC.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])
                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["ME_notemp_base"][res["states"][:, 1]<1.5],label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["ME_notemp_base"][res["states"][:, 1]<1.5],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.title("ME_notemp_base")

                plt.legend(loc='upper right')
                
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_notemp_base,xia={},xig={},psi0={},psi1={},psi2={},BC.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_notemp_base,xia={},xig={},psi0={},psi1={},psi2={},BC.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])
                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["ME_nocarb_base"][res["states"][:, 1]<1.5],label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["ME_nocarb_base"][res["states"][:, 1]<1.5],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.title("ME_nocarb_base")

                plt.legend(loc='upper right')
                
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_nocarb_base,xia={},xig={},psi0={},psi1={},psi2={},BC.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_nocarb_base,xia={},xig={},psi0={},psi1={},psi2={},BC.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])
                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["ME_noRD_base"][res["states"][:, 1]<1.5],label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["ME_noRD_base"][res["states"][:, 1]<1.5],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.title("ME_noRD_base")

                plt.legend(loc='upper right')
                
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_noRD_base,xia={},xig={},psi0={},psi1={},psi2={},BC.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_noRD_base,xia={},xig={},psi0={},psi1={},psi2={},BC.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()




for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])
                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["ME_nodmg_base"][res["states"][:, 1]<1.5],label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["ME_nodmg_base"][res["states"][:, 1]<1.5],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.title("ME_nodmg_base")

                plt.legend(loc='upper right')
                
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_nodmg_base,xia={},xig={},psi0={},psi1={},psi2={},BC.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_nodmg_base,xia={},xig={},psi0={},psi1={},psi2={},BC.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])
                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["ME_notech_base"][res["states"][:, 1]<1.5],label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], res["ME_notech_base"][res["states"][:, 1]<1.5],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                if auto==0:   
                    plt.ylim(6.75,6.95)
                plt.title("ME_notech_base")

                plt.legend(loc='upper right')
                
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_notech_base,xia={},xig={},psi0={},psi1={},psi2={},BC.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/ME_notech_base,xia={},xig={},psi0={},psi1={},psi2={},BC.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])
                
                # gt_mean1 = pd.read_csv("./data/gg_damageean1,xig={},psi0={},psi1={},psi2={}.csv".format(xigarr[0],psi0arr[0],psi1arr[0],psi2arr[0]), header=None).to_numpy()[:, 0]

                if xigarr[id_xiag]>10:

                    plt.plot(res["years"][res["states"][:, 1]<1.5], (res["Ambiguity_mean_dis"][res["states"][:, 1]<1.5]-res["Ambiguity_mean_undis"][res["states"][:, 1]<1.5])*1000,label='baseline'.format(psi2arr[id_psi2]))
                else:
                    plt.plot(res["years"][res["states"][:, 1]<1.5], (res["Ambiguity_mean_dis"][res["states"][:, 1]<1.5]-res["Ambiguity_mean_undis"][res["states"][:, 1]<1.5])*1000,label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag]))

                plt.xlabel("Years")
                plt.title("Mean Difference")
                if auto==0:   
                    plt.ylim(0,0.8)
                # plt.legend(loc='upper left')

# if auto==0:
#     plt.plot(res["years"][res["states"][:, 1]<1.5], gt_mean1[res["states"][:, 1]<1.5],label='$\\xi_m={:.3f}$'.format(xigarr[id_xiag]))
plt.legend(loc='upper left')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/MeanDiff,xia={},xig={},psi0={},psi1={},psi2={},BC.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/MeanDiff,xia={},xig={},psi0={},psi1={},psi2={},BC.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()

# for id_xiag in range(len(xiaarr)): 
#     for id_psi0 in range(len(psi0arr)):
#         for id_psi1 in range(len(psi1arr)):
#             for id_psi2 in range(len(psi2arr)):

#                 res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])
                
                
#                 if xigarr[id_xiag]>10:

#                     plt.plot(res["years"][res["states"][:, 1]<1.5], (np.zeros((res["Ambiguity_mean_dis"][res["states"][:, 1]<1.5]-res["Ambiguity_mean_undis"][res["states"][:, 1]<1.5]).shape))*1000,label='baseline'.format(psi2arr[id_psi2]))
#                 else:
#                     plt.plot(res["years"][res["states"][:, 1]<1.5], (np.zeros((res["Ambiguity_mean_dis"][res["states"][:, 1]<1.5]-res["Ambiguity_mean_undis"][res["states"][:, 1]<1.5]).shape))*1000,label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag]))

#                 plt.xlabel("Years")
#                 plt.title("Mean Difference")
#                 if auto==0:   
#                     plt.ylim(0,0.8)
#                 plt.legend(loc='upper left')


# plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/MeanDiff2,xia={},xig={},psi0={},psi1={},psi2={},BC.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
# plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/MeanDiff2,xia={},xig={},psi0={},psi1={},psi2={},BC.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
# plt.close()


# temp = 10000
# num= -5 
# for id_xiag in range(len(xiaarr)): 
#     for id_psi0 in range(len(psi0arr)):
#         for id_psi1 in range(len(psi1arr)):
#             for id_psi2 in range(len(psi2arr)):

#                 res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])
                
#                 gt_mean1 = pd.read_csv("./data/gg_damageean1,xig={},psi0={},psi1={},psi2={}.csv".format(xigarr[0],psi0arr[0],psi1arr[0],psi2arr[0]), header=None).to_numpy()[:, 0]

#                 if xigarr[id_xiag]>10:

#                     plt.plot(res["years"], (res["Ambiguity_mean_dis"]-res["Ambiguity_mean_undis"])*1000-gt_mean1,label='baseline'.format(psi2arr[id_psi2]))
#                 else:
#                     plt.plot(res["years"], (res["Ambiguity_mean_dis"]-res["Ambiguity_mean_undis"])*1000-gt_mean1,label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag]))

#                 plt.xlabel("Years")
#                 plt.title("Mean Difference-1000$\sigma_y$h")
#                 if auto==0:   
#                     plt.ylim(0,0.8)   
#                 plt.legend(loc='upper left')
                
#                 LS = np.sum(((res["Ambiguity_mean_dis"]-res["Ambiguity_mean_undis"])*1000-gt_mean1)**2)
#                 if temp > LS:
#                     temp = LS
#                     num = id_xiag
#                     print("LS at xi_a={} is minimum when xi_g={}, LS = {}" .format(xiaarr[num],xigarr[0],LS))
                
                
#                     plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/MeanDiff-h,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2]))
#                     plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/MeanDiff-h,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2]))
#                     plt.close()

# print("LS at xi_a={} is minimum when xi_g={}, final" .format(xiaarr[num],xigarr[0]))


# temp = 10000
# num= -5 
# for id_xiag in range(len(xiaarr)): 
#     for id_psi0 in range(len(psi0arr)):
#         for id_psi1 in range(len(psi1arr)):
#             for id_psi2 in range(len(psi2arr)):

#                 res = model_solution_extraction(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])
                
#                 if xigarr[0]!=xigarr[1]:
#                     gt_mean1 = np.zeros(res["Ambiguity_mean_dis"].shape)
#                 else:
#                     gt_mean1 = pd.read_csv("./data/gg_damageean1,xig={},psi0={},psi1={},psi2={}.csv".format(xigarr[0],psi0arr[0],psi1arr[0],psi2arr[0]), header=None).to_numpy()[:, 0]

#                 if xigarr[id_xiag]>10:

#                     plt.plot(res["years"], (res["Ambiguity_mean_dis"]-res["Ambiguity_mean_undis"])*1000-gt_mean1,label='baseline'.format(psi2arr[id_psi2]))
#                 else:
#                     plt.plot(res["years"], (res["Ambiguity_mean_dis"]-res["Ambiguity_mean_undis"])*1000-gt_mean1,label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag]))

#                 plt.xlabel("Years")
#                 plt.title("Mean Difference-1000$\sigma_y$h")
#                 if auto==0:   
#                     plt.ylim(0,0.8)   
#                 plt.legend(loc='upper left')

#                 LS = np.sum(((res["Ambiguity_mean_dis"]-res["Ambiguity_mean_undis"])*1000-gt_mean1)**2)
#                 if temp > LS:
#                     temp = LS
#                     num = id_xiag
#                     print("LS at xi_a={} is minimum when xi_g={}, LS = {}" .format(xiaarr[num],xigarr[0],LS))
                

# plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/MeanDiff-h,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
# plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/MeanDiff-h,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
# plt.close()

# print("LS at xi_a={} is minimum when xi_g={}, final" .format(xiaarr[num],xigarr[0]))














