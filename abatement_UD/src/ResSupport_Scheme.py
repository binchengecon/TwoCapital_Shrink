from src.supportfunctions import finiteDiff_3D
import numpy as np
import pandas as pd
import petsc4py
from petsc4py import PETSc
import petsclinearsystem
import time
import pickle
from datetime import datetime
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import os


def pde_one_interation(ksp, petsc_mat, X1_mat_1d, X2_mat_1d, X3_mat_1d, lowerLims, upperLims, dVec, increVec, v0, A, B_1, B_2, B_3, C_1, C_2, C_3, D, tol, epsilon):

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
    petsclinearsystem.formLinearSystem(X1_mat_1d, X2_mat_1d, X3_mat_1d, A_1d, B_1_1d, B_2_1d, B_3_1d, C_1_1d, C_2_1d, C_3_1d, epsilon, lowerLims, upperLims, dVec, increVec, petsc_mat)
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
    # print("PETSc preconditioned residual norm is {:g}; iterations: {}".format(ksp.getResidualNorm(), ksp.getIterationNumber()))
    return out_comp,end_ksp,bpoint1


def _FOC_partialupdate(v0, steps= (), states = (), args=(), controls=(), fraction=0.5):

    hX1, hX2, hX3 = steps
    K_mat, Y_mat, L_mat = states
    delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell, pi_c_o, pi_c, sigma_y, zeta, psi_0, psi_1, psi_2, sigma_g, V_post_tech, dG, ddG, xi_a, xi_g = args
    ii, ee, xx = controls
    
    # First order derivative
    dX1  = finiteDiff_3D(v0,0,1,hX1)
    dX1[dX1 <= 1e-16] = 1e-16
    dK = dX1
    dX2  = finiteDiff_3D(v0,1,1,hX2)
    dY = dX2
    dX3  = finiteDiff_3D(v0,2,1,hX3)
    dX3[dX3 <= 1e-16] = 1e-16
    dL = dX3
    ######## second order
    ddX1 = finiteDiff_3D(v0,0,2,hX1)
    ddX2 = finiteDiff_3D(v0,1,2,hX2)
    ddY = ddX2
    ddX3 = finiteDiff_3D(v0,2,2,hX3)

    G = dY -  dG
    F = ddY - ddG
    
    # update smooth ambiguity
    log_pi_c_ratio = - G * ee * theta_ell / xi_a
    log_pi_c_ratio += -dL * psi_0 * xx**psi_1 * np.exp( psi_1 * K_mat - (1-psi_2) * L_mat) / xi_a

    pi_c_ratio = log_pi_c_ratio - np.max(log_pi_c_ratio)
    pi_c = np.exp(pi_c_ratio) * pi_c_o
    pi_c = (pi_c <= 0) * 1e-16 + (pi_c > 0) * pi_c
    pi_c = pi_c / np.sum(pi_c, axis=0)
    entropy = np.sum(pi_c * (np.log(pi_c) - np.log(pi_c_o)), axis=0)
    
    # Technology
    g_tech = np.exp(1 / xi_g * (v0 - V_post_tech))
    g_tech[g_tech <=1e-16] = 1e-16
    
    jj =  alpha * vartheta_bar * (1 - ee / (alpha * lambda_bar * np.exp(K_mat)))**theta
    jj[jj <= 1e-16] = 1e-16
    consumption = alpha - ii - jj - xx
    consumption[consumption <= 1e-16] = 1e-16
    
    # Step (2), solve minimization problem in HJB and calculate drift distortion
    A   = - delta * np.ones(K_mat.shape) - np.exp(  L_mat - np.log(448) ) * g_tech
    B_1 = mu_k + ii - 0.5 * kappa * ii**2 - 0.5 * sigma_k**2
    B_2 = np.sum(theta_ell * pi_c, axis=0) * ee
    # B_3 = - zeta + psi_0 * (xx * np.exp(K_mat - L_mat))**psi_1 - 0.5 * sigma_g**2
    B_3 = - zeta + psi_0 * xx** psi_1 * np.exp( psi_1 * K_mat ) * np.sum(pi_c * np.exp( -( 1-psi_2) * L_mat  ), axis=0 )- 0.5 * sigma_g**2

    C_1 = 0.5 * sigma_k**2 * np.ones(K_mat.shape)
    C_2 = 0.5 * sigma_y**2 * ee**2
    C_3 = 0.5 * sigma_g**2 * np.ones(K_mat.shape)
    D = delta * np.log(consumption) + delta * K_mat  - dG * np.sum(theta_ell * pi_c, axis=0) * ee  - 0.5 * ddG * sigma_y**2 * ee**2  + xi_a * entropy + xi_g * np.exp((L_mat - np.log(448))) * (1 - g_tech + g_tech * np.log(g_tech)) + np.exp( (L_mat - np.log(448)) ) * g_tech * V_post_tech

    return A, B_1, B_2, B_3, C_1, C_2, C_3, D, dX1, dX2, dX3, ddX1, ddX2, ddX3, ii, ee, xx, pi_c, g_tech

def _FOC_noupdate(v0, steps= (), states = (), args=(), controls=(), fraction=0.5):

    hX1, hX2, hX3 = steps
    K_mat, Y_mat, L_mat = states
    delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell, pi_c_o, pi_c, sigma_y, zeta, psi_0, psi_1, psi_2, sigma_g, V_post_tech, dG, ddG, xi_a, xi_g = args
    ii, ee, xx = controls
    
    # First order derivative
    dX1  = finiteDiff_3D(v0,0,1,hX1)
    dX1[dX1 <= 1e-16] = 1e-16
    dK = dX1
    dX2  = finiteDiff_3D(v0,1,1,hX2)
    dY = dX2
    dX3  = finiteDiff_3D(v0,2,1,hX3)
    dX3[dX3 <= 1e-16] = 1e-16
    dL = dX3
    ######## second order
    ddX1 = finiteDiff_3D(v0,0,2,hX1)
    ddX2 = finiteDiff_3D(v0,1,2,hX2)
    ddY = ddX2
    ddX3 = finiteDiff_3D(v0,2,2,hX3)

    G = dY -  dG
    F = ddY - ddG
    
    pi_c = np.ones(theta_ell.shape)/ len(theta_ell)

    # update smooth ambiguity
    # log_pi_c_ratio = - G * ee * theta_ell / xi_a
    # log_pi_c_ratio += -dL * psi_0 * xx**psi_1 * np.exp( psi_1 * K_mat - (1-psi_2) * L_mat) / xi_a

    # pi_c_ratio = log_pi_c_ratio - np.max(log_pi_c_ratio)
    # pi_c = np.exp(pi_c_ratio) * pi_c_o
    # pi_c = (pi_c <= 0) * 1e-16 + (pi_c > 0) * pi_c
    # pi_c = pi_c / np.sum(pi_c, axis=0)
    entropy = np.sum(pi_c * (np.log(pi_c) - np.log(pi_c_o)), axis=0)
    
    g_tech = np.ones(v0.shape)

    # # Technology
    # g_tech = np.exp(1 / xi_g * (v0 - V_post_tech))
    # g_tech[g_tech <=1e-16] = 1e-16
    
    jj =  alpha * vartheta_bar * (1 - ee / (alpha * lambda_bar * np.exp(K_mat)))**theta
    jj[jj <= 1e-16] = 1e-16
    consumption = alpha - ii - jj - xx
    consumption[consumption <= 1e-16] = 1e-16
    
    # Step (2), solve minimization problem in HJB and calculate drift distortion
    A   = - delta * np.ones(K_mat.shape) - np.exp(  L_mat - np.log(448) ) * g_tech
    B_1 = mu_k + ii - 0.5 * kappa * ii**2 - 0.5 * sigma_k**2
    B_2 = np.sum(theta_ell * pi_c, axis=0) * ee
    # B_3 = - zeta + psi_0 * (xx * np.exp(K_mat - L_mat))**psi_1 - 0.5 * sigma_g**2
    B_3 = - zeta + psi_0 * xx** psi_1 * np.exp( psi_1 * K_mat ) * np.sum(pi_c * np.exp( -( 1-psi_2) * L_mat  ), axis=0 )- 0.5 * sigma_g**2

    C_1 = 0.5 * sigma_k**2 * np.ones(K_mat.shape)
    C_2 = 0.5 * sigma_y**2 * ee**2
    C_3 = 0.5 * sigma_g**2 * np.ones(K_mat.shape)
    D = delta * np.log(consumption) + delta * K_mat  - dG * np.sum(theta_ell * pi_c, axis=0) * ee  - 0.5 * ddG * sigma_y**2 * ee**2  + xi_a * entropy + xi_g * np.exp((L_mat - np.log(448))) * (1 - g_tech + g_tech * np.log(g_tech)) + np.exp( (L_mat - np.log(448)) ) * g_tech * V_post_tech

    return A, B_1, B_2, B_3, C_1, C_2, C_3, D, dX1, dX2, dX3, ddX1, ddX2, ddX3, ii, ee, xx, pi_c, g_tech

def hjb_pre_tech_partialupdate(
        state_grid=(), 
        model_args=(), 
        control_fixed=(),
        n_bar=(),
        V_post_damage=None, 
        tol=1e-8, 
        epsilon=0.1, 
        fraction=0.1, 
        max_iter=10000,
        v0=None,
        smart_guess=None,
        ):

    start_func = time.time()
    K, Y, L = state_grid
    delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell, sigma_y, zeta, psi_0, psi_1, psi_2, sigma_g, V_post_tech, gamma_1, gamma_2, gamma_3, y_bar, xi_a, xi_g, xi_p = model_args
    ii, ee, xx, v0 = control_fixed
    n_bar = n_bar
    
    Y = Y[:n_bar+1]
    ii = ii[:,:n_bar+1,:]
    ee = ee[:,:n_bar+1,:]
    xx = xx[:,:n_bar+1,:]
    v0 = v0[:,:n_bar+1,:]
    
    V_post_tech = V_post_tech[:,:n_bar+1,:]
    
    if V_post_damage is not None:
        V_post_damage = V_post_damage[:,:,:n_bar+1,:]

    
    K_min, K_max, Y_min, Y_max, L_min, L_max = K.min(), K.max(), Y.min(), Y.max(), L.min(), L.max()
    hK, hY, hL = K[1] - K[0], Y[1] - Y[0], L[1]-L[0]
    nK, nY, nL = len(K), len(Y), len(L)
    
    print("K=[{:.1f},{:.1f},{:.2f},{:d}], Y=[{:.1f},{:.1f},{:.2f},{:d}],L==[{:.1f},{:.1f},{:.2f},{:d}]" .format(K.min(),K.max(),hK,nK, Y.min(),Y.max(),hY,nY, L.min(),L.max(),hL,nL))


    (K_mat, Y_mat, L_mat) = np.meshgrid(K, Y, L, indexing = 'ij')
    
    
    pi_c_o = np.ones(len(theta_ell)) / len(theta_ell)
    pi_c = np.ones(len(theta_ell)) / len(theta_ell)
    
    pi_c_o = np.array([temp * np.ones(K_mat.shape) for temp in pi_c_o ])
    pi_c = np.array([temp * np.ones(K_mat.shape) for temp in pi_c ])
    theta_ell = np.array([temp * np.ones(K_mat.shape) for temp in theta_ell ])
    psi_2 = np.array([temp * np.ones(K_mat.shape) for temp in psi_2 ])
    
    pi_c_o = pi_c_o[:,:,:n_bar+1,:]
    pi_c = pi_c[:,:,:n_bar+1,:]
    theta_ell = theta_ell[:,:,:n_bar+1,:]
    psi_2 = psi_2[:,:,:n_bar+1,:]
    
    K_mat_1d = K_mat.ravel(order='F')
    Y_mat_1d = Y_mat.ravel(order='F')
    L_mat_1d = L_mat.ravel(order='F')
    lowerLims = np.array([K_min, Y_min, L_min], dtype=np.float64)
    upperLims = np.array([K_max, Y_max, L_max], dtype=np.float64)
    
    #### Model type
    if isinstance(gamma_3, (np.ndarray, list)):
        model = "Pre damage"
        pi_d_o = np.ones(len(gamma_3)) / len(gamma_3)
        pi_d_o = np.array([temp * np.ones(K_mat.shape) for temp in pi_d_o ])
        y_bar_lower = 1.5
        r_1 = 1.5
        r_2 = 2.5
        Intensity = r_1 * (np.exp(r_2 / 2 * (Y_mat - y_bar_lower)**2) -1) * (Y_mat > y_bar_lower)
        v_i = V_post_damage
        dG  = gamma_1 + gamma_2 * Y_mat
        ddG = gamma_2 
    else:
        model = "Post damage"
        dG  = gamma_1 + gamma_2 * Y_mat + gamma_3 * (Y_mat - y_bar) * (Y_mat > y_bar)
        ddG = gamma_2 + gamma_3 * (Y_mat > y_bar)

    # Initial setup of HJB
    FC_Err   = 1
    epoch    = 0



    dVec = np.array([hK, hY, hL])
    increVec = np.array([1, nK, nK * nY],dtype=np.int32)

    FOC_args = (delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell, pi_c_o, pi_c, sigma_y, zeta, psi_0, psi_1, psi_2, sigma_g, V_post_tech, dG, ddG, xi_a, xi_g )

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

    # Enter the optimization
    while FC_Err > tol and epoch < max_iter:

        start_ep = time.time()
        A, B_1, B_2, B_3, C_1, C_2, C_3, D, dK, dY, dL, ddK, ddY, ddL, ii, ee, xx, pi_c, g_tech = _FOC_partialupdate(v0, steps= (hK, hY, hL), states = (K_mat, Y_mat, L_mat), args=FOC_args, controls=(ii, ee, xx), fraction=fraction)

        if model == "Pre damage":
            g_m = np.exp(- (v_i-v0)/xi_p)

            D += xi_p * Intensity * np.sum( pi_d_o*(1-g_m+g_m*np.log(g_m)),axis=0) +Intensity*np.sum(pi_d_o*g_m*v_i,axis=0)
            A -=  Intensity*np.sum(pi_d_o*g_m,axis=0)


        out_comp,end_ksp, bpoint1 = pde_one_interation(
                ksp,
                petsc_mat,K_mat_1d, Y_mat_1d, L_mat_1d, 
                lowerLims, upperLims, dVec, increVec,
                v0, A, B_1, B_2, B_3, C_1, C_2, C_3, D, 1e-13, epsilon)
        # if epoch % 1 == 0 and reporterror:
            # Calculating PDE error and False Transient error
        PDE_rhs = A * v0 + B_1 * dK + B_2 * dY + B_3 * dL + C_1 * ddK + C_2 * ddY + C_3 * ddL + D
        PDE_Err = np.max(abs(PDE_rhs))
        FC_Err = np.max(abs((out_comp - v0)/ epsilon))
        
        if FC_Err < 2*tol:
            print("---------Epoch {:d}: False Transient Error: {:.10f}; Time: {:.4f}---------------".format(epoch, FC_Err, time.time() - start_func), flush=True)
            # print("---------Control_ii: [{},\t{}]".format(ii.min(), ii.max()), flush=True)
        elif epoch%100==0:
            print("---------Epoch {:d}: False Transient Error: {:.10f}; Time: {:.4f}---------------".format(epoch, FC_Err, time.time() - start_func), flush=True)
            # print("---------Control_ii: [{},\t{}]".format(ii.min(), ii.max()), flush=True)
            
        v0     = out_comp
        epoch += 1
        
    print("---------Converged: Epoch {:d}: False Transient Error: {:.10f}; Time: {:.4f}---------------".format(epoch, FC_Err, time.time() - start_func), flush=True)

    i_star = ii
    e_star = ee
    x_star = xx
    
    g_tech = np.exp(1. / xi_g * (v0 - V_post_tech))
    if model == "Pre damage":
        g_damage = np.exp(1 / xi_p * (v0 - v_i))
        ME = - dY * np.sum(pi_c * theta_ell, axis=0) - ddY * sigma_y**2 * ee + dG * np.sum(theta_ell * pi_c, axis=0) +  ddG * sigma_y**2 * ee
        print("---------pi_c=[{:.5f},{:.5f}], g_tech=[{:.5f},{:.5f}], g_damage=[{:.5f},{:.5f}]---------------".format(pi_c.min(), pi_c.max(), g_tech.min(), g_tech.max(), g_damage.min(), g_damage.max()), flush=True)
    else:
        print("---------pi_c=[{:.5f},{:.5f}], g_tech=[{:.5f},{:.5f}]---------------".format(pi_c.min(), pi_c.max(), g_tech.min(), g_tech.max()), flush=True)

    res = {
            "v0"    : v0,
            "i_star": i_star,
            "e_star": e_star,
            "x_star": x_star,
            "pi_c"  : pi_c,
            "g_tech": g_tech,
            "FC_Err": FC_Err,
            }
    if model == "Pre damage":
        res = {
                "v0"    : v0,
                "i_star": i_star,
                "e_star": e_star,
                "x_star": x_star,
                "pi_c"  : pi_c,
                "g_tech": g_tech,
                "g_damage": g_damage,
                "ME": ME,
                "FC_Err": FC_Err,
                }
    return res


def hjb_pre_tech_noupdate(
        state_grid=(), 
        model_args=(), 
        control_fixed=(),
        n_bar=(),
        V_post_damage=None, 
        tol=1e-8, 
        epsilon=0.1, 
        fraction=0.1, 
        max_iter=10000,
        v0=None,
        smart_guess=None,
        ):

    start_func = time.time()
    K, Y, L = state_grid
    delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell, sigma_y, zeta, psi_0, psi_1, psi_2, sigma_g, V_post_tech, gamma_1, gamma_2, gamma_3, y_bar, xi_a, xi_g, xi_p = model_args
    ii, ee, xx, v0 = control_fixed
    n_bar = n_bar
    
    Y = Y[:n_bar+1]
    ii = ii[:,:n_bar+1,:]
    ee = ee[:,:n_bar+1,:]
    xx = xx[:,:n_bar+1,:]
    v0 = v0[:,:n_bar+1,:]
    
    V_post_tech = V_post_tech[:,:n_bar+1,:]
    
    if V_post_damage is not None:
        V_post_damage = V_post_damage[:,:,:n_bar+1,:]

    
    K_min, K_max, Y_min, Y_max, L_min, L_max = K.min(), K.max(), Y.min(), Y.max(), L.min(), L.max()
    hK, hY, hL = K[1] - K[0], Y[1] - Y[0], L[1]-L[0]
    nK, nY, nL = len(K), len(Y), len(L)
    
    print("K=[{:.1f},{:.1f},{:.2f},{:d}], Y=[{:.1f},{:.1f},{:.2f},{:d}],L==[{:.1f},{:.1f},{:.2f},{:d}]" .format(K.min(),K.max(),hK,nK, Y.min(),Y.max(),hY,nY, L.min(),L.max(),hL,nL))


    (K_mat, Y_mat, L_mat) = np.meshgrid(K, Y, L, indexing = 'ij')
    
    
    pi_c_o = np.ones(len(theta_ell)) / len(theta_ell)
    pi_c = np.ones(len(theta_ell)) / len(theta_ell)
    
    pi_c_o = np.array([temp * np.ones(K_mat.shape) for temp in pi_c_o ])
    pi_c = np.array([temp * np.ones(K_mat.shape) for temp in pi_c ])
    theta_ell = np.array([temp * np.ones(K_mat.shape) for temp in theta_ell ])
    psi_2 = np.array([temp * np.ones(K_mat.shape) for temp in psi_2 ])
    
    pi_c_o = pi_c_o[:,:,:n_bar+1,:]
    pi_c = pi_c[:,:,:n_bar+1,:]
    theta_ell = theta_ell[:,:,:n_bar+1,:]
    psi_2 = psi_2[:,:,:n_bar+1,:]
    
    K_mat_1d = K_mat.ravel(order='F')
    Y_mat_1d = Y_mat.ravel(order='F')
    L_mat_1d = L_mat.ravel(order='F')
    lowerLims = np.array([K_min, Y_min, L_min], dtype=np.float64)
    upperLims = np.array([K_max, Y_max, L_max], dtype=np.float64)
    
    #### Model type
    if isinstance(gamma_3, (np.ndarray, list)):
        model = "Pre damage"
        pi_d_o = np.ones(len(gamma_3)) / len(gamma_3)
        pi_d_o = np.array([temp * np.ones(K_mat.shape) for temp in pi_d_o ])
        y_bar_lower = 1.5
        r_1 = 1.5
        r_2 = 2.5
        Intensity = r_1 * (np.exp(r_2 / 2 * (Y_mat - y_bar_lower)**2) -1) * (Y_mat > y_bar_lower)
        v_i = V_post_damage
        dG  = gamma_1 + gamma_2 * Y_mat
        ddG = gamma_2 
    else:
        model = "Post damage"
        dG  = gamma_1 + gamma_2 * Y_mat + gamma_3 * (Y_mat - y_bar) * (Y_mat > y_bar)
        ddG = gamma_2 + gamma_3 * (Y_mat > y_bar)

    # Initial setup of HJB
    FC_Err   = 1
    epoch    = 0



    dVec = np.array([hK, hY, hL])
    increVec = np.array([1, nK, nK * nY],dtype=np.int32)

    FOC_args = (delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell, pi_c_o, pi_c, sigma_y, zeta, psi_0, psi_1, psi_2, sigma_g, V_post_tech, dG, ddG, xi_a, xi_g )

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

    # Enter the optimization
    while FC_Err > tol and epoch < max_iter:

        start_ep = time.time()
        A, B_1, B_2, B_3, C_1, C_2, C_3, D, dK, dY, dL, ddK, ddY, ddL, ii, ee, xx, pi_c, g_tech = _FOC_noupdate(v0, steps= (hK, hY, hL), states = (K_mat, Y_mat, L_mat), args=FOC_args, controls=(ii, ee, xx), fraction=fraction)

        if model == "Pre damage":
            # g_m = np.exp(- (v_i-v0)/xi_p)
            g_m = np.ones(v_i.shape)
            
            D += xi_p * Intensity * np.sum( pi_d_o*(1-g_m+g_m*np.log(g_m)),axis=0) +Intensity*np.sum(pi_d_o*g_m*v_i,axis=0)
            A -=  Intensity*np.sum(pi_d_o*g_m,axis=0)


        out_comp,end_ksp, bpoint1 = pde_one_interation(
                ksp,
                petsc_mat,K_mat_1d, Y_mat_1d, L_mat_1d, 
                lowerLims, upperLims, dVec, increVec,
                v0, A, B_1, B_2, B_3, C_1, C_2, C_3, D, 1e-13, epsilon)
        # if epoch % 1 == 0 and reporterror:
            # Calculating PDE error and False Transient error
        PDE_rhs = A * v0 + B_1 * dK + B_2 * dY + B_3 * dL + C_1 * ddK + C_2 * ddY + C_3 * ddL + D
        PDE_Err = np.max(abs(PDE_rhs))
        FC_Err = np.max(abs((out_comp - v0)/ epsilon))
        
        if FC_Err < 2*tol:
            print("---------Epoch {:d}: False Transient Error: {:.10f}; Time: {:.4f}---------------".format(epoch, FC_Err, time.time() - start_func), flush=True)
            # print("---------Control_ii: [{},\t{}]".format(ii.min(), ii.max()), flush=True)
        elif epoch%100==0:
            print("---------Epoch {:d}: False Transient Error: {:.10f}; Time: {:.4f}---------------".format(epoch, FC_Err, time.time() - start_func), flush=True)
            # print("---------Control_ii: [{},\t{}]".format(ii.min(), ii.max()), flush=True)
            
        v0     = out_comp
        epoch += 1
        
    print("---------Converged: Epoch {:d}: False Transient Error: {:.10f}; Time: {:.4f}---------------".format(epoch, FC_Err, time.time() - start_func), flush=True)

    i_star = ii
    e_star = ee
    x_star = xx
    
    g_tech = np.exp(1. / xi_g * (v0 - V_post_tech))
    if model == "Pre damage":
        g_damage = np.exp(1 / xi_p * (v0 - v_i))
        ME = - dY * np.sum(pi_c * theta_ell, axis=0) - ddY * sigma_y**2 * ee + dG * np.sum(theta_ell * pi_c, axis=0) +  ddG * sigma_y**2 * ee
        print("---------pi_c=[{:.5f},{:.5f}], g_tech=[{:.5f},{:.5f}], g_damage=[{:.5f},{:.5f}]---------------".format(pi_c.min(), pi_c.max(), g_tech.min(), g_tech.max(), g_damage.min(), g_damage.max()), flush=True)
    else:
        print("---------pi_c=[{:.5f},{:.5f}], g_tech=[{:.5f},{:.5f}]---------------".format(pi_c.min(), pi_c.max(), g_tech.min(), g_tech.max()), flush=True)

    res = {
            "v0"    : v0,
            "i_star": i_star,
            "e_star": e_star,
            "x_star": x_star,
            "pi_c"  : pi_c,
            "g_tech": g_tech,
            "FC_Err": FC_Err,
            }
    if model == "Pre damage":
        res = {
                "v0"    : v0,
                "i_star": i_star,
                "e_star": e_star,
                "x_star": x_star,
                "pi_c"  : pi_c,
                "g_tech": g_tech,
                "g_damage": g_damage,
                "ME": ME,
                "FC_Err": FC_Err,
                }
    return res




def Damage_Intensity(Y, y_bar_lower=1.5):
    r_1 = 1.5
    r_2 = 2.5
    Intensity = r_1 * (np.exp(r_2 / 2 * (Y - y_bar_lower)**2) -1) * (Y > y_bar_lower)
    return Intensity


def simulate_econ(
    grid = (), 
    model_args = (), 
    controls = (),  
    initial=(np.log(85/0.115), 1.1, 2.4), 
    T0=0, 
    T=40, 
    dt=1/12,
    printing=True):

    K, Y, L = grid
    delta, alpha, kappa, mu_k, sigma_k, beta_f, sigma_y, zeta, sigma_g, gamma_1, gamma_2, y_bar, y_bar_lower, theta, lambda_bar, vartheta_bar, lambda_bar_first, vartheta_bar_first, lambda_bar_second, vartheta_bar_second, num_gamma, gamma_3_list, xi_a, xi_p, xi_g, psi_0, psi_1, psi_2= model_args
    ii, ee, xx, g_tech, g_damage, pi_c, v = controls
    K_0, Y_0, L_0 = initial
    
    (K_mat, Y_mat, L_mat) = np.meshgrid(K, Y, L, indexing = 'ij')
    hK, hY, hL = K[1] - K[0], Y[1] - Y[0], L[1]-L[0]


    print("Simulate_Econ: Grid Information: K=[{},{}], Y=[{},{}], L=[{},{}], Step size=({},{},{})" .format(K.min(),K.max(),Y.min(),Y.max(),L.min(),L.max(), K[1]-K[0], Y[1]-Y[0], L[1]-L[0]), flush=True)

    n_climate = len(pi_c)
    years  = np.arange(T0, T0 + T + dt, dt)
    pers   = len(years)
    n_damage = len(g_damage)
    
    theta_ell = pd.read_csv("./data/model144_p.csv", header=None).to_numpy()[:, 0]/1000.
    pi_c_o = np.ones(len(theta_ell)) / len(theta_ell)
    pi_c_o = np.array([temp * np.ones(K_mat.shape) for temp in pi_c_o])


    dL = finiteDiff_3D(v, 2, 1, hL )

    gridpoints = (K, Y, L)
    i_func = RegularGridInterpolator(gridpoints, ii)
    e_func = RegularGridInterpolator(gridpoints, ee)
    x_func = RegularGridInterpolator(gridpoints, xx)
    tech_func = RegularGridInterpolator(gridpoints, g_tech)
    dL_func   = RegularGridInterpolator(gridpoints, dL)
    

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

    
    def mu_K(i_x):
        return mu_k + i_x - 0.5 * kappa * i_x ** 2  - 0.5 * sigma_k ** 2
    
    def mu_L(Xt, state):
        return -zeta + psi_0 * Xt **psi_1 * (np.exp( psi_1 * state[0]) )  * np.exp( (psi_2-1) * (state[2] ) ) - 0.5 * sigma_g**2
    
    
    hist      = np.zeros([pers, 3])
    i_hist    = np.zeros([pers])
    e_hist    = np.zeros([pers])
    x_hist    = np.zeros([pers])
    scc_hist  = np.zeros([pers])
    gt_tech   = np.zeros([pers])
    gt_mean   = np.zeros([pers])
    dL_hist    = np.zeros([pers])
    gt_dmg    = np.zeros([n_damage, pers])
    pi_c_t = np.zeros([n_climate, pers])
    Ambiguity_mean_undis = np.zeros([pers])
    Ambiguity_mean_dis = np.zeros([pers])
    mu_K_hist = np.zeros([pers])
    mu_L_hist = np.zeros([pers])




    for tm in range(pers):
        
        if tm == 0:
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




        print("time={}, K={},Y={},L={},mu_K={},mu_Y={},mu_L={},ii={},ee={},xx={}" .format(tm, hist[tm,0],hist[tm,1],hist[tm,2],mu_K_hist[tm],beta_f * e_hist[tm],mu_L_hist[tm],i_hist[tm],e_hist[tm],x_hist[tm]), flush=True)
        
    
    jt = 1 - e_hist/ (alpha * lambda_bar * np.exp(hist[:, 0]))
    jt[jt <= 1e-16] = 1e-16
    MU_Emission_div_MC = theta * vartheta_bar / (lambda_bar * np.exp(hist[:, 0]) )* jt**(theta -1)
    
    MC = delta / (alpha  - i_hist - alpha * vartheta_bar * jt**theta - x_hist)
   
    scc_hist = MU_Emission_div_MC * 1000
    
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
        x = x_hist * np.exp(hist[:, 0]),
        scc = scc_hist,
        scrd = scrd_hist,
        gt_tech = gt_tech,
        gt_dmg = gt_dmg,
        distorted_damage_prob=distorted_damage_prob,
        distorted_tech_prob=distorted_tech_prob,
        pic_t = pi_c_t,
        jt = jt,
        years=years,
        true_tech_prob = true_tech_prob,
        true_damage_prob = true_damage_prob,
        Ambiguity_mean_undis = Ambiguity_mean_undis,
        Ambiguity_mean_dis = Ambiguity_mean_dis,
        )
    
    return res





def simulate_UD(
    grid = (), 
    model_args = (), 
    sol_beforeupdate = (),  
    data=(),
    initial=(np.log(85/0.115), 1.1, 2.4), 
    T0=0, 
    T=40, 
    dt=1/12,
    printing=True):
        
    
    K, Y, L, Y_short = grid
    delta, alpha, kappa, mu_k, sigma_k, beta_f, sigma_y, zeta, sigma_g, gamma_1, gamma_2, y_bar, y_bar_lower, theta, lambda_bar, vartheta_bar, lambda_bar_first, vartheta_bar_first, lambda_bar_second, vartheta_bar_second, num_gamma, gamma_3_list, xi_a, xi_p, xi_g, psi_0, psi_1, psi_2 = model_args
    dataname = data
    K_0, Y_0, L_0 = initial
    model_tech2_post_damage, model_tech2_pre_damage, model_tech1_post_damage, model_tech1_pre_damage = sol_beforeupdate

    K_min, K_max, Y_min, Y_max, L_min, L_max, Y_short_min, Y_short_max = min(K), max(K), min(Y), max(Y), min(L), max(L), max(Y_short), min(Y_short) 
    hK, hY, hL = K[1] - K[0], Y[1] - Y[0], L[1]-L[0]
    nK, nY, nL, nY_short = len(K), len(Y), len(L), len(Y_short)
    
    Output_Dir = "/scratch/bincheng/"
    Data_Dir = Output_Dir+"abatement/data_2tech/"+str(dataname)+"/"
    File_Dir = "xi_a_{}_xi_g_{}_psi_0_{}_psi_1_{}_" .format(xi_a,xi_g,psi_0,psi_1)

    (K_mat, Y_mat, L_mat) = np.meshgrid(K, Y, L, indexing = 'ij')
    (K_short_mat, Y_short_mat, L_short_mat) = np.meshgrid(K, Y_short, L, indexing = 'ij')


    print("-------------Import Parameters: psi_2={},xi_a={},xi_g={}--------------".format(psi_2, xi_a, xi_g), flush=True)
    print("K=[{:.1f},{:.1f},{:.2f},{:d}], Y=[{:.1f},{:.1f},{:.2f},{:d}],L==[{:.1f},{:.1f},{:.2f},{:d}],Y_short=[{:.1f},\t{:.1f},{:.2f},{:d}]" .format(K.min(),K.max(),hK,nK, Y.min(),Y.max(),hY,nY, L.min(),L.max(),hL,nL,min(Y_short), max(Y_short),hY,nY_short ))

    ### Prepare Climate Model and RD Models
    theta_ell_array = pd.read_csv("./data/model144_p.csv", header=None).to_numpy()[:, 0]/1000.
    psi_2_array = pd.read_csv('./data/psi2value_p.csv', header=None).to_numpy()[:, 0]
    pi_d_o = np.ones(len(gamma_3_list)) / len(gamma_3_list)
    pi_c_o = np.ones(len(theta_ell_array)) / len(theta_ell_array)

    # theta_list = pd.read_csv('data/model144.csv', header=None).to_numpy()[:, 0]/1000.
    # theta_reshape = theta_list.reshape(n_temp, n_carb)
    
    n_temp = 16
    n_carb = 9
    n_RD = 3
        
    theta_ell_reshape = theta_ell_array.reshape(n_temp, n_carb, n_RD)
    theta_ell_temp = np.mean(theta_ell_reshape, axis=(1,2))
    theta_ell_carb = np.mean(theta_ell_reshape, axis=(0,2))
    theta_ell_RD = np.mean(theta_ell_reshape, axis=(0,1))
    theta_ell_tempcarb = np.mean(theta_ell_reshape, axis=2)
    theta_ell_carbRD = np.mean(theta_ell_reshape, axis=0)
    theta_ell_RDtemp = np.mean(theta_ell_reshape, axis=1)


    psi_2_reshape = psi_2_array.reshape(n_temp, n_carb,n_RD)
    psi_2_temp = np.mean(psi_2_reshape, axis=(1,2))
    psi_2_carb = np.mean(psi_2_reshape, axis=(0,2))
    psi_2_RD = np.mean(psi_2_reshape, axis=(0,1))
    psi_2_tempcarb = np.mean(psi_2_reshape, axis=2)
    psi_2_carbRD = np.mean(psi_2_reshape, axis=0)
    psi_2_RDtemp = np.mean(psi_2_reshape, axis=1)


    # pi_d_o_wakeup = np.array([temp * np.ones((nK, nY_short, nL)) for temp in pi_d_o])
    
    ### Prepare Post Damage and Pre Tech Value Function List
    
    id_2 = np.abs(Y - y_bar).argmin()


    v_i_base = []
    for i in range(num_gamma):

        print("------------Post damage, Tech II: gamma={:.4f}----------".format(gamma_3_list[i]))

        v_post = model_tech2_post_damage[i]["v"]
        V_post_3D = np.zeros_like(K_mat)
        for j in range(nL):
            V_post_3D[:,:,j] = v_post
        
        V_post_tech2 = V_post_3D    


        print("------------Post damage, Tech I: gamma={:.4f}----------".format(gamma_3_list[i]))

        
        model_tech1_post_damage_i_ii = model_tech1_post_damage[i]['i_star']
        model_tech1_post_damage_i_ee = model_tech1_post_damage[i]['e_star']
        model_tech1_post_damage_i_xx = model_tech1_post_damage[i]['x_star']
        model_tech1_post_damage_v0 = model_tech1_post_damage[i]["v0"]

        xi_a_post = 100000.
        xi_g_post = 100000.
        xi_p_post = 100000.
        
        Guess = None
        n_bar = len(Y)-1
        
        res_tp_paru = hjb_pre_tech_partialupdate(
                state_grid=(K, Y, L), 
                model_args=(delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell_array, sigma_y, zeta, psi_0, psi_1, psi_2_array, sigma_g, V_post_tech2, gamma_1, gamma_2, gamma_3_list[i], y_bar, xi_a_post, xi_g_post, xi_p_post),
                control_fixed=(model_tech1_post_damage_i_ii, model_tech1_post_damage_i_ee, model_tech1_post_damage_i_xx, model_tech1_post_damage_v0),
                n_bar = n_bar,
                V_post_damage=None,
                tol=1e-7, 
                epsilon=0.3, 
                fraction=0.1, 
                smart_guess=Guess, 
                max_iter=10000,
                )
        print("-----------PartialUp Save Data: {}------------".format(Data_Dir+ File_Dir))

        with open(Data_Dir+ File_Dir  + "model_tech1_post_damage_gamma_{:.4f}_base_partialupdate".format(gamma_3_list[i]), "wb") as f:
            pickle.dump(res_tp_paru, f)

        res_tp_nou = hjb_pre_tech_noupdate(
                state_grid=(K, Y, L), 
                model_args=(delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell_array, sigma_y, zeta, psi_0, psi_1, psi_2_array, sigma_g, V_post_tech2, gamma_1, gamma_2, gamma_3_list[i], y_bar, xi_a_post, xi_g_post, xi_p_post),
                control_fixed=(model_tech1_post_damage_i_ii, model_tech1_post_damage_i_ee, model_tech1_post_damage_i_xx, model_tech1_post_damage_v0),
                n_bar = n_bar,
                V_post_damage=None,
                tol=1e-7, 
                epsilon=0.3, 
                fraction=0.1, 
                smart_guess=Guess, 
                max_iter=10000,
                )
        print("-----------NoUp Save Data: {}------------".format(Data_Dir+ File_Dir))

        with open(Data_Dir+ File_Dir  + "model_tech1_post_damage_gamma_{:.4f}_base_noupdate".format(gamma_3_list[i]), "wb") as f:
            pickle.dump(res_tp_nou, f)


        # res_tp = pickle.load(open(Data_Dir+ File_Dir  + "model_tech1_post_damage_gamma_{:.4f}_base".format(gamma_3_list[i]), "rb"))

        Local_Output_Dir = "./"
        Local_Data_Dir = Local_Output_Dir+"abatement/data_2tech/"+str(dataname)+"/"
        Local_File_Dir = "xi_a_{}_xi_g_{}_psi_0_{}_psi_1_{}_" .format(xi_a,xi_g,psi_0,psi_1)

        os.makedirs(Local_Data_Dir, exist_ok=True)

        # print("-----------Local PartialUp Save Data: {}------------".format(Local_Data_Dir+ Local_File_Dir))

        # with open(Local_Data_Dir+ Local_File_Dir  + "model_tech1_post_damage_gamma_{:.4f}_base_partialupdate".format(gamma_3_list[i]), "wb") as f:
        #     pickle.dump(res_tp_paru, f)
        # print("-----------Local NoUp Save Data: {}------------".format(Local_Data_Dir+ Local_File_Dir))

        # with open(Local_Data_Dir+ Local_File_Dir  + "model_tech1_post_damage_gamma_{:.4f}_base_noupdate".format(gamma_3_list[i]), "wb") as f:
        #     pickle.dump(res_tp_nou, f)

        v_post_damage_i = res_tp_paru["v0"]
        v_post_damage_temp = np.zeros((nK, nY_short, nL))
        for j in range(nY_short):
            v_post_damage_temp[:, j, :] = v_post_damage_i[:, id_2, :]
        v_i_base.append(v_post_damage_temp)
        
    v_i_base = np.array(v_i_base)

    v_post = model_tech2_pre_damage["v"]
    
    v_tech2 = np.zeros((nK, nY_short, nL))
    for i in range(nL):
        v_tech2[:, :, i] = v_post

    n_bar1 = len(Y_short)-1
    n_bar2 = np.abs(Y_short - y_bar).argmin()
    

    print("-------------Total Start--------------")

    model_tech1_pre_damage_ii = model_tech1_pre_damage["i_star"]
    model_tech1_pre_damage_ee = model_tech1_pre_damage["e_star"]
    model_tech1_pre_damage_xx = model_tech1_pre_damage["x_star"]
    model_tech1_pre_damage_v0 = model_tech1_pre_damage["v0"]
    
    model_tech1_pre_damage_jj = alpha * vartheta_bar * (1 - model_tech1_pre_damage_ee / (alpha * lambda_bar * np.exp(K_short_mat)))**theta
    model_tech1_pre_damage_jj[model_tech1_pre_damage_jj <= 1e-16] = 1e-16
    consumption = alpha - model_tech1_pre_damage_ii - model_tech1_pre_damage_jj - model_tech1_pre_damage_xx
    # ME_total = delta/ consumption  * alpha * vartheta_bar * theta * model_tech1_pre_damage_jj**(theta - 1) /( alpha * lambda_bar * np.exp(K_short_mat) )
    ME_consumption = delta/ consumption
    ME_total = delta/ consumption  * alpha * vartheta_bar * theta * (1 - model_tech1_pre_damage_ee / ( alpha * lambda_bar * np.exp(K_short_mat)))**(theta - 1) /( alpha * lambda_bar * np.exp(K_short_mat) )
    ME_total2 = delta/ consumption  * alpha * vartheta_bar * theta * (1 - model_tech1_pre_damage_ee / ( alpha * lambda_bar * np.exp(K_short_mat)))**(theta - 1) /( alpha * lambda_bar  )
    ME_SCC = ME_total/ME_consumption * 1000
    print("---------ME_total=[{:.6f},{:.6f}]---------------".format(ME_total.min(), ME_total.max()), flush=True)
    print("---------log(ME_total)=[{},{}]---------------".format(np.log(ME_total).min(), np.log(ME_total).max()), flush=True)
    # base
    print("-------------Base Start--------------")
    

    xi_a_base=100000.
    xi_g_base=100000.
    xi_p_base=100000.
    
    args = (delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell_array, sigma_y, zeta, psi_0, psi_1, psi_2_array, sigma_g, v_tech2, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a_base, xi_g_base, xi_p_base)
    grid = (K, Y_short, L)
    controls=(model_tech1_pre_damage_ii, model_tech1_pre_damage_ee, model_tech1_pre_damage_xx, model_tech1_pre_damage_v0)
    
    
    res_base = hjb_pre_tech_partialupdate(state_grid = grid, 
                                         model_args = args, 
                                         control_fixed = controls,
                                         n_bar = n_bar1, 
                                         V_post_damage=v_i_base,
                                         tol=1e-7, 
                                         epsilon=0.3, 
                                         max_iter=10000) # n_bar free
    res_base2 = hjb_pre_tech_partialupdate(state_grid = grid, 
                                         model_args = args, 
                                         control_fixed = controls,
                                         n_bar = n_bar2, 
                                         V_post_damage=v_i_base,
                                         tol=1e-7, 
                                         epsilon=0.3, 
                                         max_iter=10000) #n_bar hit 2


    with open(Data_Dir+ File_Dir  + "model_tech1_pre_damage_base_scheme1_partialupdate", "wb") as f:
        pickle.dump(res_base, f)
    with open(Data_Dir+ File_Dir  + "model_tech1_pre_damage_base2_scheme1_partialupdate", "wb") as f:
        pickle.dump(res_base2, f)

    # print("-----------Local PartialUp Save Data Final: {}------------".format(Local_Data_Dir+ Local_File_Dir))

    # with open(Local_Data_Dir+ Local_File_Dir  + "model_tech1_pre_damage_base_scheme1_partialupdate", "wb") as f:
    #     pickle.dump(res_base, f)

    # print("-----------Local PartialUp Save Data Final: {}------------".format(Local_Data_Dir+ Local_File_Dir))

    # with open(Local_Data_Dir+ Local_File_Dir  + "model_tech1_pre_damage_base2_scheme1_partialupdate", "wb") as f:
    #     pickle.dump(res_base2, f)

    res_base_nou = hjb_pre_tech_noupdate(state_grid = grid, 
                                         model_args = args, 
                                         control_fixed = controls,
                                         n_bar = n_bar1, 
                                         V_post_damage=v_i_base,
                                         tol=1e-7, 
                                         epsilon=0.3, 
                                         max_iter=10000) # n_bar free
    res_base2_nou = hjb_pre_tech_noupdate(state_grid = grid, 
                                         model_args = args, 
                                         control_fixed = controls,
                                         n_bar = n_bar2, 
                                         V_post_damage=v_i_base,
                                         tol=1e-7, 
                                         epsilon=0.3, 
                                         max_iter=10000) #n_bar hit 2


    with open(Data_Dir+ File_Dir  + "model_tech1_pre_damage_base_scheme1_noupdate", "wb") as f:
        pickle.dump(res_base_nou, f)
    with open(Data_Dir+ File_Dir  + "model_tech1_pre_damage_base2_scheme1_noupdate", "wb") as f:
        pickle.dump(res_base2_nou, f)

    # print("-----------Local NoUp Save Data Final: {}------------".format(Local_Data_Dir+ Local_File_Dir))

    # with open(Local_Data_Dir+ Local_File_Dir  + "model_tech1_pre_damage_base_scheme1_noupdate", "wb") as f:
    #     pickle.dump(res_base_nou, f)

    # print("-----------Local NoUp Save Data Final: {}------------".format(Local_Data_Dir+ Local_File_Dir))

    # with open(Local_Data_Dir+ Local_File_Dir  + "model_tech1_pre_damage_base2_scheme1_noupdate", "wb") as f:
    #     pickle.dump(res_base2_nou, f)

    # res_base = pickle.load(open(Data_Dir+ File_Dir  + "model_tech1_pre_damage_base_scheme1", "rb"))
    # res_base2 = pickle.load(open(Data_Dir+ File_Dir  + "model_tech1_pre_damage_base2_scheme1", "rb"))
 
    print("---------ME_base=[{:.6f},{:.6f}]---------------".format(res_base["ME"].min(), res_base["ME"].max()), flush=True)
    print("---------ME_base2=[{:.6f},{:.6f}]---------------".format(res_base2["ME"].min(), res_base2["ME"].max()), flush=True)


    print("Look at differences: Start")
    ME_base = res_base["ME"]
    ME_base2 = res_base2["ME"]
    print(ME_base.shape, ME_base2.shape, n_bar1,n_bar2, np.max(abs(ME_base[:,:n_bar2+1,:]-ME_base2)))
    print("Look at differences: End")
    print("-------------Base Done--------------")


    v0 = res_base["v0"]
    
    Y_short = Y_short[:n_bar1+1]
    ii = res_base["i_star"]
    ee = res_base["e_star"]
    xx = res_base["x_star"]
    

    years  = np.arange(T0, T0 + T + dt, dt)
    pers   = len(years)
       
    gridpoints = (K, Y_short, L)

    i_func = RegularGridInterpolator(gridpoints, ii)
    e_func = RegularGridInterpolator(gridpoints, ee)
    x_func = RegularGridInterpolator(gridpoints, xx)

    ME_consumption_func = RegularGridInterpolator(gridpoints, ME_consumption)
    ME_SCC_func = RegularGridInterpolator(gridpoints, ME_SCC)
    ME_total_func = RegularGridInterpolator(gridpoints, ME_total)
    ME_total2_func = RegularGridInterpolator(gridpoints, ME_total2)
    ME_base_func = RegularGridInterpolator(gridpoints, ME_base)
    
    # ME_temp_base_func = RegularGridInterpolator(gridpoints, ME_temp_base)
    # ME_carb_base_func = RegularGridInterpolator(gridpoints, ME_carb_base)
    # ME_RD_base_func = RegularGridInterpolator(gridpoints, ME_RD_base)
    # ME_dmg_base_func = RegularGridInterpolator(gridpoints, ME_dmg_base)
    # ME_tech_base_func = RegularGridInterpolator(gridpoints, ME_tech_base)
    
    # ME_notemp_base_func = RegularGridInterpolator(gridpoints, ME_notemp_base)
    # ME_nocarb_base_func = RegularGridInterpolator(gridpoints, ME_nocarb_base)
    # ME_noRD_base_func = RegularGridInterpolator(gridpoints, ME_noRD_base)
    # ME_nodmg_base_func = RegularGridInterpolator(gridpoints, ME_nodmg_base)
    # ME_notech_base_func = RegularGridInterpolator(gridpoints, ME_notech_base)
     

    def get_i(x):
        return i_func(x)

    def get_e(x):
        return e_func(x)
    
    def get_x(x):
        return x_func(x)



    def mu_K(i_x):
        return mu_k + i_x - 0.5 * kappa * i_x ** 2  - 0.5 * sigma_k ** 2
    
    def mu_L(Xt, state):
        # return -zeta + psi_0 * Xt **psi_1 * (np.exp( psi_1 * state[0]) )  * np.exp( (psi_2-1) * (state[2] - np.log(448)) ) - 0.5 * sigma_g**2
        return -zeta + psi_0 * Xt **psi_1 * (np.exp( psi_1 * state[0]) )  * np.exp( (psi_2-1) * (state[2] ) ) - 0.5 * sigma_g**2
    
    
    hist      = np.zeros([pers, 3])
    i_hist    = np.zeros([pers])
    e_hist    = np.zeros([pers])
    x_hist    = np.zeros([pers])


    
    ME_consumption_hist = np.zeros([pers])
    ME_SCC_hist = np.zeros([pers])
    ME_total_hist = np.zeros([pers])
    ME_total2_hist = np.zeros([pers])
    ME_base_hist = np.zeros([pers])
    # ME_base_hist = np.zeros([pers])
    # ME_temp_hist = np.zeros([pers])
    # ME_carb_hist = np.zeros([pers])
    # ME_RD_hist = np.zeros([pers])
    # ME_dmg_hist = np.zeros([pers])
    # ME_tech_hist = np.zeros([pers])

    # ME_notemp_hist = np.zeros([pers])
    # ME_nocarb_hist = np.zeros([pers])
    # ME_noRD_hist = np.zeros([pers])
    # ME_nodmg_hist = np.zeros([pers])
    # ME_notech_hist = np.zeros([pers])


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

            ME_consumption_hist[0] = ME_consumption_func(hist[0,:])
            ME_SCC_hist[0] = ME_SCC_func(hist[0,:])          
            ME_total_hist[0] = ME_total_func(hist[0,:])
            ME_total2_hist[0] = ME_total2_func(hist[0,:])
            ME_base_hist[0] = ME_base_func(hist[0,:])
            # ME_base_hist[tm] = ME_base_func(hist[0,:])
            # ME_temp_hist[tm] = ME_temp_func(hist[0,:])
            # ME_carb_hist[tm] = ME_carb_func(hist[0,:])
            # ME_RD_hist[tm] = ME_RD_func(hist[0,:])
            # ME_dmg_hist[tm] = ME_dmg_func(hist[0,:])
            # ME_tech_hist[tm] = ME_tech_func(hist[0,:])
            # ME_notemp_hist[tm] = ME_notemp_func(hist[0,:])
            # ME_nocarb_hist[tm] = ME_nocarb_func(hist[0,:])
            # ME_noRD_hist[tm] = ME_noRD_func(hist[0,:])
            # ME_nodmg_hist[tm] = ME_nodmg_func(hist[0,:])
            # ME_notech_hist[tm] = ME_notech_func(hist[0,:])

            # print(hist[0,:])
        else:
            # other periods
            # print(hist[tm-1,:])
            i_hist[tm] = get_i(hist[tm-1,:])
            e_hist[tm] = get_e(hist[tm-1,:])
            x_hist[tm] = get_x(hist[tm-1,:])
            
            mu_K_hist[tm] = mu_K(i_hist[tm])
            mu_L_hist[tm] = mu_L(x_hist[tm], hist[tm-1, :])
            
  
            hist[tm,0] = hist[tm-1,0] + mu_K_hist[tm] * dt #logK
            hist[tm,1] = hist[tm-1,1] + beta_f * e_hist[tm] * dt
            hist[tm,2] = hist[tm-1,2] + mu_L_hist[tm] * dt # logλ
            
            ME_consumption_hist[tm] = ME_consumption_func(hist[tm,:])
            ME_SCC_hist[tm] = ME_SCC_func(hist[tm,:])          
            ME_total_hist[tm] = ME_total_func(hist[tm,:])
            ME_total2_hist[tm] = ME_total2_func(hist[tm,:])
            ME_base_hist[tm] = ME_base_func(hist[tm,:])
            # ME_base_hist[tm] = ME_base_func(hist[tm-1,:])
            # ME_temp_hist[tm] = ME_temp_func(hist[tm-1,:])
            # ME_carb_hist[tm] = ME_carb_func(hist[tm-1,:])
            # ME_RD_hist[tm] = ME_RD_func(hist[tm-1,:])
            # ME_dmg_hist[tm] = ME_dmg_func(hist[tm-1,:])
            # ME_tech_hist[tm] = ME_tech_func(hist[tm-1,:])
            
            # ME_notemp_hist[tm] = ME_notemp_func(hist[tm-1,:])
            # ME_nocarb_hist[tm] = ME_nocarb_func(hist[tm-1,:])
            # ME_noRD_hist[tm] = ME_noRD_func(hist[tm-1,:])
            # ME_nodmg_hist[tm] = ME_nodmg_func(hist[tm-1,:])
            # ME_notech_hist[tm] = ME_notech_func(hist[tm-1,:])



        print("time={}, K={},Y={},L={},mu_K={},mu_Y={},mu_L={},ii={},ee={},xx={},ME_total_base={:.3}" .format(tm, hist[tm,0],hist[tm,1],hist[tm,2],mu_K_hist[tm],beta_f * e_hist[tm],mu_L_hist[tm],i_hist[tm],e_hist[tm],x_hist[tm],np.log(ME_total_hist[tm]/ME_base_hist[tm])*100), flush=True)
        


    print("----------------------------------------")
    print("ratio list shape------------------------")
    print(ME_total_hist.shape)
    print(ME_base_hist.shape)
    # print((ME_total_hist / ME_base_hist ).shape)
    # print((np.log(ME_total_hist / ME_base_hist ) * 100).shape)
    # print(ME_total_base_hist.shape)
    print("----------------------------------------")

    res = dict(
        ME_consumption = ME_consumption_hist,
        ME_SCC = ME_SCC_hist,
        ME_total = ME_total_hist,
        ME_total2 = ME_total2_hist,
        ME_base = ME_base_hist,
        ME_total_base = np.log(ME_total_hist / ME_base_hist ) * 100,
        ME_total2_base = np.log(ME_total2_hist / ME_base_hist ) * 100,
        # ME_temp = ME_temp_hist,
        # ME_carb = ME_carb_hist,
        # ME_RD = ME_RD_hist,
        # ME_dmg = ME_dmg_hist,
        # ME_tech = ME_tech_hist,
        # ME_notemp = ME_notemp_hist,
        # ME_nocarb = ME_nocarb_hist,
        # ME_noRD = ME_noRD_hist,
        # ME_nodmg = ME_nodmg_hist,
        # ME_notech = ME_notech_hist,
        # ME_total_base = ME_total_base_hist,
        # ME_temp_base = ME_temp_base_hist,
        # ME_carb_base = ME_carb_base_hist,
        # ME_RD_base = ME_RD_base_hist,
        # ME_dmg_base = ME_dmg_base_hist,
        # ME_tech_base = ME_tech_base_hist,
        # ME_notemp_base = ME_notemp_base_hist,
        # ME_nocarb_base = ME_nocarb_base_hist,
        # ME_noRD_base = ME_noRD_base_hist,
        # ME_nodmg_base = ME_nodmg_base_hist,
        # ME_notech_base = ME_notech_base_hist,
        )
    

    
    
    return res





def model_simulation_generate(grid=(),
                              data=(),
                              varying_argument=(),
                              constant_argument=()):

    ### Argument Extraction
    
    Xminarr, Xmaxarr, hXarr = grid
    dataname = data
    xi_a, xi_g, psi_0, psi_1, psi_2, IntPeriod, timespan = varying_argument
    delta, alpha, kappa, mu_k, sigma_k, beta_f, sigma_y, zeta, sigma_g, gamma_1, gamma_2, y_bar, y_bar_lower, theta, lambda_bar, vartheta_bar, lambda_bar_first, vartheta_bar_first, lambda_bar_second, vartheta_bar_second, num_gamma, gamma_3_list = constant_argument
    
    ### Build Grid

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

    ### Build Original Data

    Output_Dir = "/scratch/bincheng/"
    Data_Dir = Output_Dir+"abatement/data_2tech/"+dataname+"/"
    File_Dir = "xi_a_{}_xi_g_{}_psi_0_{}_psi_1_{}_" .format(xi_a,xi_g,psi_0,psi_1)
    
    print("-----------Load Data: {}------------".format(Data_Dir+ File_Dir))

    print("-----------Load Data: model_tech2_post_damage------------")
    model_tech2_post_damage = pickle.load(open(Data_Dir+ File_Dir + "model_tech2_post_damage", "rb"))
    print("-----------Load Data: model_tech1_post_damage------------")

    model_tech1_post_damage = pickle.load(open(Data_Dir+ File_Dir + "model_tech1_post_damage", "rb"))
    print("-----------Load Data: model_tech2_pre_damage------------")
    model_tech2_pre_damage = pickle.load(open(Data_Dir+ File_Dir + "model_tech2_pre_damage", "rb"))
    print("-----------Load Data: model_tech1_pre_damage------------")
    model_tech1_pre_damage = pickle.load(open(Data_Dir + File_Dir+"model_tech1_pre_damage", "rb"))

    #####Simulating Uncertainty Decomposition Path

    combined_argument_uncertainty = (delta, alpha, kappa, mu_k, sigma_k, beta_f, sigma_y, zeta, sigma_g, gamma_1, gamma_2, y_bar, y_bar_lower, theta, lambda_bar, vartheta_bar, lambda_bar_first, vartheta_bar_first, lambda_bar_second, vartheta_bar_second, num_gamma, gamma_3_list, xi_a, xi_g, xi_g, psi_0, psi_1, psi_2)

    ### Check Result Reasoning: Post tech not affected by uncerainty decomposition
    
    grid_UD = (K, Y, L, Y_short)

    for i in range(num_gamma):
        e_temp1 = model_tech2_post_damage[i]["e"]
        print("Start Checking Post tech emission: gamma={:.4f}, model_tech2_post_damage: emax={}, emin={}" .format(gamma_3_list[i], e_temp1.max(), e_temp1.min()))
    e_temp2 = model_tech2_pre_damage["e"]
    print("Start Checking Post tech emission: model_tech2_pre_damage: emax={}, emin={}" .format(e_temp2.max(), e_temp2.min()))
    print("End Checking Post Damage Emission")

    sol_beforeupdate = (model_tech2_post_damage, model_tech2_pre_damage, model_tech1_post_damage, model_tech1_pre_damage)
    res_UD = simulate_UD(grid = grid_UD,
                        model_args = combined_argument_uncertainty,
                        sol_beforeupdate = sol_beforeupdate,
                        data = dataname,
                        T0 = 0, 
                        T = IntPeriod, 
                        dt = timespan,
                        printing = True)
    
    
    #####Simulating Economic Variable Path
    # combined_argument_econ = (delta, mu_k, kappa,sigma_k, beta_f, zeta, psi_0, psi_1, psi_2, sigma_g, theta, lambda_bar, vartheta_bar, xi_a, xi_g, xi_g)
    combined_argument_econ = (delta, alpha, kappa, mu_k, sigma_k, beta_f, sigma_y, zeta, sigma_g, gamma_1, gamma_2, y_bar, y_bar_lower, theta, lambda_bar, vartheta_bar, lambda_bar_first, vartheta_bar_first, lambda_bar_second, vartheta_bar_second, num_gamma, gamma_3_list, xi_a, xi_g, xi_g, psi_0, psi_1, psi_2)

    model_tech1_pre_damage_v = model_tech1_pre_damage["v0"]
    model_tech1_pre_damage_i = model_tech1_pre_damage["i_star"]
    model_tech1_pre_damage_e = model_tech1_pre_damage["e_star"]
    model_tech1_pre_damage_x = model_tech1_pre_damage["x_star"]
    model_tech1_pre_damage_pi_c = model_tech1_pre_damage["pi_c"]
    model_tech1_pre_damage_g_tech = model_tech1_pre_damage["g_tech"]
    model_tech1_pre_damage_g_damage =  model_tech1_pre_damage["g_damage"]

    grid_econ = (K, Y_short, L)
    control_econ = (model_tech1_pre_damage_i, model_tech1_pre_damage_e, model_tech1_pre_damage_x, model_tech1_pre_damage_g_tech, model_tech1_pre_damage_g_damage, model_tech1_pre_damage_pi_c, model_tech1_pre_damage_v)
    res_econ = simulate_econ(  grid = grid_econ, 
                    model_args = combined_argument_econ,                               
                    controls = control_econ,
                    T0 = 0, 
                    T = IntPeriod, 
                    dt = timespan,
                    printing = True)

    res = {**res_UD,**res_econ}
    
    with open(Data_Dir+ File_Dir + "simulatedpath_{}".format(IntPeriod), "wb") as f:
        pickle.dump(res,f)
    
    print(res.keys())
    return 0






def model_simulation_graph(grid=(),
                              data=(),
                              varying_argument=(),
                              constant_argument=()):

    ### Argument Extraction
    
    Xminarr, Xmaxarr, hXarr = grid
    dataname = data
    xi_a, xi_g, psi_0, psi_1, psi_2, IntPeriod, timespan = varying_argument
    delta, alpha, kappa, mu_k, sigma_k, beta_f, sigma_y, zeta, sigma_g, gamma_1, gamma_2, y_bar, y_bar_lower, theta, lambda_bar, vartheta_bar, lambda_bar_first, vartheta_bar_first, lambda_bar_second, vartheta_bar_second, num_gamma, gamma_3_list = constant_argument
    
    Output_Dir = "/scratch/bincheng/"
    Data_Dir = Output_Dir+"abatement/data_2tech/"+dataname+"/"
    File_Dir = "xi_a_{}_xi_g_{}_psi_0_{}_psi_1_{}_" .format(xi_a,xi_g,psi_0,psi_1)
    

    res= pickle.load(open(Data_Dir+ File_Dir + "simulatedpath_{}".format(IntPeriod), "rb"))



    return res