from src.Utility import finiteDiff_3D
import numpy as np
import pandas as pd
import petsc4py
from petsc4py import PETSc
import petsclinearsystem
import petsclinearsystem_new
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



def pde_one_interation_noFT(ksp, petsc_mat, X1_mat_1d, X2_mat_1d, X3_mat_1d, lowerLims, upperLims, dVec, increVec, v0, A, B_1, B_2, B_3, C_1, C_2, C_3, D, tol, epsilon):

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
    petsclinearsystem_new.formLinearSystem_noFT(X1_mat_1d, X2_mat_1d, X3_mat_1d, A_1d, B_1_1d, B_2_1d, B_3_1d, C_1_1d, C_2_1d, C_3_1d, epsilon, lowerLims, upperLims, dVec, increVec, petsc_mat)
    b = D_1d 
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
        jj = alpha * vartheta_bar * (1 - ee / (alpha * lambda_bar * np.exp(K_mat)))**theta
        
        jj[jj <= 1e-16] = 1e-16
        consumption = alpha - ii - jj - xx
        ME_total = delta/ consumption  * alpha * vartheta_bar * theta * (1 - ee / ( alpha * lambda_bar * np.exp(K_mat)))**(theta - 1) /( alpha * lambda_bar * np.exp(K_mat) )

        print("log(ME_total/ME) = [{},{}]".format(np.min(np.log(ME_total / ME)), np.max(np.log(ME_total / ME))))
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
    
    # g_tech = np.exp(1. / xi_g * (v0 - V_post_tech))
    g_tech = np.ones(V_post_tech.shape)
    if model == "Pre damage":
        # g_damage = np.exp(1 / xi_p * (v0 - v_i))
        g_damage = np.ones(v_i.shape)
        ME = - dY * np.sum(pi_c * theta_ell, axis=0) - ddY * sigma_y**2 * ee + dG * np.sum(theta_ell * pi_c, axis=0) +  ddG * sigma_y**2 * ee
        jj = alpha * vartheta_bar * (1 - ee / (alpha * lambda_bar * np.exp(K_mat)))**theta
        
        jj[jj <= 1e-16] = 1e-16
        consumption = alpha - ii - jj - xx
        ME_total = delta/ consumption  * alpha * vartheta_bar * theta * (1 - ee / ( alpha * lambda_bar * np.exp(K_mat)))**(theta - 1) /( alpha * lambda_bar * np.exp(K_mat) )

        print("log(ME_total/ME) = [{},{}]".format(np.min(np.log(ME_total / ME)), np.max(np.log(ME_total / ME))))
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



def hjb_pre_tech_noupdate_noFT(
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

    start_ep = time.time()
    A, B_1, B_2, B_3, C_1, C_2, C_3, D, dK, dY, dL, ddK, ddY, ddL, ii, ee, xx, pi_c, g_tech = _FOC_noupdate(v0, steps= (hK, hY, hL), states = (K_mat, Y_mat, L_mat), args=FOC_args, controls=(ii, ee, xx), fraction=fraction)

    if model == "Pre damage":
        # g_m = np.exp(- (v_i-v0)/xi_p)
        g_m = np.ones(v_i.shape)
        
        D += xi_p * Intensity * np.sum( pi_d_o*(1-g_m+g_m*np.log(g_m)),axis=0) +Intensity*np.sum(pi_d_o*g_m*v_i,axis=0)
        A -=  Intensity*np.sum(pi_d_o*g_m,axis=0)


    out_comp,end_ksp, bpoint1 = pde_one_interation_noFT(
            ksp,
            petsc_mat,K_mat_1d, Y_mat_1d, L_mat_1d, 
            lowerLims, upperLims, dVec, increVec,
            v0, A, B_1, B_2, B_3, C_1, C_2, C_3, D, 1e-13, epsilon)
    # if epoch % 1 == 0 and reporterror:
        # Calculating PDE error and False Transient error
    PDE_rhs = A * v0 + B_1 * dK + B_2 * dY + B_3 * dL + C_1 * ddK + C_2 * ddY + C_3 * ddL + D
    PDE_Err = np.max(abs(PDE_rhs))
    FC_Err = np.max(abs((out_comp - v0)/ epsilon))
    
    # if FC_Err < 2*tol:
    #     print("---------Epoch {:d}: False Transient Error: {:.10f}; Time: {:.4f}---------------".format(epoch, FC_Err, time.time() - start_func), flush=True)
    #     # print("---------Control_ii: [{},\t{}]".format(ii.min(), ii.max()), flush=True)
    # elif epoch%100==0:
    #     print("---------Epoch {:d}: False Transient Error: {:.10f}; Time: {:.4f}---------------".format(epoch, FC_Err, time.time() - start_func), flush=True)
    #     # print("---------Control_ii: [{},\t{}]".format(ii.min(), ii.max()), flush=True)
        
    v0     = out_comp
    epoch += 1
        
    print("---------Converged: Epoch {:d}: False Transient Error: {:.10f}; Time: {:.4f}---------------".format(epoch, FC_Err, time.time() - start_func), flush=True)

    i_star = ii
    e_star = ee
    x_star = xx
    
    # g_tech = np.exp(1. / xi_g * (v0 - V_post_tech))
    g_tech = np.ones(V_post_tech.shape)
    if model == "Pre damage":
        # g_damage = np.exp(1 / xi_p * (v0 - v_i))
        g_damage = np.ones(v_i.shape)
        ME = - dY * np.sum(pi_c * theta_ell, axis=0) - ddY * sigma_y**2 * ee + dG * np.sum(theta_ell * pi_c, axis=0) +  ddG * sigma_y**2 * ee
        jj = alpha * vartheta_bar * (1 - ee / (alpha * lambda_bar * np.exp(K_mat)))**theta
        
        jj[jj <= 1e-16] = 1e-16
        consumption = alpha - ii - jj - xx
        ME_total = delta/ consumption  * alpha * vartheta_bar * theta * (1 - ee / ( alpha * lambda_bar * np.exp(K_mat)))**(theta - 1) /( alpha * lambda_bar * np.exp(K_mat) )

        print("log(ME_total/ME) = [{},{}]".format(np.min(np.log(ME_total / ME)), np.max(np.log(ME_total / ME))))
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



