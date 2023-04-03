"""
solver.py
For 3D abatement solver
"""
import os
import sys
sys.path.append("../../src/")
from src.Utility import finiteDiff_3D
import SolveLinSys
import numpy as np
import petsc4py
from petsc4py import PETSc
import petsclinearsystem
import petsclinearsystem_new
import time
from datetime import datetime


def PDESolver(stateSpace, A, B_r, B_f, B_k, C_rr, C_ff, C_kk, D, v0, ε = 1, tol = -10, smartguess = False, solverType = 'False Transient'):

    if solverType == 'False Transient':
        A = A.reshape(-1,1,order = 'F')
        B = np.hstack([B_r.reshape(-1,1,order = 'F'),B_f.reshape(-1,1,order = 'F'),B_k.reshape(-1,1,order = 'F')])
        C = np.hstack([C_rr.reshape(-1,1,order = 'F'), C_ff.reshape(-1,1,order = 'F'), C_kk.reshape(-1,1,order = 'F')])
        D = D.reshape(-1,1,order = 'F')
        v0 = v0.reshape(-1,1,order = 'F')
        out = SolveLinSys.solveFT(stateSpace, A, B, C, D, v0, ε, tol)

        return out

    elif solverType == 'Feyman Kac':
        
        if smartguess:
            iters = 1
        else:
            iters = 4000000
            
        A = A.reshape(-1, 1, order='F')
        B = np.hstack([B_r.reshape(-1, 1, order='F'), B_f.reshape(-1, 1, order='F'), B_k.reshape(-1, 1, order='F')])
        C = np.hstack([C_rr.reshape(-1, 1, order='F'), C_ff.reshape(-1, 1, order='F'), C_kk.reshape(-1, 1, order='F')])
        D = D.reshape(-1, 1, order='F')
        v0 = v0.reshape(-1, 1, order='F')
        out = SolveLinSys.solveFK(stateSpace, A, B, C, D, v0, iters)

        return out


def fk_pre_tech(
        state_grid=(), 
        model_args=(), 
        controls=(),
        VF=(),
        FFK=(),
        V_post_damage=None, 
        tol=1e-8, epsilon=0.1, fraction=0.5, max_iter=10000,
        v0=None,
        smart_guess=None,
        ):

    K, Y, L = state_grid
    delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell, pi_c_o, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3, y_bar, xi_a, xi_g, xi_p = model_args
    

    
    hL = L[1]-L[0]
    ######## post jump, 3 states
    (X1_mat, X2_mat, X3_mat) = np.meshgrid(K, Y, L, indexing = 'ij')
    stateSpace = np.hstack([X1_mat.reshape(-1,1,order = 'F'), X2_mat.reshape(-1,1,order = 'F'), X3_mat.reshape(-1, 1, order='F')])
    
    K_mat = X1_mat
    Y_mat = X2_mat
    L_mat = X3_mat
    # For PETSc
    X1_mat_1d = X1_mat.ravel(order='F')
    X2_mat_1d = X2_mat.ravel(order='F')
    X3_mat_1d = X3_mat.ravel(order='F')
    
    #### Model type
    if isinstance(gamma_3, (np.ndarray, list)):
        model = "Pre damage"
        pi_d_o = np.ones(len(gamma_3)) / len(gamma_3)
        pi_d_o = np.array([temp * np.ones(K_mat.shape) for temp in pi_d_o ])
        # v_i = V_post_damage
        y_bar_lower = 1.5
        r_1 = 1.5
        r_2 = 2.5
        Intensity = r_1 * (np.exp(r_2 / 2 * (Y_mat - y_bar_lower)**2) -1) * (Y_mat > y_bar_lower)
        i,e,x,pi_c,g_tech,g_damage = controls
        
        Phi_II, Phi = VF
        F_II, F_m = FFK


        dv0dL = finiteDiff_3D(Phi, 2, 1, hL)       

        A = -delta * np.ones(K_mat.shape) - psi_0 * psi_1 * (x * np.exp(K_mat-L_mat) )**psi_1 - np.exp(L_mat - np.log(448)) * g_tech - Intensity*np.sum(pi_d_o*g_damage,axis=0)
        B_1 = mu_k + i - 0.5 * kappa * i**2 - 0.5 * sigma_k**2
        B_2 = np.sum(theta_ell * pi_c, axis=0) * e
        B_3 = - zeta + psi_0 * (x * np.exp(K_mat - L_mat))**psi_1 - 0.5 * sigma_g**2

        C_1 = 0.5 * sigma_k**2 * np.ones(K_mat.shape)
        C_2 = 0.5 * sigma_y**2 * e**2
        C_3 = 0.5 * sigma_g**2 * np.ones(K_mat.shape)

        D = np.exp(L_mat - np.log(448)) * g_tech * (Phi_II - Phi)  + np.exp(L_mat - np.log(448)) * g_tech * F_II + Intensity * np.sum(pi_d_o*g_damage* F_m,axis=0)  
        D += xi_g * np.exp(L_mat - np.log(448)) * (1-g_tech +g_tech *np.log(g_tech))

        out = PDESolver(stateSpace, A, B_1, B_2, B_3, C_1, C_2, C_3, D, dv0dL, epsilon, solverType="Feyman Kac")
        v  = out[2].reshape(dv0dL.shape, order="F")
            
        dvdL = finiteDiff_3D(v, 2, 1, hL)    
        dvdL_orig = finiteDiff_3D(Phi, 2, 1, hL)    
        print("sanity check: {}".format(np.max(abs(dvdL-dvdL_orig))))

        
    else:
        model = "Post damage"
        i,e,x,pi_c,g_tech = controls
        
        Phi_m_II, Phi_m = VF
        F_m_II = FFK

        dv0dL = finiteDiff_3D(Phi_m, 2, 1, hL)       

        A = -delta * np.ones(K_mat.shape) - psi_0 * psi_1 * (x * np.exp(K_mat-L_mat) )**psi_1 - np.exp(L_mat - np.log(448)) * g_tech
        B_1 = mu_k + i - 0.5 * kappa * i**2 - 0.5 * sigma_k**2
        B_2 = np.sum(theta_ell * pi_c, axis=0) * e
        B_3 = - zeta + psi_0 * (x * np.exp(K_mat - L_mat))**psi_1 - 0.5 * sigma_g**2

        C_1 = 0.5 * sigma_k**2 * np.ones(K_mat.shape)
        C_2 = 0.5 * sigma_y**2 * e**2
        C_3 = 0.5 * sigma_g**2 * np.ones(K_mat.shape)

        D = np.exp(L_mat - np.log(448)) * g_tech * (Phi_m_II - Phi_m)+ np.exp(L_mat - np.log(448)) * g_tech * F_m_II
        D += xi_g * np.exp(L_mat - np.log(448)) * (1-g_tech +g_tech *np.log(g_tech))

        out = PDESolver(stateSpace, A, B_1, B_2, B_3, C_1, C_2, C_3, D, dv0dL, epsilon, solverType="Feyman Kac")
        v  = out[2].reshape(dv0dL.shape, order="F")
            
        dvdL = v
        dvdL_orig = finiteDiff_3D(Phi_m, 2, 1, hL)    
        
        print("sanity check: {}".format(np.max(abs(dvdL-dvdL_orig))))
        
    res = {
            "v0"    : v,
            "i_star": i,
            "e_star": e,
            "x_star": x,
            "pi_c"  : pi_c,
            "g_tech": g_tech,
            "dvdL": dvdL,
            }
    if model == "Pre damage":
        res = {
                "v0"    : v,
                "i_star": i,
                "e_star": e,
                "x_star": x,
                "pi_c"  : pi_c,
                "g_tech": g_tech,
                "g_damage": g_damage,
                "dvdL": dvdL,
                }
    return res


def fk_pre_tech_petsc(
        state_grid=(), 
        model_args=(), 
        controls=(),
        VF=(),
        FFK=(),
        V_post_damage=None, 
        tol=1e-8, epsilon=0.1, fraction=0.5, max_iter=10000,
        v0=None,
        smart_guess=None,
        ):

    K, Y, L = state_grid
    delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell, pi_c_o, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3, y_bar, xi_a, xi_g, xi_p = model_args
    
    K_min, K_max, Y_min, Y_max, L_min, L_max = K.min(), K.max(), Y.min(), Y.max(), L.min(), L.max()
    hK, hY, hL = K[1] - K[0], Y[1] - Y[0], L[1]-L[0]
    nK, nY, nL = len(K), len(Y), len(L)
    
    ######## post jump, 3 states
    (K_mat, Y_mat, L_mat) = np.meshgrid(K, Y, L, indexing = 'ij')
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

    
    #### Model type
    if isinstance(gamma_3, (np.ndarray, list)):
        model = "Pre damage"
        pi_d_o = np.ones(len(gamma_3)) / len(gamma_3)
        pi_d_o = np.array([temp * np.ones(K_mat.shape) for temp in pi_d_o ])
        # v_i = V_post_damage
        y_bar_lower = 1.5
        r_1 = 1.5
        r_2 = 2.5
        Intensity = r_1 * (np.exp(r_2 / 2 * (Y_mat - y_bar_lower)**2) -1) * (Y_mat > y_bar_lower)
        i,e,x,pi_c,g_tech,g_damage = controls
        
        Phi_II, Phi = VF
        F_II, F_m = FFK


        dv0dL = finiteDiff_3D(Phi, 2, 1, hL)       

        A = -delta * np.ones(K_mat.shape) - psi_0 * psi_1 * (x * np.exp(K_mat-L_mat) )**psi_1 
        B_1 = mu_k + i - 0.5 * kappa * i**2 - 0.5 * sigma_k**2
        B_2 = np.sum(theta_ell * pi_c, axis=0) * e
        B_3 = - zeta + psi_0 * (x * np.exp(K_mat - L_mat))**psi_1 - 0.5 * sigma_g**2

        C_1 = 0.5 * sigma_k**2 * np.ones(K_mat.shape)
        C_2 = 0.5 * sigma_y**2 * e**2
        C_3 = 0.5 * sigma_g**2 * np.ones(K_mat.shape)

        # D = np.zeros(A.shape)
        A += - np.exp(L_mat - np.log(448)) * g_tech - Intensity*np.sum(pi_d_o*g_damage,axis=0)
        # D = np.exp(L_mat - np.log(448)) * g_tech * (Phi_II - Phi)  + np.exp(L_mat - np.log(448)) * g_tech * F_II + Intensity * np.sum(pi_d_o*g_damage* F_m,axis=0)  
        D = np.exp(L_mat - np.log(448)) * g_tech * (Phi_II - Phi)  + np.exp(L_mat - np.log(448)) * g_tech * F_II + Intensity * np.sum(pi_d_o*g_damage* F_m,axis=0)  
        D += xi_g * np.exp(L_mat - np.log(448)) * (1-g_tech +g_tech *np.log(g_tech))

        # out = PDESolver(stateSpace, A, B_1, B_2, B_3, C_1, C_2, C_3, D, dv0dL, epsilon, solverType="Feyman Kac")

        bpoint1 = time.time()
        A_1d   = A.ravel(order = 'F')
        C_1_1d = C_1.ravel(order = 'F')
        C_2_1d = C_2.ravel(order = 'F')
        C_3_1d = C_3.ravel(order = 'F')
        B_1_1d = B_1.ravel(order = 'F')
        B_2_1d = B_2.ravel(order = 'F')
        B_3_1d = B_3.ravel(order = 'F')
        D_1d   = D.ravel(order = 'F')
        petsclinearsystem_new.formLinearSystem_noFT(K_mat_1d, Y_mat_1d, L_mat_1d, A_1d, B_1_1d, B_2_1d, B_3_1d, C_1_1d, C_2_1d, C_3_1d, epsilon, lowerLims, upperLims, dVec, increVec, petsc_mat)
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

        v  = out_comp
            
        dvdL = v    
        dvdL_orig = finiteDiff_3D(Phi, 2, 1, hL)    

        # print("sanity check: {}".format(np.mean(np.log(abs(dvdL-dvdL_orig)))))

        
    else:
        model = "Post damage"
        i,e,x,pi_c,g_tech = controls
        
        Phi_m_II, Phi_m = VF
        F_m_II = FFK

        dv0dL = finiteDiff_3D(Phi_m, 2, 1, hL)       

        A = -delta * np.ones(K_mat.shape) - psi_0 * psi_1 * (x * np.exp(K_mat-L_mat) )**psi_1 
        B_1 = mu_k + i - 0.5 * kappa * i**2 - 0.5 * sigma_k**2
        B_2 = np.sum(theta_ell * pi_c, axis=0) * e
        B_3 = - zeta + psi_0 * (x * np.exp(K_mat - L_mat))**psi_1 - 0.5 * sigma_g**2

        C_1 = 0.5 * sigma_k**2 * np.ones(K_mat.shape)
        C_2 = 0.5 * sigma_y**2 * e**2
        C_3 = 0.5 * sigma_g**2 * np.ones(K_mat.shape)
        
        # D = np.zeros(A.shape)

        A += - np.exp(L_mat - np.log(448)) * g_tech
        D = np.exp(L_mat - np.log(448)) * g_tech * (Phi_m_II - Phi_m)+ np.exp(L_mat - np.log(448)) * g_tech * F_m_II
        D += xi_g * np.exp(L_mat - np.log(448)) * (1-g_tech +g_tech *np.log(g_tech))
        
        # A += - np.exp(L_mat - np.log(448)) * g_tech
        # D =  np.exp(L_mat - np.log(448)) * g_tech * F_m_II
        # D += xi_g * np.exp(L_mat - np.log(448)) * (1-np.exp(-(Phi_m_II-Phi_m)/xi_g))

        bpoint1 = time.time()
        A_1d   = A.ravel(order = 'F')
        C_1_1d = C_1.ravel(order = 'F')
        C_2_1d = C_2.ravel(order = 'F')
        C_3_1d = C_3.ravel(order = 'F')
        B_1_1d = B_1.ravel(order = 'F')
        B_2_1d = B_2.ravel(order = 'F')
        B_3_1d = B_3.ravel(order = 'F')
        D_1d   = D.ravel(order = 'F')
        petsclinearsystem_new.formLinearSystem_noFT(K_mat_1d, Y_mat_1d, L_mat_1d, A_1d, B_1_1d, B_2_1d, B_3_1d, C_1_1d, C_2_1d, C_3_1d, epsilon, lowerLims, upperLims, dVec, increVec, petsc_mat)
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

        v  = out_comp
            
        dvdL = v   
        dvdL_orig = finiteDiff_3D(Phi_m, 2, 1, hL)    
        
    print("PETSc preconditioned residual norm is {:g}; iterations: {}".format(ksp.getResidualNorm(), ksp.getIterationNumber()))

    print("D={},{}", D.min(), D.max())

    print("absolute A={},{}", abs(A).min(), abs(A).max())
    print("sanity check: {}".format(np.max(abs(dvdL-dvdL_orig))))
    num0 = np.sum((np.ones_like(dvdL-dvdL_orig)))
    num1 = np.sum((abs(dvdL-dvdL_orig)>0.005))
    num2 = np.sum((abs(dvdL-dvdL_orig)>0.010))
    num3 = np.sum((abs(dvdL-dvdL_orig)>0.015))
    print("sanity check 0.005: {}".format((num1/num0)*100))
    print("sanity check 0.010: {}".format((num2/num0)*100))
    print("sanity check 0.015: {}".format((num3/num0)*100))
    
    num4 = np.sum((abs(dvdL_orig)>0))

    print("sanity check orig dvdL>0: {}".format((num4/num0)*100))

    pos0 = np.sum((dvdL>0))
    pos1 = np.sum((dvdL==0))
    print("sanity check positive: {}".format((pos0/num0)*100))
    print("sanity check zero: {}".format((pos1/num0)*100))

    
    if model == "Post damage":
        res = {
                "v0"    : v,
                # "i_star": i,
                # "e_star": e,
                # "x_star": x,
                # "pi_c"  : pi_c,
                # "g_tech": g_tech,
                "dvdL": dvdL,
                }
    if model == "Pre damage":
        res = {
                "v0"    : v,
                # "i_star": i,
                # "e_star": e,
                # "x_star": x,
                # "pi_c"  : pi_c,
                # "g_tech": g_tech,
                # "g_damage": g_damage,
                "dvdL": dvdL,
                }
    return res



def fk_y_pre_tech_petsc(
        state_grid=(), 
        model_args=(), 
        controls=(),
        VF=(),
        FFK=(),
        V_post_damage=None, 
        tol=1e-8, epsilon=0.1, fraction=0.5, max_iter=10000,
        v0=None,
        smart_guess=None,
        ):

    K, Y, L = state_grid
    delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell, pi_c_o, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3, y_bar, xi_a, xi_g, xi_p = model_args
    
    K_min, K_max, Y_min, Y_max, L_min, L_max = K.min(), K.max(), Y.min(), Y.max(), L.min(), L.max()
    hK, hY, hL = K[1] - K[0], Y[1] - Y[0], L[1]-L[0]
    nK, nY, nL = len(K), len(Y), len(L)
    
    ######## post jump, 3 states
    (K_mat, Y_mat, L_mat) = np.meshgrid(K, Y, L, indexing = 'ij')
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

    
    #### Model type
    if isinstance(gamma_3, (np.ndarray, list)):
        model = "Pre damage"
        pi_d_o = np.ones(len(gamma_3)) / len(gamma_3)
        pi_d_o = np.array([temp * np.ones(K_mat.shape) for temp in pi_d_o ])
        # v_i = V_post_damage
        y_bar_lower = 1.5
        r_1 = 1.5
        r_2 = 2.5
        Intensity = r_1 * (np.exp(r_2 / 2 * (Y_mat - y_bar_lower)**2) -1) * (Y_mat > y_bar_lower)
        Intensity_prime = r_1 * r_2 * np.exp(r_2 / 2 * (Y_mat - y_bar_lower)**2) * (Y_mat - y_bar_lower)* (Y_mat > y_bar_lower)
        i,e,x,pi_c,g_tech,g_damage = controls
        
        Phi_m, Phi = VF
        F_II, F_m = FFK

        A = -delta * np.ones(K_mat.shape) 
        B_1 = mu_k + i - 0.5 * kappa * i**2 - 0.5 * sigma_k**2
        B_2 = np.sum(theta_ell * pi_c, axis=0) * e
        B_3 = - zeta + psi_0 * (x * np.exp(K_mat - L_mat))**psi_1 - 0.5 * sigma_g**2

        C_1 = 0.5 * sigma_k**2 * np.ones(K_mat.shape)
        C_2 = 0.5 * sigma_y**2 * e**2
        C_3 = 0.5 * sigma_g**2 * np.ones(K_mat.shape)


        A += - np.exp(L_mat - np.log(448)) * g_tech 
        A += - Intensity*np.sum(pi_d_o*g_damage,axis=0)

        D = -( gamma_2 * np.sum( theta_ell * pi_c , axis = 0 ) * e )
        D += np.exp(L_mat - np.log(448)) * g_tech * F_II 
        D += Intensity * np.sum(pi_d_o*g_damage* F_m,axis=0)  
        D += Intensity_prime * np.sum(pi_d_o*g_damage* (Phi_m-Phi),axis=0)
        D += xi_p * Intensity_prime * np.sum(pi_d_o*(1-g_damage+g_damage*np.log(g_damage)),axis=0)
        
        # out = PDESolver(stateSpace, A, B_1, B_2, B_3, C_1, C_2, C_3, D, dv0dL, epsilon, solverType="Feyman Kac")

        bpoint1 = time.time()
        A_1d   = A.ravel(order = 'F')
        C_1_1d = C_1.ravel(order = 'F')
        C_2_1d = C_2.ravel(order = 'F')
        C_3_1d = C_3.ravel(order = 'F')
        B_1_1d = B_1.ravel(order = 'F')
        B_2_1d = B_2.ravel(order = 'F')
        B_3_1d = B_3.ravel(order = 'F')
        D_1d   = D.ravel(order = 'F')
        petsclinearsystem_new.formLinearSystem_noFT(K_mat_1d, Y_mat_1d, L_mat_1d, A_1d, B_1_1d, B_2_1d, B_3_1d, C_1_1d, C_2_1d, C_3_1d, epsilon, lowerLims, upperLims, dVec, increVec, petsc_mat)
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

        v  = out_comp
            
        dvdY = v    
        dvdY_orig = finiteDiff_3D(Phi, 1, 1, hY)    
        ddvddY = finiteDiff_3D(v, 1, 1, hY)
        ddvddY_orig = finiteDiff_3D(Phi, 1, 2, hY)    
        
        print("F range: {},{}".format(dvdY.min(),dvdY.max()))
        print("dvdY range: {},{}".format(dvdY_orig.min(),dvdY_orig.max()))
        
        print("sanity check 1st: {}".format(np.max(abs(dvdY-dvdY_orig))))
        print("sanity check 2nd: {}".format(np.max(abs(ddvddY-ddvddY_orig))))
        # print("sanity check FOC: {}".format(np.max(abs(ddvddY-ddvddY_orig))))

        
    else:
        model = "Post damage"
        i,e,x,pi_c,g_tech = controls
        
        Phi_m_II, Phi_m = VF
        F_m_II = FFK

        A = -delta * np.ones(K_mat.shape) 
        B_1 = mu_k + i - 0.5 * kappa * i**2 - 0.5 * sigma_k**2
        B_2 = np.sum(theta_ell * pi_c, axis=0) * e
        B_3 = - zeta + psi_0 * (x * np.exp(K_mat - L_mat))**psi_1 - 0.5 * sigma_g**2

        C_1 = 0.5 * sigma_k**2 * np.ones(K_mat.shape)
        C_2 = 0.5 * sigma_y**2 * e**2
        C_3 = 0.5 * sigma_g**2 * np.ones(K_mat.shape)
        

        A += - np.exp(L_mat - np.log(448)) * g_tech 

        # D =  - (    gamma_2 +gamma_3 * (Y_mat>y_bar) )  * np.sum( theta_ell * pi_c , axis = 0 ) * e
        # D += - (    gamma_3 * (Y_mat>y_bar) * sigma_y**2/2* e**2 )
        # D +=    np.exp(L_mat - np.log(448)) * g_tech * F_m_II 
        
        D =  - (    gamma_2 +gamma_3 * (Y_mat>y_bar) )  * np.sum( theta_ell * pi_c , axis = 0 ) * e
        # D += - (    gamma_3 * (Y_mat>y_bar) * sigma_y**2/2* e**2 )
        D +=    np.exp(L_mat - np.log(448)) * g_tech * F_m_II 


        A_1d   = A.ravel(order = 'F')
        C_1_1d = C_1.ravel(order = 'F')
        C_2_1d = C_2.ravel(order = 'F')
        C_3_1d = C_3.ravel(order = 'F')
        B_1_1d = B_1.ravel(order = 'F')
        B_2_1d = B_2.ravel(order = 'F')
        B_3_1d = B_3.ravel(order = 'F')
        D_1d   = D.ravel(order = 'F')
        petsclinearsystem_new.formLinearSystem_noFT(K_mat_1d, Y_mat_1d, L_mat_1d, A_1d, B_1_1d, B_2_1d, B_3_1d, C_1_1d, C_2_1d, C_3_1d, epsilon, lowerLims, upperLims, dVec, increVec, petsc_mat)
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

        v  = out_comp
            
        dvdY = v    
        dvdY_orig = finiteDiff_3D(Phi_m, 1, 1, hY)    
        ddvddY = finiteDiff_3D(v, 1, 1, hY)
        ddvddY_orig = finiteDiff_3D(dvdY_orig, 1, 1, hY)    
        
        print("F range: {},{}".format(dvdY.min(),dvdY.max()))
        print("dvdY range: {},{}".format(dvdY_orig.min(),dvdY_orig.max()))
        
        print("sanity check 1st: {}".format(np.max(abs(dvdY-dvdY_orig))))
        print("sanity check 2nd: {}".format(np.max(abs(ddvddY-ddvddY_orig)))) 
        
    print("PETSc preconditioned residual norm is {:g}; iterations: {}".format(ksp.getResidualNorm(), ksp.getIterationNumber()))

    if model == "Post damage":
        res = {
                "v0": v,
                "dvdY": dvdY,
                "ddvddY": ddvddY,
                # "v0": dvdY_orig,
                # "dvdY": dvdY_orig,
                # "ddvddY": ddvddY_orig,
                }
    if model == "Pre damage":
        res = {                
                "v0": v,
                "dvdY": dvdY,
                "ddvddY": ddvddY,
                }
    return res



def fk_y_pre_tech(
        state_grid=(), 
        model_args=(), 
        controls=(),
        VF=(),
        FFK=(),
        V_post_damage=None, 
        tol=1e-8, epsilon=0.1, fraction=0.5, max_iter=10000,
        v0=None,
        smart_guess=None,
        ):

    K_orig, Y_orig, L_orig = state_grid
    delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell, pi_c_o, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3, y_bar, xi_a, xi_g, xi_p = model_args
    
    K = K_orig
    Y = Y_orig
    L = L_orig
    pi_c_o = pi_c_o
    theta_ell = theta_ell
    
    K_min, K_max, Y_min, Y_max, L_min, L_max = K.min(), K.max(), Y.min(), Y.max(), L.min(), L.max()
    hK, hY, hL = K[1] - K[0], Y[1] - Y[0], L[1]-L[0]
    nK, nY, nL = len(K), len(Y), len(L)
    
    ######## post jump, 3 states
    (K_mat, Y_mat, L_mat) = np.meshgrid(K, Y, L, indexing = 'ij')
    stateSpace = np.hstack([K_mat.reshape(-1,1,order = 'F'), Y_mat.reshape(-1,1,order = 'F'), L_mat.reshape(-1, 1, order='F')])

    K_mat_1d = K_mat.ravel(order='F')
    Y_mat_1d = Y_mat.ravel(order='F')
    L_mat_1d = L_mat.ravel(order='F')



    
    #### Model type
    if isinstance(gamma_3, (np.ndarray, list)):
        model = "Pre damage"
        pi_d_o = np.ones(len(gamma_3)) / len(gamma_3)
        pi_d_o = np.array([temp * np.ones(K_mat.shape) for temp in pi_d_o ])
        # v_i = V_post_damage
        y_bar_lower = 1.5
        r_1 = 1.5
        r_2 = 2.5
        Intensity = r_1 * (np.exp(r_2 / 2 * (Y_mat - y_bar_lower)**2) -1) * (Y_mat > y_bar_lower)
        Intensity_prime = r_1 * r_2 * np.exp(r_2 / 2 * (Y_mat - y_bar_lower)**2) * (Y_mat - y_bar_lower)* (Y_mat > y_bar_lower)
        i,e,x,pi_c,g_tech,g_damage = controls

        
        Phi_m, Phi = VF
        F_II, F_m = FFK


        i = i
        e = e
        x = x
        pi_c = pi_c
        g_tech = g_tech
        g_damage = g_damage
        
        Phi = Phi
        Phi_m = Phi_m
        F_II = F_II
        F_m = F_m
        dv0dL = finiteDiff_3D(Phi,1,1,hY)

                
        A = -delta * np.ones(K_mat.shape) 
        B_1 = mu_k + i - 0.5 * kappa * i**2 - 0.5 * sigma_k**2
        B_2 = np.sum(theta_ell * pi_c, axis=0) * e
        B_3 = - zeta + psi_0 * (x * np.exp(K_mat - L_mat))**psi_1 - 0.5 * sigma_g**2

        C_1 = 0.5 * sigma_k**2 * np.ones(K_mat.shape)
        C_2 = 0.5 * sigma_y**2 * e**2
        C_3 = 0.5 * sigma_g**2 * np.ones(K_mat.shape)


        A += - np.exp(L_mat - np.log(448)) * g_tech 
        A += - Intensity*np.sum(pi_d_o*g_damage,axis=0)

        D = -( gamma_2 * np.sum( theta_ell * pi_c , axis = 0 ) * e )
        D += np.exp(L_mat - np.log(448)) * g_tech * F_II 
        D += Intensity * np.sum(pi_d_o*g_damage* F_m,axis=0)  
        D += Intensity_prime * np.sum(pi_d_o*g_damage* (Phi_m-Phi),axis=0)
        D += xi_p * Intensity_prime * np.sum(pi_d_o*(1-g_damage+g_damage*np.log(g_damage)),axis=0)
        
        out = PDESolver(stateSpace, A, B_1, B_2, B_3, C_1, C_2, C_3, D, dv0dL, epsilon, solverType="Feyman Kac")
        v  = out[2].reshape(dv0dL.shape, order="F")
            
        dvdY = v    
        dvdY_orig = finiteDiff_3D(Phi, 1, 1, hY)    
        ddvddY = finiteDiff_3D(v, 1, 1, hY)
        ddvddY_orig = finiteDiff_3D(Phi, 1, 2, hY)    
        
        print("F range: {},{}".format(dvdY.min(),dvdY.max()))
        print("dvdY range: {},{}".format(dvdY_orig.min(),dvdY_orig.max()))
        print("sanity check 1st: {}".format(np.max(abs(dvdY-dvdY_orig))))
        print("sanity check 2nd: {}".format(np.max(abs(ddvddY-ddvddY_orig))))
        # print("sanity check FOC: {}".format(np.max(abs(ddvddY-ddvddY_orig))))

        
    else:
        model = "Post damage"
        i,e,x,pi_c,g_tech = controls
        
        Phi_m_II, Phi_m = VF
        F_m_II = FFK


        i = i
        e = e
        x = x
        pi_c = pi_c
        g_tech = g_tech
        
        Phi_m_II = Phi_m_II
        Phi_m = Phi_m
        F_m_II = F_m_II
        dv0dL = finiteDiff_3D(Phi_m,1,1,hY)
        A = -delta * np.ones(K_mat.shape) 
        B_1 = mu_k + i - 0.5 * kappa * i**2 - 0.5 * sigma_k**2
        B_2 = np.sum(theta_ell * pi_c, axis=0) * e
        B_3 = - zeta + psi_0 * (x * np.exp(K_mat - L_mat))**psi_1 - 0.5 * sigma_g**2

        C_1 = 0.5 * sigma_k**2 * np.ones(K_mat.shape)
        C_2 = 0.5 * sigma_y**2 * e**2
        C_3 = 0.5 * sigma_g**2 * np.ones(K_mat.shape)
        

        A += - np.exp(L_mat - np.log(448)) * g_tech 

        
        D =  - (    gamma_2 +gamma_3 * (Y_mat>y_bar) )  * np.sum( theta_ell * pi_c , axis = 0 ) * e

        D +=    np.exp(L_mat - np.log(448)) * g_tech * F_m_II 


        out = PDESolver(stateSpace, A, B_1, B_2, B_3, C_1, C_2, C_3, D, dv0dL, epsilon, solverType="Feyman Kac")

        v  = out[2].reshape(dv0dL.shape, order="F")
            
        dvdY = v    
        dvdY_orig = finiteDiff_3D(Phi_m, 1, 1, hY)    
        ddvddY = finiteDiff_3D(v, 1, 1, hY)
        ddvddY_orig = finiteDiff_3D(dvdY_orig, 1, 1, hY)    

        print("F range: {},{}".format(dvdY.min(),dvdY.max()))
        print("dvdY range: {},{}".format(dvdY_orig.min(),dvdY_orig.max()))
        print("sanity check 1st: {}".format(np.max(abs(dvdY-dvdY_orig))))
        print("sanity check 2nd: {}".format(np.max(abs(ddvddY-ddvddY_orig)))) 
        
    # print("PETSc preconditioned residual norm is {:g}; iterations: {}".format(ksp.getResidualNorm(), ksp.getIterationNumber()))

    if model == "Post damage":
        res = {
                "v0": v,
                "dvdY": dvdY,
                "ddvddY": ddvddY,
                # "v0": dvdY_orig,
                # "dvdY": dvdY_orig,
                # "ddvddY": ddvddY_orig,
                }
    if model == "Pre damage":
        res = {                
                "v0": v,
                "dvdY": dvdY,
                "ddvddY": ddvddY,
                }
    return res


def fk_yshort_pre_tech_petsc(
        state_grid=(), 
        model_args=(), 
        controls=(),
        VF=(),
        FFK=(),
        n_bar = (),
        V_post_damage=None, 
        tol=1e-8, epsilon=0.1, fraction=0.5, max_iter=10000,
        v0=None,
        smart_guess=None,
        ):

    K_orig, Y_orig, L_orig = state_grid
    delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell, pi_c_o, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3, y_bar, xi_a, xi_g, xi_p = model_args
    n_bar_now = n_bar
    
    K = K_orig
    Y = Y_orig[:n_bar_now]
    L = L_orig
    pi_c_o = pi_c_o[:,:,:n_bar_now,:]
    theta_ell = theta_ell[:,:,:n_bar_now,:]
    
    K_min, K_max, Y_min, Y_max, L_min, L_max = K.min(), K.max(), Y.min(), Y.max(), L.min(), L.max()
    hK, hY, hL = K[1] - K[0], Y[1] - Y[0], L[1]-L[0]
    nK, nY, nL = len(K), len(Y), len(L)
    
    ######## post jump, 3 states
    (K_mat, Y_mat, L_mat) = np.meshgrid(K, Y, L, indexing = 'ij')
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

    
    #### Model type
    if isinstance(gamma_3, (np.ndarray, list)):
        model = "Pre damage"
        pi_d_o = np.ones(len(gamma_3)) / len(gamma_3)
        pi_d_o = np.array([temp * np.ones(K_mat.shape) for temp in pi_d_o ])
        # v_i = V_post_damage
        y_bar_lower = 1.5
        r_1 = 1.5
        r_2 = 2.5
        Intensity = r_1 * (np.exp(r_2 / 2 * (Y_mat - y_bar_lower)**2) -1) * (Y_mat > y_bar_lower)
        Intensity_prime = r_1 * r_2 * np.exp(r_2 / 2 * (Y_mat - y_bar_lower)**2) * (Y_mat - y_bar_lower)* (Y_mat > y_bar_lower)
        i,e,x,pi_c,g_tech,g_damage = controls

        
        Phi_m, Phi = VF
        F_II, F_m = FFK


        i = i[:,:n_bar_now,:]
        e = e[:,:n_bar_now,:]
        x = x[:,:n_bar_now,:]
        pi_c = pi_c[:,:,:n_bar_now,:]
        g_tech = g_tech[:,:n_bar_now,:]
        g_damage = g_damage[:,:,:n_bar_now,:]
        
        Phi = Phi[:,:n_bar_now,:]
        Phi_m = Phi_m[:,:,:n_bar_now,:]
        F_II = F_II[:,:n_bar_now,:]
        F_m = F_m[:,:,:n_bar_now,:]
                
                
        A = -delta * np.ones(K_mat.shape) 
        B_1 = mu_k + i - 0.5 * kappa * i**2 - 0.5 * sigma_k**2
        B_2 = np.sum(theta_ell * pi_c, axis=0) * e
        B_3 = - zeta + psi_0 * (x * np.exp(K_mat - L_mat))**psi_1 - 0.5 * sigma_g**2

        C_1 = 0.5 * sigma_k**2 * np.ones(K_mat.shape)
        C_2 = 0.5 * sigma_y**2 * e**2
        C_3 = 0.5 * sigma_g**2 * np.ones(K_mat.shape)


        A += - np.exp(L_mat - np.log(448)) * g_tech 
        A += - Intensity*np.sum(pi_d_o*g_damage,axis=0)

        # D = np.zeros(A.shape)
        D = -( gamma_2 * np.sum( theta_ell * pi_c , axis = 0 ) * e )
        D += np.exp(L_mat - np.log(448)) * g_tech * F_II 
        D += Intensity * np.sum(pi_d_o*g_damage* F_m,axis=0)  
        D += Intensity_prime * np.sum(pi_d_o*g_damage* (Phi_m-Phi),axis=0)
        D += xi_p * Intensity_prime * np.sum(pi_d_o*(1-g_damage+g_damage*np.log(g_damage)),axis=0)
        
        # out = PDESolver(stateSpace, A, B_1, B_2, B_3, C_1, C_2, C_3, D, dv0dL, epsilon, solverType="Feyman Kac")

        bpoint1 = time.time()
        A_1d   = A.ravel(order = 'F')
        C_1_1d = C_1.ravel(order = 'F')
        C_2_1d = C_2.ravel(order = 'F')
        C_3_1d = C_3.ravel(order = 'F')
        B_1_1d = B_1.ravel(order = 'F')
        B_2_1d = B_2.ravel(order = 'F')
        B_3_1d = B_3.ravel(order = 'F')
        D_1d   = D.ravel(order = 'F')
        petsclinearsystem_new.formLinearSystem_noFT(K_mat_1d, Y_mat_1d, L_mat_1d, A_1d, B_1_1d, B_2_1d, B_3_1d, C_1_1d, C_2_1d, C_3_1d, epsilon, lowerLims, upperLims, dVec, increVec, petsc_mat)
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

        v  = out_comp
            
        dvdY = v    
        dvdY_orig = finiteDiff_3D(Phi, 1, 1, hY)    
        ddvddY = finiteDiff_3D(v, 1, 1, hY)
        ddvddY_orig = finiteDiff_3D(Phi, 1, 2, hY)    
        
        print("F range: {},{}".format(dvdY.min(),dvdY.max()))
        print("dvdY range: {},{}".format(dvdY_orig.min(),dvdY_orig.max()))
        print("sanity check 1st: {}".format(np.max(abs(dvdY-dvdY_orig)[:,:n_bar_now,:])))
        print("sanity check 2nd: {}".format(np.max(abs(ddvddY-ddvddY_orig)[:,:n_bar_now,:])))
        # print("sanity check FOC: {}".format(np.max(abs(ddvddY-ddvddY_orig))))

        
    else:
        model = "Post damage"
        i,e,x,pi_c,g_tech = controls
        
        Phi_m_II, Phi_m = VF
        F_m_II = FFK


        i = i[:,:n_bar_now,:]
        e = e[:,:n_bar_now,:]
        x = x[:,:n_bar_now,:]
        pi_c = pi_c[:,:,:n_bar_now,:]
        g_tech = g_tech[:,:n_bar_now,:]
        
        Phi_m_II = Phi_m_II[:,:n_bar_now,:]
        Phi_m = Phi_m[:,:n_bar_now,:]
        F_m_II = F_m_II[:,:n_bar_now,:]
                
        A = -delta * np.ones(K_mat.shape) 
        B_1 = mu_k + i - 0.5 * kappa * i**2 - 0.5 * sigma_k**2
        B_2 = np.sum(theta_ell * pi_c, axis=0) * e
        B_3 = - zeta + psi_0 * (x * np.exp(K_mat - L_mat))**psi_1 - 0.5 * sigma_g**2

        C_1 = 0.5 * sigma_k**2 * np.ones(K_mat.shape)
        C_2 = 0.5 * sigma_y**2 * e**2
        C_3 = 0.5 * sigma_g**2 * np.ones(K_mat.shape)
        

        A += - np.exp(L_mat - np.log(448)) * g_tech 

        
        # D =  - (    gamma_2 +gamma_3 * (Y_mat>y_bar) )  * np.sum( theta_ell * pi_c , axis = 0 ) * e
        # # D =  - (    gamma_2 )  * np.sum( theta_ell * pi_c , axis = 0 ) * e
        # # D += - (    gamma_3 * (Y_mat>y_bar) * sigma_y**2/2* e**2 )
        # D +=    np.exp(L_mat - np.log(448)) * g_tech * F_m_II 


        D = np.zeros(A.shape)
        D +=  np.exp(L_mat - np.log(448)) * g_tech * F_m_II 
        D +=  - (    gamma_2 )  * np.sum( theta_ell * pi_c , axis = 0 ) * e
        D +=  - (    gamma_3 * (Y_mat>y_bar) )  * np.sum( theta_ell * pi_c , axis = 0 ) * e

        # D +=    np.exp(L_mat - np.log(448)) * g_tech * (Phi_m_II - Phi_m) 
        # D +=    xi_g * np.exp(L_mat - np.log(448)) * (1 - g_tech + g_tech * np.log(g_tech))
        
        A_1d   = A.ravel(order = 'F')
        C_1_1d = C_1.ravel(order = 'F')
        C_2_1d = C_2.ravel(order = 'F')
        C_3_1d = C_3.ravel(order = 'F')
        B_1_1d = B_1.ravel(order = 'F')
        B_2_1d = B_2.ravel(order = 'F')
        B_3_1d = B_3.ravel(order = 'F')
        D_1d   = D.ravel(order = 'F')
        petsclinearsystem_new.formLinearSystem_noFT(K_mat_1d, Y_mat_1d, L_mat_1d, A_1d, B_1_1d, B_2_1d, B_3_1d, C_1_1d, C_2_1d, C_3_1d, epsilon, lowerLims, upperLims, dVec, increVec, petsc_mat)
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

        v  = out_comp
            
        dvdY = v    
        dvdY_orig = finiteDiff_3D(Phi_m, 1, 1, hY)    
        ddvddY = finiteDiff_3D(v, 1, 1, hY)
        ddvddY_orig = finiteDiff_3D(dvdY_orig, 1, 1, hY)    

        print("F range: {},{}".format(dvdY.min(),dvdY.max()))
        print("dvdY range: {},{}".format(dvdY_orig.min(),dvdY_orig.max()))
        print("sanity check 1st: {}".format(np.max(abs(dvdY-dvdY_orig)[:,:n_bar_now,:])))
        print("sanity check 2nd: {}".format(np.max(abs(ddvddY-ddvddY_orig)[:,:n_bar_now,:]))) 
        
    print("PETSc preconditioned residual norm is {:g}; iterations: {}".format(ksp.getResidualNorm(), ksp.getIterationNumber()))

    if model == "Post damage":
        res = {
                "v0": v,
                "dvdY": dvdY,
                "ddvddY": ddvddY,
                # "v0": dvdY_orig,
                # "dvdY": dvdY_orig,
                # "ddvddY": ddvddY_orig,
                }
    if model == "Pre damage":
        res = {                
                "v0": v,
                "dvdY": dvdY,
                "ddvddY": ddvddY,
                }
    return res


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


def hjb_pre_tech_check(
        state_grid=(), 
        model_args=(), 
        controls=(),
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
    delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell, pi_c_o, sigma_y, zeta, psi_0, psi_1, sigma_g, v_tech2, gamma_1, gamma_2, gamma_3, y_bar, xi_a, xi_g, xi_p = model_args

    
    V_post_tech = v_tech2
    
    if V_post_damage is not None:
        V_post_damage = V_post_damage

    # v0 = np.mean(V_post_damage,axis=0)

    K_min, K_max, Y_min, Y_max, L_min, L_max = K.min(), K.max(), Y.min(), Y.max(), L.min(), L.max()
    hK, hY, hL = K[1] - K[0], Y[1] - Y[0], L[1]-L[0]
    nK, nY, nL = len(K), len(Y), len(L)
    
    print("K=[{:.1f},{:.1f},{:.2f},{:d}], Y=[{:.1f},{:.1f},{:.2f},{:d}],L==[{:.1f},{:.1f},{:.2f},{:d}]" .format(K.min(),K.max(),hK,nK, Y.min(),Y.max(),hY,nY, L.min(),L.max(),hL,nL))


    (K_mat, Y_mat, L_mat) = np.meshgrid(K, Y, L, indexing = 'ij')
    
    
    pi_c_o = np.ones(len(theta_ell)) / len(theta_ell)
    # pi_c = np.ones(len(theta_ell)) / len(theta_ell)
    
    pi_c_o = np.array([temp * np.ones(K_mat.shape) for temp in pi_c_o ])
    theta_ell = np.array([temp * np.ones(K_mat.shape) for temp in theta_ell ])

    
    K_mat_1d = K_mat.ravel(order='F')
    Y_mat_1d = Y_mat.ravel(order='F')
    L_mat_1d = L_mat.ravel(order='F')
    lowerLims = np.array([K_min, Y_min, L_min], dtype=np.float64)
    upperLims = np.array([K_max, Y_max, L_max], dtype=np.float64)
    
    #### Model type
    if isinstance(gamma_3, (np.ndarray, list)):
        model = "Pre damage"
        ii, ee, xx, pi_c, g_tech, g_damage, v0 = controls

        pi_d_o = np.ones(len(gamma_3)) / len(gamma_3)
        pi_d_o = np.array([temp * np.ones(K_mat.shape) for temp in pi_d_o ])
        y_bar_lower = 1.5
        r_1 = 1.5
        r_2 = 2.5
        Intensity = r_1 * (np.exp(r_2 / 2 * (Y_mat - y_bar_lower)**2) -1) * (Y_mat > y_bar_lower)
        v_i = V_post_damage
        dG  = gamma_1 + gamma_2 * Y_mat
        ddG = gamma_2 
        
        
        jj =  alpha * vartheta_bar * (1 - ee / (alpha * lambda_bar * np.exp(K_mat)))**theta
        jj[jj <= 1e-16] = 1e-16
        consumption = alpha - ii - jj - xx
        consumption[consumption <= 1e-16] = 1e-16
        entropy = np.sum(pi_c * (np.log(pi_c) - np.log(pi_c_o)), axis=0)

        A   = - delta * np.ones(K_mat.shape) - np.exp(  L_mat - np.log(448) ) * g_tech
        B_1 = mu_k + ii - 0.5 * kappa * ii**2 - 0.5 * sigma_k**2
        B_2 = np.sum(theta_ell * pi_c, axis=0) * ee
        B_3 = - zeta + psi_0 * (xx * np.exp(K_mat - L_mat))**psi_1 - 0.5 * sigma_g**2
        # B_3 = - zeta + psi_0 * xx** psi_1 * np.exp( psi_1 * K_mat ) * np.sum(pi_c * np.exp( -( 1-psi_2) * L_mat  ), axis=0 )- 0.5 * sigma_g**2

        C_1 = 0.5 * sigma_k**2 * np.ones(K_mat.shape)
        C_2 = 0.5 * sigma_y**2 * ee**2
        C_3 = 0.5 * sigma_g**2 * np.ones(K_mat.shape)
        D = delta * np.log(consumption) + delta * K_mat  - dG * np.sum(theta_ell * pi_c, axis=0) * ee  - 0.5 * ddG * sigma_y**2 * ee**2  + xi_a * entropy + xi_g * np.exp((L_mat - np.log(448))) * (1 - g_tech + g_tech * np.log(g_tech)) + np.exp( (L_mat - np.log(448)) ) * g_tech * V_post_tech
        D += xi_p * Intensity * np.sum( pi_d_o*(1-g_damage+g_damage*np.log(g_damage)),axis=0) +Intensity*np.sum(pi_d_o*g_damage*v_i,axis=0)
        A -=  Intensity*np.sum(pi_d_o*g_damage,axis=0)

    else:
        model = "Post damage"
        
        ii, ee, xx, pi_c, g_tech, v0 = controls

        dG  = gamma_1 + gamma_2 * Y_mat + gamma_3 * (Y_mat - y_bar) * (Y_mat > y_bar)
        ddG = gamma_2 + gamma_3 * (Y_mat > y_bar)
        
        jj =  alpha * vartheta_bar * (1 - ee / (alpha * lambda_bar * np.exp(K_mat)))**theta
        jj[jj <= 1e-16] = 1e-16
        consumption = alpha - ii - jj - xx
        consumption[consumption <= 1e-16] = 1e-16
        entropy = np.sum(pi_c * (np.log(pi_c) - np.log(pi_c_o)), axis=0)

        A   = - delta * np.ones(K_mat.shape) - np.exp(  L_mat - np.log(448) ) * g_tech
        B_1 = mu_k + ii - 0.5 * kappa * ii**2 - 0.5 * sigma_k**2
        B_2 = np.sum(theta_ell * pi_c, axis=0) * ee
        B_3 = - zeta + psi_0 * (xx * np.exp(K_mat - L_mat))**psi_1 - 0.5 * sigma_g**2
        # B_3 = - zeta + psi_0 * xx** psi_1 * np.exp( psi_1 * K_mat ) * np.sum(pi_c * np.exp( -( 1-psi_2) * L_mat  ), axis=0 )- 0.5 * sigma_g**2

        C_1 = 0.5 * sigma_k**2 * np.ones(K_mat.shape)
        C_2 = 0.5 * sigma_y**2 * ee**2
        C_3 = 0.5 * sigma_g**2 * np.ones(K_mat.shape)
        D = delta * np.log(consumption) + delta * K_mat  - dG * np.sum(theta_ell * pi_c, axis=0) * ee  - 0.5 * ddG * sigma_y**2 * ee**2  + xi_a * entropy + xi_g * np.exp((L_mat - np.log(448))) * (1 - g_tech + g_tech * np.log(g_tech)) + np.exp( (L_mat - np.log(448)) ) * g_tech * V_post_tech

    # Initial setup of HJB
    FC_Err   = 1
    epoch    = 0



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

    # Enter the optimization

    start_ep = time.time()


    out_comp,end_ksp, bpoint1 = pde_one_interation_noFT(
            ksp,
            petsc_mat,K_mat_1d, Y_mat_1d, L_mat_1d, 
            lowerLims, upperLims, dVec, increVec,
            v0, A, B_1, B_2, B_3, C_1, C_2, C_3, D, 1e-13, epsilon)
    # if epoch % 1 == 0 and reporterror:
        # Calculating PDE error and False Transient error

    dX1  = finiteDiff_3D(v0,0,1,hK)
    # dX1[dX1 <= 1e-16] = 1e-16
    dK = dX1
    dX2  = finiteDiff_3D(v0,1,1,hY)
    dY = dX2
    dX3  = finiteDiff_3D(v0,2,1,hL)
    # dX3[dX3 <= 1e-16] = 1e-16
    dL = dX3
    ######## second order
    ddX1 = finiteDiff_3D(v0,0,2,hK)
    ddK = ddX1
    ddX2 = finiteDiff_3D(v0,1,2,hY)
    ddY = ddX2
    ddX3 = finiteDiff_3D(v0,2,2,hL)
    ddL = ddX3
    
    PDE_rhs1 = A * v0 + B_1 * dK + B_2 * dY + B_3 * dL + C_1 * ddK + C_2 * ddY + C_3 * ddL + D
    PDE_Err1 = np.max(abs(PDE_rhs1))
    
    
    dX1  = finiteDiff_3D(out_comp,0,1,hK)
    # dX1[dX1 <= 1e-16] = 1e-16
    dK = dX1
    dX2  = finiteDiff_3D(out_comp,1,1,hY)
    dY = dX2
    dX3  = finiteDiff_3D(out_comp,2,1,hL)
    # dX3[dX3 <= 1e-16] = 1e-16
    dL = dX3
    ######## second order
    ddX1 = finiteDiff_3D(out_comp,0,2,hK)
    ddK = ddX1
    ddX2 = finiteDiff_3D(out_comp,1,2,hY)
    ddY = ddX2
    ddX3 = finiteDiff_3D(out_comp,2,2,hL)
    ddL = ddX3
    
    PDE_rhs2 = A * out_comp + B_1 * dK + B_2 * dY + B_3 * dL + C_1 * ddK + C_2 * ddY + C_3 * ddL + D
    PDE_Err2 = np.max(abs(PDE_rhs2))
    FC_Err = np.max(abs((out_comp - v0)/ epsilon))
    

    dY_orig = finiteDiff_3D(v0,1,1,hY)
    dY = finiteDiff_3D(out_comp,1,1,hY)
    print("sanity check: {}" .format(np.max(abs(dY-dY_orig))))
    v0     = out_comp
    epoch += 1
        
    print("---------Converged: Epoch {:d}: PDE Error: 1={}, 2={}, False Transient Error: {:.10f}; Time: {:.4f}---------------".format(epoch, PDE_Err1, PDE_Err2, FC_Err, time.time() - start_func), flush=True)

    i_star = ii
    e_star = ee
    x_star = xx
    

    res = {
            "v0"    : v0,
            "i_star": i_star,
            "e_star": e_star,
            "x_star": x_star,
            "pi_c"  : pi_c,
            "g_tech": g_tech,
            "FC_Err": FC_Err,
            "dvdL": dL,
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
                "FC_Err": FC_Err,
                "dvdL": dL,
                }
    return res




