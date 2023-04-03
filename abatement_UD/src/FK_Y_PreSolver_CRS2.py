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
import petsclinearsystem_new
import time
from datetime import datetime


def pde_one_interation(ksp, petsc_mat, X1_mat_1d, X2_mat_1d, X3_mat_1d, lowerLims, upperLims, dVec, increVec, A, B_1, B_2, B_3, C_1, C_2, C_3, D, tol, epsilon):

    bpoint1 = time.time()
    A_1d   = A.ravel(order = 'F')
    C_1_1d = C_1.ravel(order = 'F')
    C_2_1d = C_2.ravel(order = 'F')
    C_3_1d = C_3.ravel(order = 'F')
    B_1_1d = B_1.ravel(order = 'F')
    B_2_1d = B_2.ravel(order = 'F')
    B_3_1d = B_3.ravel(order = 'F')
    D_1d   = D.ravel(order = 'F')
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
    print("PETSc preconditioned residual norm is {:g}; iterations: {}".format(ksp.getResidualNorm(), ksp.getIterationNumber()))
    return out_comp,end_ksp,bpoint1

def _FK_update(steps= (), states = (), args=(), controls=(), fraction=0.5):

    hX1, hX2, hX3 = steps
    K_mat, Y_mat, L_mat = states
    delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell, pi_c_o, sigma_y, zeta, psi_0, psi_1, sigma_g, dG, ddG, xi_a, xi_g = args

    ii, ee, xx, pi_c = controls
    

    # Step (2), solve minimization problem in HJB and calculate drift distortion
    A   = - delta * np.ones(K_mat.shape) 
    B_1 = mu_k + ii - 0.5 * kappa * ii**2 - 0.5 * sigma_k**2
    B_2 = np.sum(theta_ell * pi_c, axis=0) * ee
    B_3 = - zeta + psi_0 * (xx * np.exp(K_mat - L_mat))**psi_1 - 0.5 * sigma_g**2

    C_1 = 0.5 * sigma_k**2 * np.ones(K_mat.shape)
    C_2 = 0.5 * sigma_y**2 * ee**2
    C_3 = 0.5 * sigma_g**2 * np.ones(K_mat.shape)
    D = - ddG * np.sum(theta_ell * pi_c, axis=0) * ee 
    
    return A, B_1, B_2, B_3, C_1, C_2, C_3, D

def fk_y_pre_tech(
        state_grid=(), model_args=(), V=(), F=(), controls=(),
        tol=1e-8, epsilon=0.1, fraction=0.5, max_iter=10000,
        v0=None,
        ):

    now = datetime.now()
    current_time = now.strftime("%d-%H:%M")
    K, Y, L = state_grid

    delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell, pi_c_o, pi_c, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3, y_bar, xi_a, xi_g, xi_p = model_args


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
    
    
    #### Model type
    if isinstance(gamma_3, (np.ndarray, list)):
        model = "Pre damage"
        pi_d_o = np.ones(len(gamma_3)) / len(gamma_3)
        pi_d_o = np.array([temp * np.ones(K_mat.shape) for temp in pi_d_o ])
        y_bar_lower = 1.5
        r_1 = 1.5
        r_2 = 2.5
        Intensity = r_1 * (np.exp(r_2 / 2 * (Y_mat - y_bar_lower)**2) -1) * (Y_mat > y_bar_lower)
        Intensity_prime = r_1 * r_2 * np.exp(r_2 / 2 * (Y_mat - y_bar_lower)**2) * (Y_mat - y_bar_lower)* (Y_mat > y_bar_lower)

        dG  = gamma_1 + gamma_2 * Y_mat
        ddG = gamma_2 
        
        Phi_m, Phi = V
        F_II, F_m = F
        
        i_star, e_star, x_star, pi_c, g_tech, g_damage = controls
        
    else:
        model = "Post damage"
        dG  = gamma_1 + gamma_2 * Y_mat + gamma_3 * (Y_mat - y_bar) * (Y_mat > y_bar)
        ddG = gamma_2 + gamma_3 * (Y_mat > y_bar)
        
        Phi_m_II, Phi_m = V
        F_m_II = F
        
        i_star, e_star, x_star, pi_c, g_tech = controls


    dVec = np.array([hX1, hX2, hX3])
    increVec = np.array([1, nX1, nX1 * nX2],dtype=np.int32)
    petsc_mat = PETSc.Mat().create()
    petsc_mat.setType('aij')
    petsc_mat.setSizes([nX1 * nX2 * nX3, nX1 * nX2 * nX3])
    petsc_mat.setPreallocationNNZ(13)
    petsc_mat.setUp()
    ksp = PETSc.KSP()
    ksp.create(PETSc.COMM_WORLD)
    ksp.setType('bcgs')
    ksp.getPC().setType('ilu')
    ksp.setFromOptions()

    epoch = 0
    
    FK_args = (delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell, pi_c_o, sigma_y, zeta, psi_0, psi_1, sigma_g, dG, ddG, xi_a, xi_g )

    start_ep = time.time()
    A, B_1, B_2, B_3, C_1, C_2, C_3, D = _FK_update(steps= (hX1, hX2, hX3), states = (K_mat, Y_mat, L_mat), args=FK_args, controls=(i_star, e_star, x_star, pi_c), fraction=fraction)

    if model == "Pre damage":
        
        D += np.exp(L_mat - np.log(448)) * F_II
        D += Intensity * np.sum( pi_d_o * g_damage *(F_m-F))
        D += xi_p * Intensity_prime * np.sum( pi_d_o*(1-g_damage+g_damage*np.log(g_damage)),axis=0) 
        D += Intensity_prime * np.sum(pi_d_o*g_damage*(Phi_m - Phi),axis=0)
        A -=  Intensity*np.sum(pi_d_o*g_damage,axis=0)

    if model == "Post damage":
        

        D +=  np.exp(L_mat - np.log(448)) * g_tech * F_m_II
        
        A +=  - np.exp(L_mat - np.log(448)) * g_tech
        
    out_comp,end_ksp, bpoint1 = pde_one_interation(
            ksp,
            petsc_mat,X1_mat_1d, X2_mat_1d, X3_mat_1d, 
            lowerLims, upperLims, dVec, increVec,
            A, B_1, B_2, B_3, C_1, C_2, C_3, D, 1e-13, epsilon)

    if model == "Post damage":

        dvdY_orig = finiteDiff_3D(Phi_m, 1, 1, hX2)
        
    if model == "Post damage":

        dvdY_orig = finiteDiff_3D(Phi, 1, 1, hX2)
        
    dFdX1 = finiteDiff_3D(dvdY_orig, 0, 1, hX1)
    dFdX2 = finiteDiff_3D(dvdY_orig, 1, 1, hX2)
    dFdX3 = finiteDiff_3D(dvdY_orig, 2, 1, hX3)
    ddFddX1 = finiteDiff_3D(dvdY_orig, 0, 2, hX1)
    ddFddX2 = finiteDiff_3D(dvdY_orig, 1, 2, hX2)
    ddFddX3 = finiteDiff_3D(dvdY_orig, 2, 2, hX3)
    PDE_orig_rhs = A * dvdY_orig + B_1 * dFdX1 + B_2 * dFdX2 + B_3 * dFdX3 + C_1 * ddFddX1 + C_2 * ddFddX2 + C_3 * ddFddX3 + D
    PDE_orig_Err = np.max(abs(PDE_orig_rhs))

    
    
    F = out_comp
    dFdX1 = finiteDiff_3D(F, 0, 1, hX1)
    dFdX2 = finiteDiff_3D(F, 1, 1, hX2)
    dFdX3 = finiteDiff_3D(F, 2, 1, hX3)
    ddFddX1 = finiteDiff_3D(F, 0, 2, hX1)
    ddFddX2 = finiteDiff_3D(F, 1, 2, hX2)
    ddFddX3 = finiteDiff_3D(F, 2, 2, hX3)
    
    PDE_rhs = A * F + B_1 * dFdX1 + B_2 * dFdX2 + B_3 * dFdX3 + C_1 * ddFddX1 + C_2 * ddFddX2 + C_3 * ddFddX3 + D
    
    PDE_Err = np.max(abs(PDE_rhs))
    Sanity_Err = np.max(abs((F - dvdY_orig)))
    

    print("Epoch {:d} (PETSc): PDE_orig Error: {:.10f}; PDE Error: {:.10f}; Sanity Error: {:.10f}" .format(epoch, PDE_orig_Err, PDE_Err, Sanity_Err))
        
    print("F=[{},{}]".format(F.min(),F.max()))
    print("dvdY=[{},{}]".format(dvdY_orig.min(),dvdY_orig.max()))
    
    res = {
            "F"    : F,
            "i_star": i_star,
            "e_star": e_star,
            "x_star": x_star,
            "pi_c"  : pi_c,
            "g_tech": g_tech,
            "Sanity_Err": Sanity_Err,
            }
    if model == "Pre damage":
        res = {
                "F"    : F,
                "i_star": i_star,
                "e_star": e_star,
                "x_star": x_star,
                "pi_c"  : pi_c,
                "g_tech": g_tech,
                "g_damage": g_damage,
                "Sanity_Err": Sanity_Err,
                }
    return res
