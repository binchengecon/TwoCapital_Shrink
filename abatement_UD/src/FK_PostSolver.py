"""
file for post HJB with k and y
"""
import os
import sys
sys.path.append("../src/")
import numpy as np
import pandas as pd
# from numba import njit
from supportfunctions import finiteDiff
import SolveLinSys

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

def decompose(v0, stateSpace, states=(), controls=(), args=()):
    i_star, e_star, x_star = controls
    K_mat, Y_mat, L_mat = states
    delta, alpha, kappa, mu_k, sigma_k, gamma_1, gamma_2, theta_ell, pi_c_o, sigma_y,  theta, vartheta_bar, lambda_bar = args
    
    j_star = 1 - e_star / (alpha * lambda_bar * np.exp(K_mat))
    j_star[j_star <= 1e-16] = 1e-16
    
    mc = delta / (alpha - i_star - alpha * vartheta_bar * j_star**theta - x_star)
    dG  = gamma_1 + gamma_2 * Y_mat
    ddG = gamma_2
    
    tol = 1e-7
    episode = 0
    epsilon = 0.1
    max_iter = 3000
    error = 1.
    
    A = np.zeros(K_mat.shape)
    B_1 = np.zeros(K_mat.shape)
    B_2 = np.sum(theta_ell * pi_c_o, axis=0) 
    B_3 = np.zeros(K_mat.shape)

    C_1 = np.zeros(K_mat.shape)
    C_2 = e_star * sigma_y**2
    C_3 = np.zeros(K_mat.shape)

    D = mc * theta * vartheta_bar / (lambda_bar * np.exp(K_mat)) * j_star**(theta-1) - dG * np.sum(theta_ell * pi_c_o, axis=0) - ddG * sigma_y**2 * e_star

    out = PDESolver(stateSpace, A, B_1, B_2, B_3, C_1, C_2, C_3, D, v0, epsilon, solverType="Feyman Kac")
    v  = out[2].reshape(v0.shape, order="F")
        

    dvdy = finiteDiff_3D(v, 1, 1, hY)
    ddvdyy = finiteDiff_3D(v, 1, 2, hY)
    RHS = - dvdy * np.sum(pi_c_o * theta_ell, axis=0) - ddvdyy * sigma_y**2 * e_star + dG * np.sum(theta_ell * pi_c_o, axis=0) + ddG * sigma_y**2 * e_star
    LHS = mc * theta * vartheta_bar /(lambda_bar * np.exp(K_mat)) * j_star**(theta-1)
    diff = np.max(abs(RHS - LHS))
    
    ME_base = RHS
    return v, ME_base, diff



def FK_one_iteration_cpp(stateSpace, A, B1, B2, C1, C2, D, v0, ε):
    
    iters=4000000
    
    A = A.reshape(-1, 1, order='F')
    B = np.hstack([B1.reshape(-1, 1, order='F'), B2.reshape(-1, 1, order='F')])
    C = np.hstack([C1.reshape(-1, 1, order='F'), C2.reshape(-1, 1, order='F')])
    D = D.reshape(-1, 1, order='F')
    v0 = v0.reshape(-1, 1, order='F')
    out = SolveLinSys.solveFK(stateSpace, A, B, C, D, v0, iters)
    return out[2].reshape(v0.shape, order = "F")

def _FK_iteration(
        v0, k_mat, y_mat, dk, dy, d_Delta, dd_Delta, theta, lambda_bar, vartheta_bar, delta, alpha, kappa, mu_k, sigma_k, pi_c_o, pi_c, theta_ell, sigma_y, xi_a, xi_b, i, e, fraction):

    # temp = alpha - i - alpha * vartheta_bar * (1 - e / (alpha * lambda_bar * np.exp(k_mat))) ** theta
    # mc = delta / temp

    # entropy = np.sum(pi_c * (np.log(pi_c) - np.log(pi_c_o)), axis=0)

    A    = np.ones_like(y_mat) * (- delta)
    B_k  = mu_k + i - kappa / 2. * i ** 2 - sigma_k ** 2 / 2.
    B_y  = np.sum(pi_c * theta_ell, axis=0) * e
    C_kk = sigma_k ** 2 / 2 * np.ones_like(y_mat)
    C_yy = .5 * sigma_y **2 * e**2

    consumption = alpha - i - alpha * vartheta_bar * ( 1 - e /(alpha * lambda_bar * np.exp(k_mat)))**theta
    consumption[consumption <= 1e-16] = 1e-16
    D = np.zeros(i.shape)

    return A, B_k, B_y, C_kk, C_yy, D



def FK_post_damage_post_tech(
        k_grid, y_grid, model_args=(), control=(), 
        epsilon=1., fraction=.1, tol=1e-8, max_iter=10000, print_iteration=True
        ):

    delta, alpha, kappa, mu_k, sigma_k, theta_ell, pi_c_o, sigma_y, xi_a, xi_b, gamma_1, gamma_2, gamma_3, y_bar, theta, lambda_bar, vartheta_bar = model_args
    e, i, g, pi_c, v0 = control
    
    dk = k_grid[1] - k_grid[0]
    dy = y_grid[1] - y_grid[0]
    (k_mat, y_mat) = np.meshgrid(k_grid, y_grid, indexing = 'ij')
    

    d_Delta  = gamma_1 + gamma_2 * y_mat + gamma_3 * (y_mat > y_bar) * (y_mat - y_bar)
    dd_Delta = gamma_2 + gamma_3 * (y_mat > y_bar)
    
    pi_c_o = np.array([temp * np.ones_like(y_mat) for temp in pi_c_o])
    theta_ell = np.array([temp * np.ones_like(y_mat) for temp in theta_ell])

    state_space = np.hstack([k_mat.reshape(-1, 1, order = 'F'),
                             y_mat.reshape(-1, 1, order = 'F')])

    count = 0
    error = 1.


    A, B_k, B_y, C_kk, C_yy, D =  _FK_iteration(
            v0, k_mat, y_mat, dk, dy, d_Delta, dd_Delta, theta, lambda_bar, vartheta_bar,
            delta, alpha, kappa, mu_k, sigma_k, pi_c_o, pi_c, theta_ell, sigma_y, xi_a, xi_b, i, e, fraction
            )

    v = FK_one_iteration_cpp(state_space, A, B_k, B_y, C_kk, C_yy, D, v0, epsilon)

    # rhs_error = A * v0 + B_k * dvdk + B_y * dvdy + C_kk * dvdkk + C_yy * dvdyy + D
    # rhs_error = np.max(abs(rhs_error))
    rhs_error = 0
    lhs_error = np.max(abs((v - v0)/epsilon))

    error = lhs_error
    v0 = v
    count += 1

    # if print_iteration:
    #     print("Iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))

    print("End. Total iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))

    res = {
        'v': v,
        'k': k_grid,
        'y': y_grid,
        'e': e,
        'i': i,
        'pi_c': pi_c,
        'error': error,
        }

    return res

def FK_pre_damage_post_tech(
        k_grid, y_grid, model_args=(), control=(),
        smart_guess=None, epsilon=1., fraction=.1,
        tol=1e-8, max_iter=10000, print_iteration=True
        ):

    delta, alpha, kappa, mu_k, sigma_k, theta_ell, pi_c_o, sigma_y, xi_a, xi_b, xi_p, pi_d_o, v_i, gamma_1, gamma_2, theta, lambda_bar, vartheta_bar, y_bar_lower = model_args
    e, i, g, pi_c, v0 = control

    dk = k_grid[1] - k_grid[0]
    dy = y_grid[1] - y_grid[0]
    (k_mat, y_mat) = np.meshgrid(k_grid, y_grid, indexing = 'ij')

    d_Delta  = gamma_1 + gamma_2 * y_mat
    dd_Delta = gamma_2

    # pi_c_o = np.array([temp * np.ones_like(y_mat) for temp in pi_c_o])
    # pi_d_o = np.array([temp * np.ones_like(y_mat) for temp in pi_d_o])
    # theta_ell = np.array([temp * np.ones_like(y_mat) for temp in theta_ell])

    r1=1.5
    r2=2.5
    intensity = r1*(np.exp(r2/2*(y_mat - y_bar_lower)**2)-1)*(y_mat >= y_bar_lower)

    state_space = np.hstack([k_mat.reshape(-1, 1, order = 'F'),
                             y_mat.reshape(-1, 1, order = 'F')])

    error = 1.
    count = 0

    A, B_k, B_y, C_kk, C_yy, D =  _FK_iteration(v0, k_mat, y_mat, dk, dy, d_Delta, dd_Delta, theta, lambda_bar, vartheta_bar,
                        delta, alpha, kappa, mu_k, sigma_k, pi_c_o, pi_c, theta_ell, sigma_y, xi_a, xi_b, i, e, fraction)


    v = FK_one_iteration_cpp(state_space, A, B_k, B_y, C_kk, C_yy, D, v0, epsilon)

    # rhs_error = A * v0 + B_k * dvdk + B_y * dvdy + C_kk * dvdkk + C_yy * dvdyy + D
    # rhs_error = np.max(abs(rhs_error))
    rhs_error = 0
    lhs_error = np.max(abs((v - v0)/epsilon))

    error = lhs_error
    v0 = v
    count += 1

    # if print_iteration:
    #     print("Iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))

    print("End. Total iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))


    # if smart_guess:
    #     iteration_error = np.max(abs(smart_guess["v"]-v0))
    #     print(iteration_error)
    
    # g = np.exp(1. / xi_p * (v - v_i))

    res = {'v': v,
           'e': e,
           'i': i,
           'g': g,
           'pi_c': pi_c,
           'h': h,
           'error': error,
           }

    return res
