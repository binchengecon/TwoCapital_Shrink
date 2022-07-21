"""
file for post HJB with k and y
"""
import os
import sys
sys.path.append("../src/")
import numpy as np
import pandas as pd
from numba import njit
from supportfunctions import finiteDiff
import SolveLinSys

def false_transient_one_iteration_cpp(stateSpace, A, B1, B2, C1, C2, D, v0, ε):
    A = A.reshape(-1, 1, order='F')
    B = np.hstack([B1.reshape(-1, 1, order='F'), B2.reshape(-1, 1, order='F')])
    C = np.hstack([C1.reshape(-1, 1, order='F'), C2.reshape(-1, 1, order='F')])
    D = D.reshape(-1, 1, order='F')
    out = SolveLinSys.solveFT(stateSpace, A, B, C, D, v0.reshape(-1, 1, order='F'), ε, -10)
    return out[2].reshape(v0.shape, order = "F")

def _hjb_iteration(
        v0, k_mat, y_mat, dk, dy, d_Delta, dd_Delta, theta, lambda_bar, vartheta_bar, delta, alpha, kappa, mu_k, sigma_k, pi_c_o, pi_c, theta_ell, sigma_y, xi_a, xi_b, i, e, fraction):

    dvdk  = finiteDiff(v0, 0, 1, dk)
    dvdkk = finiteDiff(v0, 0, 2, dk)
    dvdy  = finiteDiff(v0, 1, 1, dy)
    dvdyy = finiteDiff(v0, 1, 2, dy)

    temp = alpha - i - alpha * vartheta_bar * (1 - e / (alpha * lambda_bar * np.exp(k_mat))) ** theta
    mc = delta / temp

    i_new =  (1 - mc / dvdk) / kappa



    # Method 1 : Solve second order equation
    if vartheta_bar != 0 and theta == 3:
        G = dvdy  - d_Delta
        F = dvdyy - dd_Delta
        temp = mc * vartheta_bar * theta / (lambda_bar * np.exp(k_mat))
        a = temp / (alpha * lambda_bar * np.exp(k_mat)) ** 2
        b = - 2 * temp / (alpha * lambda_bar * np.exp(k_mat)) + F  * sigma_y ** 2
        c = temp + G * np.sum(pi_c * theta_ell, axis=0)
        temp = b ** 2 - 4 * a * c
        temp = temp * (temp > 0)
        root1 = (- b - np.sqrt(temp)) / (2 * a)
        root2 = (- b + np.sqrt(temp)) / (2 * a)
        if root1.all() > 0 :
            e_new = root1
        else:
            e_new = root2
    elif vartheta_bar != 0 and theta == 2:
        temp =  mc * vartheta_bar * theta / (lambda_bar * np.exp(k_mat))
        a = - mc * temp / (alpha * lambda_bar * np.exp(k_mat)) + F * sigma_y**2
        b = mc * temp + G * np.sum(pi_c * theta_ell, axis=0)
        e_new = - b / a
    else:
        G = dvdy  - d_Delta
        F = dvdyy - dd_Delta
        temp = mc * vartheta_bar * theta / (lambda_bar * np.exp(k_mat))
        b = - 2 * temp / (alpha * lambda_bar * np.exp(k_mat)) + F * sigma_y ** 2
        c = temp + G * np.sum(pi_c * theta_ell, axis=0)
        # e_new = c / (-b)
        e_new = np.zeros(k_mat.shape)

#     # Method 2 : Fix a and solve
#     e_new = (a * e**2 + c) / (-b)

    e_new = e_new * (e_new > 0) + 1e-16 * (e_new <= 0)
    
    i = i_new * fraction + i * (1-fraction)
    e = e_new * fraction + e * (1-fraction)

    log_pi_c_ratio = - G * e * theta_ell / xi_a
    pi_c_ratio = log_pi_c_ratio - np.max(log_pi_c_ratio)
    pi_c = np.exp(pi_c_ratio) * pi_c_o
    pi_c = pi_c / np.sum(pi_c, axis=0)
    pi_c = (pi_c <= 0) * 1e-16 + (pi_c > 0) * pi_c
    entropy = np.sum(pi_c * (np.log(pi_c) - np.log(pi_c_o)), axis=0)

    A    = np.ones_like(y_mat) * (- delta)
    B_k  = mu_k + i - kappa / 2. * i ** 2 - sigma_k ** 2 / 2.
    B_y  = np.sum(pi_c * theta_ell, axis=0) * e
    C_kk = sigma_k ** 2 / 2 * np.ones_like(y_mat)
    C_yy = .5 * sigma_y **2 * e**2

    consumption = alpha - i - alpha * vartheta_bar * ( 1 - e /(alpha * lambda_bar * np.exp(k_mat)))**theta
    consumption[consumption <= 1e-16] = 1e-16
    D = delta * np.log(consumption) + delta * k_mat - (d_Delta * np.sum(pi_c * theta_ell, axis=0) * e + .5 * dd_Delta * sigma_y ** 2 * e ** 2) + xi_a * entropy

    h = - G * e * sigma_y / xi_b

    return pi_c, A, B_k, B_y, C_kk, C_yy, D, dvdk, dvdy, dvdkk, dvdyy, i, e, h



def hjb_post_damage_post_tech(
        k_grid, y_grid, model_args=(), v0=None, 
        epsilon=1., fraction=.1, tol=1e-8, max_iter=10000, print_iteration=True
        ):

    delta, alpha, kappa, mu_k, sigma_k, theta_ell, pi_c_o, sigma_y, xi_a, xi_b, gamma_1, gamma_2, gamma_3, y_bar, theta, lambda_bar, vartheta_bar = model_args
    
    dk = k_grid[1] - k_grid[0]
    dy = y_grid[1] - y_grid[0]
    (k_mat, y_mat) = np.meshgrid(k_grid, y_grid, indexing = 'ij')
    
    a_i = kappa
    b_i = - (1. + alpha * kappa)
    c_i = alpha  - delta
    i = (- b_i - np.sqrt(b_i ** 2 - 4 * a_i * c_i)) / (2 * a_i)

    i = np.ones_like(k_mat) * i
    e = np.zeros_like(k_mat)

    if v0 is None:
        v0 =  k_mat - delta * y_mat ** 2

    d_Delta  = gamma_1 + gamma_2 * y_mat + gamma_3 * (y_mat > y_bar) * (y_mat - y_bar)
    dd_Delta = gamma_2 + gamma_3 * (y_mat > y_bar)

    pi_c_o = np.array([temp * np.ones_like(y_mat) for temp in pi_c_o])
    theta_ell = np.array([temp * np.ones_like(y_mat) for temp in theta_ell])
    pi_c = pi_c_o.copy()

    state_space = np.hstack([k_mat.reshape(-1, 1, order = 'F'),
                             y_mat.reshape(-1, 1, order = 'F')])

    count = 0
    error = 1.

    while error > tol and count < max_iter:

        pi_c, A, B_k, B_y, C_kk, C_yy, D, dvdk, dvdy, dvdkk, dvdyy, i, e, h =  _hjb_iteration(
                v0, k_mat, y_mat, dk, dy, d_Delta, dd_Delta, theta, lambda_bar, vartheta_bar,
                delta, alpha, kappa, mu_k, sigma_k, pi_c_o, pi_c, theta_ell, sigma_y, xi_a, xi_b, i, e, fraction
                )

        v = false_transient_one_iteration_cpp(state_space, A, B_k, B_y, C_kk, C_yy, D, v0, epsilon)

        rhs_error = A * v0 + B_k * dvdk + B_y * dvdy + C_kk * dvdkk + C_yy * dvdyy + D
        rhs_error = np.max(abs(rhs_error))
        lhs_error = np.max(abs((v - v0)/epsilon))

        error = lhs_error
        v0 = v
        count += 1

        if print_iteration:
            print("Iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))

    print("Converged. Total iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))

    res = {
        'v': v,
        'k': k_grid,
        'y': y_grid,
        'e': e,
        'i': i,
        'pi_c': pi_c,
        'h': h,
        'error': error,
        }

    return res

def hjb_pre_damage_post_tech(
        k_grid, y_grid, model_args=(), v0=None, epsilon=1., fraction=.1,
        tol=1e-8, max_iter=10000, print_iteration=True
        ):

    delta, alpha, kappa, mu_k, sigma_k, theta_ell, pi_c_o, sigma_y, xi_a, xi_b, xi_p, pi_d_o, v_i, gamma_1, gamma_2, theta, lambda_bar, vartheta_bar, y_bar_lower = model_args
    dk = k_grid[1] - k_grid[0]
    dy = y_grid[1] - y_grid[0]
    (k_mat, y_mat) = np.meshgrid(k_grid, y_grid, indexing = 'ij')

    a_i = kappa
    b_i = - (1. + alpha * kappa)
    c_i = alpha - 1.
    i = (- b_i - np.sqrt(b_i ** 2 - 4 * a_i * c_i)) / (2 * a_i)

    i = np.ones_like(k_mat) * i
    e = np.zeros_like(k_mat)

    if v0 is None:
        v0 = k_mat -  np.average(theta_ell, axis=0) * y_mat ** 2

    d_Delta  = gamma_1 + gamma_2 * y_mat
    dd_Delta = gamma_2

    # pi_c_o = np.array([temp * np.ones_like(y_mat) for temp in pi_c_o])
    # pi_d_o = np.array([temp * np.ones_like(y_mat) for temp in pi_d_o])
    # theta_ell = np.array([temp * np.ones_like(y_mat) for temp in theta_ell])
    pi_c = pi_c_o.copy()

    r1=1.5
    r2=2.5
    intensity = r1*(np.exp(r2/2*(y_mat - y_bar_lower)**2)-1)*(y_mat >= y_bar_lower)

    state_space = np.hstack([k_mat.reshape(-1, 1, order = 'F'),
                             y_mat.reshape(-1, 1, order = 'F')])

    error = 1.
    count = 0
    while error > tol and count < max_iter:
        pi_c, A, B_k, B_y, C_kk, C_yy, D, dvdk, dvdy, dvdkk, dvdyy, i, e, h= \
            _hjb_iteration(v0, k_mat, y_mat, dk, dy, d_Delta, dd_Delta, theta, lambda_bar, vartheta_bar,
                           delta, alpha, kappa, mu_k, sigma_k, pi_c_o, pi_c, theta_ell, sigma_y, xi_a, xi_b, i, e, fraction)

        D -= xi_p * intensity * (np.sum(pi_d_o * np.exp(- v_i / xi_p), axis=0) - np.exp(- v0 / xi_p)) / np.exp(- v0 / xi_p)

        v = false_transient_one_iteration_cpp(state_space, A, B_k, B_y, C_kk, C_yy, D, v0, epsilon)

        rhs_error = A * v0 + B_k * dvdk + B_y * dvdy + C_kk * dvdkk + C_yy * dvdyy + D
        rhs_error = np.max(abs(rhs_error))
        lhs_error = np.max(abs((v - v0)/epsilon))

        error = lhs_error
        v0 = v
        count += 1

        if print_iteration:
            print("Iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))

#     print("Converged. Total iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))

    g = np.exp(1. / xi_p * (v - v_i))

    res = {'v': v,
           'e': e,
           'i': i,
           'g': g,
           'pi_c': pi_c,
           'h': h,
           'error': error,
           }

    return res
