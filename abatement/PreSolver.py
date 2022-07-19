"""
Solver for 3-state pre jump
"""
import sys
sys.path.append("../src/")
import numpy as np
import pandas as pd
from numba import njit
from supportfunctions import finiteDiff
import SolveLinSys

def false_transient_one_iteration_3d(stateSpace, A, B1, B2, B3,  C1, C2, C3,  D, v0, ε):
    A = A.reshape(-1, 1, order='F')
    B = np.hstack([B1.reshape(-1, 1, order='F'), B2.reshape(-1, 1, order='F'), B3.reshape(-1,1, order="F")])
    C = np.hstack([C1.reshape(-1, 1, order='F'), C2.reshape(-1, 1, order='F'), C3.reshape(-1,1, order="F")])
    D = D.reshape(-1, 1, order='F')
    out = SolveLinSys.solveFT(stateSpace, A, B, C, D, v0.reshape(-1, 1, order='F'), ε, -10)
    return out[2].reshape(v0.shape, order = "F")


def _hjb_iteration_pre(
        v0, k_mat, y_mat, logI_mat, dk, dy, dI, d_Delta, dd_Delta, theta, lambda_bar, vartheta_bar, delta, alpha, kappa, mu_k, sigma_k, pi_c_o, pi_c, theta_ell, sigma_y, zeta, psi_0, psi_1, sigma_g,  xi_a, xi_b, i, e, x, fraction
        ):

    dvdk  = finiteDiff(v0, 0, 1, dk)
    # dvdk[dvdk < 1e-15] = 1e-15
    dvdkk = finiteDiff(v0, 0, 2, dk)
    dvdy  = finiteDiff(v0, 1, 1, dy)
    dvdyy = finiteDiff(v0, 1, 2, dy)
    dvdI  = finiteDiff(v0, 2, 1, dI)
    dvdI[dvdI <1e-10] = 1e-10
    dvdII = finiteDiff(v0, 2, 2, dI)

    temp = alpha - i - alpha * vartheta_bar * (1 - e / (alpha * lambda_bar * np.exp(k_mat))) ** theta - x
    temp[temp < 1e-16] = 1e-16
    mc = 1. / temp

    i_new = - (mc / dvdk - 1) / kappa

    G = dvdy  - d_Delta
    F = dvdyy - dd_Delta

    temp = mc * vartheta_bar * theta / (lambda_bar * np.exp(k_mat))
    a = temp / (alpha * lambda_bar * np.exp(k_mat)) ** 2
    b = - 2 * temp / (alpha * lambda_bar * np.exp(k_mat)) + (F - G**2/xi_b) * sigma_y ** 2
    c = temp + G * np.sum(pi_c * theta_ell, axis=0)

    # Method 1 : Solve second order equation
    if vartheta_bar != 0 and theta == 3:
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
        e_new = c / (-b)

#     # Method 2 : Fix a and solve
#     e_new = (a * e**2 + c) / (-b)


    # find x_new
    if psi_1 ==1 and theta == 2:
        G = dvdy  - d_Delta
        F = dvdyy - dd_Delta
        temp2 = dvdI * psi_1 * psi_0 * np.exp(k_mat - logI_mat)
        Omega_1 = temp2 * theta * vartheta_bar / lambda_bar * np.exp(- k_mat) + G * np.sum(pi_c * theta_ell, axis=0)
        Omega_2 = temp2 * theta * vartheta_bar / lambda_bar * np.exp(- k_mat) / (alpha * lambda_bar * np.exp(k_mat))- F * sigma_y**2
        e_new = Omega_1 / Omega_2
        i_new = - (temp2 / dvdk - 1) / kappa
        temp3 = alpha - i_new - alpha * vartheta_bar * (1 - e_new / (alpha * lambda_bar * np.exp(k_mat)))**theta
        x_new = temp3 * np.exp(k_mat - logI_mat) - 1 / (dvdI * psi_0 * psi_1)

    elif psi_1 ==1 and theta == 3:
        G = dvdy  - d_Delta
        F = dvdyy - dd_Delta
        mc = dvdI * psi_1 * psi_0 * np.exp(k_mat - logI_mat)
        temp = mc * vartheta_bar * theta / (lambda_bar * np.exp(k_mat))
        a = temp / (alpha * lambda_bar * np.exp(k_mat)) ** (theta - 1)
        b = - 2 * temp / (alpha * lambda_bar * np.exp(k_mat)) + F * sigma_y ** 2
        c = temp + G * np.sum(pi_c * theta_ell, axis=0)
        temp = b ** 2 - 4 * a * c
        temp[temp <=0] = 0
        # temp = temp * (temp > 0)
        root1 = (- b - np.sqrt(temp)) / (2 * a)
        root2 = (- b + np.sqrt(temp)) / (2 * a)
        if root1.all() > 0 :
            e_new = root1
        else:
            e_new = root2

        i_new = (1 - mc/ dvdk) / kappa
        temp3 = alpha - i_new - alpha * vartheta_bar * (1 - e_new / (alpha * lambda_bar * np.exp(k_mat)))**theta
        x_new = temp3 * np.exp(k_mat - logI_mat) - 1 / (dvdI * psi_0 * psi_1)

    elif psi_1 != 1 and theta == 2:
        temp = alpha - i - alpha * vartheta_bar * (1 - e / (alpha * lambda_bar * np.exp(k_mat))) ** theta - x
        temp[temp < 1e-16] = 1e-16
        mc = 1. / temp

        temp1 =  mc * vartheta_bar * theta / (lambda_bar * np.exp(k_mat))
        a = - mc * temp1 / (alpha * lambda_bar * np.exp(k_mat)) + F * sigma_y**2
        b = mc * temp1 + G * np.sum(pi_c * theta_ell, axis=0)
        e_new = - b / a
        i_new = (1 - mc / dvdk) / kappa
        x_new = (mc / dvdI * psi_0 * psi_1)**(1 / (psi_1 - 1))
    elif psi_1 != 1 and vartheta_bar != 0 and theta == 3:
        temp = b ** 2 - 4 * a * c
        temp = temp * (temp > 0)
        root1 = (- b - np.sqrt(temp)) / (2 * a)
        root2 = (- b + np.sqrt(temp)) / (2 * a)
        if root1.all() > 0 :
            e_new = root1
        else:
            e_new = root2
        temp = alpha - i - alpha * vartheta_bar * (1 - e / (alpha * lambda_bar * np.exp(k_mat))) ** theta - x
        temp[temp < 1e-16] = 1e-16
        mc = 1. / temp
        i_new = - (mc / dvdk - 1) / kappa
        x_new = (mc / dvdI * psi_0 * psi_1)**(1 / (psi_1 - 1))

    # x_new = np.zeros(k_mat.shape)
    # i_new[i_new <= 1e-15] = 1e-15
    # e_new[e_new <= 1e-15] = 1e-15
    # x_new[x_new <= 1e-15] = 1e-15
    i = i_new * fraction + i * (1-fraction)
    e = e_new * fraction + e * (1-fraction)
    x = x_new * fraction + x * (1-fraction)
    print("min i: {},\t max i: {}\t".format(i.min(), i.max()))
    print("min e: {},\t max e: {}\t".format(e.min(), e.max()))
    print("min x: {},\t max x: {}\t".format(x.min(), x.max()))

    log_pi_c_ratio = - G * e * theta_ell / xi_a
    pi_c_ratio = log_pi_c_ratio - np.max(log_pi_c_ratio)
    pi_c = np.exp(pi_c_ratio) * pi_c_o
    pi_c = (pi_c <= 0) * 1e-16 + (pi_c > 0) * pi_c
    pi_c = pi_c / np.sum(pi_c, axis=0)
    entropy = np.sum(pi_c * (np.log(pi_c) - np.log(pi_c_o)), axis=0)

    A = np.ones_like(y_mat) * (- delta)
    B_k = mu_k + i - kappa / 2. * i ** 2 - sigma_k ** 2 / 2.
    B_y = np.sum(pi_c * theta_ell, axis=0) * e
    B_I = - zeta + psi_0 * (x * np.exp(k_mat - logI_mat) )**psi_1 - sigma_g**2/2
    C_kk = sigma_k ** 2 / 2 * np.ones_like(y_mat)
    C_yy = .5 * sigma_y **2 * e**2
    C_II = sigma_g**2 / 2 * np.ones_like(logI_mat)

    consumption = alpha - i - alpha * vartheta_bar * (1 - e / (alpha * lambda_bar * np.exp(k_mat)))**theta - x
    consumption[consumption <= 1e-16] = 1e-16
    print("min consum: {},\t max consum: {}\t".format(consumption.min(), consumption.max()))

    D = delta * np.log(consumption) + delta * k_mat - (d_Delta * np.sum(pi_c * theta_ell, axis=0) * e + .5 * dd_Delta * sigma_y ** 2 * e ** 2) + xi_a * entropy

    h = - G * e * sigma_y / xi_b

    return pi_c, A, B_k, B_y, B_I, C_kk, C_yy, C_II,  D, dvdk, dvdy, dvdI, dvdkk, dvdyy, dvdII, i, e, x, h


def hjb_post_damage_pre_tech(
        k_grid, y_grid, logI_grid, model_args=(),
        v0=None, ϵ=1., fraction=.1, tol=1e-8, max_iter=10_000,
        print_iteration=True,
        Guess=None,
        ):

    delta, alpha, kappa, mu_k, sigma_k, theta_ell, pi_c_o, sigma_y, xi_a, xi_b, xi_g, v_post, gamma_1, gamma_2, gamma_3, y_bar, zeta, psi_0, psi_1, sigma_g, theta, lambda_bar, vartheta_bar = model_args
    dk = k_grid[1] - k_grid[0]
    dy = y_grid[1] - y_grid[0]
    dI = logI_grid[1] - logI_grid[0]
    (k_mat, y_mat, logI_mat) = np.meshgrid(k_grid, y_grid, logI_grid, indexing = 'ij')

    if v0 is None:
        v0 = k_mat -  y_mat ** 2
    # expand v_post
    V_post = np.zeros_like(k_mat)
    for i in range(len(logI_grid)):
        V_post[:, :, i] = v_post

    if Guess is None:

        a_i = kappa
        b_i = - (1. + alpha * kappa) 
        c_i = alpha - 1.
        i = (- b_i - np.sqrt(b_i ** 2 - 4 * a_i * c_i)) / (2 * a_i)

        i = np.ones_like(k_mat) * i
        e = np.zeros_like(k_mat)
        x = np.zeros_like(k_mat)

    else:
        v0 = Guess["v0"]
        i  = Guess["i"]
        e  = Guess["e"]
        x  = Guess["x"]


    d_Delta = gamma_1 + gamma_2 * y_mat + gamma_3 * (y_mat > y_bar) * (y_mat - y_bar)
    dd_Delta = gamma_2 + gamma_3 * (y_mat > y_bar)

    pi_c_o = np.array([temp * np.ones_like(y_mat) for temp in pi_c_o])
    theta_ell = np.array([temp * np.ones_like(y_mat) for temp in theta_ell])
    pi_c = pi_c_o.copy()

    state_space = np.hstack([k_mat.reshape(-1, 1, order = 'F'),
                             y_mat.reshape(-1, 1, order = 'F'),
                             logI_mat.reshape(-1, 1, order = 'F')])

    count = 0
    error = 1.

    while error > tol and count < max_iter:
        print("---------------------------------------------")
        print("---------------Iteration: {}--------------------".format(count))
        print("---------------------------------------------")
        pi_c, A, B_k, B_y, B_I, C_kk, C_yy, C_II,  D, dvdk, dvdy, dvdI, dvdkk, dvdyy, dvdII, i, e, x, h = _hjb_iteration_pre(
        v0, k_mat, y_mat, logI_mat, dk, dy, dI, d_Delta, dd_Delta, theta, lambda_bar, vartheta_bar, delta, alpha, kappa, mu_k, sigma_k, pi_c_o, pi_c, theta_ell, sigma_y, zeta, psi_0, psi_1, sigma_g,  xi_a, xi_b, i, e, x, fraction
                    )

        g_tech = np.exp(1. / xi_g * (v0 - V_post))
        g_tech[g_tech <= 1e-16] = 1e-16
        A -= np.exp(logI_mat) * g_tech
        D += np.exp(logI_mat) * g_tech * (V_post ) + xi_g * np.exp(logI_mat) * (1 - g_tech + g_tech * np.log(g_tech))

        v = false_transient_one_iteration_3d(state_space, A, B_k, B_y, B_I, C_kk, C_yy, C_II, D, v0, ε)

        rhs_error = A * v0 + B_k * dvdk + B_y * dvdy + B_I * dvdI + C_kk * dvdkk + C_yy * dvdyy + C_II * dvdII + D
        rhs_error = np.max(abs(rhs_error))
        lhs_error = np.max(abs((v - v0)/ϵ))

        error = lhs_error
        v0 = v
        count += 1

        if print_iteration:
            print("Iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))

#     print("Converged. Total iteration %s: LHS Error: %s; RHS Error %s" % (count, lhs_error, rhs_error))

    g_tech = np.exp(1. / xi_g * (v - V_post))

    import pickle
    from datetime import datetime
    now = datetime.now()
    current_time = now.strftime("%d-%H:%M")
    filename = "pre_jump_res_{}".format(current_time)

    my_shelf = {}
    for key in dir():
        if isinstance(locals()[key], (int,float, float, str, bool, np.ndarray,list)):
            try:
                my_shelf[key] = locals()[key]
            except TypeError:
                #
                # __builtins__, my_shelf, and imported modules can not be shelved.
                #
                print('ERROR shelving: {0}'.format(key))
        else:
            pass


    file = open("./res_data/" + filename, 'wb')
    pickle.dump(my_shelf, file)
    file.close()
    res = {'v': v,
           'e': e,
           'i': i,
           'g_tech': g_tech,
           'pi_c': pi_c,
           'h': h}

    return res
