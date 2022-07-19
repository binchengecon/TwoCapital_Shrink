import numpy as np
import pandas as pd
from PostSolver import hjb_post_damage_post_tech
from PreSolver import hjb_post_damage_pre_tech
import pickle
##################
xi_a = 100.
xi_p = 100.
xi_b = 100.
xi_g = 1000.
arrival = 20
y_bar = 2.
y_bar_lower = 1.5
n_model = 20
##################
# Model parameters
delta   = 0.010
alpha   = 0.115
kappa   = 6.667
mu_k    = -0.043
sigma_k = np.sqrt(0.0087**2 + 0.0038**2)
# Technology
theta        = 3 # 3
lambda_bar   = 0.1206
vartheta_bar = 0.0453

gamma_1   = 1.7675/10000
gamma_2   = 0.0022*2
gamma_3_i = 0.0



theta_ell = pd.read_csv('../data/model144.csv', header=None).to_numpy()[:, 0]/1000.
pi_c_o = np.ones_like(theta_ell)/len(theta_ell)
sigma_y = 1.2 * np.mean(theta_ell)

# Grid setting
k_step = .1
k_grid = np.arange(4., 8.5 + k_step, k_step)
nk = len(k_grid)
y_step = .1
y_grid_long = np.arange(0., 3. +y_step, y_step)
y_grid_short = np.arange(0., 2.5+y_step, y_step)
ny = len(y_grid_long)
# n_bar = find_nearest_value(y_grid_long, τ) + 1

logI_step = 0.2
logI_min  = -5.
logI_max  = -0.
logI_grid = np.arange(logI_min, logI_max , logI_step)
nlogI = len(logI_grid)

zeta    = 0.0
psi_0   = 0.05
psi_1   = 1
sigma_g = 0.016
# Tech jump
lambda_bar_first = lambda_bar / 2
vartheta_bar_first = vartheta_bar / 2
lambda_bar_second = 1e-9
vartheta_bar_second = 0.


# After second jump
model_args = (delta, alpha, kappa, mu_k, sigma_k, theta_ell, pi_c_o, sigma_y, xi_a, xi_b, gamma_1, gamma_2, gamma_3_i, y_bar, theta, lambda_bar_first, vartheta_bar_first)


model_res = hjb_post_damage_post_tech(
        k_grid, y_grid_long, model_args, v0=None,
        epsilon=0.5, fraction=.5,tol=1e-10, max_iter=4000, print_iteration=True)

with open("./res_data/post_jump_3", "wb") as f:
    pickle.dump(model_res, f)

# with open("./res_data/post_jump_3", "rb") as f:
    # model_res = pickle.load(f)

# v_post = model_res["v"]

# V_post = np.zeros((nk, ny, nlogI))
# for i in range(nlogI):
    # V_post[:,:,i] = v_post

# Guess = pickle.load(open("./res_data/pre_jump_res_01-04:30", "rb"))


# model_args_pre = ( delta, alpha, kappa, mu_k, sigma_k, theta_ell, pi_c_o, sigma_y, xi_a, xi_b, xi_g, v_post, gamma_1, gamma_2, gamma_3_i, y_bar, zeta, psi_0, psi_1, sigma_g, theta, lambda_bar, vartheta_bar)

# res_pre = hjb_post_damage_pre_tech(
        # k_grid, y_grid_long, logI_grid, model_args_pre, v0=V_post,
        # ε=0.0001, fraction=0.5, tol=1e-8, max_iter=20000,
        # Guess=Guess
        # )
# with open("./res_data/pre_jump", "wb") as f:
    # pickle.dump(res_pre, f, protocol=pickle.HIGHEST_PROTOCOL)
