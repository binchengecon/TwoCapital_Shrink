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




def extract(
    grid = (), 
    model_args = (), 
    sol_beforeupdate = (),  
    data=(),
    initial=(np.log(85/0.115), 1.1, 2.4), 
    T0=0, 
    T=40, 
    dt=1/12,
    printing=True):
        
    
    ### Argument Extraction
    

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
    theta_ell_array = pd.read_csv("/home/bcheng4/TwoCapital_Shrink/data/model144_p.csv", header=None).to_numpy()[:, 0]/1000.
    psi_2_array = pd.read_csv('/home/bcheng4/TwoCapital_Shrink/data/psi2value_p.csv', header=None).to_numpy()[:, 0]
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
    res_tpset_paru = []
    res_tpset_nou = []
    res_tpset_nou_noFT = []
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
        
        # res_tp_paru = hjb_pre_tech_partialupdate(
        #         state_grid=(K, Y, L), 
        #         model_args=(delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell_array, sigma_y, zeta, psi_0, psi_1, psi_2_array, sigma_g, V_post_tech2, gamma_1, gamma_2, gamma_3_list[i], y_bar, xi_a_post, xi_g_post, xi_p_post),
        #         control_fixed=(model_tech1_post_damage_i_ii, model_tech1_post_damage_i_ee, model_tech1_post_damage_i_xx, model_tech1_post_damage_v0),
        #         n_bar = n_bar,
        #         V_post_damage=None,
        #         tol=1e-7, 
        #         epsilon=0.3, 
        #         fraction=0.1, 
        #         smart_guess=Guess, 
        #         max_iter=10000,
        #         )
        # print("-----------PartialUp Save Data: {}------------".format(Data_Dir+ File_Dir))

        # with open(Data_Dir+ File_Dir  + "model_tech1_post_damage_gamma_{:.4f}_base_partialupdate".format(gamma_3_list[i]), "wb") as f:
        #     pickle.dump(res_tp_paru, f)
            
        res_tp_paru = pickle.load(open(Data_Dir+ File_Dir  + "model_tech1_post_damage_gamma_{:.4f}_base_partialupdate".format(gamma_3_list[i]), "rb"))

        # res_tp_nou = hjb_pre_tech_noupdate(
        #         state_grid=(K, Y, L), 
        #         model_args=(delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell_array, sigma_y, zeta, psi_0, psi_1, psi_2_array, sigma_g, V_post_tech2, gamma_1, gamma_2, gamma_3_list[i], y_bar, xi_a_post, xi_g_post, xi_p_post),
        #         control_fixed=(model_tech1_post_damage_i_ii, model_tech1_post_damage_i_ee, model_tech1_post_damage_i_xx, model_tech1_post_damage_v0),
        #         n_bar = n_bar,
        #         V_post_damage=None,
        #         tol=1e-7, 
        #         epsilon=0.3, 
        #         fraction=0.1, 
        #         smart_guess=Guess, 
        #         max_iter=10000,
        #         )
        # print("-----------NoUp Save Data: {}------------".format(Data_Dir+ File_Dir))

        # with open(Data_Dir+ File_Dir  + "model_tech1_post_damage_gamma_{:.4f}_base_noupdate".format(gamma_3_list[i]), "wb") as f:
        #     pickle.dump(res_tp_nou, f)


        res_tp_nou = pickle.load(open(Data_Dir+ File_Dir  + "model_tech1_post_damage_gamma_{:.4f}_base_noupdate".format(gamma_3_list[i]), "rb"))

        # res_tp_nou_noFT = hjb_pre_tech_noupdate_noFT(
        #         state_grid=(K, Y, L), 
        #         model_args=(delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell_array, sigma_y, zeta, psi_0, psi_1, psi_2_array, sigma_g, V_post_tech2, gamma_1, gamma_2, gamma_3_list[i], y_bar, xi_a_post, xi_g_post, xi_p_post),
        #         control_fixed=(model_tech1_post_damage_i_ii, model_tech1_post_damage_i_ee, model_tech1_post_damage_i_xx, model_tech1_post_damage_v0),
        #         n_bar = n_bar,
        #         V_post_damage=None,
        #         tol=1e-7, 
        #         epsilon=0.3, 
        #         fraction=0.1, 
        #         smart_guess=Guess, 
        #         max_iter=10000,
        #         )
        # print("-----------NoUp NoFT Save Data: {}------------".format(Data_Dir+ File_Dir))

        # with open(Data_Dir+ File_Dir  + "model_tech1_post_damage_gamma_{:.4f}_base_noupdate_noFT".format(gamma_3_list[i]), "wb") as f:
        #     pickle.dump(res_tp_nou_noFT, f)

        res_tp_nou_noFT = pickle.load(open(Data_Dir+ File_Dir  + "model_tech1_post_damage_gamma_{:.4f}_base_noupdate_noFT".format(gamma_3_list[i]), "rb"))


        # Local_Output_Dir = "./"
        # Local_Data_Dir = Local_Output_Dir+"abatement/data_2tech/"+str(dataname)+"/"
        # Local_File_Dir = "xi_a_{}_xi_g_{}_psi_0_{}_psi_1_{}_" .format(xi_a,xi_g,psi_0,psi_1)

        # os.makedirs(Local_Data_Dir, exist_ok=True)

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
        
        res_tpset_paru.append(res_tp_paru)
        res_tpset_nou.append(res_tp_nou)
        res_tpset_nou_noFT.append(res_tp_nou_noFT)
        
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
    
    
    # res_base = hjb_pre_tech_partialupdate(state_grid = grid, 
    #                                      model_args = args, 
    #                                      control_fixed = controls,
    #                                      n_bar = n_bar1, 
    #                                      V_post_damage=v_i_base,
    #                                      tol=1e-7, 
    #                                      epsilon=0.3, 
    #                                      max_iter=10000) # n_bar free
    # res_base2 = hjb_pre_tech_partialupdate(state_grid = grid, 
    #                                      model_args = args, 
    #                                      control_fixed = controls,
    #                                      n_bar = n_bar2, 
    #                                      V_post_damage=v_i_base,
    #                                      tol=1e-7, 
    #                                      epsilon=0.3, 
    #                                      max_iter=10000) #n_bar hit 2


    # with open(Data_Dir+ File_Dir  + "model_tech1_pre_damage_base_scheme1_partialupdate", "wb") as f:
    #     pickle.dump(res_base, f)
    # with open(Data_Dir+ File_Dir  + "model_tech1_pre_damage_base2_scheme1_partialupdate", "wb") as f:
    #     pickle.dump(res_base2, f)

    # print("-----------Local PartialUp Save Data Final: {}------------".format(Local_Data_Dir+ Local_File_Dir))

    # with open(Local_Data_Dir+ Local_File_Dir  + "model_tech1_pre_damage_base_scheme1_partialupdate", "wb") as f:
    #     pickle.dump(res_base, f)

    # print("-----------Local PartialUp Save Data Final: {}------------".format(Local_Data_Dir+ Local_File_Dir))

    # with open(Local_Data_Dir+ Local_File_Dir  + "model_tech1_pre_damage_base2_scheme1_partialupdate", "wb") as f:
    #     pickle.dump(res_base2, f)
    
    res_base_paru = pickle.load(open(Data_Dir+ File_Dir  + "model_tech1_pre_damage_base_scheme1_partialupdate", "rb"))


    # res_base_nou = hjb_pre_tech_noupdate(state_grid = grid, 
    #                                      model_args = args, 
    #                                      control_fixed = controls,
    #                                      n_bar = n_bar1, 
    #                                      V_post_damage=v_i_base,
    #                                      tol=1e-7, 
    #                                      epsilon=0.3, 
    #                                      max_iter=10000) # n_bar free
    # res_base2_nou = hjb_pre_tech_noupdate(state_grid = grid, 
    #                                      model_args = args, 
    #                                      control_fixed = controls,
    #                                      n_bar = n_bar2, 
    #                                      V_post_damage=v_i_base,
    #                                      tol=1e-7, 
    #                                      epsilon=0.3, 
    #                                      max_iter=10000) #n_bar hit 2


    # with open(Data_Dir+ File_Dir  + "model_tech1_pre_damage_base_scheme1_noupdate", "wb") as f:
    #     pickle.dump(res_base_nou, f)
    # with open(Data_Dir+ File_Dir  + "model_tech1_pre_damage_base2_scheme1_noupdate", "wb") as f:
    #     pickle.dump(res_base2_nou, f)

    # print("-----------Local NoUp Save Data Final: {}------------".format(Local_Data_Dir+ Local_File_Dir))

    # with open(Local_Data_Dir+ Local_File_Dir  + "model_tech1_pre_damage_base_scheme1_noupdate", "wb") as f:
    #     pickle.dump(res_base_nou, f)

    # print("-----------Local NoUp Save Data Final: {}------------".format(Local_Data_Dir+ Local_File_Dir))

    # with open(Local_Data_Dir+ Local_File_Dir  + "model_tech1_pre_damage_base2_scheme1_noupdate", "wb") as f:
    #     pickle.dump(res_base2_nou, f)

    res_base_nou = pickle.load(open(Data_Dir+ File_Dir  + "model_tech1_pre_damage_base_scheme1_noupdate", "rb"))
    # res_base2 = pickle.load(open(Data_Dir+ File_Dir  + "model_tech1_pre_damage_base2_scheme1", "rb"))
 
    # res_base_nou_noFT = hjb_pre_tech_noupdate_noFT(state_grid = grid, 
    #                                      model_args = args, 
    #                                      control_fixed = controls,
    #                                      n_bar = n_bar1, 
    #                                      V_post_damage=v_i_base,
    #                                      tol=1e-7, 
    #                                      epsilon=0.3, 
    #                                      max_iter=10000) # n_bar free
    # res_base2_nou_noFT = hjb_pre_tech_noupdate_noFT(state_grid = grid, 
    #                                      model_args = args, 
    #                                      control_fixed = controls,
    #                                      n_bar = n_bar2, 
    #                                      V_post_damage=v_i_base,
    #                                      tol=1e-7, 
    #                                      epsilon=0.3, 
    #                                      max_iter=10000) #n_bar hit 2


    # with open(Data_Dir+ File_Dir  + "model_tech1_pre_damage_base_scheme1_noupdate_noFT", "wb") as f:
    #     pickle.dump(res_base_nou_noFT, f)
    # with open(Data_Dir+ File_Dir  + "model_tech1_pre_damage_base2_scheme1_noupdate_noFT", "wb") as f:
    #     pickle.dump(res_base2_nou_noFT, f)

    res_base_nou_noFT = pickle.load(open(Data_Dir+ File_Dir  + "model_tech1_pre_damage_base_scheme1_noupdate_noFT", "rb"))
    # res_base2_nou_noFT = pickle.load(open(Data_Dir+ File_Dir  + "model_tech1_pre_damage_base2_scheme1_noupdate_noFT", "rb"))


    
    return res_tpset_paru, res_tpset_nou, res_tpset_nou_noFT, res_base_paru, res_base_nou, res_base_nou_noFT, model_tech1_pre_damage, K, Y, L, Y_short




def model_extract(grid=(),
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
    
    res_tpset_paru, res_tpset_nou, res_tpset_nou_noFT, res_base_paru, res_base_nou, res_base_nou_noFT, model_tech1_pre_damage, K, Y, L, Y_short = extract(grid = grid_UD,
                        model_args = combined_argument_uncertainty,
                        sol_beforeupdate = sol_beforeupdate,
                        data = dataname,
                        T0 = 0, 
                        T = IntPeriod, 
                        dt = timespan,
                        printing = True)
    
    

    return res_tpset_paru, res_tpset_nou, res_tpset_nou_noFT, res_base_paru, res_base_nou, res_base_nou_noFT, model_tech1_pre_damage, K, Y, L, Y_short