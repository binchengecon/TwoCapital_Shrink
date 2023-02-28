
    # # Temp
    # print("-------------Temp Start--------------")
    # xi_a_temp = xi_a
    # xi_g_temp = 1000
    # xi_p_temp = 1000
    # args = (delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a_temp, xi_g_temp, xi_p_temp)
    # variable_args = (theta_ell_temp_wakeup, psi_2_temp_wakeup)
    # controls=(ii,ee,xx, g_tech, g_damage,  pi_c_o_temp_wakeup, pi_c_temp_wakeup, pi_d_o_wakeup, v0, v_i, v_tech2)
    # ME_temp = minimize_pi_c(grid, args, n_bar1, variable_args, controls, tol=1e-6, epsilon=0.1, max_iter=10000) # n_bar free
    # ME_temp2 = minimize_pi_c(grid, args, n_bar2, variable_args, controls, tol=1e-6, epsilon=0.1, max_iter=10000) #n_bar hit 2

    


    # print("Look at differences")
    # print(ME_temp.shape)
    # print(ME_temp2.shape)
    # print(n_bar1,n_bar2)
    # print(np.max(abs(ME_temp[:,:n_bar2+1,:]-ME_temp2)))
    # print("Look at differences")
    # print("-------------Temp Done--------------")
    # print("---------------------------")
    # print("---------------------------")
    # print("---------------------------")

    # # Carb
    # print("-------------Carb Start--------------")
    # xi_a_carb = xi_a
    # xi_g_carb = 1000
    # xi_p_carb = 1000
    
    # args = (delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a_carb, xi_g_carb, xi_p_carb)
    # variable_args = (theta_ell_carb_wakeup, psi_2_carb_wakeup)
    # controls=(ii,ee,xx, g_tech, g_damage,  pi_c_o_carb_wakeup, pi_c_carb_wakeup, pi_d_o_wakeup, v0, v_i, v_tech2)
    # ME_carb = minimize_pi_c(grid, args, n_bar1, variable_args, controls, tol=1e-6, epsilon=0.1, max_iter=10000) # n_bar free
    # ME_carb2 = minimize_pi_c(grid, args, n_bar2, variable_args, controls, tol=1e-6, epsilon=0.1, max_iter=10000) #n_bar hit 2

    
    # n_bar1 = len(Y)-1
    # n_bar2 = np.abs(Y - y_bar).argmin()

    # print("Look at differences")
    # print(ME_carb.shape)
    # print(ME_carb2.shape)
    # print(n_bar1,n_bar2)
    # print(np.max(abs(ME_carb[:,:n_bar2+1,:]-ME_carb2)))
    # print("Look at differences")
    # print("-------------Carb Done--------------")
    # print("---------------------------")
    # print("---------------------------")
    # print("---------------------------")

    # # RD
    # print("-------------RD Start--------------")
    # xi_a_RD = xi_a
    # xi_g_RD = 1000
    # xi_p_RD = 1000
    
    # args = (delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a_RD, xi_g_RD, xi_p_RD)
    # variable_args = (theta_ell_RD_wakeup, psi_2_RD_wakeup)
    # controls=(ii,ee,xx, g_tech, g_damage,  pi_c_o_RD_wakeup, pi_c_RD_wakeup, pi_d_o_wakeup, v0, v_i, v_tech2)
    # ME_RD = minimize_pi_c(grid, args, n_bar1, variable_args, controls, tol=1e-6, epsilon=0.1, max_iter=10000) # n_bar free
    # ME_RD2 = minimize_pi_c(grid, args, n_bar2, variable_args, controls, tol=1e-6, epsilon=0.1, max_iter=10000) #n_bar hit 2

    
    # n_bar1 = len(Y)-1
    # n_bar2 = np.abs(Y - y_bar).argmin()

    # print("Look at differences")
    # print(ME_RD.shape)
    # print(ME_RD2.shape)
    # print(n_bar1,n_bar2)
    # print(np.max(abs(ME_RD[:,:n_bar2+1,:]-ME_RD2)))
    # print("Look at differences")
    # print("-------------RD Done--------------")

    # print("---------------------------")
    # print("---------------------------")
    # print("---------------------------")

    # # Damage
    # print("-------------Damage Start--------------")
    # xi_a_dmg = 1000
    # xi_g_dmg = 1000
    # xi_p_dmg = xi_p # xi_p is associated with damage
    
    # args = (delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a_dmg, xi_g_dmg, xi_p_dmg)
    # variable_args = (theta_ell_wakeup, psi_2_wakeup)
    # controls=(ii,ee,xx, g_tech, g_damage,  pi_c_o_wakeup, pi_c_wakeup, pi_d_o_wakeup, v0, v_i, v_tech2)
    # ME_dmg = minimize_dmg(grid, args, n_bar1, variable_args, controls, tol=1e-6, epsilon=0.1, max_iter=10000) # n_bar free
    # ME_dmg2 = minimize_dmg(grid, args, n_bar2, variable_args, controls, tol=1e-6, epsilon=0.1, max_iter=10000) #n_bar hit 2

    
    # n_bar1 = len(Y)-1
    # n_bar2 = np.abs(Y - y_bar).argmin()

    # print("Look at differences")
    # print(ME_dmg.shape)
    # print(ME_dmg2.shape)
    # print(n_bar1,n_bar2)
    # print(np.max(abs(ME_dmg[:,:n_bar2+1,:]-ME_dmg2)))
    # print("Look at differences")
    # print("-------------Damage Done--------------")
    # print("---------------------------")
    # print("---------------------------")
    # print("---------------------------")

    # # Tech
    # print("-------------Tech Start--------------")
    # xi_a_tech = 1000
    # xi_g_tech = xi_g
    # xi_p_tech = 1000
    
    # args = (delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a_tech, xi_g_tech, xi_p_tech)
    # variable_args = (theta_ell_wakeup, psi_2_wakeup)
    # controls=(ii,ee,xx, g_tech, g_damage,  pi_c_o_wakeup, pi_c_wakeup, pi_d_o_wakeup, v0, v_i, v_tech2)
    # ME_tech = minimize_tech(grid, args, n_bar1, variable_args, controls, tol=1e-6, epsilon=0.1, max_iter=10000) # n_bar free
    # ME_tech2 = minimize_tech(grid, args, n_bar2, variable_args, controls, tol=1e-6, epsilon=0.1, max_iter=10000) #n_bar hit 2

    
    # n_bar1 = len(Y)-1
    # n_bar2 = np.abs(Y - y_bar).argmin()

    # print("Look at differences")
    # print(ME_tech.shape)
    # print(ME_tech2.shape)
    # print(n_bar1,n_bar2)
    # print(np.max(abs(ME_tech[:,:n_bar2+1,:]-ME_tech2)))
    # print("Look at differences")
    # print("-------------Tech Done--------------")

    # print("---------------------------")
    # print("---------------------------")
    # print("---------------------------")


    # # Omit Temp
    # print("-------------Omit Temp Start--------------")
    # xi_a_notemp = xi_a
    # xi_g_notemp = xi_g
    # xi_p_notemp = xi_g
    
    # args = (delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a_notemp, xi_g_notemp, xi_p_notemp)
    # variable_args = (theta_ell_carbRD_wakeup, psi_2_carbRD_wakeup)
    # controls=(ii,ee,xx, g_tech, g_damage,  pi_c_o_carbRD_wakeup, pi_c_carbRD_wakeup, pi_d_o_wakeup, v0, v_i, v_tech2)
    # ME_notemp = minimize_pi_c(grid, args, n_bar1, variable_args, controls, tol=1e-6, epsilon=0.1, max_iter=10000) # n_bar free
    # ME_notemp2 = minimize_pi_c(grid, args, n_bar2, variable_args, controls, tol=1e-6, epsilon=0.1, max_iter=10000) #n_bar hit 2

    


    # print("Look at differences")
    # print(ME_notemp.shape)
    # print(ME_notemp2.shape)
    # print(n_bar1,n_bar2)
    # print(np.max(abs(ME_notemp[:,:n_bar2+1,:]-ME_notemp2)))
    # print("Look at differences")
    # print("-------------Omit Temp Done--------------")
    # print("---------------------------")
    # print("---------------------------")
    # print("---------------------------")

    # # Omit Carb
    # print("-------------Omit Carb Start--------------")
    # xi_a_nocarb = xi_a
    # xi_g_nocarb = xi_g
    # xi_p_nocarb = xi_g
    
    # args = (delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a_nocarb, xi_g_nocarb, xi_p_nocarb)
    # variable_args = (theta_ell_RDtemp_wakeup, psi_2_RDtemp_wakeup)
    # controls=(ii,ee,xx, g_tech, g_damage,  pi_c_o_RDtemp_wakeup, pi_c_RDtemp_wakeup, pi_d_o_wakeup, v0, v_i, v_tech2)
    # ME_nocarb = minimize_pi_c(grid, args, n_bar1, variable_args, controls, tol=1e-6, epsilon=0.1, max_iter=10000) # n_bar free
    # ME_nocarb2 = minimize_pi_c(grid, args, n_bar2, variable_args, controls, tol=1e-6, epsilon=0.1, max_iter=10000) #n_bar hit 2

    
    # n_bar1 = len(Y)-1
    # n_bar2 = np.abs(Y - y_bar).argmin()

    # print("Look at differences")
    # print(ME_nocarb.shape)
    # print(ME_nocarb2.shape)
    # print(n_bar1,n_bar2)
    # print(np.max(abs(ME_nocarb[:,:n_bar2+1,:]-ME_nocarb2)))
    # print("Look at differences")
    # print("-------------Omit Carb Done--------------")
    # print("---------------------------")
    # print("---------------------------")
    # print("---------------------------")

    # # Omit RD
    # print("-------------Omit RD Start--------------")
    # xi_a_noRD = xi_a
    # xi_g_noRD = xi_g
    # xi_p_noRD = xi_g
    
    # args = (delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a_noRD, xi_g_noRD, xi_p_noRD)
    # variable_args = (theta_ell_tempcarb_wakeup, psi_2_tempcarb_wakeup)
    # controls=(ii,ee,xx, g_tech, g_damage,  pi_c_o_tempcarb_wakeup, pi_c_tempcarb_wakeup, pi_d_o_wakeup, v0, v_i, v_tech2)
    # ME_noRD = minimize_pi_c(grid, args, n_bar1, variable_args, controls, tol=1e-6, epsilon=0.1, max_iter=10000) # n_bar free
    # ME_noRD2 = minimize_pi_c(grid, args, n_bar2, variable_args, controls, tol=1e-6, epsilon=0.1, max_iter=10000) #n_bar hit 2

    
    # n_bar1 = len(Y)-1
    # n_bar2 = np.abs(Y - y_bar).argmin()

    # print("Look at differences")
    # print(ME_noRD.shape)
    # print(ME_noRD2.shape)
    # print(n_bar1,n_bar2)
    # print(np.max(abs(ME_noRD[:,:n_bar2+1,:]-ME_noRD2)))
    # print("Look at differences")
    # print("-------------Omit RD Done--------------")

    # print("---------------------------")
    # print("---------------------------")
    # print("---------------------------")

    # # Omit Damage
    # print("-------------Omit Damage Start--------------")
    # xi_a_noRD = xi_a
    # xi_g_noRD = xi_g
    # xi_p_noRD = 1000
    
    # args = (delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a_noRD, xi_g_noRD, xi_p_noRD)
    # variable_args = (theta_ell_wakeup, psi_2_wakeup)
    # controls=(ii,ee,xx, g_tech, g_damage,  pi_c_o_wakeup, pi_c_wakeup, pi_d_o_wakeup, v0, v_i, v_tech2)
    # ME_nodmg = minimize_pi_c(grid, args, n_bar1, variable_args, controls, tol=1e-6, epsilon=0.1, max_iter=10000) # n_bar free
    # ME_nodmg2 = minimize_pi_c(grid, args, n_bar2, variable_args, controls, tol=1e-6, epsilon=0.1, max_iter=10000) #n_bar hit 2

    
    # n_bar1 = len(Y)-1
    # n_bar2 = np.abs(Y - y_bar).argmin()

    # print("Look at differences")
    # print(ME_nodmg.shape)
    # print(ME_nodmg2.shape)
    # print(n_bar1,n_bar2)
    # print(np.max(abs(ME_nodmg[:,:n_bar2+1,:]-ME_nodmg2)))
    # print("Look at differences")
    # print("-------------Omit Damage Done--------------")
    # print("---------------------------")
    # print("---------------------------")
    # print("---------------------------")

    # # Omit Tech
    # print("-------------Omit Tech Start--------------")
    # xi_a_notech = xi_a
    # xi_g_notech = 1000
    # xi_p_notech = xi_p
    
    # args = (delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, sigma_y, zeta, psi_0, psi_1, sigma_g, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a_notech, xi_g_notech, xi_p_notech)
    # variable_args = (theta_ell_wakeup, psi_2_wakeup)
    # controls=(ii,ee,xx, g_tech, g_damage,  pi_c_o_wakeup, pi_c_wakeup, pi_d_o_wakeup, v0, v_i, v_tech2)
    # ME_notech = minimize_pi_c(grid, args, n_bar1, variable_args, controls, tol=1e-6, epsilon=0.1, max_iter=10000) # n_bar free
    # ME_notech2 = minimize_pi_c(grid, args, n_bar2, variable_args, controls, tol=1e-6, epsilon=0.1, max_iter=10000) #n_bar hit 2

    
    # n_bar1 = len(Y)-1
    # n_bar2 = np.abs(Y - y_bar).argmin()

    # print("Look at differences")
    # print(ME_notech.shape)
    # print(ME_notech2.shape)
    # print(n_bar1,n_bar2)
    # print(np.max(abs(ME_notech[:,:n_bar2+1,:]-ME_notech2)))
    # print("Look at differences")
    # print("-------------Omit Tech Done--------------")

    # print("---------------------------")
    # print("---------------------------")
    # print("---------------------------")
