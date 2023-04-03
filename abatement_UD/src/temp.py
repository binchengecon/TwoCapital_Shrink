
def fk_yshort_pre_tech(
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

        
        D =  - (    gamma_2 +gamma_3 * (Y_mat>y_bar) )  * np.sum( theta_ell * pi_c , axis = 0 ) * e
        # D =  - (    gamma_2 )  * np.sum( theta_ell * pi_c , axis = 0 ) * e
        # D += - (    gamma_3 * (Y_mat>y_bar) * sigma_y**2/2* e**2 )
        D +=    np.exp(L_mat - np.log(448)) * g_tech * F_m_II 


        out = PDESolver(stateSpace, A, B_1, B_2, B_3, C_1, C_2, C_3, D, dv0dL, epsilon, solverType="Feyman Kac")

        v  = out[2].reshape(dv0dL.shape, order="F")
            
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
