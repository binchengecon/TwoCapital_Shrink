def minimize_pi_c2(grid=(),  args = (), controls = (), tol=1e-7, epsilon=0.1, max_iter=10000):
    """
    compute jump model with ambiguity over climate models
    """    
    
    K, Y, L = grid

    
    K_min, K_max, Y_min, Y_max, L_min, L_max = min(K), max(K), min(Y), max(Y), min(L), max(L)
    hK, hY, hL = K[1] - K[0], Y[1] - Y[0], L[1]-L[0]
    nK, nY, nL = len(K), len(Y), len(L)
    
    print("K_min={},K_max={},Y_min={},Y_max={},L_min={},L_max={}" .format(K.min(),K.max(),Y.min(),Y.max(),L.min(),L.max()))
    print("hK={},hY={},hL={}" .format(hK, hY, hL))
    print("nK={},nY={},nL={}" .format(nK, nY, nL))

    (K_mat, Y_mat, L_mat) = np.meshgrid(K, Y, L, indexing = 'ij')

    n_bar = np.abs(Y - y_bar).argmin()

    delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell_parted, pi_c_o_parted, pi_c_parted, pi_d_o, sigma_y, zeta, psi_0, psi_1, psi_2_parted, sigma_g, v_tech2, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a, xi_g, xi_p = args

    ii, ee, xx, g_tech, g_damage, pi_c_parted, v0, v_i, V_post_tech = controls
    
    ii = ii[:,:n_bar+1,:]
    ee  = ee[:,:n_bar+1,:]
    xx = xx[:,:n_bar+1,:]
    g_tech = g_tech[:,:n_bar+1,:]
    g_damage = g_damage[:,:n_bar+1,:]
    pi_c = pi_c[:,:n_bar+1,:]
    v0  = 
    # v_i = ee[:,:n_bar+1,:] additional treatment 
    V_post_tech = V_post_tech[:,:n_bar+1,:]

    # v = vf
    
    # args = (delta, alpha, kappa, mu_k, sigma_k, gamma_1, gamma_2, theta_ell_temp, psi_2_temp, pi_c_o_temp, sigma_y,  theta, vartheta_bar, lambda_bar)
    # controls=(i,e,x, g_tech, g_damage, pi_c, v0, v_i, v_tech2)
    # model_args =(delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell, pi_c_o, pi_c, sigma_y, zeta, psi_0, psi_1, psi_2, sigma_g, v_tech2, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a, xi_g, xi_p)

    n_climate = len(pi_c_parted)

    method = 'linear'
    years  = np.arange(T0, T0 + T + dt, dt)
    pers   = len(years)
       

    # setting up grids
    stateSpace = np.hstack([
        K_mat.reshape(-1,1,order = "F"), 
        Y_mat.reshape(-1,1,order = "F"),
        L_mat.reshape(-1,1,order = "F"),
    ])

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


    # tol=1e-8
    # epsilon=0.1
    # fraction=0.5
    # max_iter=10000
    
    while FC_Err > tol and epoch < max_iter:
        
        
        
        dVdK  = finiteDiff_3D(v0,0,1,hK)
        dVdK[dVdK <= 1e-16] = 1e-16
        dK = dVdK
        dVdY  = finiteDiff_3D(v0,1,1,hY)
        dY = dVdY
        dVdL  = finiteDiff_3D(v0,2,1,hL)
        dVdL[dVdL <= 1e-16] = 1e-16
        dL = dVdL
        ######## second order
        ddVdK = finiteDiff_3D(v0,0,2,hK)
        ddVdY = finiteDiff_3D(v0,1,2,hY)
        ddY = ddVdY
        ddVdL = finiteDiff_3D(v0,2,2,hL)
        
        dG  = gamma_1 + gamma_2 * Y_mat
        ddG = gamma_2 
        G = dY -  dG
        
        
        log_pi_c_ratio = - G * ee * theta_ell_parted / xi_a

        log_pi_c_ratio += -dL * psi_0 * x_new**psi_1 * np.exp( psi_1 * K_mat - (1-psi_2_parted) * L_mat) / xi_a

        pi_c_ratio = log_pi_c_ratio - np.max(log_pi_c_ratio)
        pi_c = np.exp(pi_c_ratio) * pi_c_o_parted
        pi_c = (pi_c <= 0) * 1e-16 + (pi_c > 0) * pi_c
        pi_c = pi_c / np.sum(pi_c, axis=0)
        entropy = np.sum(pi_c * (np.log(pi_c) - np.log(pi_c_o_parted)), axis=0)


        A   = - delta * np.ones(K_mat.shape) - np.exp(  L_mat - np.log(448) ) * g_tech
        B_1 = mu_k + ii - 0.5 * kappa * ii**2 - 0.5 * sigma_k**2
        B_2 = np.sum(theta_ell_parted * pi_c, axis=0) * ee
        # B_3 = - zeta + psi_0 * (xx * np.exp(K_mat - L_mat))**psi_1 - 0.5 * sigma_g**2
        B_3 = - zeta + psi_0 * xx** psi_1 * np.exp( psi_1 * K_mat ) * np.sum(pi_c * np.exp( -( 1-psi_2_parted) * L_mat  ), axis=0 )- 0.5 * sigma_g**2

        C_1 = 0.5 * sigma_k**2 * np.ones(K_mat.shape)
        C_2 = 0.5 * sigma_y**2 * ee**2
        C_3 = 0.5 * sigma_g**2 * np.ones(K_mat.shape)
        D = delta * np.log(consumption) + delta * K_mat  - dG * np.sum(theta_ell_parted * pi_c, axis=0) * ee  - 0.5 * ddG * sigma_y**2 * ee**2  + xi_a * entropy + xi_g * np.exp((L_mat - np.log(448))) * (1 - g_tech + g_tech * np.log(g_tech)) + np.exp( (L_mat - np.log(448)) ) * g_tech * V_post_tech
        # g_damage = np.exp(1 / xi_p * (v0 - v_i))

        D += xi_p * Intensity * np.sum( pi_d_o*(1-g_damage+g_damage*np.log(g_damage)),axis=0) +Intensity*np.sum(pi_d_o*g_damage*v_i,axis=0)
        A -=  Intensity*np.sum(pi_d_o*g_damage,axis=0)

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
        print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
        print("PETSc preconditioned residual norm is {:g}; iterations: {}".format(ksp.getResidualNorm(), ksp.getIterationNumber()))
        PDE_rhs = A * v0 + B_1 * dVdK + B_2 * dVdY + B_3 * dVdL + C_1 * ddVdK + C_2 * ddVdY + C_3 * ddVdL + D
        PDE_Err = np.max(abs(PDE_rhs))
        FC_Err = np.max(abs((out_comp - v0)/ epsilon))
        
        if FC_Err < 10*tol:
            
            print("-----------------------------------")
            print("---------Epoch {}---------------".format(epoch))
            print("-----------------------------------")
            print("min i: {},\t max i: {}\t".format(ii.min(), ii.max()))
            print("min e: {},\t max e: {}\t".format(ee.min(), ee.max()))
            print("min x: {},\t max x: {}\t".format(xx.min(), xx.max()))
            print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
            print("Epoch {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))
            print("Epoch time: {:.4f}".format(time.time() - start_ep))
        elif epoch%100==0:
            
            print("-----------------------------------")
            print("---------Epoch {}---------------".format(epoch))
            print("-----------------------------------")
            print("min i: {},\t max i: {}\t".format(ii.min(), ii.max()))
            print("min e: {},\t max e: {}\t".format(ee.min(), ee.max()))
            print("min x: {},\t max x: {}\t".format(xx.min(), xx.max()))
            print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
            print("Epoch {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))
            print("Epoch time: {:.4f}".format(time.time() - start_ep))
          
          
        v0     = out_comp
        # pi_c
        epoch += 1
        
    dVdY = finiteDiff_3D(v0, 1, 1, hY)
    ddVdY = finiteDiff_3D(v0, 1, 2, hY)
    ME = - dVdY * np.sum(pi_c_o * theta_ell, axis=0) -  ddVdY * sigma_y**2 * ee + dG * np.sum(theta_ell * pi_c_o, axis=0) + 1/2 * ddG * sigma_y**2 * ee

    
    return ME



def minimize_g2(grid=(),  args = (), controls = (), tol=1e-7, epsilon=0.1, max_iter=10000):
    """
    compute jump model with ambiguity over climate models
    """

    K, Y, L = grid

    
    K_min, K_max, Y_min, Y_max, L_min, L_max = min(K), max(K), min(Y), max(Y), min(L), max(L)
    hK, hY, hL = K[1] - K[0], Y[1] - Y[0], L[1]-L[0]
    nK, nY, nL = len(K), len(Y), len(L)
    
    print("K_min={},K_max={},Y_min={},Y_max={},L_min={},L_max={}" .format(K.min(),K.max(),Y.min(),Y.max(),L.min(),L.max()))
    print("hK={},hY={},hL={}" .format(hK, hY, hL))
    print("nK={},nY={},nL={}" .format(nK, nY, nL))

    (K_mat, Y_mat, L_mat) = np.meshgrid(K, Y, L, indexing = 'ij')

    n_bar = np.abs(Y - y_bar).argmin()


    delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell, pi_c_o, pi_c, pi_d_o, sigma_y, zeta, psi_0, psi_1, psi_2, sigma_g, v_tech2, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a, xi_g, xi_p = args

    ii, ee, xx, g_tech, g_damage, pi_c, v0, v_i, V_post_tech = controls
    # v = vf
    
    # args = (delta, alpha, kappa, mu_k, sigma_k, gamma_1, gamma_2, theta_ell_temp, psi_2_temp, pi_c_o_temp, sigma_y,  theta, vartheta_bar, lambda_bar)
    # controls=(i,e,x, g_tech, g_damage, pi_c, v0, v_i, v_tech2)
    # model_args =(delta, alpha, theta, vartheta_bar, lambda_bar, mu_k, kappa, sigma_k, theta_ell, pi_c_o, pi_c, sigma_y, zeta, psi_0, psi_1, psi_2, sigma_g, v_tech2, gamma_1, gamma_2, gamma_3_list, y_bar, xi_a, xi_g, xi_p)

    n_climate = len(pi_c)

    method = 'linear'
    years  = np.arange(T0, T0 + T + dt, dt)
    pers   = len(years)
       

    # setting up grids
    stateSpace = np.hstack([
        K_mat.reshape(-1,1,order = "F"), 
        Y_mat.reshape(-1,1,order = "F"),
        L_mat.reshape(-1,1,order = "F"),
    ])

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


    # tol=1e-8
    # epsilon=0.1
    # fraction=0.5
    # max_iter=10000
    
    while FC_Err > tol and epoch < max_iter:
        
        
        
        dVdK  = finiteDiff_3D(v0,0,1,hK)
        dVdK[dVdK <= 1e-16] = 1e-16
        dK = dVdK
        dVdY  = finiteDiff_3D(v0,1,1,hY)
        dY = dVdY
        dVdL  = finiteDiff_3D(v0,2,1,hL)
        dVdL[dVdL <= 1e-16] = 1e-16
        dL = dVdL
        ######## second order
        ddVdK = finiteDiff_3D(v0,0,2,hK)
        ddVdY = finiteDiff_3D(v0,1,2,hY)
        ddY = ddVdY
        ddVdL = finiteDiff_3D(v0,2,2,hL)
        
        dG  = gamma_1 + gamma_2 * Y_mat
        G = dY -  dG
        
        
        g_damage = np.exp(- (v_i-v0)/xi_p)


        A   = - delta * np.ones(K_mat.shape) - np.exp(  L_mat - np.log(448) ) * g_tech
        B_1 = mu_k + ii - 0.5 * kappa * ii**2 - 0.5 * sigma_k**2
        B_2 = np.sum(theta_ell * pi_c, axis=0) * ee
        # B_3 = - zeta + psi_0 * (xx * np.exp(K_mat - L_mat))**psi_1 - 0.5 * sigma_g**2
        B_3 = - zeta + psi_0 * xx** psi_1 * np.exp( psi_1 * K_mat ) * np.sum(pi_c * np.exp( -( 1-psi_2) * L_mat  ), axis=0 )- 0.5 * sigma_g**2

        C_1 = 0.5 * sigma_k**2 * np.ones(K_mat.shape)
        C_2 = 0.5 * sigma_y**2 * ee**2
        C_3 = 0.5 * sigma_g**2 * np.ones(K_mat.shape)
        D = delta * np.log(consumption) + delta * K_mat  - dG * np.sum(theta_ell * pi_c, axis=0) * ee  - 0.5 * ddG * sigma_y**2 * ee**2  + xi_a * entropy + xi_g * np.exp((L_mat - np.log(448))) * (1 - g_tech + g_tech * np.log(g_tech)) + np.exp( (L_mat - np.log(448)) ) * g_tech * V_post_tech
        D -= xi_p * Intensity * (np.sum(pi_d_o * np.exp(- v_i / xi_p), axis=0) - np.exp(- v0 / xi_p)) / np.exp(- v0 / xi_p)

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
        print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
        print("PETSc preconditioned residual norm is {:g}; iterations: {}".format(ksp.getResidualNorm(), ksp.getIterationNumber()))
        PDE_rhs = A * v0 + B_1 * dVdK + B_2 * dVdY + B_3 * dVdL + C_1 * ddVdK + C_2 * ddVdY + C_3 * ddVdL + D
        PDE_Err = np.max(abs(PDE_rhs))
        FC_Err = np.max(abs((out_comp - v0)/ epsilon))
        
        if FC_Err < 10*tol:
            
            print("-----------------------------------")
            print("---------Epoch {}---------------".format(epoch))
            print("-----------------------------------")
            print("min i: {},\t max i: {}\t".format(ii.min(), ii.max()))
            print("min e: {},\t max e: {}\t".format(ee.min(), ee.max()))
            print("min x: {},\t max x: {}\t".format(xx.min(), xx.max()))
            print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
            print("Epoch {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))
            print("Epoch time: {:.4f}".format(time.time() - start_ep))
        elif epoch%100==0:
            
            print("-----------------------------------")
            print("---------Epoch {}---------------".format(epoch))
            print("-----------------------------------")
            print("min i: {},\t max i: {}\t".format(ii.min(), ii.max()))
            print("min e: {},\t max e: {}\t".format(ee.min(), ee.max()))
            print("min x: {},\t max x: {}\t".format(xx.min(), xx.max()))
            print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
            print("Epoch {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))
            print("Epoch time: {:.4f}".format(time.time() - start_ep))
          
          
        v0     = out_comp
        # pi_c
        epoch += 1
        
    dVdY = finiteDiff_3D(v0, 1, 1, hY)
    ddVdY = finiteDiff_3D(v0, 1, 2, hY)
    ME = - dVdY * np.sum(pi_c_o * theta_ell, axis=0) - 1/2* ddVdY * sigma_y**2 * ee + dG * np.sum(theta_ell * pi_c_o, axis=0) + 1/2 * ddG * sigma_y**2 * ee

    
    return ME

