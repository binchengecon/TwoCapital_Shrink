import os
import sys
sys.path.append('./src')
from supportfunctions import finiteDiff
import numpy as np
import pandas
# import petsc4py
# import petsclinearsystem
# from petsc4py import PETSc
import time

def FOC_func(logK, R, Y, q, id_star, ig_star, dK, dR, dY, args, fraction):
    """ update control"""

    delta, eta, A_d, A_g, alpha_d, alpha_g, sigma_d, sigma_g, phi_d, phi_g, gamma_1, \
            gamma_2, y_bar, varphi, varsigma, beta_f = args

    Converged = 0
    num = 0
    
    i_d = id_star
    i_g = ig_star

    while Converged == 0 and num < 4000:
        i_g_1 = (1 - q / (dR * (1 - R) + dK )) / phi_g
        i_d_1 = (1 - q / (-dR * R + dK)) / phi_d
        i_d_1[i_d_1 >= A_d] = A_d - 1e-8
        # i_d_1[i_d_1 <= 1e-8] = 1e-8
        i_g_1[i_g_1 >= A_g] = A_g - 1e-8
        # i_g_1[i_g_1 <= 1e-8] = 1e-8

        if np.max(abs(i_g_1 - i_g)) <= 1e-8 and np.max(abs(i_d_1 - i_d)) <= 1e-8:
            Converged = 1
            i_g = i_g_1
            i_d = i_d_1
        else:
            i_g = i_g_1
            i_d = i_d_1

            q = delta * (
                (A_g * R - i_g * R) + (A_d * (1 - R) - i_d * (1 - R))) ** (-1) * fraction + (1 - fraction) * q
        num += 1 

    return i_d, i_g, q

def GetCoeff_2(logK, R, Y, args):
    """ get A C11 C22 C33"""

    delta, eta, A_d, A_g, alpha_d, alpha_g, sigma_d, sigma_g, phi_d, phi_g, gamma_1, \
            gamma_2, y_bar, varphi, varsigma, beta_f = args

    A = - delta * np.ones(logK.shape)
    C_kk = 0.5 * (( sigma_d * (1 - R) )**2 + (sigma_g * R )**2)
    C_rr = 0.5 * (1- R)**2 * R**2 * (sigma_d**2 + sigma_g**2)
    C_yy = 0.5 * (eta * varsigma * A_d * np.exp(logK) * (1 - R))** 2

    return A, C_kk, C_rr, C_yy

def GetCoeff(logK, R, Y, i_d, i_g, args):
    """ get B1, B2, B2, D"""

    delta, eta, A_d, A_g, alpha_d, alpha_g, sigma_d, sigma_g, phi_d, phi_g, gamma_1, \
            gamma_2, y_bar, varphi, varsigma, beta_f = args

    B_k = (alpha_d + i_d - 0.5* phi_d * i_d**2) * (1 - R) +  (alpha_g + i_g - 0.5 * phi_g * i_g**2) * R - 0.5 * (( sigma_d * (1 - R) )**2 + (sigma_g * R )**2)
    B_r = ((alpha_g + i_g - 0.5 * phi_g * i_g**2) -  (alpha_d + i_d - 0.5* phi_d * i_d**2)) * R * (1 - R)
    B_y = beta_f * eta * A_d * np.exp(logK) * (1 - R)

    consumption = (A_g - i_g )* R + (A_d - i_d) * (1 - R)
    consumption[consumption <= 1e-15] = 1e-15
    D = delta * np.log(consumption) + delta * logK  - (gamma_1 + gamma_2 * Y)* beta_f * eta * A_d * np.exp(logK) * (1 - R)  - 0.5 * gamma_2 * (varsigma * eta * A_d * np.exp(logK) * (1 - R) )**2

    return B_k, B_r, B_y, D

def solver_3d(X_mat, Y_mat, Z_mat, FOC_func=FOC_func, GetCoeff=GetCoeff, GetCoeff_2=GetCoeff_2,
        args=(),
        linearsolver = "petsc",
        reporterror = True,
        v0=None, tol=1e-6, max_iter=10000, epsilon=1., fraction=1.,
        saveRes=False,
        ):
    """Solve 3D HJB"""

    delta, eta, A_d, A_g, alpha_d, alpha_g, sigma_d, sigma_g, phi_d, phi_g, gamma_1, \
            gamma_2, y_bar, varphi, varsigma, beta_f = args

    if v0.all() == None:
        v0 = np.zeros(X_mat.shape)

    n1, n2, n3 = X_mat.shape
    h1 = X_mat[1,0,0] - X_mat[0,0,0]
    h2 = Y_mat[0,1,0] - Y_mat[0,0,0]
    h3 = Z_mat[0,0,1] - Z_mat[0,0,0]

    X_min = X_mat.min()
    X_max = X_mat.max()
    Y_min = Y_mat.min()
    Y_max = Y_mat.max()
    Z_min = Z_mat.min()
    Z_max = Z_mat.max()

    # For PETSc
    X_mat_1d = X_mat.ravel(order='F')
    Y_mat_1d = Y_mat.ravel(order='F')
    Z_mat_1d = Z_mat.ravel(order='F')
    lowerLims = np.array([X_min, Y_min, Z_min], dtype=np.float64)
    upperLims = np.array([X_max, Y_max, Z_max], dtype=np.float64)

    FC_Err = 1
    epoch = 0
    while FC_Err > tol and epoch < max_iter:
        print("-----------------------------------")
        print("---------Epoch {}---------------".format(epoch))
        print("-----------------------------------")
        start_ep = time.time()
        vold = v0.copy()
        # Applying finite difference scheme to the value function
        ######## first order
        d1 = finiteDiff(v0,0,1,h1)
        # d1[d1 < 1e-8] = 1e-8
        d2 = finiteDiff(v0,1,1,h2)
        # d2[d2 < 1e-8] = 1e-8
        d3 = finiteDiff(v0,2,1,h3)
        ######## second order
        dd1 = finiteDiff(v0,0,2,h1)
        dd2 = finiteDiff(v0,1,2,h2)
        dd3 = finiteDiff(v0,2,2,h3)

        # if epoch > 2000:
            # epsilon = 0.01
        # elif epoch > 1000:
            # epsilon = 0.05
        # else:
            # pass

        # update control
        if epoch == 0:
            # i_d = np.zeros(X_mat.shape)
            # i_g = np.zeros(Y_mat.shape)
            consumption_0 = A_d * (1 - Y_mat) + A_g * Y_mat
            consumption = consumption_0
            mc = delta / consumption
            i_d = 1 -  mc / (d1 - Y_mat *  d2)
            i_d /= phi_d
            i_d[i_d < 0] = 0
            # i_d[i_d > A_d] = A_d
            i_g = 1 - mc / (d1 + (1 - Y_mat) * d2)
            i_g /= phi_g
            # i_g[i_g < 0] = 0
            # i_g[i_g > A_g] = A_g
            q = delta * ((A_g * Y_mat - i_g * Y_mat) + (A_d * (1 - Y_mat) - i_d * (1 - Y_mat))) ** (-1)
        else:

            # i_d, i_g = FOC_func(X_mat, Y_mat, Z_mat, args)
            
            i_d, i_g, q = FOC_func(X_mat, Y_mat, Z_mat, q, id_star, ig_star, d1, d2, d3, args, fraction)

            # Converged = 0
            # num = 0
            
            # i_d = id_star
            # i_g = ig_star

            # while Converged == 0:
                # i_g_1 = (1 - q / (dR * (1 - R) + dK )) / phi_g
                # i_d_1 = (1 - q / (-dR * R + dK)) / phi_d
                # i_d_1[i_d_1 >= A_d] = A_d - 1e-8
                # # i_d_1[i_d_1 <= 1e-8] = 1e-8
                # i_g_1[i_g_1 >= A_g] = A_g - 1e-8
                # # i_g_1[i_g_1 <= 1e-8] = 1e-8

                # if np.max(abs(i_g_1 - i_g)) <= 1e-8 and np.max(abs(i_d_1 - i_d)) <= 1e-8:
                    # Converged = 1
                    # i_g = i_g_1
                    # i_d = i_d_1
                # else:
                    # i_g = i_g_1
                    # i_d = i_d_1

                    # q = delta * (
                        # (A_g * R - i_g * R) + (A_d * (1 - R) - i_d * (1 - R))) ** (-1) * fraction + (1 - fraction) * q
                # num += 1 
            # consumption_new = (A_d - id_star) * X_mat + (A_g - ig_star) * Y_mat
            # consumption_new[consumption_new <= 1e-8] = 1e-8
            # # consumption_new[consumption_new > consumption_0] = consumption_0[consumption_new > consumption_0]
            # mc = delta / consumption_new
            # id_new = 1 / phi_d * (1 - mc / (d1 - Y_mat * d2 )) * fraction + i_d * (1 - fraction)
            # id_new[id_new < -1] = -1
            # # id_new[id_new > A_d] = A_d- 1e-8
            # # id_new[id_new > 1/phi_d] = 1 / phi_d
            # ig_new = 1 / phi_g * (1 - mc / (d1  + (1 - Y_mat) * d2)) * fraction + i_g * (1 - fraction)
            # ig_new[ig_new <-1] = -1
            # # ig_new[ig_new > A_g] = A_g
            # # ig_new[ig_new > 1 / phi_g] = 1 / phi_g

            # # mc_new = fraction * delta / ((A_d -id_new) * (1 - Y_mat) + (A_g - ig_new) * Y_mat) + mc * (1 - fraction)
            # i_d = id_new
            # i_g = ig_new

            # nums = 0
            # converge = False
            # while not converge:

                # id_new = 1 / phi_d * (1 - mc / (d1 - Y_mat * d2 )) * fraction + i_d * (1 - fraction)
                # id_new[id_new < 0] = 0
                # id_new[id_new > A_d] = A_d- 1e-8
                # # id_new[id_new > 1/phi_d] = 1 / phi_d
                # ig_new = 1 / phi_g * (1 - mc / (d1  + (1 - Y_mat) * d2)) * fraction + i_g * (1 - fraction)
                # ig_new[ig_new < 0] = 0
                # # ig_new[ig_new > A_g] = A_g
                # ig_new[ig_new > 1 / phi_g] = 1 / phi_g

                # mc_new = fraction * delta / ((A_d -id_new) * (1 - Y_mat) + (A_g - ig_new) * Y_mat) + mc * (1 - fraction)
                # i_d = id_new
                # i_g = ig_new
                # diff = np.max(np.abs(mc - mc_new) / fraction)
                # if diff  < 1e-5 or nums > 10000:
                    # converge = True
                    # mc = mc_new
                # else:
                    # mc = mc_new
                    # pass
                # nums += 1

            # print(diff)
        print(np.min(i_d), np.min(i_g))
        # i_d = np.zeros(X_mat.shape)
        # i_g = np.zeros(Y_mat.shape)
        i_d[i_d < 0] = 0
        i_g[i_g < 0] = 0
        # consumption = (A_d -i_d) * (1 - Y_mat) + (A_g - i_g) * Y_mat
        # consumption[consumption < 1e-8] = 1e-8
        # i_d[i_d >= A_d] = A_d - 1e-8
        # i_g[i_g >= A_g] = A_g - 1e-8
        # Step (2), solve minimization problem in HJB and calculate drift distortion
        # See remark 2.1.3 for more details
        start_time2 = time.time()
        if epoch == 0:
            dVec = np.array([h1, h2, h3])
            increVec = np.array([1, n1, n1 * n2],dtype=np.int32)
            # These are constant
            A, C_11, C_22, C_33 = GetCoeff_2(X_mat, Y_mat, Z_mat, args)
            # A = - delta * np.ones(X_mat.shape)
            # C_11 = 0.5 * ( sigma_d * (1 - Y_mat) + sigma_g * Y_mat )**2
            # C_22 = np.zeros(X_mat.shape)
            # C_33 = 0.5 * (eta * varsigma * A_d * np.exp(X_mat) * (1 - Y_mat))** 2

            if linearsolver == 'petsc4py' or linearsolver == 'petsc' or linearsolver == 'both':
                petsc_mat = PETSc.Mat().create()
                petsc_mat.setType('aij')
                petsc_mat.setSizes([n1*n2*n3, n1*n2*n3])
                petsc_mat.setPreallocationNNZ(13)
                petsc_mat.setUp()
                ksp = PETSc.KSP()
                ksp.create(PETSc.COMM_WORLD)
                ksp.setType('bcgs')
                ksp.getPC().setType('ilu')
                ksp.setFromOptions()

                A_1d = A.ravel(order = 'F')
                C_11_1d = C_11.ravel(order = 'F')
                C_22_1d = C_22.ravel(order = 'F')
                C_33_1d = C_33.ravel(order = 'F')

                if linearsolver == 'petsc4py':
                    I_LB_1 = (stateSpace[:,0] == X_min)
                    I_UB_1 = (stateSpace[:,0] == X_max)
                    I_LB_2 = (stateSpace[:,1] == Y_min)
                    I_UB_2 = (stateSpace[:,1] == Y_max)
                    I_LB_3 = (stateSpace[:,2] == Z_min)
                    I_UB_3 = (stateSpace[:,2] == Z_max)
                    diag_0_base = A_1d[:]
                    diag_0_base += (I_LB_1 * C_11_1d[:] + I_UB_1 * C_11_1d[:] - 2 * (1 - I_LB_1 - I_UB_1) * C_11_1d[:]) / dVec[0] ** 2
                    diag_0_base += (I_LB_2 * C_22_1d[:] + I_UB_2 * C_22_1d[:] - 2 * (1 - I_LB_2 - I_UB_2) * C_22_1d[:]) / dVec[1] ** 2
                    diag_0_base += (I_LB_K * C_kk_1d[:] + I_UB_K * C_kk_1d[:] - 2 * (1 - I_LB_K - I_UB_K) * C_kk_1d[:]) / dVec[2] ** 2
                    diag_1_base = - 2 * I_LB_1 * C_11_1d[:] / dVec[0] ** 2 + (1 - I_LB_1 - I_UB_1) * C_11_1d[:] / dVec[0] ** 2
                    diag_1m_base = - 2 * I_UB_1 * C_11_1d[:] / dVec[0] ** 2 + (1 - I_LB_1 - I_UB_1) * C_11_1d[:] / dVec[0] ** 2
                    diag_2_base = - 2 * I_LB_2 * C_22_1d[:] / dVec[1] ** 2 + (1 - I_LB_2 - I_UB_2) * C_22_1d[:] / dVec[1] ** 2
                    diag_2m_base = - 2 * I_UB_2 * C_22_1d[:] / dVec[1] ** 2 + (1 - I_LB_2 - I_UB_2) * C_22_1d[:] / dVec[1] ** 2
                    diag_3_base = - 2 * I_LB_3 * C_33_1d[:] / dVec[2] ** 2 + (1 - I_LB_3 - I_UB_3) * C_33_1d[:] / dVec[2] ** 2
                    diag_3m_base = - 2 * I_UB_3 * C_33_1d[:] / dVec[2] ** 2 + (1 - I_LB_3 - I_UB_3) * C_33_1d[:] / dVec[2] ** 2
                    diag_11 = I_LB_1 * C_11_1d[:] / dVec[0] ** 2
                    diag_11m = I_UB_1 * C_11_1d[:] / dVec[0] ** 2
                    diag_22 = I_LB_2 * C_22_1d[:] / dVec[1] ** 2
                    diag_22m = I_UB_2 * C_22_1d[:] / dVec[1] ** 2
                    diag_33 = I_LB_3 * C_33_1d[:] / dVec[2] ** 2
                    diag_33m = I_UB_3 * C_33_1d[:] / dVec[2] ** 2


        # Step (6) and (7) Formulating HJB False Transient parameters
        # See remark 2.1.4 for more details
        B_1, B_2, B_3, D = GetCoeff(X_mat, Y_mat, Z_mat, i_d, i_g, args)

        # B_1 = (alpha_d + i_d - 0.5* phi_d * i_d**2) * (1 - Y_mat) +  (alpha_g + i_g - 0.5 * phi_g * i_g**2) * Y_mat - C_11
        # B_2 = ((alpha_g + i_g - 0.5 * phi_g * i_g**2) -  (alpha_d + i_d - 0.5* phi_d * i_d**2)) * Y_mat * (1 - Y_mat)
        # B_3 = beta_f * eta * A_d * np.exp(X_mat) * (1 - Y_mat)

        # D = delta * np.log(consumption) + delta * X_mat  - (gamma_1 + gamma_2 * Z_mat)* beta_f * eta * A_d * np.exp(X_mat) * (1 - Y_mat)  - 0.5 * gamma_2 * (varsigma * eta * A_d * np.exp(X_mat) * (1 - Y_mat) )**2

        if linearsolver == 'eigen' or linearsolver == 'both':
            start_eigen = time.time()
            out_eigen = PDESolver(stateSpace, A, B_1, B_2, B_3, C_11, C_22, C_33, D, v0, epsilon, solverType = 'False Transient')
            out_comp = out_eigen[2].reshape(v0.shape,order = "F")
            print("Eigen solver: {:3f}s".format(time.time() - start_eigen))
            if epoch % 1 == 0 and reporterror:
                v = np.array(out_eigen[2])
                res = np.linalg.norm(out_eigen[3].dot(v) - out_eigen[4])
                print("Eigen residual norm: {:g}; iterations: {}".format(res, out_eigen[0]))
                PDE_rhs = A * v0 + B_1 * d1 + B_2 * d2 + B_3 * d3 + C_11 * dd1 + C_22 * dd2 + C_33 * dd3 + D
                PDE_Err = np.max(abs(PDE_rhs))
                FC_Err = np.max(abs((out_comp - v0)))
                print("Episode {:d} (Eigen): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))

        if linearsolver == 'petsc4py':
            bpoint1 = time.time()
            # ==== original impl ====
            B_1_1d = B_1.ravel(order = 'F')
            B_2_1d = B_2.ravel(order = 'F')
            B_3_1d = B_3.ravel(order = 'F')
            D_1d = D.ravel(order = 'F')
            v0_1d = v0.ravel(order = 'F')
            # profiling
            # bpoint2 = time.time()
            # print("reshape: {:.3f}s".format(bpoint2 - bpoint1))
            diag_0 = diag_0_base - 1 / epsilon + I_LB_R * B_r_1d[:] / -dVec[0] + I_UB_R * B_r_1d[:] / dVec[0] - (1 - I_LB_R - I_UB_R) * np.abs(B_r_1d[:]) / dVec[0] + I_LB_F * B_f_1d[:] / -dVec[1] + I_UB_F * B_f_1d[:] / dVec[1] - (1 - I_LB_F - I_UB_F) * np.abs(B_f_1d[:]) / dVec[1] + I_LB_K * B_k_1d[:] / -dVec[2] + I_UB_K * B_k_1d[:] / dVec[2] - (1 - I_LB_K - I_UB_K) * np.abs(B_k_1d[:]) / dVec[2]
            diag_R = I_LB_R * B_r_1d[:] / dVec[0] + (1 - I_LB_R - I_UB_R) * B_r_1d.clip(min=0.0) / dVec[0] + diag_R_base
            diag_Rm = I_UB_R * B_r_1d[:] / -dVec[0] - (1 - I_LB_R - I_UB_R) * B_r_1d.clip(max=0.0) / dVec[0] + diag_Rm_base
            diag_F = I_LB_F * B_f_1d[:] / dVec[1] + (1 - I_LB_F - I_UB_F) * B_f_1d.clip(min=0.0) / dVec[1] + diag_F_base
            diag_Fm = I_UB_F * B_f_1d[:] / -dVec[1] - (1 - I_LB_F - I_UB_F) * B_f_1d.clip(max=0.0) / dVec[1] + diag_Fm_base
            diag_K = I_LB_K * B_k_1d[:] / dVec[2] + (1 - I_LB_K - I_UB_K) * B_k_1d.clip(min=0.0) / dVec[2] + diag_K_base
            diag_Km = I_UB_K * B_k_1d[:] / -dVec[2] - (1 - I_LB_K - I_UB_K) * B_k_1d.clip(max=0.0) / dVec[2] + diag_Km_base
            # profiling
            # bpoint3 = time.time()
            # print("prepare: {:.3f}s".format(bpoint3 - bpoint2))

            data = [diag_0, diag_R, diag_Rm, diag_RR, diag_RRm, diag_F, diag_Fm, diag_FF, diag_FFm, diag_K, diag_Km, diag_KK, diag_KKm]
            diags = np.array([0,-increVec[0],increVec[0],-2*increVec[0],2*increVec[0],
                            -increVec[1],increVec[1],-2*increVec[1],2*increVec[1],
                            -increVec[2],increVec[2],-2*increVec[2],2*increVec[2]])
            # The transpose of matrix A_sp is the desired. Create the csc matrix so that it can be used directly as the transpose of the corresponding csr matrix.
            A_sp = spdiags(data, diags, len(diag_0), len(diag_0), format='csc')
            b = -v0_1d/epsilon - D_1d
            # A_sp = spdiags(data, diags, len(diag_0), len(diag_0))
            # A_sp = csr_matrix(A_sp.T)
            # b = -v0/ε - D
            # profiling
            # bpoint4 = time.time()
            # print("create matrix and rhs: {:.3f}s".format(bpoint4 - bpoint3))
            petsc_mat = PETSc.Mat().createAIJ(size=A_sp.shape, csr=(A_sp.indptr, A_sp.indices, A_sp.data))
            petsc_rhs = PETSc.Vec().createWithArray(b)
            x = petsc_mat.createVecRight()
            # profiling
            # bpoint5 = time.time()
            # print("assemble: {:.3f}s".format(bpoint5 - bpoint4))

            # dump to files
            #x.set(0)
            #viewer = PETSc.Viewer().createBinary('TCRE_MacDougallEtAl2017_A.dat', 'w')
            #petsc_mat.view(viewer)
            #viewer = PETSc.Viewer().createBinary('TCRE_MacDougallEtAl2017_b.dat', 'w')
            #petsc_rhs.view(viewer)

            # create linear solver
            start_ksp = time.time()
            ksp.setOperators(petsc_mat)
            ksp.setTolerances(rtol=1e-12)
            ksp.solve(petsc_rhs, x)
            petsc_mat.destroy()
            petsc_rhs.destroy()
            x.destroy()
            out_comp = np.array(ksp.getSolution()).reshape(R_mat.shape,order = "F")
            end_ksp = time.time()
            # print("ksp solve: {:.3f}s".format(end_ksp - start_ksp))
            print("petsc4py total: {:.3f}s".format(end_ksp - bpoint1))
            print("PETSc preconditioned residual norm is {:g}; iterations: {}".format(ksp.getResidualNorm(), ksp.getIterationNumber()))
            if epoch % 1 == 0 and reporterror:
                # Calculating PDE error and False Transient error
                PDE_rhs = A * v0 + B_1 * d1 + B_2 * d2 + B_3 * d3 + C_11 * dd1 + C_22 * dd2 + C_33 * dd3 + D
                PDE_Err = np.max(abs(PDE_rhs))
                FC_Err = np.max(abs((out_comp - v0)))
                print("Epoch {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))
                # profling
                # bpoint7 = time.time()
                # print("compute error: {:.3f}s".format(bpoint7 - bpoint6))
            # if linearsolver == 'both':
                # compare
                # csr_mat = csr_mat*(-ε)
                # b = b*(-ε)
                # A_diff =  np.max(np.abs(out_eigen[3] - csr_mat))
                #
                # print("Coefficient matrix difference: {:.3f}".format(A_diff))
                # b_diff = np.max(np.abs(out_eigen[4] - np.squeeze(b)))
                # print("rhs difference: {:.3f}".format(b_diff))

        if linearsolver == 'petsc' or linearsolver == 'both':
            bpoint1 = time.time()
            B_1_1d = B_1.ravel(order = 'F')
            B_2_1d = B_2.ravel(order = 'F')
            B_3_1d = B_3.ravel(order = 'F')
            D_1d = D.ravel(order = 'F')
            v0_1d = v0.ravel(order = 'F')
            petsclinearsystem.formLinearSystem(X_mat_1d, Y_mat_1d, Z_mat_1d, A_1d, B_1_1d, B_2_1d, B_3_1d, C_11_1d, C_22_1d, C_33_1d, epsilon, lowerLims, upperLims, dVec, increVec, petsc_mat)
            # profiling
            # bpoint2 = time.time()
            # print("form petsc mat: {:.3f}s".format(bpoint2 - bpoint1))
            b = v0_1d + D_1d*epsilon
            # petsc4py setting
            # petsc_mat.scale(-1./ε)
            # b = -v0_1d/ε - D_1d
            petsc_rhs = PETSc.Vec().createWithArray(b)
            x = petsc_mat.createVecRight()
            # profiling
            # bpoint3 = time.time()
            # print("form rhs and workvector: {:.3f}s".format(bpoint3 - bpoint2))


            # create linear solver
            start_ksp = time.time()
            ksp.setOperators(petsc_mat)
            ksp.setTolerances(rtol=1e-12)
            ksp.solve(petsc_rhs, x)
            # petsc_mat.destroy()
            petsc_rhs.destroy()
            x.destroy()
            out_comp = np.array(ksp.getSolution()).reshape(X_mat.shape,order = "F")
            end_ksp = time.time()
            # profiling
            # print("ksp solve: {:.3f}s".format(end_ksp - start_ksp))
            num_iter = ksp.getIterationNumber()
            # file_iter.write("%s \n" % num_iter)
            print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
            print("PETSc preconditioned residual norm is {:g}; iterations: {}".format(ksp.getResidualNorm(), ksp.getIterationNumber()))
            if epoch % 1 == 0 and reporterror:
                # Calculating PDE error and False Transient error
                PDE_rhs = A * v0 + B_1 * d1 + B_2 * d2 + B_3 * d3 + C_11 * dd1 + C_22 * dd2 + C_33 * dd3 + D
                PDE_Err = np.max(abs(PDE_rhs))
                FC_Err = np.max(abs((out_comp - v0)/ epsilon))
                print("Epoch {:d} (PETSc): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))
        print("Epoch time: {:.4f}".format(time.time() - start_ep))
        # step 9: keep iterating until convergence
        # rowcontent = {
            # "epoch": epoch,
            # "iterations": num_iter,
            # "residual norm": ksp.getResidualNorm(),
            # "PDE_Err": PDE_Err,
            # "FC_Err": FC_Err
        # }
        # writer.writerow(rowcontent)
        id_star = i_d
        ig_star = i_g
        v0 = out_comp
        epoch += 1

    if saveRes:

        from datetime import datetime
        
        current_time = datetime.now()
        filename =  "res" + '-' + "{:d}-{:d}-{:d}".format(current_time.day, current_time.hour, current_time.minute)

        import pickle
        # filename = filename
        my_shelf = {}
        for key in dir():
            if isinstance(globals()[key], (int,float, float, str, bool, np.ndarray,list)):
                try:
                    my_shelf[key] = globals()[key]
                except TypeError:
                    #
                    # __builtins__, my_shelf, and imported modules can not be shelved.
                    #
                    print('ERROR shelving: {0}'.format(key))
            else:
                pass


        file = open("data/" + filename, 'wb')
        pickle.dump(my_shelf, file)
        file.close()

        return file
