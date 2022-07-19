#  Optimization of pre technology jump HJB
#Required packages
import os
import sys
import csv
sys.path.append('../src')
from supportfunctions import *
sys.stdout.flush()
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
# import petsclinearsystem
from scipy.sparse import spdiags
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from datetime import datetime

from numpy.linalg import matrix_rank, norm, cond

reporterror = True
# Linear solver choices
# Chosse among petsc, petsc4py, eigen, both
# petsc: matrix assembled in C
# petsc4py: matrix assembled in Python
# eigen: matrix assembled in C++
# both: petsc+petsc4py
#
linearsolver = 'petsc4py'
current_time = datetime.now()


print("Script starts: {:d}/{:d}-{:d}:{:d}".format(current_time.month, current_time.day, current_time.hour, current_time.minute))
print("Linear solver: " + linearsolver)


start_time = time.time()
# Parameters as defined in the paper
with open("./data/PostJump/res-post", "rb") as f:
    postjump = pickle.load(f)

delta = postjump["delta"]
A_d = postjump["A_d"]
A_g_prime = postjump["A_g"]
A_g = 0.1000

alpha_d = postjump["alpha_d"]
alpha_g = postjump["alpha_g"]
sigma_d = postjump["sigma_d"]
sigma_g = postjump["sigma_g"]

varsigma = postjump["varsigma"]
beta_f = postjump["beta_f"]
phi_d = postjump["phi_d"]
phi_g = postjump["phi_g"]
########## arrival rate
# varphi = postjump["varphi"]
varphi  = 0.010
sigma_l = 0.016
alpha_l = 0.000
########## Scaling factor
eta = postjump["eta"]

# Grids Specification

K_min = postjump["K_min"]
K_max = postjump["K_max"]
hK = postjump["hK"]
K = postjump["K"]
R_min = postjump["R_min"]
R_max = postjump["R_max"]
hR = postjump["hR"]
R = postjump["R"]
Temp_min = postjump["Y_min"]
Temp_max = postjump["Y_max"]
hTemp = postjump["hY"]
Temp = postjump["Y"]
v_post = postjump["v0"]

# jump intensity
logL_min = - 4.
logL_max = 0.
hL = 0.5
logL = np.arange(logL_min, logL_max, hL)

X = K[10:]
Y = R[:-40]
Z = Temp[:]
W = logL

###### damage
gamma_1 = postjump["gamma_1"]
gamma_2 = postjump["gamma_2"]
gamma_3 = postjump["gamma_3"]

filename =  "varphi-" + str(varphi) +"-gamma-" + str(gamma_3)  + '-' + "Agp" + "-" + str(A_g_prime) + "-{:d}-{:d}-{:d}".format(current_time.day, current_time.hour, current_time.minute)

hX = X[1] - X[0]
nX = len(X)
X_min = X.min()
X_max = X.max()
hY = Y[1] - Y[0]
nY = len(Y)
Y_min = Y.min()
Y_max = Y.max()
hZ = Z[1] - Z[0]
nZ = len(Z)
Z_min = Z.min()
Z_max = Z.max()
hW = W[1] - W[0]
nW = len(W)
W_min = W.min()
W_max = W.max()

print("Grid dimension: [{}, {}, {}, {}]".format(nX, nY, nZ, nW))
print("difference: [{}, {}, {}, {}]".format(hX, hY, hZ, hW))
print(Z.min(), Z.max())
# Discretization of the state space for numerical PDE solution.
######## post jump, 3 states
(X_mat, Y_mat, Z_mat, W_mat) = np.meshgrid(X, Y, Z, W,  indexing = 'ij')
stateSpace = np.hstack([X_mat.reshape(-1,1,order = 'F'), Y_mat.reshape(-1,1,order = 'F'), Z_mat.reshape(-1, 1, order='F'), W_mat.reshape(-1, 1, order="F")])

K_mat    = X_mat
R_mat    = Y_mat
Temp_mat = Z_mat
logL_mat = W_mat

print(X_mat.shape)
# For PETSc
X_1d = X_mat.ravel(order='F')
Y_1d = Y_mat.ravel(order='F')
Z_1d = Z_mat.ravel(order='F')
W_1d = W_mat.ravel(order="F")

lowerLims = np.array([X_min, Y_min, Z_min, W_min], dtype=np.float64)
upperLims = np.array([X_max, Y_max, Z_max, W_max], dtype=np.float64)

v0 = np.zeros(X_mat.shape)
for i in range(len(W)):
    v0[:,:,:, i] = v_post[10:, :-40, :]
V_post = v0


with open("./data/PreJump/varphi-0.01-gamma-0.0-Agp-0.15-24-11-10", "rb") as f:
    data = pickle.load(f)
v0 = data["v0"]
# v0 = V_post
continue_mode = True

csvfile = open("test3-logL.csv", "w")
fieldnames = [
        "epoch", "iterations", "residual norm", "PDE_Err", "FC_Err",
        "id_min", "id_max",
        "ig_min", "ig_max",
        "consum_min", "consum_max",
        "mc_min", "mc_max",
]

writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
writer.writeheader()
############# step up of optimization

FC_Err   = 1
epoch    = 0
tol      = 1e-6
epsilon  = 0.003
fraction = 0.5
max_iter = 10000
# file_iter = open("iter_c_compile.txt", "w")
while  FC_Err > tol and epoch < max_iter:
    print("-----------------------------------")
    print("---------Epoch {}---------------".format(epoch))
    print("-----------------------------------")
    start_ep = time.time()
    vold = v0.copy()
    # Applying finite difference scheme to the value function
    ######## first order
    dX = finiteDiff(v0,0,1,hX)
    # dX[dX < 0.5] = 0.5
    dK = dX
    dY = finiteDiff(v0,1,1,hY)
    # dY[dY < 0.5] = 0.5
    dR = dY
    dZ = finiteDiff(v0,2,1,hZ)
    # dY[dY > -  1e-15] = -1e-15
    dTemp = dZ
    dW = finiteDiff(v0,3,1,hW)
    dW[dW <= 1e-5] = 1e-5
    dLam = dW
    ######## second order
    ddX = finiteDiff(v0,0,2,hX)
    ddY = finiteDiff(v0,1,2,hY)
    ddZ = finiteDiff(v0,2,2,hZ)
    ddW = finiteDiff(v0,3,2,hW)

    # if epoch > 100:
        # epsilon = 0.01
    # if epoch > 2000:
        # epsilon = 0.001
    # elif epoch > 1000:
        # epsilon = 0.1
    # else:
        # pass

    # if FC_Err < 0.0031:
        # epsilon = 0.0001

    # update control
    if epoch == 0:
        # i_d = np.zeros(Kd_mat.shape)
        # i_g = np.zeros(Kg_mat.shape)
        consumption =  A_d * (1 - R_mat ) + A_g * R_mat
        mc = delta / consumption
        mc_min = mc.min()
        mc_max = mc.max()
        i_d = 1 / phi_d - mc / phi_d / (dK - R_mat * dR)
        i_d_min = i_d.min()
        i_d_max = i_d.max()
        i_g = 1 / phi_g - mc / phi_g / (dK + (1 - R_mat) * dR)
        i_g_min = i_g.min()
        i_g_max = i_g.max()
        # i_l = (A_d - i_d ) * (1 - R_mat) + (A_g - i_g) * R_mat - delta / (np.exp(K_mat) * varphi * dLam)
        # i_l = 1e-15 * np.ones(X_mat.shape)
        i_l = np.zeros(X_mat.shape)
        i_l_min = i_l.min()
        i_l_max = i_l.max()
        q = delta  * ( (A_d - i_d) * (1 - R_mat) + (A_g - i_g)  * R_mat - i_l  ) ** (-1)


        if continue_mode:
            i_d = data["i_d"]
            i_g = data["i_g"]
            i_l = data["i_l"]

    else:
        # pass
        # i_d = (1 / 8 - np.sqrt(0.678) / 8) * np.ones(K_mat.shape) + 0.00001
        # i_g = (1 / 8 - np.sqrt(0.678) / 8) * np.ones(K_mat.shape) + 0.00001
        mc = np.exp(K_mat-logL_mat) * varphi * dLam
        mc_min = mc.min()
        mc_max = mc.max()

        i_d = 1 / phi_d - varphi * dLam * np.exp(K_mat)  /np.exp(logL_mat) / phi_d / (dK - R_mat * dR)
        i_d = i_d * fraction + id_star * (1. - fraction)
        i_d_min = i_d.min()
        i_d_max = i_d.max()
        # i_d[i_d > A_d - 1e-16 ] = A_d - 1e-16
        # i_d[i_d <  0.00] =  0.000
        i_g = 1 / phi_g - varphi * dLam  * np.exp(K_mat) /np.exp(logL_mat)/ phi_g / (dK + (1 - R_mat) * dR)
        i_g = i_g * fraction + ig_star * (1. - fraction)
        i_g_min = i_g.min()
        i_g_max = i_g.max()
        # i_g[i_g > A_g  - 1e-16] = A_g - 1e-16
        # i_g[i_g <  0.000] = 0.0
        temp = (A_d - i_d ) * (1 - R_mat) + (A_g - i_g) * R_mat  - delta / (np.exp(K_mat - logL_mat) * varphi * dLam)
        # temp[ temp < 0.000 ] = 0.000
        # temp[ temp > A_g - 1e-15] = A_g - 1e-15
        i_l =  temp
        # i_l[i_l > 1.  - 1e-16] = 1. - 1e-16
        # i_l[i_l < 0.0] = 0.000000000
        # i_l = temp * fraction + il_star * (1. - fraction)
        i_l_min = i_l.min()
        i_l_max = i_l.max()
        # i_l[i_l > A_d - 1e-16] = A_d - 1e-16
        # i_l[i_l < 0.0 ] = 0.0
        # i_l = np.zeros(X_mat.shape)
        # i_l = 1e-5 * np.ones(X_mat.shape)
        print("min il/K: {:.10f}, max il/K: {:.10f}".format(np.min(i_l), np.max(i_l)))

        ## updating using quardratic equation
        # a = 1 / phi_d - 1 / phi_d * (dK + (1-R_mat)*dR) / (dK - R_mat * dR)
        # print(np.min(a), np.max(a))
        # b = phi_g / phi_d * (dK + (1-R_mat)*dR) / (dK - R_mat * dR)
        # print(np.min(b), np.max(b))
        # AA = ((1 - R_mat) * b + R_mat) * phi_g
        # BB = - phi_g * ((A_d - a) * (1 - R_mat) + A_g * R_mat) - ((1 - R_mat) * b + R_mat)
        # CC = ((A_d - a) * (1 - R_mat) + A_g * R_mat) - delta / (dK + (1 - R_mat) * dR)
        # DELTA = BB**2  - 4 * AA * CC
        # DELTA[DELTA <= 0] = 0.0
        # i_g = (- BB -  np.sqrt(DELTA) ) / (2 * AA)
        # print(np.min(i_g), np.max(i_g))
        # i_g[i_g > A_g] = A_g
        # i_g[i_g < 0] = 0.0
        # i_d = a + b * i_g
        # i_d[i_d > A_d] = A_d
        # i_d[i_d < 0] = 0.0


     # # updating controls
        # Converged = 0
        # num = 0

        # while Converged == 0 and num < 2000:
            # i_d_1 = (1 - q / (dK - R_mat * dR) ) / phi_d
            # i_d_1[ i_d_1 > A_d - 1e-15 ] = A_d - 1e-15
            # # i_d_1[i_d_1 < 1e-15] = 1e-15
            # i_g_1 = (1 - q / (dK + (1 - R_mat) * dR) ) / phi_g
            # i_g_1[ i_g_1 > A_g - 1e-15 ] = A_g  - 1e-15
            # # i_g_1[i_g_1 < 1e-15] = 1e-15
            # if np.max(abs(i_d_1 - i_d)) <= 1e-8 and np.max(abs(i_g_1 - i_g)) <= 1e-8:
                # Converged = 1
                # err1 = np.max(abs(i_d_1 - i_d))
                # err2 = np.max(abs(i_g_1 - i_g))
                # i_d = i_d_1
                # i_g = i_g_1
            # else:
                # err1 = np.max(abs(i_d_1 - i_d))
                # err2 = np.max(abs(i_g_1 - i_g))
                # i_d = i_d_1
                # i_g = i_g_1

                # temp =  (A_d - i_d) * (1 - R_mat) + (A_g - i_g) * R_mat - i_l * np.exp(logL_mat)
                # temp[temp < 1e-15] = 1e-15
                # q = delta / temp * fraction + q * (1 - fraction)
            # num += 1

        # print(err1, err2)
    print("mc min: {:.10f}, mc max: {:.10f}".format(mc_min, mc_max))
    print("min id: {},\t min ig: {},\t min il: {}".format(np.min(i_d), np.min(i_g), np.min(i_l)))
    print("max id: {},\t max ig: {},\t max il: {}".format(np.max(i_d), np.max(i_g), np.max(i_l)))
    # i_d[ i_d > 1/phi_d - 1e-15 ] = 1 / phi_d - 1e-15
    # i_d[i_d > A_d - 1e-15] = A_d - 1e-15
    # i_d[i_d < 1e-15] = 1e-15
    # i_g[i_g > A_g - 1e-15] = A_g - 1e-15
    # i_g[i_g < 1e-15] = 1e-15
    # i_g[ i_g > 1 / phi_g ] = 1 / phi_g - 1e-15
    # i_l[i_l > 1 - 1e-15] = 1 - 1e-15
    # i_l[i_l < 1e-16] = 1e-16
    # i_l[i_l > 700.  - 1e-16] = 700. - 1e-16
    # i_l[i_l < 0.] = 0.000000000


    # Step (2), solve minimization problem in HJB and calculate drift distortion
    # See remark 2.1.3 for more details
    start_time2 = time.time()
    if epoch == 0:
        dVec = np.array([hX, hY, hZ, hW])
        increVec = np.array([1, nX, nX * nY, nX * nY * nZ], dtype=np.int32)
        # These are constant
        A = - delta   * np.ones(K_mat.shape)  - np.exp(logL_mat)
        C_11 = 0.5 * (sigma_d * (1 - R_mat) + sigma_g * R_mat)**2
        C_22 = 0.5 * R_mat**2 * (1 - R_mat)**2 * (sigma_d + sigma_g)**2
        C_33 = 0.5 * (varsigma * eta * A_d * np.exp(K_mat) * (1 - R_mat) )**2
        C_44 = 0.5 *  sigma_l**2 * np.ones(X_mat.shape)
        B_3 = beta_f * eta * A_d * np.exp(K_mat) * (1 - R_mat)

    if linearsolver == 'petsc4py' or linearsolver == 'petsc' or linearsolver == 'both':
        petsc_mat = PETSc.Mat().create()
        petsc_mat.setType('aij')
        petsc_mat.setSizes([nX * nY * nZ * nW, nX * nY * nZ * nW])
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
        C_44_1d = C_44.ravel(order = 'F')

        if linearsolver == 'petsc4py':
            I_LB_1 = (stateSpace[:,0] == X_min)
            I_UB_1 = (stateSpace[:,0] == X_max)
            I_LB_2 = (stateSpace[:,1] == Y_min)
            I_UB_2 = (stateSpace[:,1] == Y_max)
            I_LB_3 = (stateSpace[:,2] == Z_min)
            I_UB_3 = (stateSpace[:,2] == Z_max)
            I_LB_4 = (stateSpace[:,3] == W_min)
            I_UB_4 = (stateSpace[:,3] == W_max)
            diag_0_base = A_1d[:]
            diag_0_base += (I_LB_1 * C_11_1d[:] + I_UB_1 * C_11_1d[:] - 2 * (1 - I_LB_1 - I_UB_1) * C_11_1d[:]) / dVec[0] ** 2
            diag_0_base += (I_LB_2 * C_22_1d[:] + I_UB_2 * C_22_1d[:] - 2 * (1 - I_LB_2 - I_UB_2) * C_22_1d[:]) / dVec[1] ** 2
            diag_0_base += (I_LB_3 * C_33_1d[:] + I_UB_3 * C_33_1d[:] - 2 * (1 - I_LB_3 - I_UB_3) * C_33_1d[:]) / dVec[2] ** 2
            diag_0_base += (I_LB_4 * C_44_1d[:] + I_UB_4 * C_44_1d[:] - 2 * (1 - I_LB_4 - I_UB_4) * C_44_1d[:]) / dVec[3] ** 2

            diag_1_base = - 2 * I_LB_1 * C_11_1d[:] / dVec[0] ** 2 + (1 - I_LB_1 - I_UB_1) * C_11_1d[:] / dVec[0] ** 2
            diag_1m_base = - 2 * I_UB_1 * C_11_1d[:] / dVec[0] ** 2 + (1 - I_LB_1 - I_UB_1) * C_11_1d[:] / dVec[0] ** 2
            diag_2_base = - 2 * I_LB_2 * C_22_1d[:] / dVec[1] ** 2 + (1 - I_LB_2 - I_UB_2) * C_22_1d[:] / dVec[1] ** 2
            diag_2m_base = - 2 * I_UB_2 * C_22_1d[:] / dVec[1] ** 2 + (1 - I_LB_2 - I_UB_2) * C_22_1d[:] / dVec[1] ** 2
            diag_3_base = - 2 * I_LB_3 * C_33_1d[:] / dVec[2] ** 2 + (1 - I_LB_3 - I_UB_3) * C_33_1d[:] / dVec[2] ** 2
            diag_3m_base = - 2 * I_UB_3 * C_33_1d[:] / dVec[2] ** 2 + (1 - I_LB_3 - I_UB_3) * C_33_1d[:] / dVec[2] ** 2
            diag_4_base = - 2 * I_LB_4 * C_44_1d[:] / dVec[3] ** 2 + (1 - I_LB_4 - I_UB_4) * C_44_1d[:] / dVec[3] ** 2
            diag_4m_base = - 2 * I_UB_4 * C_44_1d[:] / dVec[3] ** 2 + (1 - I_LB_4 - I_UB_4) * C_44_1d[:] / dVec[3] ** 2
            diag_11 = I_LB_1 * C_11_1d[:] / dVec[0] ** 2
            diag_11m = I_UB_1 * C_11_1d[:] / dVec[0] ** 2
            diag_22 = I_LB_2 * C_22_1d[:] / dVec[1] ** 2
            diag_22m = I_UB_2 * C_22_1d[:] / dVec[1] ** 2
            diag_33 = I_LB_3 * C_33_1d[:] / dVec[2] ** 2
            diag_33m = I_UB_3 * C_33_1d[:] / dVec[2] ** 2
            diag_44 = I_LB_4 * C_44_1d[:] / dVec[3] ** 2
            diag_44m = I_UB_4 * C_44_1d[:] / dVec[3] ** 2


    # Step (6) and (7) Formulating HJB False Transient parameters
    # See remark 2.1.4 for more details
    mu_d = alpha_d + i_d - 0.5 * phi_d * i_d**2
    mu_g = alpha_g + i_g - 0.5 * phi_g * i_g**2
    B_1 = mu_d * (1 - R_mat) + mu_g * R_mat - C_11
    B_2 = (mu_g - mu_d - R_mat * sigma_g**2 + (1 - R_mat) * sigma_d**2) * R_mat * (1 - R_mat)
    temp2 =  (A_d - i_d ) * (1 - R_mat)  + (A_g - i_g) * R_mat
    B_4 =  varphi * temp2 * np.exp(K_mat - logL_mat) - alpha_l - 0.5 * sigma_l**2
    # B_4[B_4 <=0] = 0.
    # B_4 = ( (A_d - i_d) * (1 - R_mat)  +  (A_g - i_g) * R_mat ) * varphi * np.exp(K_mat)  - alpha_l * Lambda_mat

    # B_4  = 0.001 * np.ones(X_mat.shape)
    # B_4 = np.zeros(X_mat.shape)
    consumption = (A_d - i_d) * (1 - R_mat)  +  (A_g - i_g) * R_mat  - i_l
    consumption_min = consumption.min()
    consumption_max = consumption.max()
    print("max consum: {},\t min consum: {}\t".format(np.max(consumption), np.min(consumption)))
    consumption[consumption <= 0] = 1e-300

    D = delta * np.log(consumption) + delta * K_mat - delta - (gamma_1 + gamma_2 * Temp_mat )* beta_f * eta * A_d * np.exp(K_mat) * (1 - R_mat)  - 0.5 * (gamma_2) * (varsigma * eta * A_d * np.exp(K_mat) * (1 - R_mat) )**2 + np.exp(logL_mat) * V_post

    if linearsolver == 'eigen' or linearsolver == 'both':
        start_eigen = time.time()
        out_eigen = PDESolver(stateSpace, A, B_1, B_2, B_3, C_11, C_22, C_33, D, v0, epsilon, tol=-9, solverType = 'False Transient')
        out_comp = out_eigen[2].reshape(v0.shape,order = "F")
        print("Eigen solver: {:3f}s".format(time.time() - start_eigen))
        if epoch % 1 == 0 and reporterror:
            v = np.array(out_eigen[2])
            res = np.linalg.norm(out_eigen[3].dot(v) - out_eigen[4])
            print("Eigen residual norm: {:g}; iterations: {}".format(res, out_eigen[0]))
            PDE_rhs = A * v0 + B_1 * dX + B_2 * dY + B_3 * dY + C_11 * ddX + C_22 * ddY + C_33 * ddY + D
            PDE_Err = np.max(abs(PDE_rhs))
            FC_Err = np.max(abs((out_comp - v0)/ epsilon))
            print("Episode {:d} (Eigen): PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch, PDE_Err, FC_Err))

    if linearsolver == 'petsc4py':
        bpoint1 = time.time()
        # ==== original impl ====
        B_1_1d = B_1.ravel(order = 'F')
        B_2_1d = B_2.ravel(order = 'F')
        B_3_1d = B_3.ravel(order = 'F')
        B_4_1d = B_4.ravel(order = 'F')
        D_1d = D.ravel(order = 'F')
        v0_1d = v0.ravel(order = 'F')
        # profiling
        # bpoint2 = time.time()
        # print("reshape: {:.3f}s".format(bpoint2 - bpoint1))
        diag_0 = diag_0_base - 1 / epsilon + I_LB_1 * B_1_1d[:] / -dVec[0] + I_UB_1 * B_1_1d[:] / dVec[0] - (1 - I_LB_1 - I_UB_1) * np.abs(B_1_1d[:]) / dVec[0] + I_LB_2 * B_2_1d[:] / -dVec[1] + I_UB_2 * B_2_1d[:] / dVec[1] - (1 - I_LB_2 - I_UB_2) * np.abs(B_2_1d[:]) / dVec[1] + I_LB_3 * B_3_1d[:] / -dVec[2] + I_UB_3 * B_3_1d[:] / dVec[2] - (1 - I_LB_3 - I_UB_3) * np.abs(B_3_1d[:]) / dVec[2]+ I_LB_4 * B_4_1d[:] / -dVec[3] + I_UB_4 * B_4_1d[:] / dVec[3] - (1 - I_LB_4 - I_UB_4) * np.abs(B_4_1d[:]) / dVec[3]
        diag_1 = I_LB_1 * B_1_1d[:] / dVec[0] + (1 - I_LB_1- I_UB_1) * B_1_1d.clip(min=0.0) / dVec[0] + diag_1_base
        diag_1m = I_UB_1 * B_1_1d[:] / -dVec[0] - (1 - I_LB_1 - I_UB_1) * B_1_1d.clip(max=0.0) / dVec[0] + diag_1m_base
        diag_2 = I_LB_2 * B_2_1d[:] / dVec[1] + (1 - I_LB_2 - I_UB_2) * B_2_1d.clip(min=0.0) / dVec[1] + diag_2_base
        diag_2m = I_UB_2 * B_2_1d[:] / -dVec[1] - (1 - I_LB_2 - I_UB_2) * B_2_1d.clip(max=0.0) / dVec[1] + diag_2m_base
        diag_3 = I_LB_3 * B_3_1d[:] / dVec[2] + (1 - I_LB_3 - I_UB_3) * B_3_1d.clip(min=0.0) / dVec[2] + diag_3_base
        diag_3m = I_UB_3 * B_3_1d[:] / -dVec[2] - (1 - I_LB_3 - I_UB_3) * B_3_1d.clip(max=0.0) / dVec[2] + diag_3m_base
        diag_4 = I_LB_4 * B_4_1d[:] / dVec[3] + (1 - I_LB_4 - I_UB_4) * B_4_1d.clip(min=0.0) / dVec[3] + diag_4_base
        diag_4m = I_UB_4 * B_4_1d[:] / -dVec[3] - (1 - I_LB_4 - I_UB_4) * B_4_1d.clip(max=0.0) / dVec[3] + diag_4m_base
        # profiling
        # bpoint3 = time.time()
        # print("prepare: {:.3f}s".format(bpoint3 - bpoint2))

        data = [diag_0, diag_1, diag_1m, diag_11, diag_11m, diag_2, diag_2m, diag_22, diag_22m, diag_3, diag_3m, diag_33, diag_33m, diag_4, diag_4m, diag_44, diag_44m]
        diags = np.array([0,-increVec[0],increVec[0],-2*increVec[0],2*increVec[0],
                        -increVec[1],increVec[1],-2*increVec[1],2*increVec[1],
                        -increVec[2],increVec[2],-2*increVec[2],2*increVec[2],
                        -increVec[3],increVec[3],-2*increVec[3],2*increVec[3]])
        # The transpose of matrix A_sp is the desired. Create the csc matrix so that it can be used directly as the transpose of the corresponding csr matrix.
        A_sp = spdiags(data, diags, len(diag_0), len(diag_0), format='csc')
        A_sp = A_sp * epsilon
        # A_dense = A_sp.todense()
        # b = -v0_1d/epsilon - D_1d
        b = -v0_1d - D_1d * epsilon
        # A_rank = matrix_rank(A_dense)
        # b_rank = matrix_rank(b)
        # A_norm = norm(A_dense, ord=2)
        # b_norm = norm(b)
        # A_cond = cond(A_dense)
        # b_cond = cond(b)

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
        ksp.setTolerances(rtol=1e-9)
        ksp.solve(petsc_rhs, x)
        petsc_mat.destroy()
        petsc_rhs.destroy()
        x.destroy()
        out_comp = np.array(ksp.getSolution()).reshape(X_mat.shape,order = "F")
        end_ksp = time.time()
        # print("ksp solve: {:.3f}s".format(end_ksp - start_ksp))
        print("petsc4py total: {:.3f}s".format(end_ksp - bpoint1))
        print("PETSc preconditioned residual norm is {:g}; iterations: {}".format(ksp.getResidualNorm(), ksp.getIterationNumber()))
        if epoch % 1 == 0 and reporterror:
            # Calculating PDE error and False Transient error
            PDE_rhs = A * v0 + B_1 * dX + B_2 * dY + B_3 * dZ  + B_4 * dW + C_11 * ddX + C_22 * ddY + C_33 * ddZ + C_44 * ddW +  D
            PDE_Err = np.max(abs(PDE_rhs))
            FC_Err = np.max(abs((out_comp - v0)/epsilon))
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
        # rowcontent = {
            # "epoch": epoch,
            # "iterations": num_iter,
            # "residual norm": ksp.getResidualNorm(),
            # "PDE_Err": PDE_Err,
            # "FC_Err": FC_Err,
            # # "A_rank": A_rank,
            # "A_norm": A_norm,
            # "A_cond": A_cond,
            # "b_rank": b_rank,
            # "b_norm": b_norm,
            # "b_cond": b_cond
        # }
        # writer.writerow(rowcontent)
        rowcontent = {
            "epoch": epoch,
            "iterations": ksp.getIterationNumber(),
            "residual norm": ksp.getResidualNorm(),
            "PDE_Err": PDE_Err,
            "FC_Err": FC_Err,
            "id_min": i_d_min,
            "id_max": i_d_max,
            "ig_min": i_g_min,
            "ig_max": i_g_max,
            "consum_min": consumption_min,
            "consum_max": consumption_max,
            "mc_min": mc_min,
            "mc_max": mc_max,
        }
        writer.writerow(rowcontent)

    if linearsolver == 'petsc' or linearsolver == 'both':
        bpoint1 = time.time()
        B_1_1d = B_1.ravel(order = 'F')
        B_2_1d = B_2.ravel(order = 'F')
        B_3_1d = B_3.ravel(order = 'F')
        B_4_1d = B_4.ravel(order = 'F')
        D_1d = D.ravel(order = 'F')
        v0_1d = v0.ravel(order = 'F')
        petsclinearsystem.formLinearSystem(X_1d, Y_1d, Z_1d, W_1d, A_1d, B_1_1d, B_2_1d, B_3_1d, B_4_1d, C_11_1d, C_22_1d, C_33_1d, C_44_1d, epsilon, lowerLims, upperLims, dVec, increVec, petsc_mat)
        # profiling
        # bpoint2 = time.time()
        # print("form petsc mat: {:.3f}s".format(bpoint2 - bpoint1))
        b = v0_1d + D_1d*epsilon
        # petsc4py setting
        # petsc_mat.scale(-1./ε)
        # b = -v0_1d/ε - D_1d
        petsc_rhs = PETSc.Vec().createWithArray(b)
        print(matrix_rank(petsc_rhs))
        # print(matrix_rank(petsc_mat))
        x = petsc_mat.createVecRight()
        # profiling
        # bpoint3 = time.time()
        # print("form rhs and workvector: {:.3f}s".format(bpoint3 - bpoint2))
        # x.set(0)
        # viewer = PETSc.Viewer().createBinary('A.dat', 'w')
        # petsc_mat.view(viewer)
        # viewer = PETSc.Viewer().createBinary('TCRE_MacDougallEtAl2017_b.dat', 'w')
        # petsc_rhs.view(viewer)
        # ai, aj, av = petsc_mat.getValuesCSR()
        # x.set(0)
        # viewer = PETSc.Viewer().createBinary('A.dat', 'w')
        # petsc_mat.view(viewer)
        # viewer = PETSc.Viewer().createBinary('TCRE_MacDougallEtAl2017_b.dat', 'w')
        # petsc_rhs.view(viewer)
        # ai, aj, av = petsc_mat.getValuesCSR()
        # print(type(x))
        print(type(petsc_mat))
        print(type(petsc_rhs))
        # print(aj)


        # create linear solver
        start_ksp = time.time()
        ksp.setOperators(petsc_mat)
        print(petsc_mat.norm())
        ksp.setTolerances(rtol=1e-12)
        ksp.solve(petsc_rhs, x)
        # petsc_mat.destroy()
        petsc_rhs.destroy()
        x.destroy()
        out_comp = np.array(ksp.getSolution()).reshape(Kd_mat.shape,order = "F")
        end_ksp = time.time()
        # profiling
        # print("ksp solve: {:.3f}s".format(end_ksp - start_ksp))
        num_iter = ksp.getIterationNumber()
        # file_iter.write("%s \n" % num_iter)
        print("petsc total: {:.3f}s".format(end_ksp - bpoint1))
        print("PETSc preconditioned residual norm is {:g}; iterations: {}".format(ksp.getResidualNorm(), ksp.getIterationNumber()))
        if epoch % 1 == 0 and reporterror:
            # Calculating PDE error and False Transient error
            PDE_rhs = A * v0 + B_1 * dX + B_2 * dY + B_3 * dZ + B_4 * dW + C_11 * ddX + C_22 * ddY + C_33 * ddZ + C_44 * ddW + D
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
    # step 9: keep iterating until convergence
    id_star = i_d
    ig_star = i_g
    il_star = i_l
    v0 = out_comp
    epoch += 1

if reporterror:
    print("===============================================")
    print("Fianal epoch {:d}: PDE Error: {:.10f}; False Transient Error: {:.10f}" .format(epoch -1, PDE_Err, FC_Err))
print("--- Total running time: %s seconds ---" % (time.time() - start_time))



# exit()

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


file = open("./data/PreJump/" + filename, 'wb')
pickle.dump(my_shelf, file)
file.close()
