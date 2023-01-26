import sys
sys.path.append("../src/")
import numpy as np
import pandas as pd
import pickle
from scipy.integrate import solve_ivp
from scipy.fft import fft, fftfreq
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import SymLogNorm
import matplotlib.mlab
import scipy.io as sio
import pandas as pd
import scipy.optimize as optim
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy import fft, arange, signal
from scipy.interpolate import RegularGridInterpolator
import SolveLinSys
from supportfunctions import finiteDiff_3D
import petsclinearsystem
from petsc4py import PETSc
import petsc4py
import argparse
import time


petsc4py.init(sys.argv)
reporterror = True

rcParams["figure.figsize"] = (8,5)
rcParams["savefig.bbox"] = 'tight'

parser = argparse.ArgumentParser(description="xi_r values")
parser.add_argument("--name",type=str,default="ReplicateSuri")
parser.add_argument("--hXarr",nargs='+',type=float)
parser.add_argument("--Xminarr",nargs='+',type=float)
parser.add_argument("--Xmaxarr",nargs='+',type=float)
parser.add_argument("--epsilon",type=float)
parser.add_argument("--fraction",type=float)
parser.add_argument("--maxiter",type=int)
parser.add_argument("--delta",type=float)

parser.add_argument("--cearth",type=float)
parser.add_argument("--tauc",type=float)


args = parser.parse_args()

name = args.name

Xminarr = args.Xminarr
Xmaxarr = args.Xmaxarr
hXarr = args.hXarr

epsilon = args.epsilon
fraction = args.fraction
maxiter = args.maxiter

cearth = args.cearth
tauc = args.tauc

k = 0


# Pre-industrial: 282.87K

sa = 1
Ts = 282.9
Cs = 275.5

Q0 = 342.5
p = 0.3
# outgoing radiation linearized
kappa = 1.74
Tkappa = 154
## CO2 radiative forcing
# Greenhouse effect parameter
B = 5.35

alphaland = 0.28
bP = 0.05
sigma_P = 0.000
bB = 0.08
sigma_B = 0.0
cod = 0. # 2.5563471547779937 #3.035

coc0 = 0. #350
## Ocean albedo parameters
Talphaocean_low = 219
Talphaocean_high = 299
alphaocean_max = 0.84
alphaocean_min = 0.255

Cbio_low = 50
Cbio_high = 700

T0 = 298
C0 = 280

## CO2 uptake by vegetation
wa = 0.015
vegcover = 0.4

Thigh = 315
Tlow = 282
Topt1 = 295
Topt2 = 310
acc = 5


# lower and upper bounds of opt and viable temp
Tbiopt1_low = Topt1
Tbiopt1_high = Topt1 + 5
Tbiolow_low = Tlow
Tbiolow_high = Tlow + 5

To = 282.87  # Mean with no anthropogenic carbon emissions, in Fᵒ

## Volcanism
Volcan = 0.028


# alphaocean = (0.255 + 0.37 ) /2.
alphaocean = 0.3444045881126172

#Fraction of ocean covered by ice
def fracseaice(T):
    
    temp = np.zeros(T.shape)
    temp[ T< Talphaocean_low ] = 1
    temp[ (T>= Talphaocean_low)&(T< Talphaocean_high)] = 1 - 1 / (Talphaocean_high - Talphaocean_low) * (T[(T>= Talphaocean_low)&(T< Talphaocean_high)] - Talphaocean_low)
    temp[T>= Talphaocean_high] = 0

    return temp
    
# fracseaice = 0.15
# fracseaice = 0.1

T = np.arange(-30,30,0.1)

# plot for fracseaice

plt.plot(s)

C0v = 1000
VC_min = 0
VC_max = 5/12 * C0v

def biopump(F):
    """F, accumulated anthrpogenic emission"""
    temp = np.zeros(F.shape)
    
    temp[F < Cbio_low] = 1
    temp[(F >= Cbio_low)&(F < Cbio_high)] = 1 - 1/(Cbio_high - Cbio_low) * (F[(F >= Cbio_low)&(F < Cbio_high)] - Cbio_low)
    temp[F >= Cbio_high] = 0
    return temp

def Tvegoptlow(G):

    temp = np.zeros(G.shape)

    temp[G <= Cbio_low] = VC_min
    temp[(G > Cbio_low) & (G < Cbio_high)] = VC_min + \
        (VC_max - VC_min) / \
        (Cbio_high - Cbio_low) * \
        (G[(G > Cbio_low) & (G < Cbio_high)] - Cbio_low)
    temp[G >= Cbio_high] = VC_max

    return temp

def Tveglow(G):

    temp = np.zeros(G.shape)

    temp[G <= Cbio_low] = Tbiolow_low
    temp[(G > Cbio_low) & (G < Cbio_high)] = Tbiolow_low + \
        (Tbiolow_high - Tbiolow_low) / \
        (Cbio_high - Cbio_low) * \
        (G[(G > Cbio_low) & (G < Cbio_high)] - Cbio_low)
    temp[G >= Cbio_high] = Tbiolow_high

    return temp


def veggrowthdyn(T, G):

    temp = np.zeros(T.shape)

    temp[T < Tveglow(G)] = 0

    temp[(T >= Tveglow(G)) & (T < Tvegoptlow(G))] = acc / (Tvegoptlow(G) - Tveglow(G))[(T >= Tveglow(G)) & (T < Tvegoptlow(G))] * \
        (T[(T >= Tveglow(G)) & (T < Tvegoptlow(G))] - Tveglow(G)[(T >= Tveglow(G)) & (T < Tvegoptlow(G))])

    temp[(T >= Tvegoptlow(G)) & (T <= Topt2)] = acc

    temp[(T >= Topt2) & (T < Thigh)] = acc / (Topt2 - Thigh) * \
        (T[(T >= Topt2) & (T < Thigh)] - Thigh)

    temp[T > Thigh] = 0

    return temp


def veggrowth(T):
    
    temp = np.zeros(T.shape)
    
    temp[T < Tlow] = 0
    temp[(T >= Tlow)&(T < Topt1)] = acc / (Topt1 - Tlow) * (T[(T >= Tlow)&(T < Topt1)] - Tlow)
    temp[(T >= Topt1)&(T < Topt2)] = acc
    temp[(T >= Topt2)&(T < Thigh)] = acc / (Topt2 - Thigh) * (T[(T >= Topt2)&(T < Thigh)] - Thigh)
    temp[T > Thigh] = 0
    
    return temp


def alphaocean(T):
    temp = np.zeros(T.shape)
    temp[T<=Talphaocean_low] = alphaocean_max
    temp[(T>Talphaocean_low)  & (T<Talphaocean_high)] = alphaocean_max + (alphaocean_min-alphaocean_max)/(Talphaocean_high-Talphaocean_low) * (T[(T>Talphaocean_low)  & (T<Talphaocean_high)]-Talphaocean_low)
    temp[T>=Talphaocean_high] = alphaocean_min

    return temp

#Incoming radiation modified by albedo
def Ri(T):
    return 1/cearth * (Q0 * (1 - p * alphaland - (1 - p) * alphaocean(T)) )

# Ri = 1 / cearth * Q0 * (1 - p * alphaland - (1 - p) * alphaocean)

# Outgoing radiation modified by greenhouse effect
def Ro(T, C):
    return 1/cearth * (kappa * (T - Tkappa) -  B * np.log(C / C0))

#Solubility of atmospheric carbon into the oceans
# carbon pumps
def kappaP(T, W):
    return np.exp(- bP * (T - T0))

def oceanatmphysflux(T):
    return 1 / tauc * (coc0 * (np.exp(-bP * (T - T0))))

def oceanbioflux(T):
    
    # if sa == 1:
        
    return 1/tauc * (coc0 * (np.exp( bB * (T - T0))))
    
    # elif sa == 0:
        
    #     return 1/tauc * (coc0 * (np.exp(bB * (T - T0))))
    
    # else:
    #     return ValueError("Wrong input value: 0 or 1.")

def oceanatmcorrflux(C):
    return 1 / tauc * (- cod * C)

T_preindustrial = 286.85
C_preindustrial = 280


# Economic paramaters
gamma_1 = 1.7675 / 10000.
gamma_2 = 2 * 0.0022
delta   = args.delta
eta     = 0.032

# State variable
# Temperature anomaly, in celsius
T_min  = Xminarr[0]
T_max  = Xmaxarr[0] 
hT     = hXarr[0]
T_grid = np.arange(T_min, T_max + hT, hT)
nT_grid = len(T_grid)



# atmospheric carbon concentration, in gigaton
C_min  = Xminarr[1]
C_max  = Xmaxarr[1] 
hC     = hXarr[1]
C_grid = np.arange(C_min, C_max + hC, hC)
nC_grid = len(C_grid)


# F, Sa in the notes, accumulative anthropogenic carbon, in gigaton
F_min = Xminarr[2]
F_max = Xmaxarr[2] 
hF = hXarr[2]
F_grid = np.arange(F_min, F_max + hF, hF)
nF_grid = len(F_grid)


# meshgrid
(T_mat, C_mat, F_mat) = np.meshgrid(T_grid, C_grid, F_grid, indexing="ij")
stateSpace = np.hstack([
    T_mat.reshape(-1, 1, order="F"),
    C_mat.reshape(-1, 1, order="F"),
    F_mat.reshape(-1, 1, order="F")
])

T_mat_1d = T_mat.ravel(order='F')
C_mat_1d = C_mat.ravel(order='F')
F_mat_1d = F_mat.ravel(order='F')



v0 = ( - eta* T_mat - eta * delta * C_mat - eta * F_mat)*1

# v0 =  delta * eta * np.log(delta /4 * (9000/2.13 - F_mat)) + (eta - 1) * gamma_2 * T_mat / cearth * (B * np.log(C_mat/ C0) + kappa * (T_mat + To - Tkappa))

dG  = gamma_1 + gamma_2 * T_mat
# epsilon  = 0.1
count    = 0
error    = 1.
tol      = 1e-7

lowerLims = np.array([T_grid.min(), C_grid.min(), F_grid.min()], dtype=np.float64)
upperLims = np.array([T_grid.max(), C_grid.max(), F_grid.max()], dtype=np.float64)

dVec = np.array([hT, hC, hF])
increVec = np.array([1, nT_grid, nT_grid*nC_grid], dtype=np.int32)


petsc_mat = PETSc.Mat().create()
petsc_mat.setType('aij')
petsc_mat.setSizes([nT_grid * nC_grid * nF_grid, nT_grid * nC_grid * nF_grid])
petsc_mat.setPreallocationNNZ(13)
petsc_mat.setUp()
ksp = PETSc.KSP()
ksp.create(PETSc.COMM_WORLD)
ksp.setType('bcgs')
ksp.getPC().setType('ilu')
ksp.setFromOptions()


while error > tol and count < maxiter:
    
    dvdT  = finiteDiff_3D(v0, 0, 1, hT)
    dvdTT = finiteDiff_3D(v0, 0, 2, hT)
    dvdC  = finiteDiff_3D(v0, 1, 1, hC)
    dvdCC = finiteDiff_3D(v0, 1, 2, hC)
    dvdF  = finiteDiff_3D(v0, 2, 1, hF)
    dvdFF = finiteDiff_3D(v0, 2, 2, hF)
        

    Ca = - eta * delta / (dvdC + dvdF)-k
    # Ca = eta * delta / (dvdC + dvdF)-k

    # Ca = np.ones(v0.shape)
    # print("dvdC_max={}, dvdC_min={}".format(dvdC.max(),dvdC.min()))
    # print("dvdF_max={}, dvdF_min={}".format(dvdF.max(),dvdF.min()))
    print("g_min={}, gmax={}".format(Ca.max(),Ca.min()))

    # if count >=1:
    #     Ca = Ca * fraction + Ca_star * (1 - fraction)

    
    Ca[Ca <= 1e-16] = 1e-16
    
    # Ca = 1. * np.ones(T_mat.shape)
    
    A  = - delta * np.ones(T_mat.shape)
    B1 = Ri(T_mat + To) - Ro(T_mat + To, C_mat)
    # print("B11_min={}, B1max={}".format(Ri(T_mat + To).max(),Ri(T_mat + To).min()))
    # print("B12_min={}, B1max={}".format(Ro(T_mat + To, C_mat).max(),Ro(T_mat + To, C_mat).min()))

    B2 = Volcan
    # print("B2min={}, B2max={}".format(B2.max(),B2.min()))

    B2 += Ca * sa
    # print("B2min={}, B2max={}".format(B2.max(),B2.min()))
    B2 -= wa * C_mat * vegcover  * veggrowthdyn(T_mat + To, F_mat)
    # print("B2min={}, B2max={}".format(B2.max(),B2.min()))
    B2 += oceanatmphysflux(T_mat + To) * (1 - fracseaice(T_mat + To))
    # print("B2min={}, B2max={}".format(B2.max(),B2.min()))
    B2 += oceanbioflux(T_mat + To) *     (1 - fracseaice(T_mat + To))
    # print("B2min={}, B2max={}".format(B2.max(),B2.min()))
    B2 += oceanatmcorrflux(C_mat) * (1 - fracseaice(T_mat + To))
    # print("B2min={}, B2max={}".format(B2.max(),B2.min()))
    B3 = Ca
    C1 = 0.0 * np.ones(T_mat.shape)
    C2 = 0.0 * np.ones(T_mat.shape)
    C3 = np.zeros(T_mat.shape)
    D = eta * delta * np.log(Ca+k ) + (eta - 1)  * dG * B1  

    start_ksp = time.time()

    A_1d = A.ravel(order='F')
    C_1_1d = C1.ravel(order='F')
    C_2_1d = C2.ravel(order='F')
    C_3_1d = C3.ravel(order='F')
    B_1_1d = B1.ravel(order='F')
    B_2_1d = B2.ravel(order='F')
    B_3_1d = B3.ravel(order='F')
    D_1d = D.ravel(order='F')
    v0_1d = v0.ravel(order='F')
    petsclinearsystem.formLinearSystem(T_mat_1d, C_mat_1d, F_mat_1d, A_1d, B_1_1d, B_2_1d,
                                       B_3_1d, C_1_1d, C_2_1d, C_3_1d, epsilon, lowerLims, upperLims, dVec, increVec, petsc_mat)
    # F_init_1d = F_init.ravel(order='F')
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
    v = np.array(ksp.getSolution()).reshape(A.shape, order="F")
    end_ksp = time.time()
    num_iter = ksp.getIterationNumber()

    print("A_min={}, Amax={}".format(A.max(),A.min()))
    print("B1_min={}, B1max={}".format(B1.max(),B1.min()))
    print("B2min={}, B2max={}".format(B2.max(),B2.min()))
    print("B3min={}, B3max={}".format(B3.max(),B3.min()))

    rhs_error = A * v0 + B1 * dvdT + B2 * dvdC + B3* dvdF + C1 * dvdTT + C2 * dvdCC + C3*dvdFF + D
    rhs_error = np.max(abs(rhs_error))
    lhs_error = np.max(abs((v - v0)/epsilon))

    error = lhs_error
    v0 = v
    Ca_star = Ca
    count += 1

    print("Iteration: %s;\t False Transient Error: %s;\t PDE Error: %s\t" % (count, lhs_error, rhs_error))

print("Total iteration: %s;\t LHS Error: %s;\t RHS Error %s\t" % (count, lhs_error, rhs_error))