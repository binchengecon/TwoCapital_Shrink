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
from supportfunctions import finiteDiff
rcParams["figure.figsize"] = (8,5)
rcParams["savefig.bbox"] = 'tight'
                                                                                
def PDESolver(stateSpace, A, B1, B2, C1, C2, D, v0, 
              ε = 1, tol = -10):                                              
                                                                                 

    A = A.reshape(-1,1,order = 'F')                                         
    B = np.hstack([B1.reshape(-1,1,order = 'F'),B2.reshape(-1,1,order = 'F')])
    C = np.hstack([C1.reshape(-1,1,order = 'F'),C2.reshape(-1,1,order = 'F')])
    D = D.reshape(-1,1,order = 'F')                                         
    v0 = v0.reshape(-1,1,order = 'F')                                       
    out = SolveLinSys.solveFT(stateSpace, A, B, C, D, v0, ε, tol)           

    return out                                                            

# Anthropogenic emissions (zero or one)
Can = pd.read_csv("rcp30co2eqv3.csv")
#times2co2eq
#rcp85co2eq.csv
Ca = Can[(Can["YEARS"] > 1799) & (Can["YEARS"] < 2801)]
Ca1 = Can[(Can["YEARS"] > 1799) & (Can["YEARS"] < 2801)]

Ca = Ca["CO2EQ"]
Ca = Ca - 281.69873
Ca = Ca.to_numpy()

Ce = np.arange(1001) * 1.0
#np.min(Ca)
for i in range(len(Ce)):
    if i == 0:
        Ce[i] = 0
    else:
        Ce[i] = Ca[i] - Ca[i-1] 
        
t_val = np.linspace(0, 1000, 1001)
def Yam(t):
    t_points = t_val
    em_points = Ce
    
    tck = interpolate.splrep(t_points, em_points)
    return interpolate.splev(t,tck)
        
Cebis = np.arange(1001) * 1.0
#np.min(Ca)
for i in range(len(Cebis)):
    if i == 0:
        Cebis[i] = 0
    else:
        Cebis[i] = max( Ca[i] - Ca[i-1], 0) 
        
Cc = np.arange(1001) * 1.0
#np.min(Ca)
for i in range(len(Cc)):
    if i == 0:
        Cc[i] = 0
    else:
        Cc[i] = sum(Cebis[0:i])

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
cearth = 0.107
# cearth = 10.
tauc = 20
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


## Volcanism
Volcan = 0.028


# alphaocean = (0.255 + 0.37 ) /2.
alphaocean = 0.3444045881126172

#Fraction of ocean covered by ice
# def fracseaice(T):
    
#     temp = np.zeros(T.shape)
#     temp[ T< Talphaocean_low ] = 1
#     temp[ (T>= Talphaocean_low)&(T< Talphaocean_high)] = 1 - 1 / (Talphaocean_high - Talphaocean_low) * (T[(T>= Talphaocean_low)&(T< Talphaocean_high)] - Talphaocean_low)
#     temp[T>= Talphaocean_high] = 0

#     return temp
    
fracseaice = 0.15
# fracseaice = 0.1


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

    temp[(T >= Tveglow(G)) & (T < Tvegoptlow(G))] = acc / (Tvegoptlow(G) - Tveglow(G)) * \
        (T[(T >= Tveglow(G)) & (T < Tvegoptlow(G))] - Tveglow(G))

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


#Incoming radiation modified by albedo
# def Ri(T):
#     return 1/cearth * (Q0 * (1 - p * alphaland - (1 - p) * alphaocean) )

Ri = 1 / cearth * Q0 * (1 - p * alphaland - (1 - p) * alphaocean)

# Outgoing radiation modified by greenhouse effect
def Ro(T, C):
    return 1/cearth * (kappa * (T - Tkappa) -  B * np.log(C / C0))

#Solubility of atmospheric carbon into the oceans
# carbon pumps
def kappaP(T, W):
    return np.exp(-(bP + sigma_P * W) * (T - T0))

def oceanatmphysflux(T, W):
    return 1 / tauc * (coc0 * (np.exp(-(bP + sigma_P * W) * (T - T0))))

def oceanbioflux(T, W, sa):
    
    if sa == 1:
        
        return 1/tauc * (coc0 * (np.exp( (bB + sigma_B * W) * (T - T0))))
    
    elif sa == 0:
        
        return 1/tauc * (coc0 * (np.exp(bB * (T - T0))))
    
    else:
        return ValueError("Wrong input value: 0 or 1.")

def oceanatmcorrflux(C):
    return 1 / tauc * (- cod * C)

T_preindustrial = 286.85
C_preindustrial = 280

def dydt(t, y):
    T = y[0]
    C = y[1]

    dT = 1. / cearth * ( - kappa * T + B * np.log(C + C0) - B * np.log(C0))
#     dT -= Ro(T, C)
    Ws = np.random.normal(size=(2,1))
#     dC = Volcan
    dC = Yam(t) - (Volcan / C_preindustrial * C + (1 - fracseaice) * cod / tauc * C) + coc0 / tauc * (1 - fracseaice) * (np.exp(-bP * (T + T_preindustrial - T0) - np.exp(-bP * (T_preindustrial - T0)))) + coc0 / tauc * (1 - fracseaice) * (np.exp(bB * (T + T_preindustrial - T0) - np.exp(bB * (T_preindustrial - T0))))   # biological pump flux * fraction sea ice
#     dC += oceanbioflux(T) * (1 - fracseaice(T))      # biological pump flux * fraction sea ice
#     dC += oceanatmcorrflux(C) * (1 - fracseaice)    # correction parameter

    return dT, dC

# Economic paramaters
gamma_1 = 1.7675 / 10000.
gamma_2 = 2 * 0.0022
delta   = 0.01
eta     = 0.032

# State variable
# Temperature anomaly, in celsius
T_min  = 0. 
T_max  = 3. # 
hT     = 0.1
T_grid = np.arange(T_min, T_max + hT, hT)

# atmospheric carbon concentration, in gigaton
C_min  = 0.
C_max  = 500.
hC     = 10.
C_grid = np.arange(C_min, C_max + hC, hC)

# F, Sa in the notes, accumulative anthropogenic carbon, in gigaton
F_min = 10. 
F_max = 2500. 
hF = 50.
F_grid = np.arange(F_min, F_max + hF, hF)

# meshgrid
(T_mat, C_mat, F_mat) = np.meshgrid(T_grid, C_grid, F_grid, indexing="ij")
stateSpace = np.hstack([
    T_mat.reshape(-1, 1, order="F"),
    C_mat.reshape(-1, 1, order="F"),
    F_mat.reshape(-1, 1, order="F")
])

T_mat.shape
To = 282.87  # Mean with no anthropogenic carbon emissions, in Fᵒ

v0 =  - eta * delta * C_mat - eta * F_mat

# v0 =  delta * eta * np.log(delta /4 * (9000/2.13 - F_mat)) + (eta - 1) * gamma_2 * T_mat / cearth * (B * np.log(C_mat/ C0) + kappa * (T_mat + To - Tkappa))

dG  = gamma_1 + gamma_2 * T_mat
epsilon  = 0.1
count    = 0
error    = 1.
tol      = 1e-8
max_iter = 6000
fraction = 0.1
cearth = 0.107
# cearth = 10.

while error > tol and count < max_iter:
    
    dvdT  = finiteDiff(v0, 0, 1, hT)
    dvdTT = finiteDiff(v0, 0, 2, hT)
    dvdC  = finiteDiff(v0, 1, 1, hC)
#     dvdC[dvdC >= - 1e-16] = - 1e-16
    dvdCC = finiteDiff(v0, 1, 2, hC)
    dvdF  = finiteDiff(v0, 2, 1, hF)
    dvdFF = finiteDiff(v0, 2, 2, hF)
        

    Ca = - eta * delta / (dvdC + dvdF)

    
    if count >=1:
        Ca = Ca * fraction + Ca_star * (1 - fraction)

    
    Ca[Ca <= 1e-32] = 1e-32
    
#     Ca = 1. * np.ones(T_mat.shape)
    
    A  = - delta * np.ones(T_mat.shape)
    B1 = Ri(T_mat + To) - Ro(T_mat + To, C_mat)
    B2 = Volcan
    B2 += Ca * sa
    # B2 -= wa * C_mat * veggrowth(T_mat + To)
    B2 -= wa * C_mat * veggrowthdyn(T_mat + To, F_mat)
    B2 += oceanatmphysflux(T_mat + To) * (1 - fracseaice(T_mat + To))
    B2 += oceanbioflux(T_mat + To) * \
        (1 - fracseaice(T_mat + To))
    B2 += oceanatmcorrflux(C_mat) * (1 - fracseaice(T_mat + To))
    B3 = Ca
    C1 = 0.0 * np.ones(T_mat.shape)
    C2 = 0.0 * np.ones(T_mat.shape)
    C3 = np.zeros(T_mat.shape)
    D = eta * delta * np.log(Ca) + (eta - 1) * \
        dG * B1  # relation between versions?

    out = PDESolver(stateSpace, A, B1, B2, C1, C2, D, v0, epsilon)
    v = out[2].reshape(v0.shape, order="F")

    rhs_error = A * v0 + B1 * dvdT + B2 * dvdC + C1 * dvdTT + C2 * dvdCC + D
    rhs_error = np.max(abs(rhs_error))
    lhs_error = np.max(abs((v - v0)/epsilon))

    error = lhs_error
    v0 = v
    Ca_star = Ca
    count += 1

    print("Iteration: %s;\t False Transient Error: %s;\t PDE Error: %s\t" % (count, lhs_error, rhs_error))

print("Total iteration: %s;\t LHS Error: %s;\t RHS Error %s\t" % (count, lhs_error, rhs_error))