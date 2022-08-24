##################################################################
## Section 1: Global Parameters and Library
##################################################################


##################################################################
## Section 1.1: Library Loading
##################################################################

import os
from unittest import TestCase
import numpy as np
import configparser 
import sys
from scipy.integrate import solve_ivp
from scipy.fft import fft, fftfreq
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import SymLogNorm
import matplotlib.mlab
import scipy.io as sio
import matplotlib as mpl
import pandas as pd
import scipy.optimize as optim
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy import fft, arange, signal

mpl.rcParams["lines.linewidth"] = 3.5
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["figure.figsize"] = (8,5)
mpl.rcParams["font.size"] = 20
mpl.rcParams["legend.frameon"] = False

##################################################################
## Section 1.2: Parameter Initialization
##################################################################

# cearth = 0.373 # heat capacity

Q0 = 342.5 #Incoming radiation



## land fraction and albedo

p = 0.3 #Fraction of land on the planet
alphaland = 0.28 # land albedo (0.28328) 


## Ocean albedo parameters
Talphaocean_low = 219
Talphaocean_high = 299
# alphaocean_max = 0.84
# alphaocean_min = 0.255
alphaocean_max = 0.85
alphaocean_min = 0.25


## outgoing radiation linearized
kappa = 1.74
T_kappa = 154


## CO2 radiative forcing
a = 5.35 # CO2 radiative forcing W/m^2
C0 = 280 # CO2 params. C0 is the reference C02 level, unit ppm


## ocean carbon pumps

bP = 0.029 # Solubility dependence on temperature (value from Fowler et al) K^(-1) (0.077)
bB = 0.069 # Biopump dependence on temperature (Value from Fowler) unit K^(-1) (0.090)


## Characteristics Time
# tauc=30 # Characteristics Time

## CO2 uptake by vegetation
W_a = 0.0056 # (0.015)


T_vh = 307.15 # (305)
T_vl = 286.15 # (275)


T_ol = 290.15 # (285)
T_oh = 302.15 # (295)

T_vl_minus= T_vl
T_vl_plus = T_vl_minus + 5

T_ol_minus= T_ol
T_ol_plus = T_ol_minus + 5

k = 8 #  Maximum Vegetaion Development Factor


lamb = 2.2 # Ocean carbon pump modulation parameter

Gl = 150 # Cumulative carbon lower bound
Gh = 750 # Cumulative carbon upper bound

## C_0,oc ocean carbon depending on depth
C_0_ocean = 330 # (73.78)


T0 = 288 # Ocean Pumps Reference Temperature Factor


## Volcanism
V = 0.028 # Volcanism




## Anthropogenic carbon
sa = 1 # Switch to take anthropogenic emissions

T_mean = 286.5 # unit K 282.9
C_mean = 269 # unit ppm 275.5 
G_mean = 0

##################################################################
##################################################################
##################################################################









##################################################################
## Section 2: Function Definitions
##################################################################

##################################################################
## Section 2.1: Definitions
################################################################## 

## Time Period: 100 Years

t_span = 100
t_val = np.linspace(0, t_span-1, t_span)

def Yam(t,CcTemp):
    t_points = t_val
    em_points = CcTemp
    
    tck = interpolate.splrep(t_points, em_points)
    return interpolate.splev(t,tck)

    
def alphaocean(T):
    """"Ocean albedo"""
    if T < Talphaocean_low:
        return alphaocean_max
    elif T < Talphaocean_high:
        return alphaocean_max + (alphaocean_min - alphaocean_max) / (Talphaocean_high - Talphaocean_low) * (T - Talphaocean_low)
    else: # so T is higher
        return alphaocean_min



def beta_func(T):
    """Fraction of ocean covered by ice"""
    if T < Talphaocean_low:
        return 1
    elif T < Talphaocean_high:
        return  1 -  1 / (Talphaocean_high - Talphaocean_low) * (T - Talphaocean_low)
    else: # so T is higher
        return 0



#
def Ri(T, cearth, tauc):
    """Incoming radiation modified by albedo"""
    return 1/cearth * (Q0 * (1 - p * alphaland - (1 - p) * alphaocean(T)))



# 
def Ro(T, C, cearth, tauc):
    """Outgoing radiation modified by greenhouse effect"""
    return 1/cearth * (kappa * (T - T_kappa) -  a * np.log(C / C0))


def kappaP(T):
    """Solubility of atmospheric carbon into the oceans: Carbon Pumps"""
    np.exp(-bP * (T - T0))





def oceanatmphysflux(T):
    """Sum of two terms that reflect, respectively, the physical (or solubility) carbon pump in the ocean and Wally """
    """Broecker’s “biopump”, due to thermally enhanced bioproductivity (Fowler et al., 2013)"""
    return 1 / tauc * (C_0_ocean * (np.exp(-bP * (T - T0))))




def oceanbioflux(T):
    return 1/tauc * (C_0_ocean * (np.exp(bB  * (T - T0))))



def oceanatmcorrflux(C):
    return 1 / tauc * (- lamb * C)

def T_ol_func(G):
    if G < Gl:
        return T_ol_minus
    elif G < Gh:
        return T_ol_minus+(T_ol_plus-T_ol_minus)*(G-Gl)/(Gh-Gl)
    else :
        return T_ol_plus

def T_ol_func_vec(CcTemp):
    """Goal: vectorize Function biopump"""
    """Output: Grid Points for Interpolation"""        
    T_ol_func_vec = np.vectorize(T_ol_func)
    T_ol_func_vec_modulation = [T_ol_func_vec(val) for val in CcTemp]
    T_ol_mod = np.float_(T_ol_func_vec_modulation)
    return T_ol_mod

def T_vl_func(G):
    if G<Gl:
        return T_vl_minus
    elif G<Gh:
        return T_vl_minus+(T_vl_plus-T_vl_minus)*(G-Gl)/(Gh-Gl)
    else :
        return T_vl_plus

def T_vl_func_vec(CcTemp):
    """Goal: vectorize Function biopump"""
    """Output: Grid Points for Interpolation"""
    T_vl_func_vec = np.vectorize(T_vl_func)
    T_vl_func_vec_modulation = [T_vl_func_vec(val) for val in CcTemp]
    T_vl_mod = np.float_(T_vl_func_vec_modulation)
    return T_vl_mod


def T_l_interpolation(t,T_l_mod):
    t_points = t_val
    em_points = T_l_mod
    
    tck = interpolate.splrep(t_points, em_points)
    return interpolate.splev(t,tck)



def veggrowth2(T,t,T_vl_mod,T_ol_mod):

    if T < T_l_interpolation(t,T_vl_mod):
        return 0

    if (T >= T_l_interpolation(t,T_vl_mod)) and (T < T_l_interpolation(t,T_ol_mod)):
        return k / (T_l_interpolation(t,T_ol_mod)- T_l_interpolation(t,T_vl_mod)) * (T - T_l_interpolation(t,T_vl_mod))
    
    if (T >= T_l_interpolation(t,T_ol_mod)) and (T <= T_oh):
        return k
    
    if (T > T_oh) and (T < T_vh):
        return k / (T_oh - T_vh) * (T - T_vh)
    
    if T > T_vh:
        return 0






def model3(T_mean, C_mean, G_mean, cearth, tauc, Ce=np.zeros(t_span)):
    """Input: Starting Temp, CO2, Impulse """
    """Transition: Dynamic Equation"""
    """Output: Whole Array of Temp, Temp Anamoly, CO2"""
    Ce[Ce <0] = 0
    Cc = G_mean/2.13 + np.cumsum(Ce) 
    Cc[0]=0
    T_vl_mod = T_vl_func_vec(Cc)
    T_ol_mod = T_ol_func_vec(Cc)

    def dydt(t, y):
        """Transition: Dynamic Equation (3 Input)"""
        """t: time sequence"""
        """y: Starting Temp, CO2"""
        """Cc: Cumulated Carbon **Constant Array** """
        T = y[0]
        C = y[1]

        dT = Ri(T, cearth, tauc) 
        dT -= Ro(T, C, cearth, tauc)
    
        dC = V
        dC += Yam(t,Ce) * sa                                  #  anthropogenic emissions from Ca spline                                                # volcanism 
        dC -= W_a * C  * veggrowth2(T,t,T_vl_mod,T_ol_mod)             # carbon uptake by vegetation
        dC += oceanatmphysflux(T) * (1 - beta_func(T))    # physical solubility into ocean * fraction of ice-free ocean
        dC +=  oceanbioflux(T) * (1 - beta_func(T)) # biological pump flux * fraction sea ice
        
        
        dC += oceanatmcorrflux(C) * (1 - beta_func(T))    # correction parameter



        return dT, dC

    init = [T_mean, C_mean]

    t_eval = np.linspace(0, 100, 10000)

    sol = solve_ivp(dydt, t_eval[[0, -1]], init, t_eval=t_eval, method='RK45', max_step=0.1)

    # sol = solve_ivp(dydt, t_eval[[0, -1]], init, t_eval=t_eval, method='BDF')
    # -

    #Extract values of temperature and CO2
    Tv = sol.y[0, :]
    Cv = sol.y[1, :]
    tv = sol.t


    # Tvmid = Tv - 286.7 
    Tvmid = Tv - 286.6

    return tv, Tvmid, Cv
##################################################################
##################################################################
##################################################################



## Impulse Path
ImpulsePattern = 0

## Pattern = 1
if ImpulsePattern == 0:
    """Equivalent Impulse Horizon with Hetero-Value"""
    # ImpulseMin = 0 
    # ImpulseMax = 1100
    # ImpulseStep = 100
    # ImpulsePathSize = int((ImpulseMax-ImpulseMin)/ImpulseStep )

    # Carbon   = np.array([0, 100, 150, 200])
    Carbon   = np.array([0])

    ImpulsePathSize = len(Carbon)
    CeMatrix = np.zeros((ImpulsePathSize,t_span))

    CeMatrix[:,0] =     Carbon[:] /2.13

elif ImpulsePattern ==1:
    """Heterogenous Impulse Horizon with Homo-value"""
    ImpulsePathSize = 10
    ImpulseValue = 100
    
    CeMatrix = ImpulseValue*np.eye(ImpulsePathSize, t_span)

elif ImpulsePattern ==2:
    """Fixed Impulse Response"""
    ImpulsePathSize = 2
    ImpulseValue = 10
    
    CeMatrix = np.zeros((ImpulsePathSize, t_span))
    CeMatrix[1,:] = ImpulseValue*np.ones((1,t_span))/2.13


## cearth, tauc Path

# cearth_taucMatrix = [[35., 6603. ],
#                      [0.107, 20],
#                      [0.373, 30],
#                      [10, 1886]]

cearth_taucMatrix = [[0.3725, 30]]

cearth_taucMatrixSize = len(cearth_taucMatrix)



## Looping
Figure_Dir = "./nonlinearCarbon/figure/"+"Adapt_"

for ctpathnum in range(cearth_taucMatrixSize):
    figwidth = 10
    # fig, axs = plt.subplots(4, 1, sharex=True, figsize=(12, 2 *figwidth))
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(12, 2 *figwidth))
    # fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12, 2 *figwidth))
    TvmidBase = np.zeros(10000)

    for pathnum in range(ImpulsePathSize):


        Ce = CeMatrix[pathnum,:]
        cearth, tauc = cearth_taucMatrix[ctpathnum]
        print(cearth)
        tv, Tvmid, Cv = model3(T_mean, C_mean, G_mean, cearth, tauc, Ce)

        plotnum = ImpulsePattern*pathnum

        if pathnum ==0:
            TvmidBase = Tvmid

        if pathnum==0:
            axs[0].plot(tv, Tvmid, label="baseline")
        else: 
            axs[0].plot(tv, Tvmid, label=f"CarbonImpulse={CeMatrix[pathnum,plotnum]*2.13}")
        # axs[0].plot(tv, Tvmid, label=f"CarbonImpulse={CeMatrix[pathnum,plotnum]*2.13}")
        axs[0].set_xlabel('Time (year)')
        axs[0].set_ylabel('Temperature (K)')
        axs[0].set_title('Temperature Anomaly Dynamics T')
        axs[0].grid(linestyle=':')
        axs[0].legend()
        if pathnum==0:
            axs[1].plot(tv, Cv, label="baseline")
        else: 
            axs[1].plot(tv, Cv, label=f"CarbonImpulse={CeMatrix[pathnum,plotnum]*2.13}")
        # axs[1].plot(tv, Cv, label=f"CarbonImpulse={CeMatrix[pathnum,plotnum]*2.13}")
        axs[1].set_xlabel('Time (year)')
        axs[1].set_ylabel('Carbon (ppm)')
        axs[1].set_title('Carbon Concentration Dynamics C')
        axs[1].grid(linestyle=':')
        axs[1].legend()
        # if pathnum==0:
        #     axs[2].plot(tv, Gv, label="baseline")
        #     # axs[2].legend()        
        # else: 
        #     axs[2].plot(tv, Gv, label=f"CarbonImpulse={CeMatrix[pathnum,plotnum]*2.13}")
        # # axs[2].plot(tv, Gv, label=f"CarbonImpulse={CeMatrix[pathnum,plotnum]*2.13}")
        # axs[2].set_xlabel('Time (year)',fontsize = 16)
        # axs[2].set_ylabel('Total',fontsize = 16)
        # axs[2].set_title('Total Emission Dynamics G')
        # axs[2].grid(linestyle=':')
        # axs[2].legend()
        if pathnum==0:
            axs[2].plot(tv, Tvmid-TvmidBase, label="baseline")
            # axs[2].legend()        
        else: 
            axs[2].plot(tv, Tvmid-TvmidBase, label=f"CarbonImpulse={CeMatrix[pathnum,plotnum]*2.13}")
        axs[2].set_xlabel('Time (year)')
        axs[2].set_ylabel('Degree Celsius')
        axs[2].set_title('Impulse Response of temperature anomaly per Gigatonne of Carbon')
        axs[2].grid(linestyle=':')
        axs[2].legend()        

        # np.save(f"ImpPtnPath_{ImpulsePattern}_{CeMatrix[pathnum,plotnum]*2.13}_cearth_{cearth}_tauc_{tauc}.npy", [tv, Tvmid, Cv])


    plt.tight_layout()
    plt.savefig(Figure_Dir+f"ImpulsePtn={ImpulsePattern}, cearth={cearth}, tauc={tauc}_new1.pdf")
    plt.savefig(Figure_Dir+f"ImpulsePtn={ImpulsePattern}, cearth={cearth}, tauc={tauc}_new1.png")    
    # plt.savefig(Figure_Dir+"sample_with0.pdf")





