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

## heat capacity, incoming radiation
Q0 = 342.5 #Incoming radiation

cearth = 0.373


## land fraction and albedo
p = 0.3 #Fraction of land on the planet
# alphaland = 0.28328 # land albedo
alphaland = 0.28


## outgoing radiation linearized
kappa = 1.74
Tkappa = 154

## CO2 radiative forcing
a = 5.35 # CO2 radiative forcing
# renamed into "B"->"a"

C0 = 280 # CO2 params. C0 is the reference C02 level


## ocean carbon pumps
# bP = 0.077 # Solubility dependence on temperature (value from Fowler et al)
# bB = 0.090 # Biopump dependence on temperature (Value from Fowler)

bP = 0.029 # Solubility dependence on temperature (value from Fowler et al)
bB = 0.069 # Biopump dependence on temperature (Value from Fowler)

cod = 0.54 # Ocean carbon pump modulation parameter


## timescale and reference temperature (from Fowler)
# tauc = 20 # timescale 
tauc = 30 # timescale 

T0 = 288 # Temperature reference

# Tmean = 286.6

## Coc0 ocean carbon depending on depth
# coc0 = 73.78
coc0 = 330

## CO2 uptake by vegetation
# wa = 0.015
wa = 0.006


vegcover = 0.4

# Thigh = 305
# Tlow = 275
Thigh = 307.15
Tlow = 286.15

# Topt1 = 285
# Topt2 = 295
Topt1 = 290.16
Topt2 = 302.15

T_vl_minus= 286.15
T_vl_plus = T_vl_minus+4

T_ol_minus= 290.15
T_ol_plus = T_ol_minus+4


# acc = 5 actually renamed "acc"-> "k"
acc=8

Gl= 150
Gh=750

## Volcanism
V = 0.028

## Anthropogenic carbon
sa = 1 # Switch to take anthropogenic emissions

# Ts = 286.7 + 0.56 # 282.9
# Cs = 389 # 275.5

Ts = 286.6 # 282.9
Cs = 268.6 # 275.5
Gs = 0

## Ocean albedo parameters
Talphaocean_low = 219
Talphaocean_high = 299
# alphaocean_max = 0.84
# alphaocean_min = 0.255
alphaocean_max = 0.843
alphaocean_min = 0.254
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


#
# def fracseaice(T):
#     """Fraction of ocean covered by ice"""
#     if T < Talphaocean_low:
#         return 1
#     elif T < Talphaocean_high:
#         return 1 - 1 / (Talphaocean_high - Talphaocean_low) * (T - Talphaocean_low)
#     else: # so T is higher
#         return 0

def fracseaice(T):
    """Fraction of ocean covered by ice"""
    if T < Talphaocean_low:
        return 1
    elif T < Talphaocean_high:
        return  1 / (Talphaocean_high - Talphaocean_low) * (T - Talphaocean_low)
    else: # so T is higher
        return 0



#
def Ri(T, cearth, tauc):
    """Incoming radiation modified by albedo"""
    return 1/cearth * (Q0 * (1 - p * alphaland - (1 - p) * alphaocean(T)))



# 
def Ro(T, C, cearth, tauc):
    """Outgoing radiation modified by greenhouse effect"""
    return 1/cearth * (kappa * (T - Tkappa) -  a * np.log(C / C0))


def kappaP(T):
    """Solubility of atmospheric carbon into the oceans: Carbon Pumps"""
    np.exp(-bP * (T - T0))





def biopump_vec(CcTemp):
    """Goal: vectorize Function biopump"""
    """Output: Grid Points for Interpolation"""
    def biopump(CcTempW1):
        if CcTempW1 < Cbio_low:
            return 1
        elif CcTempW1 < Cbio_high:
            return 1 - 1 / (Cbio_high - Cbio_low) * (CcTempW1 - Cbio_low)
        else: # so Cc is higher
            return 0
    
    biopump = np.vectorize(biopump)
    biomodulation = [biopump(val) for val in CcTemp]
    biomod = np.float_(biomodulation)
    return biomod



def bioefficiency(t,biomodTemp):
    t_points = t_val
    em_points = biomodTemp
    
    tck = interpolate.splrep(t_points, em_points)
    return interpolate.splev(t,tck)


def oceanatmphysflux(T):
    """Sum of two terms that reflect, respectively, the physical (or solubility) carbon pump in the ocean and Wally """
    """Broecker’s “biopump”, due to thermally enhanced bioproductivity (Fowler et al., 2013)"""
    return 1 / tauc * (coc0 * (np.exp(-bP * (T - T0))))

# def oceanbioflux(T, t, biomodTemp):
#      return 1/tauc * (coc0 * (np.exp(bB * bioefficiency(t, biomodTemp) * (T - T0))))

# def oceanbioflux(T, t, biomodTemp):
#      return 1/tauc * (coc0 * (np.exp(bB  * (T - T0))))

def oceanbioflux(T):
     return 1/tauc * (coc0 * (np.exp(bB  * (T - T0))))




def oceanatmcorrflux(C):
    return 1 / tauc * (- cod * C)


def veggrowth(T):
    """Vegetation growth function"""
    if T < Tlow:
        return 0
    if (T >= Tlow) and (T < Topt1):
        return acc / (Topt1 - Tlow) * (T - Tlow)
    if (T >= Topt1) and (T <= Topt2):
        return acc
    if (T > Topt2) and (T < Thigh):
        #return acc
        return acc / (Topt2 - Thigh) * (T - Thigh)
    if T > Thigh:
        #return acc
        return 0


def veggrowth2(T,G):

    if T < T_vl(G):
        return 0
    if (T > T_vl(G)) and (T < T_ol(G)):
        return acc / (Topt1 - Tlow) * (T - Tlow)
    if (T > T_ol(G)) and (T <= Topt2):
        return acc
    if (T > Topt2) and (T <= Thigh):
        #return acc
        return acc / (Topt2 - Thigh) * (T - Thigh)
    if T > Thigh:
        #return acc
        return 0


def T_ol(G):
    if G<Gl:
        return T_ol_minus
    if (G>Gl) and (G<=Gh):
        return T_ol_minus+(T_ol_plus-T_ol_minus)*(G-Gl)/(Gh-Gl)
    if (G>Gh):
        return T_ol_plus

def T_vl(G):
    if G<Gl:
        return T_vl_minus
    if (G>Gl) and (G<=Gh):
        return T_vl_minus+(T_vl_plus-T_vl_minus)*(G-Gl)/(Gh-Gl)
    if (G>Gh):
        return T_vl_plus



def model(Ts, Cs, cearth, tauc, Ce=np.zeros(t_span)):
    """Input: Starting Temp, CO2, Impulse """
    """Transition: Dynamic Equation"""
    """Output: Whole Array of Temp, Temp Anamoly, CO2"""
    Ce[Ce <0] = 0
    Cc = np.cumsum(Ce)
    # Cc = 1.34*12/44*1000/2.13 + np.cumsum(Ce) 
    biomod = biopump_vec(Cc)

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
        dC -= wa * C * vegcover * veggrowth(T)             # carbon uptake by vegetation
        dC += oceanatmphysflux(T) * (1 - fracseaice(T))    # physical solubility into ocean * fraction of ice-free ocean
        
        
        dC +=  oceanbioflux(T, t, biomod) * (1 - fracseaice(T))
        
        # dC += oceanbioflux(T, t, Cc) * (1 - fracseaice(T))      # biological pump flux * fraction sea ice
        
        
        
        dC += oceanatmcorrflux(C) * (1 - fracseaice(T))    # correction parameter

        return dT, dC

    init = [Ts, Cs]

    t_eval = np.linspace(0, t_span, 10000)

    sol = solve_ivp(dydt, t_eval[[0, -1]], init, t_eval=t_eval, method='RK45', max_step=0.1)

    # sol = solve_ivp(dydt, t_eval[[0, -1]], init, t_eval=t_eval, method='BDF')
    # -

    #Extract values of temperature and CO2
    Tv = sol.y[0, :]
    Cv = sol.y[1, :]
    tv = sol.t


    Tvmid = Tv - 286.7 

    return tv, Tvmid, Cv


def model2(Ts, Cs, Gs, cearth, tauc, Ce=np.zeros(t_span)):
    """Input: Starting Temp, CO2, Impulse """
    """Transition: Dynamic Equation"""
    """Output: Whole Array of Temp, Temp Anamoly, CO2"""
    Ce[Ce <0] = 0
    # Cc = np.cumsum(Ce)
    # Cc = 1.34*12/44*1000/2.13 + np.cumsum(Ce) 
    Cc = Gs/2.13 + np.cumsum(Ce) 
    # biomod = biopump_vec(Cc)

    def dydt(t, y):
        """Transition: Dynamic Equation (3 Input)"""
        """t: time sequence"""
        """y: Starting Temp, CO2"""
        """Cc: Cumulated Carbon **Constant Array** """
        T = y[0]
        C = y[1]
        G = y[2]

        dT = Ri(T, cearth, tauc) 
        dT -= Ro(T, C, cearth, tauc)
    
        dC = V
        dC += Yam(t,Ce) * sa                                  #  anthropogenic emissions from Ca spline                                                # volcanism 
        dC -= wa * C * vegcover * veggrowth2(T,G)             # carbon uptake by vegetation
        dC += oceanatmphysflux(T) * (1 - fracseaice(T))    # physical solubility into ocean * fraction of ice-free ocean
        
        
        dC +=  oceanbioflux(T) * (1 - fracseaice(T))
        
        # dC += oceanbioflux(T, t, Cc) * (1 - fracseaice(T))      # biological pump flux * fraction sea ice
        
        
        
        dC += oceanatmcorrflux(C) * (1 - fracseaice(T))    # correction parameter

        dG = Yam(t,Ce) * sa


        return dT, dC, dG

    init = [Ts, Cs, Cc[0]]

    t_eval = np.linspace(0, t_span, 10000)

    sol = solve_ivp(dydt, t_eval[[0, -1]], init, t_eval=t_eval, method='RK45', max_step=0.1)

    # sol = solve_ivp(dydt, t_eval[[0, -1]], init, t_eval=t_eval, method='BDF')
    # -

    #Extract values of temperature and CO2
    Tv = sol.y[0, :]
    Cv = sol.y[1, :]
    Gv = sol.y[2, :]
    tv = sol.t


    # Tvmid = Tv - 286.7 
    Tvmid = Tv - 286.6

    return tv, Tvmid, Cv, Gv

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

cearth_taucMatrix = [[35., 6603. ],
                     [0.107, 20],
                     [0.373, 30],
                     [10, 1886]]

cearth_taucMatrixSize = len(cearth_taucMatrix)



## Looping
Figure_Dir = "./nonlinearCarbon/figure/"

for ctpathnum in range(cearth_taucMatrixSize):
    figwidth = 10
    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(12, 2 *figwidth))
    # fig, axs = plt.subplots(3, 1, sharex=True, figsize=(12, 2 *figwidth))
    # fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12, 2 *figwidth))
    TvmidBase = np.zeros(10000)

    for pathnum in range(ImpulsePathSize):


        Ce = CeMatrix[pathnum,:]
        cearth, tauc = cearth_taucMatrix[ctpathnum]

        tv, Tvmid, Cv, Gv = model2(Ts, Cs, Gs, cearth, tauc, Ce)

        plotnum = ImpulsePattern*pathnum

        if pathnum ==0:
            TvmidBase = Tvmid

        if pathnum==0:
            axs[0].plot(tv, Tvmid, label="baseline")
        else: 
            axs[0].plot(tv, Tvmid, label=f"CarbonImpulse={CeMatrix[pathnum,plotnum]*2.13}")
        # axs[0].plot(tv, Tvmid, label=f"CarbonImpulse={CeMatrix[pathnum,plotnum]*2.13}")
        axs[0].set_xlabel('Time (year)')
        axs[0].set_ylabel('Temperature  (K)')
        axs[0].set_title('Temperature Anomaly Dynamics')
        axs[0].grid(linestyle=':')
        axs[0].legend()
        if pathnum==0:
            axs[1].plot(tv, Cv, label="baseline")
        else: 
            axs[1].plot(tv, Cv, label=f"CarbonImpulse={CeMatrix[pathnum,plotnum]*2.13}")
        # axs[1].plot(tv, Cv, label=f"CarbonImpulse={CeMatrix[pathnum,plotnum]*2.13}")
        axs[1].set_xlabel('Time (year)')
        axs[1].set_ylabel('Carbon (ppm)')
        axs[1].set_title('Carbon Concentration Dynamics')
        axs[1].grid(linestyle=':')
        axs[1].legend()
        if pathnum==0:
            axs[2].plot(tv, Gv, label="baseline")
            # axs[2].legend()        
        else: 
            axs[2].plot(tv, Gv, label=f"CarbonImpulse={CeMatrix[pathnum,plotnum]*2.13}")
        # axs[2].plot(tv, Gv, label=f"CarbonImpulse={CeMatrix[pathnum,plotnum]*2.13}")
        axs[2].set_xlabel('Time (year)',fontsize = 16)
        axs[2].set_ylabel('Total',fontsize = 16)
        axs[2].set_title('Total Emission')
        axs[2].grid(linestyle=':')
        axs[2].legend()
        if pathnum==0:
            axs[3].plot(tv, Tvmid-TvmidBase, label="baseline")
            # axs[2].legend()        
        else: 
            axs[3].plot(tv, Tvmid-TvmidBase, label=f"CarbonImpulse={CeMatrix[pathnum,plotnum]*2.13}")
        axs[3].set_xlabel('Time (year)')
        axs[3].set_ylabel('Degree Celsius')
        axs[3].set_title('Impulse Response of temperature anomaly per Gigatonne of Carbon')
        axs[3].grid(linestyle=':')
        axs[3].legend()        

        # np.save(f"ImpPtnPath_{ImpulsePattern}_{CeMatrix[pathnum,plotnum]*2.13}_cearth_{cearth}_tauc_{tauc}.npy", [tv, Tvmid, Cv])


    plt.tight_layout()
    plt.savefig(Figure_Dir+f"ImpulsePtn={ImpulsePattern}, cearth={cearth}, tauc={tauc}_new1.pdf")
    plt.savefig(Figure_Dir+f"ImpulsePtn={ImpulsePattern}, cearth={cearth}, tauc={tauc}_new1.png")    
    # plt.savefig(Figure_Dir+"sample_with0.pdf")







##################################################################
## Section 3.3: Model Testing
##################################################################

# TestCode = 2

# Ce_test = np.load("test2.npy")


# ImpulsePathSize = 1
    
# CeMatrix = np.zeros((ImpulsePathSize, t_span))
# CeMatrix[0,:] = Ce_test[0:t_span]



# ## cearth, tauc Path

# cearth_taucMatrix = [# [35., 6603. ],
#                      [0.107, 20]    ]

# cearth_taucMatrixSize = len(cearth_taucMatrix)



# ## Looping

# for ctpathnum in range(cearth_taucMatrixSize):
#     figwidth = 10
#     fig, axs = plt.subplots(1, 2, sharex=True, figsize=(2 * figwidth, 0.5 * figwidth))
#     for pathnum in range(ImpulsePathSize):

#         Ce = CeMatrix[pathnum,:]
#         cearth, tauc = cearth_taucMatrix[ctpathnum]

#         tv, Tvmid, Cv = model(Ts, Cs, cearth, tauc, Ce)

#         axs[0].plot(tv, Tvmid, label=f"ImpPtnPath_{ImpulsePattern}_{CeMatrix[pathnum,0]}_cearth_{cearth}_tauc_{tauc}")
#         axs[0].set_xlabel('Time (year)',fontsize = 16)
#         axs[0].set_ylabel('Temperature (K)',fontsize = 16)
#         axs[0].set_title('Temperature dynamics')
#         axs[0].grid(linestyle=':')
#         axs[0].legend()
#         axs[1].plot(tv, Cv, label=f"ImpPtnPath_{ImpulsePattern}_{CeMatrix[pathnum,0]}_cearth_{cearth}_tauc_{tauc}")
#         axs[1].set_xlabel('Time (year)')
#         axs[1].set_ylabel('Carbon (ppm)')
#         axs[1].set_title('Carbon dynamics')
#         axs[1].grid(linestyle=':')
#         axs[1].legend()
#         np.save(f"TestCode_{TestCode}_{CeMatrix[pathnum,0]}_cearth_{cearth}_tauc_{tauc}.npy", [tv, Tvmid, Cv])

#     plt.tight_layout()
#     plt.savefig(f"TestCode_{TestCode}_cearth_{cearth}_tauc_{tauc}.pdf")


# res = np.load("ImpPtnPath_0_469.4835680751174_cearth_0.107_tauc_20.npy")
# res2 = np.load("ImpPtnPath_0_422.53521126760563_cearth_0.107_tauc_20.npy")

# print(res-res2)