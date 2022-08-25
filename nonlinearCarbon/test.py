#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created on Wed Jan 13 20:48:20 2021

@author: erik
"""

## version including anthropogenic emissions

import os
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

mpl.rcParams["lines.linewidth"] = 1.5
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["figure.figsize"] = (16,10)
mpl.rcParams["font.size"] = 15
mpl.rcParams["legend.frameon"] = False

# os.chdir('/Users/erikchavez/Documents/Economic_Policy/C-T-dynamic-ODEs/')
def model(impulse):
    
    #### INPUT PARAMETERS

    ## heat capacity, incoming radiation
    # Earth heat capacity
    cearth = 0.3725
    #Incoming radiation
    Q0 = 342.5

    ## land fraction and albedo
    #Fraction of land on the planet
    p = 0.3
    # land albedo
    alphaland = 0.28

    ## outgoing radiation linearized
    kappa = 1.74
    T_kappa = 154

    ## CO2 radiative forcing
    # Greenhouse effect parameter
    a = 5.35
    # CO2 params. C0 is the reference C02 level
    C0 = 280

    ## ocean carbon pumps
    # Solubility dependence on temperature (value from Fowler et al)
    bP = 0.029
    # Biopump dependence on temperature (Value from Fowler)
    bB = 0.069
    # Ocean carbon pump modulation parameter
    lamb = 2.2

    ## timescale and reference temperature (from Fowler)
    # timescale 
    tauc = 30
    # Temperature reference
    T0 = 288

    ## Coc0 ocean carbon depending on depth
    coc0 = 330

    ## CO2 uptake by vegetation
    W_a = 0.0056
    #vegcover = 0.4
    T_vh = 307.15
    T_vl = 286.15
    T_ol = 290.15
    T_oh = 302.15
    k = 8

    ## Volcanism
    V = 0.028

    ## Anthropogenic carbon
    # Switch to take anthropogenic emissions
    sa = 1
    # Anthropogenic emissions (zero or one)

    # Biopump modulation


    Gl = 150
    Gh = 750


    tspan = 1000

    Ce = np.arange(tspan) * 1.0
    #np.min(Ca)
    for i in range(len(Ce)):
        if i == 0:
            Ce[i] = impulse/2.13

        else:
            # Ce[i] = Ca[i] - Ca[i-1] 
            Ce[i] = 0

    Cebis = np.arange(tspan) * 1.0
    #np.min(Ca)
    for i in range(len(Cebis)):
        if i == 0:
            Cebis[i] = impulse/2.13
        else:
            # Cebis[i] = max( Ca[i] - Ca[i-1], 0) 
            Cebis[i] = 0

    Cc = np.arange(tspan) * 1.0
    #np.min(Ca)
    for i in range(len(Cc)):
        if i == 0:
            Cc[i] = 0
        else:
            Cc[i] = sum(Cebis[0:i])


    ## Ocean albedo parameters
    Talphaocean_low = 219
    Talphaocean_high = 299
    alphaocean_max = 0.85
    alphaocean_min = 0.25

    ## Biopump parameters


    #### FUNCTIONS
    # Anthropogenic carbon fitting with cubic spline
    t_val = np.linspace(0, tspan-1, tspan)


    def Yam(t):
        t_points = t_val
        em_points = Ce
        
        tck = interpolate.splrep(t_points, em_points)
        return interpolate.splev(t,tck)


    # Ocean albedo
    def alphaocean(T):
        if T < Talphaocean_low:
            return alphaocean_max
        elif T < Talphaocean_high:
            return alphaocean_max + (alphaocean_min - alphaocean_max) / (Talphaocean_high - Talphaocean_low) * (T - Talphaocean_low)
        else: # so T is higher
            return alphaocean_min



    #Fraction of ocean covered by ice
    def fracseaice(T):
        if T < Talphaocean_low:
            return 1
        elif T < Talphaocean_high:
            return 1 - 1 / (Talphaocean_high - Talphaocean_low) * (T - Talphaocean_low)
        else: # so T is higher
            return 0



    # Vegetation growth function


    T_vh = 307.15
    T_vl = 286.15
    T_ol = 290.15
    T_oh = 302.15
    k = 8



    T_ol_minus = T_ol
    T_ol_plus = T_ol + 5


    def T_ol_func(Cc):
        if Cc < Gl:
            return T_ol_minus
        elif Cc < Gh:
            return T_ol_minus + (T_ol_plus - T_ol_minus)/ (Gh - Gl) * (Cc - Gl)
            #return 1 - 2 / (Gh - Gl) * (Cc - Gl)
        else: # so Cc is higher
            return T_ol_plus
            #return -1

    T_ol_func = np.vectorize(T_ol_func)

    Toptmodulation = [T_ol_func(val) for val in Cc]
    Toptmod = np.float_(Toptmodulation)

    def Tvegoptlow(t):
        t_points = t_val
        em_points = Toptmod
        
        tck = interpolate.splrep(t_points, em_points)
        return interpolate.splev(t,tck)



    ## T_vl_func

    T_vl_minus = T_vl
    T_vl_plus = T_vl + 5

    #Gl = 150
    #Gh = 600

    def T_vl_func(Cc):
        if Cc < Gl:
            return T_vl_minus
        elif Cc < Gh:
            return T_vl_minus + (T_vl_plus - T_vl_minus)/ (Gh - Gl) * (Cc - Gl)
            #return 1 - 2 / (Gh - Gl) * (Cc - Gl)
        else: # so Cc is higher
            return T_vl_plus

    T_vl_func = np.vectorize(T_vl_func)

    Tlowmodulation = [T_vl_func(val) for val in Cc]
    Tlowmod = np.float_(Tlowmodulation)

    def Tveglow(t):
        t_points = t_val
        em_points = Tlowmod
        
        tck = interpolate.splrep(t_points, em_points)
        return interpolate.splev(t,tck)

    # t_val_v2 = np.linspace(1800, 2800, 1001)
    # plt.plot(t_val_v2[0:600], Tveglow(t_val)[0:600])
    # plt.tick_params(axis='both', which='major', labelsize=13)
    # plt.xlabel('Time (years)',fontsize = 14);
    # plt.ylabel('Lower veg. lim temperature (K)',fontsize = 14);
    # plt.grid(linestyle=':')

    #def vegetationgrowth(T,t):
    #    return veggrowth(T) * bioefficiency(t)

    # Vegetation growth function
    def veggrowthdyn(T,t):
        if T < Tveglow(t):
            return 0
        if (T >= Tveglow(t)) and (T < Tvegoptlow(t)):
            return k / (Tvegoptlow(t)- Tveglow(t)) * (T - Tveglow(t))
        if (T >= Tvegoptlow(t)) and (T <= T_oh):
            return k
        if (T > T_oh) and (T < T_vh):
            #return k
            return k / (T_oh - T_vh) * (T - T_vh)
        if T > T_vh:
            #return k
            return 0

    #Incoming radiation modified by albedo
    def Ri(T):
        return 1/cearth * (Q0 * (1 - p * alphaland - (1 - p) * alphaocean(T)))



    # Outgoing radiation modified by greenhouse effect
    def Ro(T, C):
        return 1/cearth * (kappa * (T - T_kappa) -  a * np.log(C / C0))


    #Sum of two terms that reflect, respectively, the physical (or solubility) carbon pump in the ocean and Wally Broecker’s “biopump”, due to thermally enhanced bioproductivity (Fowler et al., 2013)

    def oceanatmphysflux(T):
        return 1 / tauc * (coc0 * (np.exp(-bP * (T - T0))))

    def oceanbioflux(T):
        return 1/tauc * (coc0 * (np.exp(bB * (T - T0))))

    def oceanatmcorrflux(C):
        return 1 / tauc * (- lamb * C)

    #def vegflux(T,C,t):
    #    return W_a * bioefficiency(t) * C * vegcover * veggrowth(T)

    #def vegfluxdyn(T,C,t):
    #    return W_a * C * vegcover * veggrowthdyn(T,t)

    def vegfluxdyn(T,C,t):
        return W_a * C * veggrowthdyn(T,t)

    #def oceanbioflux(T,t):
    #    return 1/tauc * (coc0 * (np.exp(bB * bioefficiency(t) * (T - T0))))



    #### MODEL EQUATIONS

    def dydt(t, y):
        T = y[0]
        C = y[1]
        G = y[2]

        dT = Ri(T) 
        dT -= Ro(T, C)
        
        dC = V
        dC += Yam(t) * sa                                  #  anthropogenic emissions from Ca spline                                                # volcanism 
        #dC += Ca * sa                                       # added for bif diagrams
        #dC -= W_a * C * vegcover * veggrowth(T)             # carbon uptake by vegetation
        #dC -= vegflux(T, C, t)
        dC -= vegfluxdyn(T,C,t)
        dC += oceanatmphysflux(T) * (1 - fracseaice(T))    # physical solubility into ocean * fraction of ice-free ocean
        #dC += oceanbioflux(T,t) * (1 - fracseaice(T))      # biological pump flux * fraction sea ice
        dC += oceanbioflux(T) * (1 - fracseaice(T))      # biological pump flux * fraction sea ice
        dC += oceanatmcorrflux(C) * (1 - fracseaice(T))    # correction parameter

        dG = Yam(t) * sa
        return dT, dC, dG


    #Integrate the ODE

    sa = 1
    #Ts = 282.9
    Ts = 286.5
    # Cs = 269
    Cs = 389
    Gs = 0

    alphaland = 0.28
    bP = 0.029
    bB = 0.069
    lamb = 2.2
    #cearth = 35
    cearth = 0.3725
    tauc = 30
    coc0 = 330
    ## Ocean albedo parameters
    Talphaocean_low = 219
    Talphaocean_high = 299
    alphaocean_max = 0.843
    alphaocean_min = 0.254


    #Gl = 50
    #Gh = 700

    T0 = 288
    C0 = 280

    ## CO2 uptake by vegetation
    W_a = 0.0056
    #vegcover = 0.4

    T_vh = 307.15
    T_vl = 286.15
    T_ol = 290.15
    T_oh = 302.15
    k = 8

    #T_vh = 305
    #T_vl = 282
    #T_ol = 295
    #T_oh = 282.5
    #k = 2
    #Cs = C0
    init = [Ts, Cs, Gs]
    t_eval = np.linspace(0, tspan, 100000)
    sol = solve_ivp(dydt, t_eval[[0, -1]], init, t_eval=t_eval, method='RK45', max_step=0.1)

    #sol = solve_ivp(dydt, t_eval[[0, -1]], init, t_eval=t_eval, method='BDF')

    #Extract values of temperature and C02
    Tv = sol.y[0, :]
    Cv = sol.y[1, :]
    Gv = sol.y[2, :]
    tv = sol.t


    #Fixed points
    print('Tp = {:.1f}'.format(Tv[-1]))
    print('Cp = {:.1f}'.format(Cv[-1]))

    Tvmid = Tv - 286.6181299517094
    Cvmid = Cv - 268.6226981649593
    #Tvmid = Tv - 271.0298639974771
    Tvmean = np.mean(Tv) 
    Tvmin = np.min(Tv)
    Tvmax = np.max(Tv) 
    np.mean(Cv) 
    Te = np.arange(tspan) * 1.0
    return Tvmid, Cv, tv, Gv, Te, Cc



Figure_Dir="./nonlinearCarbon/figure/"

figwidth = 10

baseline = 100

# for max in (150, 200, 300, 500, 700):

fig, axs = plt.subplots(4, 1, sharex=True, figsize=(12, 2 *figwidth))
# fig, axs = plt.subplots(3, 1, sharex=True, figsize=(12, 2 *figwidth))
# fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12, 2 *figwidth))
TvmidBase = np.zeros(10000)

pathnum=0

array = np.array((baseline, 150, 200, 300))

for impulse in array:


    Tvmid, Cv, tv, Cebis, Te, Cc = model(impulse)

    print(Cc.shape)
    if pathnum ==0:
        TvmidBase = Tvmid


    if pathnum==0:
        axs[0].plot(tv, Tvmid, label="baseline")
    else: 
        axs[0].plot(tv, Tvmid, label=f"CarbonImpulse={impulse}")
    axs[0].set_xlabel('Time (year)')
    axs[0].set_ylabel('Temperature (K)')
    axs[0].set_title('Temperature Anomaly Dynamics T')
    axs[0].grid(linestyle=':')
    axs[0].legend()

    if pathnum==0:
        axs[1].plot(tv, Cv, label="baseline")
    else: 
        axs[1].plot(tv, Cv, label=f"CarbonImpulse={impulse}")
    axs[1].set_xlabel('Time (year)')
    axs[1].set_ylabel('Carbon (ppm)')
    axs[1].set_title('Carbon Concentration Dynamics C')
    axs[1].grid(linestyle=':')
    axs[1].legend()

    print(tv.shape)
    if pathnum==0:
        axs[2].plot(Te, Cc*2.13, label="baseline")
        # axs[2].legend()        
    else: 
        axs[2].plot(Te, Cc*2.13, label=f"CarbonImpulse={impulse}")
    # axs[2].plot(tv, Gv, label=f"CarbonImpulse={CeMatrix[pathnum,plotnum]*2.13}")
    axs[2].set_xlabel('Time (year)',fontsize = 16)
    axs[2].set_ylabel('Total',fontsize = 16)
    axs[2].set_title('Total Emission Dynamics G')
    axs[2].grid(linestyle=':')
    axs[2].legend()

    if pathnum==0:
        axs[3].plot(tv, Tvmid-TvmidBase, label="baseline")
    else: 
        axs[3].plot(tv, Tvmid-TvmidBase, label=f"CarbonImpulse={impulse}")
    axs[3].set_xlabel('Time (year)')
    axs[3].set_ylabel('Degree Celsius')
    axs[3].set_title('Impulse Response of temperature anomaly per Gigatonne of Carbon')
    axs[3].grid(linestyle=':')
    axs[3].legend()        

    pathnum =  pathnum + 1
    print(pathnum)

plt.tight_layout()
plt.savefig(Figure_Dir+f"Pulse={array[0]},{array[1]},{array[2]},{array[3]}.pdf")
plt.savefig(Figure_Dir+f"Pulse={array[0]},{array[1]},{array[2]},{array[3]}.png")
# plt.savefig(Figure_Dir+"sample_with0.pdf")



