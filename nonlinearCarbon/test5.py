#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created on Wed Jan 13 20:48:20 2021

@author: erik
"""

# version including anthropogenic emissions

import cv2
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
import matplotlib as mpl
import scipy.io as sio
import pandas as pd
import scipy.optimize as optim
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy import fft, arange, signal
# from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

mpl.rcParams["lines.linewidth"] = 1.5
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["figure.figsize"] = (16, 10)
mpl.rcParams["font.size"] = 15
mpl.rcParams["legend.frameon"] = False
# INPUT PARAMETERS


def model(time, length, pulse):
    # heat capacity, incoming radiation
    # Earth heat capacity
    cearth = 0.3725
    # Incoming radiation
    Q0 = 342.5

    # land fraction and albedo
    # Fraction of land on the planet
    p = 0.3
    # land albedo
    alphaland = 0.28

    # outgoing radiation linearized
    kappa = 1.74
    Tkappa = 154

    # CO2 radiative forcing
    # Greenhouse effect parameter
    B = 5.35
    # CO2 params. C0 is the reference C02 level
    C0 = 280

    # ocean carbon pumps
    # Solubility dependence on temperature (value from Fowler et al)
    bP = 0.029
    # Biopump dependence on temperature (Value from Fowler)
    bB = 0.069
    # Ocean carbon pump modulation parameter
    cod = 2.2

    # timescale and reference temperature (from Fowler)
    # timescale
    tauc = 30
    # Temperature reference
    T0 = 288

    # Coc0 ocean carbon depending on depth
    coc0 = 330

    # CO2 uptake by vegetation
    wa = 0.006
    #vegcover = 0.4
    Thigh = 307.15
    Tlow = 286.15
    Topt1 = 286.15
    Topt2 = 302.15
    acc = 8

    # Volcanism
    V = 0.028

    # Anthropogenic carbon
    # Switch to take anthropogenic emissions
    sa = 1
    # Anthropogenic emissions (zero or one)
    file_name = "rcp45co2eqv3"
    # time = 245
    # pulse = 25
    Can = pd.read_csv("./nonlinearCarbon/data/"+file_name+".csv")
    #Can = pd.read_csv("Et-sim2.csv")
    # times2co2eq
    # rcp85co2eq.csv
    #Ca = Can[(Can["YEARS"] > 1899) & (Can["YEARS"] < 2201)]
    #Ca = Can[(Can["YEARS"] > 1799) & (Can["YEARS"] < 2501)]
    Ca = Can[(Can["YEARS"] > 1799) & (Can["YEARS"] < 2801)]
    # Ca = Can[(Can["YEARS"] < 2801)]
    Ca1 = Can[(Can["YEARS"] > 1799) & (Can["YEARS"] < 2801)]
    #Ca["YEARS"] = np.arange(start=0,stop=401,step=1)
    #Ca = Ca.pd.DataFrame()
    Ca = Ca["CO2EQ"]
    #Ca = Ca - 286.76808
    Ca = Ca - 281.69873
    Ca = Ca.to_numpy()

    tspan = len(Ca)

    # #Ce = np.arange(401)
    # # Ce = np.arange(601)
    # # Ce = np.arange(1001) * 1.0
    # Ce = np.zeros(tspan)
    # #np.min(Ca)
    # for i in range(len(Ce)):
    #     if i == 0:
    #         Ce[i] = 0
    #     else:
    #         Ce[i] = Ca[i] - Ca[i-1]

    # Cebis = np.zeros(tspan) * 1.0
    # #np.min(Ca)
    # for i in range(len(Cebis)):
    #     if i == 0:
    #         Cebis[i] = 0
    #     else:
    #         Cebis[i] = max( Ca[i] - Ca[i-1], 0)

    # Cc = np.zeros(tspan) * 1.0
    # #np.min(Ca)
    # for i in range(len(Cc)):
    #     if i == 0:
    #         Cc[i] = 0
    #     else:
    #         Cc[i] = sum(Cebis[0:i])

    #Ce = np.arange(401)
    # Ce = np.arange(601)
    # Ce = np.arange(1001) * 1.0
    Ce = np.zeros(tspan)
    # np.min(Ca)
    for i in range(len(Ce)):
        if i == 0:
            Ce[i] = 0
        elif i >= time and i <= time + length-1:
            Ce[i] = Ca[i] - Ca[i-1] + pulse/2.13/length

        else:
            Ce[i] = Ca[i] - Ca[i-1]

    Cebis = np.zeros(tspan) * 1.0
    # np.min(Ca)
    for i in range(len(Cebis)):
        if i == 0:
            Cebis[i] = 0
        elif i >= time and i <= time + length-1:
            Cebis[i] = np.max(Ca[i] - Ca[i-1], 0) + pulse/2.13/length
        else:
            Cebis[i] = np.max(Ca[i] - Ca[i-1], 0)

    Cc = np.zeros(tspan) * 1.0
    # np.min(Ca)
    for i in range(len(Cc)):
        if i == 0:
            Cc[i] = 0
        else:
            Cc[i] = sum(Cebis[0:i])

    # Ocean albedo parameters
    Talphaocean_low = 219
    Talphaocean_high = 299
    alphaocean_max = 0.85
    alphaocean_min = 0.25

    # Biopump parameters
    #Cbio_low = 100
    #Cbio_high = 700

    # FUNCTIONS

    # Anthropogenic carbon fitting with cubic spline
    t_val = np.linspace(0, tspan-1, tspan)
    #t_val = np.linspace(0, 700, 701)
    # t_val = np.linspace(0, 1000, 1001)

    def Yem(t):
        t_points = t_val
        em_points = Ca

        tck = interpolate.splrep(t_points, em_points)
        return interpolate.splev(t, tck)

    def Yam(t):
        t_points = t_val
        em_points = Ce

        tck = interpolate.splrep(t_points, em_points)
        return interpolate.splev(t, tck)

    def Ycm(t):
        t_points = t_val
        em_points = Cc

        tck = interpolate.splrep(t_points, em_points)
        return interpolate.splev(t, tck)

    #plt.plot(t_val, Ca)
    #plt.plot(t_val, Yem(t_val))

    #plt.plot(t_val, Ce)
    #plt.plot(t_val, Yam(t_val))

    # plt.plot(t_val, Cc)
    # plt.plot(t_val, Ycm(t_val))

    #t_val_v2 = np.linspace(1800, 2800, 1001)
    #tv2 = np.linspace(1800, 2800, 100000)
    #Ycm30 = Ycm(t_val_v2)

    #plt.figure(figsize=(10, 5))
    #plt.plot(tv, Tvmid)

    #plt.plot(t_val_v2[0:600], Ycm[0:600])
    #plt.plot(t_val_v2[0:600], Ycm60[0:600])
    #plt.plot(t_val_v2[0:600], Ycm45[0:600])
    #plt.plot(t_val_v2[0:600], Ycm30[0:600])

    #plt.tick_params(axis='both', which='major', labelsize=18)
    #plt.xlabel('Time (year)',fontsize = 18);
    #plt.ylabel('Temperature annomaly (K)',fontsize = 18);
    # plt.grid(linestyle=':')

    #t_val_v2 = np.linspace(1800, 2800, 1001)
    #plt.plot(t_val_v2[0:600], Yam(t_val)[0:600])
    #plt.tick_params(axis='both', which='major', labelsize=13)
    #plt.xlabel('Time (years)',fontsize = 14);
    #plt.ylabel('Annual emissions (ppm/year)',fontsize = 14);
    # plt.grid(linestyle=':')

    # Ocean albedo
    def alphaocean(T):
        if T < Talphaocean_low:
            return alphaocean_max
        elif T < Talphaocean_high:
            return alphaocean_max + (alphaocean_min - alphaocean_max) / (Talphaocean_high - Talphaocean_low) * (T - Talphaocean_low)
        else:  # so T is higher
            return alphaocean_min

    # T_values = np.linspace(200, 330, 201)
    # plt.plot(T_values, [alphaocean(val) for val in T_values])
    # plt.tick_params(axis='both', which='major', labelsize=13)
    # plt.grid(linestyle=':')
    # plt.xlabel('Temperature (K)',fontsize = 14);
    # plt.ylabel('Ocean albedo',fontsize = 14);

    # Fraction of ocean covered by ice

    def fracseaice(T):
        if T < Talphaocean_low:
            return 1
        elif T < Talphaocean_high:
            return 1 - 1 / (Talphaocean_high - Talphaocean_low) * (T - Talphaocean_low)
        else:  # so T is higher
            return 0

    # plt.plot(T_values, [fracseaice(val) for val in T_values])
    # plt.tick_params(axis='both', which='major', labelsize=13)
    # plt.xlabel('Temperature (K)',fontsize = 14);
    # plt.ylabel('Fraction of sea ice',fontsize = 14);
    # plt.grid(linestyle=':')

    # Biopump modulation

    #Cbio_low = 150
    #Cbio_high = 615

    Cbio_low = 150
    Cbio_high = 750

    # def biopump(Cc):
    #    if Cc < Cbio_low:
    #        return 1
    #    elif Cc < Cbio_high:
    #        return 1 - 1 / (Cbio_high - Cbio_low) * (Cc - Cbio_low)
    # return 1 - 2 / (Cbio_high - Cbio_low) * (Cc - Cbio_low)
    #    else: # so Cc is higher
    #        return 0
    #        #return -1

    #biopump = np.vectorize(biopump)

    #Cc_values = np.linspace(0, 1500, 201)
    #plt.plot(Cc_values, [biopump(val) for val in Cc_values])
    #plt.tick_params(axis='both', which='major', labelsize=13)
    #plt.xlabel('Cumulative anthropogenic emissions (ppm)',fontsize = 14);
    #plt.ylabel('Bio pump efficiency',fontsize = 14);
    # plt.grid(linestyle=':')

    # Time varying biopump modulation

    #plt.plot(t_val, [biopump(val) for val in Ca])
    #plt.tick_params(axis='both', which='major', labelsize=13)
    #plt.xlabel('Time (years)',fontsize = 14);
    #plt.ylabel('Bio pump efficiency',fontsize = 14);
    # plt.grid(linestyle=':')

    # switch between Ca and Cc to allow for reversal of ocean acidity or locking it

    #biomodulation = [biopump(val) for val in Cc]
    #biomod = np.float_(biomodulation)

    # def bioefficiency(t):
    #    t_points = t_val
    #    em_points = biomod

    #    tck = interpolate.splrep(t_points, em_points)
    #    return interpolate.splev(t,tck)

    #plt.plot(t_val, bioefficiency(t_val))
    #plt.tick_params(axis='both', which='major', labelsize=18)
    #plt.xlabel('Time (years)',fontsize = 18);
    #plt.ylabel('Bio pump efficiency',fontsize = 18);
    # plt.grid(linestyle=':')

    #t_val_v2 = np.linspace(1800, 2800, 1001)
    #plt.plot(t_val_v2, bioefficiency(t_val))
    #plt.tick_params(axis='both', which='major', labelsize=18)
    #plt.xlabel('Time (years)',fontsize = 18);
    #plt.ylabel('Bio efficiency',fontsize = 18);
    # plt.grid(linestyle=':')

    #t_val_v2 = np.linspace(1800, 2800, 1001)
    #plt.plot(t_val_v2[0:600], bioefficiency(t_val)[0:600])
    #plt.tick_params(axis='both', which='major', labelsize=18)
    #plt.xlabel('Time (years)',fontsize = 18);
    #plt.ylabel('Bio pump efficiency',fontsize = 18);
    # plt.grid(linestyle=':')

    #t_val_v3 = np.linspace(1, 1, 1001)
    #plt.plot(t_val_v2[0:600], t_val_v3[0:600])
    #plt.tick_params(axis='both', which='major', labelsize=18)
    #plt.xlabel('Time (years)',fontsize = 18);
    #plt.ylabel('Bio pump efficiency',fontsize = 18);
    # plt.grid(linestyle=':')

    # Vegetation growth function

    def veggrowth(T):
        if T < Tlow:
            return 0
        if (T >= Tlow) and (T < Topt1):
            return acc / (Topt1 - Tlow) * (T - Tlow)
        if (T >= Topt1) and (T <= Topt2):
            return acc
        if (T > Topt2) and (T < Thigh):
            # return acc
            return acc / (Topt2 - Thigh) * (T - Thigh)
        if T >= Thigh:
            # return acc
            return 0

    Thigh = 307.15
    Tlow = 286.15
    Topt1 = 290.15
    Topt2 = 302.15
    acc = 8

    # T_values = np.linspace(280, 315, 201)
    # plt.plot(T_values, [veggrowth(val) for val in T_values])
    # plt.tick_params(axis='both', which='major', labelsize=13)
    # plt.xlabel('Temperature (K)',fontsize = 14);
    # plt.ylabel('Vegetation growth',fontsize = 14);
    # plt.grid(linestyle=':')

    # veggrowth(286.6181299517094)
    #veggrowth_vectorized = np.vectorize(veggrowth)
    # veggrowth_vectorized(Tv)

    # veggrowth(Tvmean)
    # veggrowth(Tvmax)
    # veggrowth(Tvmin)
    # veggrowth(Tv)

    Tbiopt1_low = Topt1
    Tbiopt1_high = Topt1 + 5

    #Cbio_low = 150
    #Cbio_high = 600

    def Tbioptlow(Cc):
        if Cc < Cbio_low:
            return Tbiopt1_low
        elif Cc < Cbio_high:
            return Tbiopt1_low + (Tbiopt1_high - Tbiopt1_low) / (Cbio_high - Cbio_low) * (Cc - Cbio_low)
            # return 1 - 2 / (Cbio_high - Cbio_low) * (Cc - Cbio_low)
        else:  # so Cc is higher
            return Tbiopt1_high
            # return -1

    Tbioptlow = np.vectorize(Tbioptlow)

    # Cc_values = np.linspace(0, 1500, 201)
    # plt.plot(Cc_values, [Tbioptlow(val) for val in Cc_values])
    # plt.tick_params(axis='both', which='major', labelsize=13)
    # plt.xlabel('Cumulative anthropogenic emissions (ppm)',fontsize = 14);
    # plt.ylabel('Veg. lower opt. temperature (K)',fontsize = 14);
    # plt.grid(linestyle=':')

    # Time varying biopump modulation

    #plt.plot(t_val, [biopump(val) for val in Ca])
    #plt.tick_params(axis='both', which='major', labelsize=13)
    #plt.xlabel('Time (years)',fontsize = 14);
    #plt.ylabel('Bio pump efficiency',fontsize = 14);
    # plt.grid(linestyle=':')

    # switch between Ca and Cc to allow for reversal of ocean acidity or locking it

    Toptmodulation = [Tbioptlow(val) for val in Cc]
    Toptmod = np.float_(Toptmodulation)

    def Tvegoptlow(t):
        t_points = t_val
        em_points = Toptmod

        tck = interpolate.splrep(t_points, em_points)
        return interpolate.splev(t, tck)

    # t_val_v2 = np.linspace(1800, 2800, 1001)
    # plt.plot(t_val_v2[0:600], Tvegoptlow(t_val)[0:600])
    # plt.tick_params(axis='both', which='major', labelsize=13)
    # plt.xlabel('Time (years)',fontsize = 14);
    # plt.ylabel('Lower opt. temperature (K)',fontsize = 14);
    # plt.grid(linestyle=':')

    # Tbiolow

    Tbiolow_low = Tlow
    Tbiolow_high = Tlow + 5

    #Cbio_low = 150
    #Cbio_high = 600

    def Tbiolow(Cc):
        if Cc < Cbio_low:
            return Tbiolow_low
        elif Cc < Cbio_high:
            return Tbiolow_low + (Tbiolow_high - Tbiolow_low) / (Cbio_high - Cbio_low) * (Cc - Cbio_low)
            # return 1 - 2 / (Cbio_high - Cbio_low) * (Cc - Cbio_low)
        else:  # so Cc is higher
            return Tbiolow_high
            # return -1

    Tbiolow = np.vectorize(Tbiolow)

    # Cc_values = np.linspace(0, 1500, 201)
    # plt.plot(Cc_values, [Tbiolow(val) for val in Cc_values])
    # plt.tick_params(axis='both', which='major', labelsize=13)
    # plt.xlabel('Cumulative anthropogenic emissions (ppm)',fontsize = 14);
    # plt.ylabel('Lower veg. lim temperature (K)',fontsize = 14);
    # plt.grid(linestyle=':')

    # Time varying biopump modulation

    #plt.plot(t_val, [biopump(val) for val in Ca])
    #plt.tick_params(axis='both', which='major', labelsize=13)
    #plt.xlabel('Time (years)',fontsize = 14);
    #plt.ylabel('Bio pump efficiency',fontsize = 14);
    # plt.grid(linestyle=':')

    # switch between Ca and Cc to allow for reversal of ocean acidity or locking it

    Tlowmodulation = [Tbiolow(val) for val in Cc]
    Tlowmod = np.float_(Tlowmodulation)

    def Tveglow(t):
        t_points = t_val
        em_points = Tlowmod

        tck = interpolate.splrep(t_points, em_points)
        return interpolate.splev(t, tck)

    # t_val_v2 = np.linspace(1800, 2800, 1001)
    # plt.plot(t_val_v2[0:600], Tveglow(t_val)[0:600])
    # plt.tick_params(axis='both', which='major', labelsize=13)
    # plt.xlabel('Time (years)',fontsize = 14);
    # plt.ylabel('Lower veg. lim temperature (K)',fontsize = 14);
    # plt.grid(linestyle=':')

    # def vegetationgrowth(T,t):
    #    return veggrowth(T) * bioefficiency(t)

    # Vegetation growth function
    def veggrowthdyn(T, t):
        if T < Tveglow(t):
            return 0
        if (T >= Tveglow(t)) and (T < Tvegoptlow(t)):
            return acc / (Tvegoptlow(t) - Tveglow(t)) * (T - Tveglow(t))
        if (T >= Tvegoptlow(t)) and (T <= Topt2):
            return acc
        if (T > Topt2) and (T < Thigh):
            # return acc
            return acc / (Topt2 - Thigh) * (T - Thigh)
        if T > Thigh:
            # return acc
            return 0

    # Incoming radiation modified by albedo
    def Ri(T):
        return 1/cearth * (Q0 * (1 - p * alphaland - (1 - p) * alphaocean(T)))

    #plt.plot(T_values, [Ri(val) for val in T_values])
    #plt.xlabel('Temperature (K)')
    # plt.grid(linestyle=':')

    # Outgoing radiation modified by greenhouse effect
    def Ro(T, C):
        return 1/cearth * (kappa * (T - Tkappa) - B * np.log(C / C0))

    # Solubility of atmospheric carbon into the oceans
    # carbon pumps
    # def kappaP(T):
    #    np.exp(-bP * (T - T0))

    # def kappaB(T):
    #    np.exp(bB * (T - T0))

    # Sum of two terms that reflect, respectively, the physical (or solubility) carbon pump in the ocean and Wally Broecker’s “biopump”, due to thermally enhanced bioproductivity (Fowler et al., 2013)

    def oceanatmphysflux(T):
        return 1 / tauc * (coc0 * (np.exp(-bP * (T - T0))))

    def oceanbioflux(T):
        return 1/tauc * (coc0 * (np.exp(bB * (T - T0))))

    def oceanatmcorrflux(C):
        return 1 / tauc * (- cod * C)

    # def vegflux(T,C,t):
    #    return wa * bioefficiency(t) * C * vegcover * veggrowth(T)

    # def vegfluxdyn(T,C,t):
    #    return wa * C * vegcover * veggrowthdyn(T,t)

    def vegfluxdyn(T, C, t):
        return wa * C * veggrowthdyn(T, t)

    # def oceanbioflux(T,t):
    #    return 1/tauc * (coc0 * (np.exp(bB * bioefficiency(t) * (T - T0))))

    # MODEL EQUATIONS

    def dydt(t, y):
        T = y[0]
        C = y[1]

        dT = Ri(T)
        dT -= Ro(T, C)

        dC = V
        # anthropogenic emissions from Ca spline                                                # volcanism
        dC += Yam(t) * sa
        # dC += Ca * sa                                       # added for bif diagrams
        # dC -= wa * C * vegcover * veggrowth(T)             # carbon uptake by vegetation
        #dC -= vegflux(T, C, t)
        dC -= vegfluxdyn(T, C, t)
        # physical solubility into ocean * fraction of ice-free ocean
        dC += oceanatmphysflux(T) * (1 - fracseaice(T))
        # dC += oceanbioflux(T,t) * (1 - fracseaice(T))      # biological pump flux * fraction sea ice
        # biological pump flux * fraction sea ice
        dC += oceanbioflux(T) * (1 - fracseaice(T))
        dC += oceanatmcorrflux(C) * (1 - fracseaice(T)
                                     )    # correction parameter

        return dT, dC

    # Integrate the ODE

    sa = 1
    #Ts = 282.9
    Ts = 286.5
    Cs = 269

    #wa = 0.05
    #cod = 0.15
    alphaland = 0.28
    bP = 0.029
    bB = 0.069
    cod = 2.2
    #cearth = 35
    cearth = 0.3725
    tauc = 30
    coc0 = 330
    # Ocean albedo parameters
    Talphaocean_low = 219
    Talphaocean_high = 299
    alphaocean_max = 0.843
    alphaocean_min = 0.254

    #Cbio_low = 50
    #Cbio_high = 700

    T0 = 288
    C0 = 280

    # CO2 uptake by vegetation
    wa = 0.006
    #vegcover = 0.4

    Thigh = 307.15
    Tlow = 286.15
    Topt1 = 290.15
    Topt2 = 302.15
    acc = 8

    #Thigh = 305
    #Tlow = 282
    #Topt1 = 295
    #Topt2 = 282.5
    #acc = 2
    #Cs = C0
    init = [Ts, Cs]
    t_eval = np.linspace(0, tspan, 100000)
    sol = solve_ivp(dydt, t_eval[[0, -1]], init,
                    t_eval=t_eval, method='RK45', max_step=0.1)

    #sol = solve_ivp(dydt, t_eval[[0, -1]], init, t_eval=t_eval, method='BDF')

    # Extract values of temperature and C02
    Tv = sol.y[0, :]
    Cv = sol.y[1, :]
    tv = sol.t

    # Fixed points
    print('Tp = {:.1f}'.format(Tv[-1]))
    print('Cp = {:.1f}'.format(Cv[-1]))

    Tvmid = Tv - 286.6181299517094
    Cvmid = Cv - 268.6226981649593
    #Tvmid = Tv - 271.0298639974771
    Tvmean = np.mean(Tv)
    Tvmin = np.min(Tv)
    Tvmax = np.max(Tv)
    np.mean(Cv)

    T = np.arange(tspan)*1.0

    return tv, Tvmid, Cv, T, Cc, file_name


Figure_Dir = "./nonlinearCarbon/figure/"

timearray = np.array((230, 240, 245, 250))
# maxarray = np.array((10, 25, 50, 75, 100, 150, 200))
maxarray2 = np.array((10, 25, 50, 75, 100, 150, 200))
# maxarray2 = np.array((150, 200))
lengtharray = np.array((1, 5, 10))
# maxarray = np.array((10, 25))

# time =245


for max in maxarray2:
    # max = 10
    baseline = 0

    figwidth = 10

    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(12, 2 * figwidth))

    pathnum = 0

    time = 210
    tvBase, TvmidBase, CvBase, TeBase, CcBase, file_name = model(time, 1, 0)

    TeBase = TeBase + 1800
    tvBase = tvBase + 1800

    axs[0].plot(tvBase, TvmidBase, label="baseline")
    axs[0].set_xlabel('Time (year)')
    axs[0].set_ylabel('Temperature (K)')
    axs[0].set_title('Temperature Anomaly Dynamics T')
    axs[0].grid(linestyle=':')
    axs[0].legend()

    axs[1].plot(tvBase, CvBase, label="baseline")
    axs[1].set_xlabel('Time (year)')
    axs[1].set_ylabel('Carbon (ppm)')
    axs[1].set_title('Carbon Concentration Dynamics C')
    axs[1].grid(linestyle=':')
    axs[1].legend()

    axs[2].plot(TeBase, CcBase*2.13, label="baseline")
    axs[2].set_xlabel('Time (year)', fontsize=16)
    axs[2].set_ylabel('Total (Gigatonne)', fontsize=16)
    axs[2].set_title('Total Emission Dynamics G')
    axs[2].grid(linestyle=':')
    axs[2].legend()

    axs[3].plot(tvBase, TvmidBase-TvmidBase, label="baseline")
    axs[3].set_xlabel('Time (year)')
    axs[3].set_ylabel('Degree Celsius')
    axs[3].set_title(
        'Impulse Response of temperature anomaly per Gigatonne of Carbon')
    axs[3].grid(linestyle=':')
    axs[3].legend()

    plt.tight_layout()
    plt.savefig(Figure_Dir+"Pulse="+file_name+".pdf")
    plt.savefig(Figure_Dir+"Pulse="+file_name+".png")

    pathnum = pathnum + 1
    print(pathnum)

    for length in lengtharray:

        tv, Tvmid, Cv, Te, Cc, file_name = model(time, length, max)
        year = time + 1800

        Te = Te + 1800
        tv = tv + 1800

        axs[0].plot(tv, Tvmid, label=f"Impulse={max},Length={length}")
        axs[0].set_xlabel('Time (year)')
        axs[0].set_ylabel('Temperature (K)')
        axs[0].set_title('Temperature Anomaly Dynamics T')
        axs[0].grid(linestyle=':')
        axs[0].legend()

        axs[1].plot(tv, Cv, label=f"Impulse={max},length={length}")
        axs[1].set_xlabel('Time (year)')
        axs[1].set_ylabel('Carbon (ppm)')
        axs[1].set_title('Carbon Concentration Dynamics C')
        axs[1].grid(linestyle=':')
        axs[1].legend()

        axs[2].plot(Te, Cc*2.13, label=f"Impulse={max},length={length}")
        axs[2].set_xlabel('Time (year)', fontsize=16)
        axs[2].set_ylabel('Total (Gigatonne)', fontsize=16)
        axs[2].set_title('Total Emission Dynamics G')
        axs[2].grid(linestyle=':')
        axs[2].legend()

        axs[3].plot(tv, Tvmid-TvmidBase,
                    label=f"Impulse={max},length={length}")
        axs[3].set_xlabel('Time (year)')
        axs[3].set_ylabel('Degree Celsius')
        axs[3].set_title(
            'Impulse Response of temperature anomaly per Gigatonne of Carbon')
        axs[3].grid(linestyle=':')
        axs[3].legend()

        pathnum = pathnum + 1
        print(pathnum)

    plt.tight_layout()
    plt.savefig(Figure_Dir+"Pulse="+file_name+",pulseyear="+str(time+1800) +
                ",pulselength="+str(lengtharray)+",pulsesize="+str(max)+".pdf")
    plt.savefig(Figure_Dir+"Pulse="+file_name+",pulseyear="+str(time+1800) +
                ",pulselength="+str(lengtharray)+",pulsesize="+str(max)+".jpg")
    plt.savefig(Figure_Dir+"Pulse="+file_name+",pulseyear="+str(time+1800) +
                ",pulselength="+str(lengtharray)+",pulsesize="+str(max)+".png")
    plt.close()


# code for displaying multiple images in one figure

#import libraries

# create figure
fig = plt.figure(figsize=(100, 25))

# setting values to rows and column variables
rows = 1
columns = len(maxarray2)

# reading images

num = 0

for max in maxarray2:
    Image = cv2.imread(Figure_Dir+"Pulse="+file_name+",pulseyear="+str(time+1765) +
                       ",pulselength="+str(lengtharray)+",pulsesize="+str(max)+".png")
    Image = np.flip(Image, axis=-1)
    fig.add_subplot(rows, columns, num+1)
    plt.imshow(Image)
    plt.axis('off')
    plt.title(f"Carbon Impulse={max}")
    num = num + 1

plt.savefig(Figure_Dir+"Pulse="+file_name+",pulselength=" +
            str(lengtharray)+",back2back.png")
plt.close()


# Figure_Dir="./nonlinearCarbon/figure/"

# figwidth = 10
# # fig, axs = plt.subplots(4, 1, sharex=True, figsize=(12, 2 *figwidth))
# # fig, axs = plt.subplots(3, 1, sharex=True, figsize=(12, 2 *figwidth))
# fig, axs = plt.subplots(3, 1, sharex=True, figsize=(12, 2 *figwidth))
# TvmidBase = np.zeros(10000)


# pathnum=0

# axs[0].plot(tv, Tvmid, label="baseline")

# axs[0].set_xlabel('Time (year)')
# axs[0].set_ylabel('Temperature (K)')
# axs[0].set_title('Temperature Anomaly Dynamics T')
# axs[0].grid(linestyle=':')
# axs[0].legend()

# axs[1].plot(tv, Cv, label="baseline")
# axs[1].set_xlabel('Time (year)')
# axs[1].set_ylabel('Carbon (ppm)')
# axs[1].set_title('Carbon Concentration Dynamics C')
# axs[1].grid(linestyle=':')
# axs[1].legend()


# #Plot results
# #Time series

# #t_val_years = np.linspace(1800, 2800, 100000)
# #tv2 = np.linspace(1397, 2399, 100000)

# #plt.figure(figsize=(10, 5))
# #plt.plot(tv, Tvmid)
# #plt.plot(tv2[35229:99999], Tvmid[35229:99999])
# #plt.tick_params(axis='both', which='major', labelsize=18)
# #plt.xlabel('Time (year)',fontsize = 18);
# #plt.ylabel('Temperature annomaly (K)',fontsize = 18);
# #plt.grid(linestyle=':')


# #figwidth = 10
# #fig, axs = plt.subplots(1, 2, sharex=True, figsize=(2 * figwidth, 0.5 * figwidth))
# #axs[0].plot(tv, Tvmid)
# #axs[0].set_xlabel('Time (year)',fontsize = 16)
# #axs[0].set_ylabel('Temperature (K)',fontsize = 16)
# #axs[0].set_title('Temperature dynamics')
# #axs[0].grid(linestyle=':')
# #axs[1].plot(tv, Cv)
# #axs[1].set_xlabel('Time (year)')
# #axs[1].set_ylabel('Carbon (ppm)')
# #axs[1].set_title('Carbon dynamics');
# #axs[1].grid(linestyle=':')


# #Tv85 = Tv
# #Cv85 = Cv
# #tv85 = tv
# #Tvmid85 = Tvmid
# #Cvmid85 = Cvmid


# tv2 = np.linspace(1800, 2800, 100000)

# plt.figure(figsize=(10, 5))
# #plt.plot(tv, Tvmid)
# plt.plot(tv2[0:59999], Tvmid[0:59999])
# #plt.plot(tv2[0:59999], Tvmidv3[0:59999])
# plt.tick_params(axis='both', which='major', labelsize=18)
# plt.xlabel('Time (year)',fontsize = 18);
# plt.ylabel('Temperature annomaly (K)',fontsize = 18);
# plt.grid(linestyle=':')


# tv2 = np.linspace(1800, 2800, 100000)

# plt.figure(figsize=(10, 5))
# #plt.plot(tv, Tvmid)
# plt.plot(tv2[0:5999], Tvmid[0:5999])
# #plt.plot(tv2[0:59999], Tvmidv3[0:59999])
# plt.tick_params(axis='both', which='major', labelsize=18)
# plt.xlabel('Time (year)',fontsize = 18);
# plt.ylabel('Temperature annomaly (K)',fontsize = 18);
# plt.grid(linestyle=':')


# tv2 = np.linspace(1800, 2800, 100000)

# plt.figure(figsize=(10, 5))
# #plt.plot(tv, Tvmid)
# plt.plot(tv2[0:59999], Tv[0:59999])
# #plt.plot(tv2[0:59999], Tvmidv3[0:59999])
# plt.tick_params(axis='both', which='major', labelsize=18)
# plt.xlabel('Time (year)',fontsize = 18);
# plt.ylabel('Temperature annomaly (K)',fontsize = 18);
# plt.grid(linestyle=':')

# Tvmidv5 = Tvmid[0:59999]


# tv2 = np.linspace(1800, 2800, 100000)

# plt.figure(figsize=(10, 5))
# #plt.plot(tv, Tvmid)

# plt.plot(tv2[0:59999], Tvmid[0:59999])

# plt.plot(tv2[0:59999], Tvmidv3[0:59999])
# plt.plot(tv2[0:59999], Tvmidv4[0:59999])
# plt.plot(tv2[0:59999], Tvmidv5[0:59999])

# plt.tick_params(axis='both', which='major', labelsize=18)
# plt.xlabel('Time (year)',fontsize = 18);
# plt.ylabel('Temperature annomaly (K)',fontsize = 18);
# plt.grid(linestyle=':')

# #os.chdir('/Users/erikchavez/Documents/Economic_Policy/Figures/')
# #plt.savefig('temp-model5-allscenarios.pdf')


# Tcel = Tv - 273.15
# tv2 = np.linspace(1800, 2800, 100000)

# plt.figure(figsize=(10, 5))
# #plt.plot(tv, Tvmid)
# plt.plot(tv2[0:59999], Tcel[0:59999])
# plt.tick_params(axis='both', which='major', labelsize=18)
# plt.xlabel('Time (year)',fontsize = 18);
# plt.ylabel('Temperature annomaly (C)',fontsize = 18);
# plt.grid(linestyle=':')


# plt.figure(figsize=(10, 5))
# #plt.plot(tv, Tvmid)
# plt.plot(tv2[0:5999], Tcel[0:5999])
# plt.tick_params(axis='both', which='major', labelsize=18)
# plt.xlabel('Time (year)',fontsize = 18);
# plt.ylabel('Temperature annomaly (C)',fontsize = 18);
# plt.grid(linestyle=':')


# tv2 = np.linspace(1800, 2800, 100000)

# plt.figure(figsize=(10, 5))
# #plt.plot(tv, Tvmid)
# plt.plot(tv2[0:59999], Cv[0:59999])
# plt.tick_params(axis='both', which='major', labelsize=18)
# plt.xlabel('Time (year)',fontsize = 18);
# plt.ylabel('Carbon concentration (ppm)',fontsize = 18);
# plt.grid(linestyle=':')


# veggrowth_vectorized = np.vectorize(veggrowth)
# veggrowthcoeff = veggrowth_vectorized(Tv)

# plt.plot(tv2, veggrowth_vectorized(Tv))


# plt.figure(figsize=(10, 5))
# #plt.plot(tv, Tvmid)
# plt.plot(tv2[0:59999], veggrowthcoeff[0:59999])
# plt.tick_params(axis='both', which='major', labelsize=18)
# plt.xlabel('Time (year)',fontsize = 18);
# plt.ylabel('Vegetegation grwoth factor',fontsize = 18);
# plt.grid(linestyle=':')


# vegfluxdyn_vectorized = np.vectorize(vegfluxdyn)
# vegfluxdyn_values = vegfluxdyn_vectorized(Tv, Cv, tv2)

# oceanatmphysflux_vectorized = np.vectorize(oceanatmphysflux)
# oceanatmphysflux_values = oceanatmphysflux_vectorized(Tv)

# fracseaice_vectorized = np.vectorize(fracseaice)
# fracseaice_values = fracseaice_vectorized(Tv)

# oceanbioflux_vectorized = np.vectorize(oceanbioflux)
# oceanbioflux_values = oceanbioflux_vectorized(Tv)

# oceanatmcorrflux_vectorized = np.vectorize(oceanatmcorrflux)
# oceanatmcorrflux_values = oceanatmcorrflux_vectorized(Cv)

# lis = [oceanbioflux_values, oceanatmphysflux_values, oceanatmcorrflux_values]
# total_ocean_flux = [sum(x) for x in zip(*lis)]


# lis_full = [vegfluxdyn_values, total_ocean_flux]
# ratio_flux = [i / j for i, j in zip(*lis_full)]

# Fo_mean = np.mean(total_ocean_flux[0:59999])
# Fv_mean = np.mean(vegfluxdyn_values[0:59999])

# fig = plt.figure(figsize=(10, 5))
# #plt.plot(tv, Tvmid)
# plt.plot(tv2[0:59999], total_ocean_flux[0:59999], color = 'b')
# plt.plot(tv2[0:59999], vegfluxdyn_values[0:59999], color ='orange')
# #plt.axhline(y = Fo_mean, color = 'b', linestyle = '--')
# #plt.axhline(y = Fv_mean, color = 'orange', linestyle = '--')
# plt.tick_params(axis='both', which='major', labelsize=18)
# plt.xlabel('Time (year)',fontsize = 18);
# plt.ylabel('Carbon flux (ppm/yr)',fontsize = 18);
# plt.grid(linestyle=':')
# os.chdir('/Users/erikchavez/Documents/Economic_Policy/Figures/')
# fig.savefig('ocean-vegetation-fluxes.pdf')


# fig = plt.figure(figsize=(10, 5))
# #plt.plot(tv, Tvmid)
# plt.plot(tv2[0:59999], total_ocean_flux[0:59999], color = 'b')
# plt.tick_params(axis='both', which='major', labelsize=18)
# plt.xlabel('Time (year)',fontsize = 18);
# plt.ylabel('Carbon flux (ppm/yr)',fontsize = 18);
# plt.grid(linestyle=':')
# os.chdir('/Users/erikchavez/Documents/Economic_Policy/Figures/')
# fig.savefig('ocean-flux.pdf')


# fig = plt.figure(figsize=(10, 5))
# #plt.plot(tv, Tvmid)
# plt.plot(tv2[0:59999], vegfluxdyn_values[0:59999], color ='orange')
# plt.tick_params(axis='both', which='major', labelsize=18)
# plt.xlabel('Time (year)',fontsize = 18);
# plt.ylabel('Carbon flux (ppm/yr)',fontsize = 18);
# plt.grid(linestyle=':')
# os.chdir('/Users/erikchavez/Documents/Economic_Policy/Figures/')
# fig.savefig('vegetation-flux.pdf')


# fig = plt.figure(figsize=(10, 5))
# #plt.plot(tv, Tvmid)
# plt.plot(tv2[0:5999], total_ocean_flux[0:5999], color = 'b')
# plt.tick_params(axis='both', which='major', labelsize=18)
# plt.xlabel('Time (year)',fontsize = 18);
# plt.ylabel('Carbon flux (ppm/yr)',fontsize = 18);
# plt.grid(linestyle=':')
# os.chdir('/Users/erikchavez/Documents/Economic_Policy/Figures/')
# fig.savefig('ocean-flux-zoom.pdf')


# fig = plt.figure(figsize=(10, 5))
# #plt.plot(tv, Tvmid)
# plt.plot(tv2[0:5999], vegfluxdyn_values[0:5999], color ='orange')
# plt.tick_params(axis='both', which='major', labelsize=18)
# plt.xlabel('Time (year)',fontsize = 18);
# plt.ylabel('Carbon flux (ppm/yr)',fontsize = 18);
# plt.grid(linestyle=':')
# os.chdir('/Users/erikchavez/Documents/Economic_Policy/Figures/')
# fig.savefig('vegetation-flux-zoom.pdf')


# fig = plt.figure(figsize=(10, 5))
# #plt.plot(tv, Tvmid)
# plt.plot(tv2[0:5999], total_ocean_flux[0:5999], color = 'b')
# plt.plot(tv2[0:5999], vegfluxdyn_values[0:5999], color ='orange')
# plt.tick_params(axis='both', which='major', labelsize=18)
# plt.xlabel('Time (year)',fontsize = 18);
# plt.ylabel('Carbon flux (ppm/yr)',fontsize = 18);
# plt.grid(linestyle=':')
# os.chdir('/Users/erikchavez/Documents/Economic_Policy/Figures/')
# fig.savefig('ocean-vegetation-fluxes-zoom.pdf')

# fig = plt.figure(figsize=(10, 5))
# #plt.plot(tv, Tvmid)
# plt.plot(tv2[0:59999], ratio_flux[0:59999], color ='m')
# plt.tick_params(axis='both', which='major', labelsize=18)
# plt.xlabel('Time (year)',fontsize = 18);
# plt.ylabel('Ratio of fluxes (vegetation/ocean)',fontsize = 18);
# plt.grid(linestyle=':')
# os.chdir('/Users/erikchavez/Documents/Economic_Policy/Figures/')
# fig.savefig('vegetation-over-ocean-ratio.pdf')

# fig = plt.figure(figsize=(10, 5))
# #plt.plot(tv, Tvmid)
# plt.plot(tv2[0:5999], ratio_flux[0:5999], color ='m')
# plt.tick_params(axis='both', which='major', labelsize=18)
# plt.xlabel('Time (year)',fontsize = 18);
# plt.ylabel('Ratio of fluxes (vegetation/ocean)',fontsize = 18);
# plt.grid(linestyle=':')
# os.chdir('/Users/erikchavez/Documents/Economic_Policy/Figures/')
# fig.savefig('vegetation-over-ocean-ratio-zoom.pdf')


# plt.figure(figsize=(10, 5))
# #plt.plot(tv, Tvmid)

# plt.plot(tv2[0:59999], vegfluxdyn_values[0:59999])

# plt.plot(tv2[0:59999], oceantotalflux_values[0:59999])
# #plt.plot(tv2[0:59999], Tvmidv4[0:59999])
# #plt.plot(tv2[0:59999], Tvmidv5[0:59999])

# plt.tick_params(axis='both', which='major', labelsize=18)
# plt.xlabel('Time (year)',fontsize = 18);
# plt.ylabel('Carbon flux (ppm)',fontsize = 18);
# plt.grid(linestyle=':')


# figwidth = 10
# fig, axs = plt.subplots(1, 2, sharex=True, figsize=(2 * figwidth, 0.5 * figwidth))
# axs[0].plot(tv, oceanphyflux)
# axs[0].set_xlabel('Time (year)')
# axs[0].set_ylabel('Temperature (K)')
# axs[0].set_title('Temperature dynamics')
# axs[0].grid(linestyle=':')
# axs[1].plot(tv, oceanphyflux)
# axs[1].set_xlabel('Time (year)')
# axs[1].set_ylabel('Carbon (ppm)')
# axs[1].set_title('Carbon dynamics');
# axs[1].grid(linestyle=':')


# Flux = oceanatmflux(Cv,Tv)

# figwidth = 10
# fig, axs = plt.subplots(1, 2, sharex=True, figsize=(2 * figwidth, 0.5 * figwidth))
# axs[0].plot(tv, Flux)
# axs[0].set_xlabel('Time (year)')
# axs[0].set_ylabel('Temperature (K)')
# axs[0].set_title('Temperature dynamics')
# axs[0].grid(linestyle=':')
# axs[1].plot(tv, Cv)
# axs[1].set_xlabel('Time (year)')
# axs[1].set_ylabel('Carbon (ppm)')
# axs[1].set_title('Carbon dynamics');
# axs[1].grid(linestyle=':')


# #Plot results
# #Time series

# tv2 = np.linspace(1397, 2399, 100000)

# figwidth = 10
# fig, axs = plt.subplots(1, 2, sharex=True, figsize=(2 * figwidth, 0.5 * figwidth))
# axs[0].plot(tv2[35000:75001], Tvmid[35000:75001])
# axs[0].set_xlabel('Time (year)')
# axs[0].set_ylabel('Temperature (K)')
# axs[0].set_title('Temperature dynamics')
# axs[0].grid(linestyle=':')
# axs[1].plot(tv2[35000:65001], Cv[35000:65001])
# axs[1].set_xlabel('Time (year)')
# axs[1].set_ylabel('Carbon (ppm)')
# axs[1].set_title('Carbon dynamics');
# axs[1].grid(linestyle=':')


# Tvs = np.linspace(0, 1000, 100000)

# Tvs[0:2999] = Tvmid[0:2999]
# for i in range(2999, 100000):
#     Tvs[i] = np.mean(Tvmid[i-2999:i])

# #Plot results
# #Time series
# figwidth = 10
# fig, axs = plt.subplots(1, 2, sharex=True, figsize=(2 * figwidth, 0.5 * figwidth))
# axs[0].plot(tv, Tvs)
# axs[0].set_xlabel('Time (year)')
# axs[0].set_ylabel('Temperature (K)')
# axs[0].set_title('Temperature dynamics')
# axs[0].grid(linestyle=':')
# axs[1].plot(tv, Cv)
# axs[1].set_xlabel('Time (year)')
# axs[1].set_ylabel('Carbon (ppm)')
# axs[1].set_title('Carbon dynamics');
# axs[1].grid(linestyle=':')


# tv2 = np.linspace(1397, 2399, 100000)

# figwidth = 10
# fig, axs = plt.subplots(1, 2, sharex=True, figsize=(2 * figwidth, 0.5 * figwidth))
# axs[0].plot(tv2[35000:75001], Tvs[35000:75001])
# axs[0].set_xlabel('Time (year)')
# axs[0].set_ylabel('Temperature (K)')
# axs[0].set_title('Temperature dynamics')
# axs[0].grid(linestyle=':')
# axs[1].plot(tv2[35000:65001], Cv[35000:65001])
# axs[1].set_xlabel('Time (year)')
# axs[1].set_ylabel('Carbon (ppm)')
# axs[1].set_title('Carbon dynamics');
# axs[1].grid(linestyle=':')


# os.chdir('/Users/erik/Documents/Papers/Economic_Policy/C-T-dynamic-ODEs/results/')
# #plt.savefig("figure_CT_"+str(sim_id)+"_index.png",  bbox_inches='tight',pad_inches = 1)
# #plt.savefig("figure_CT_"+"no-carbon"+"_index.png",  bbox_inches='tight',pad_inches = 1)
# os.chdir('/Users/erik/Documents/Papers/Economic_Policy/C-T-dynamic-ODEs/')


# #Scatter plots

# fig, ax = plt.subplots(1, 1, figsize=(0.5 * figwidth, 0.5 * figwidth))
# ax.plot(Tvmid, Cv)
# plt.tick_params(axis='both', which='major', labelsize=18)
# ax.set_xlabel('Temperature anomaly (K)',fontsize = 16)
# ax.set_ylabel('Carbon (ppm)',fontsize = 16)
# ax.grid(linestyle=':')


# #Scatter plots

# fig, ax = plt.subplots(1, 1, figsize=(0.5 * figwidth, 0.5 * figwidth))
# ax.plot(Tv, Cv)
# plt.tick_params(axis='both', which='major', labelsize=18)
# ax.set_xlabel('Temperature (K)',fontsize = 16)
# ax.set_ylabel('Carbon (ppm)',fontsize = 16)
# ax.grid(linestyle=':')


# #Nullclines
# T_values = np.linspace(180, 330, 501)
# C_values = np.linspace(1, 1000, 500)

# z = np.zeros((len(C_values), len(T_values), 2))
# for i, C_value in enumerate(C_values):
#     for j, T_value in enumerate(T_values):
#         z[i, j, :] = dydt(0, [T_value, C_value])

# z = np.abs(z)**0.2 * np.sign(z)

# plt.plot(T_values, [dydt(0, [val, C0])[0] for val in T_values], label=r'$C=C_{0}$')
# plt.rcParams.update({'font.size': 16})
# #ax.tick_params(axis='both', which='major', labelsize=12)
# plt.axhline(0, color='gray')
# plt.legend()
# plt.xlabel('Temperature (K)', fontsize = 16)
# plt.ylabel('dT/dt', fontsize = 16)
# #plt.ylabel(r'$\dot{T}$')
# plt.grid(linestyle=':')
# # plt.ylim([-10, 10])

# #T_values = np.linspace(150, 310, 201)
# #C_values = np.linspace(0, 700, 200)


# fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(2 * figwidth, figwidth))
# plt.rcParams.update({'font.size': 16})
# plt.xticks(fontsize=16)
# zmax = np.percentile(np.abs(z), 95, axis=(0, 1))
# #axs[0].contourf(T_values, C_values, z[..., 0], 20, cmap='coolwarm', vmin=-zmax[0], vmax=zmax[0])
# axs[0].contour(T_values, C_values, z[..., 0], levels=0, linestyles='-', linewidths=2, colors='b')
# axs[0].contour(T_values, C_values, z[..., 1], levels=0, linestyles='-',linewidths=2, colors='r')
# axs[0].quiver(T_values[::10], C_values[::10], z[::10, ::10, 0], z[::10, ::10, 1])
# axs[0].axvline(Talphaocean_low, color='k', linestyle='--')
# axs[0].axvline(Talphaocean_high, color='k', linestyle='--')
# axs[0].axhline(C0, color='k', linestyle='--')
# axs[0].text(Talphaocean_low, 1.01 * C_values[-1], r'$T_{\alpha,l}$', horizontalalignment='center', fontsize='x-large')
# axs[0].text(Talphaocean_high, 1.01 * C_values[-1], r'$T_{\alpha,h}$', horizontalalignment='center', fontsize='x-large')
# axs[0].text(1.0025 * T_values[-1], C0, r'$C_{0}$', verticalalignment='center', fontsize='x-large')
# axs[0].plot(Tv, Cv, 'magenta', linewidth=2.5)
# axs[0].set_xlabel('Temperature (K)', fontsize = 'x-large')
# axs[0].set_ylabel('Carbon (ppm)', fontsize = 'x-large')
# #axs[0].set_title('dT/dt', fontsize = 'x-large')
# #axs[0].set_title('$\dot T$')
# os.chdir('/Users/erikchavez/Documents/Economic_Policy/Figures/')
# fig.savefig('nullclines.pdf')


# fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(2 * figwidth, figwidth))
# plt.rcParams.update({'font.size': 16})
# plt.xticks(fontsize=16)
# zmax = np.percentile(np.abs(z), 95, axis=(0, 1))
# #axs[0].contourf(T_values, C_values, z[..., 0], 20, cmap='coolwarm', vmin=-zmax[0], vmax=zmax[0])
# axs[0].contour(T_values, C_values, z[..., 0], levels=0, linestyles='-', linewidths=2.5, colors='b')
# axs[0].contour(T_values, C_values, z[..., 1], levels=0, linestyles='-', linewidths=2.5, colors='r')
# axs[0].quiver(T_values[::25], C_values[::25], z[::25, ::25, 0], z[::25, ::25, 1])
# axs[0].axvline(Talphaocean_low, color='k', linestyle='--')
# axs[0].axvline(Talphaocean_high, color='k', linestyle='--')
# axs[0].axhline(C0, color='k', linestyle='--')
# axs[0].text(Talphaocean_low, 1.01 * C_values[-1], r'$T_{\alpha,l}$', horizontalalignment='center', fontsize='x-large')
# axs[0].text(Talphaocean_high, 1.01 * C_values[-1], r'$T_{\alpha,h}$', horizontalalignment='center', fontsize='x-large')
# axs[0].text(1.0025 * T_values[-1], C0, r'$C_{0}$', verticalalignment='center', fontsize='x-large')
# #axs[0].plot(Tv, Cv, 'magenta', linewidth=2.5)
# axs[0].set_xlabel('Temperature (K)', fontsize = 'x-large')
# axs[0].set_ylabel('$CO_{2}$ conentration (ppm)', fontsize = 'x-large')
# os.chdir('/Users/erikchavez/Documents/Economic_Policy/Figures/')
# fig.savefig('nullclines-v2.png')


# #Nullclines
# T_values = np.linspace(260, 310, 501)
# C_values = np.linspace(150, 350, 500)

# z = np.zeros((len(C_values), len(T_values), 2))
# for i, C_value in enumerate(C_values):
#     for j, T_value in enumerate(T_values):
#         z[i, j, :] = dydt(0, [T_value, C_value])

# z = np.abs(z)**0.2 * np.sign(z)

# plt.plot(T_values, [dydt(0, [val, C0])[0] for val in T_values], label=r'$C=C_{0}$')
# plt.rcParams.update({'font.size': 16})
# #ax.tick_params(axis='both', which='major', labelsize=12)
# plt.axhline(0, color='gray')
# plt.legend()
# plt.xlabel('Temperature (K)', fontsize = 16)
# plt.ylabel('dT/dt', fontsize = 16)
# #plt.ylabel(r'$\dot{T}$')
# plt.grid(linestyle=':')
# # plt.ylim([-10, 10])

# #T_values = np.linspace(150, 310, 201)
# #C_values = np.linspace(0, 700, 200)


# fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(2 * figwidth, figwidth))
# plt.rcParams.update({'font.size': 16})
# plt.xticks(fontsize=16)
# zmax = np.percentile(np.abs(z), 95, axis=(0, 1))
# #axs[0].contourf(T_values, C_values, z[..., 0], 20, cmap='coolwarm', vmin=-zmax[0], vmax=zmax[0])
# axs[0].contour(T_values, C_values, z[..., 0], levels=0, linestyles='-', linewidths=3, colors='b')
# axs[0].contour(T_values, C_values, z[..., 1], levels=0, linestyles='-', linewidths=3, colors='r')
# axs[0].quiver(T_values[::35], C_values[::35], z[::35, ::35, 0], z[::35, ::35, 1])
# axs[0].axvline(Topt1, color='k', linestyle='--')
# axs[0].axvline(Topt2, color='k', linestyle='--')
# axs[0].axvline(Tlow, color='k', linestyle='--')
# axs[0].axvline(Thigh, color='k', linestyle='--')
# axs[0].axhline(C0, color='k', linestyle='--')
# axs[0].text(Topt1, 1.01 * C_values[-1], r'$T_{o,l}$', horizontalalignment='center', fontsize='x-large')
# axs[0].text(Topt2, 1.01 * C_values[-1], r'$T_{o,h}$', horizontalalignment='center', fontsize='x-large')
# axs[0].text(Thigh, 1.01 * C_values[-1], r'$T_{v,h}$', horizontalalignment='center', fontsize='x-large')
# axs[0].text(Tlow, 1.01 * C_values[-1], r'$T_{v,l}$', horizontalalignment='center', fontsize='x-large')


# axs[0].text(1.0025 * T_values[-1], C0, r'$C_{0}$', verticalalignment='center', fontsize='x-large')
# axs[0].plot(Tv, Cv, 'magenta', linewidth=2.5)
# axs[0].set_xlabel('Temperature (K)', fontsize = 'x-large')
# axs[0].set_ylabel('$CO_{2}$ conentration (ppm)', fontsize = 'x-large')
# #axs[0].set_title('dT/dt', fontsize = 'x-large')
# #axs[0].set_title('$\dot T$')
# os.chdir('/Users/erikchavez/Documents/Economic_Policy/Figures/')
# fig.savefig('nullclines-zoom.png')


# Thigh = 307.15
# Tlow = 286.15
# Topt1 = 290.15
# Topt2 = 302.15


# #Nullclines
# T_values = np.linspace(278, 305, 501)
# C_values = np.linspace(230, 350, 500)

# z = np.zeros((len(C_values), len(T_values), 2))
# for i, C_value in enumerate(C_values):
#     for j, T_value in enumerate(T_values):
#         z[i, j, :] = dydt(0, [T_value, C_value])

# z = np.abs(z)**0.2 * np.sign(z)

# plt.plot(T_values, [dydt(0, [val, C0])[0] for val in T_values], label=r'$C=C_{0}$')
# plt.rcParams.update({'font.size': 16})
# #ax.tick_params(axis='both', which='major', labelsize=12)
# plt.axhline(0, color='gray')
# plt.legend()
# plt.xlabel('Temperature (K)', fontsize = 16)
# plt.ylabel('dT/dt', fontsize = 16)
# #plt.ylabel(r'$\dot{T}$')
# plt.grid(linestyle=':')
# # plt.ylim([-10, 10])


# #Nullclines
# #T_values = np.linspace(260, 305, 201)
# #C_values = np.linspace(170, 310, 200)
# T_values = np.linspace(250, 305, 301)
# C_values = np.linspace(150, 350, 300)

# fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(2 * figwidth, figwidth))
# plt.rcParams.update({'font.size': 16})
# plt.xticks(fontsize=16)
# zmax = np.percentile(np.abs(z), 95, axis=(0, 1))
# #axs[0].contourf(T_values, C_values, z[..., 0], 20, cmap='coolwarm', vmin=-zmax[0], vmax=zmax[0])
# axs[0].contour(T_values, C_values, z[..., 0], levels=0, linestyles='-', colors='b')
# axs[0].contour(T_values, C_values, z[..., 1], levels=0, linestyles='--', colors='r')
# axs[0].quiver(T_values[::10], C_values[::10], z[::10, ::10, 0], z[::10, ::10, 1])
# #axs[0].axvline(Talphaocean_low, color='k', linestyle='--')
# #axs[0].axvline(Talphaocean_high, color='k', linestyle='--')
# #axs[0].axhline(C0, color='k', linestyle='--')
# #axs[0].text(Talphaocean_low, 1.01 * C_values[-1], r'$T_{\alpha,l}$', horizontalalignment='center', fontsize='x-large')
# #axs[0].text(Talphaocean_high, 1.01 * C_values[-1], r'$T_{\alpha,h}$', horizontalalignment='center', fontsize='x-large')
# axs[0].text(1.0025 * T_values[-1], C0, r'$C_{0}$', verticalalignment='center', fontsize='x-large')
# axs[0].plot(Tv, Cv, 'magenta', linewidth=2.5)
# axs[0].set_xlabel('Temperature (K)', fontsize = 'x-large')
# axs[0].set_ylabel('Carbon (ppm)', fontsize = 'x-large')
# axs[0].set_title('dT/dt', fontsize = 'x-large')
# #axs[0].set_title('$\dot T$')
# os.chdir('/Users/erikchavez/Documents/Economic_Policy/Figures/')
# fig.savefig('nullclines-zoom.pdf')
