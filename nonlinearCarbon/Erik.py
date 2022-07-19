#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created on Wed Jan 13 20:48:20 2021

@author: erik
"""

# # version including anthropogenic emissions

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
import pandas as pd
import scipy.optimize as optim
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy import fft, arange, signal

# os.chdir('/Users/erik/Documents/Papers/Economic_Policy/C-T-dynamic-ODEs/')

# ### INPUT PARAMETERS



## heat capacity, incoming radiation
# Earth heat capacity
cearth = 2.28
#Incoming radiation
Q0 = 342.5

## land fraction and albedo
#Fraction of land on the planet
p = 0.3
# land albedo
alphaland = 0.28328

## outgoing radiation linearized
kappa = 1.74
Tkappa = 154

## CO2 radiative forcing
# Greenhouse effect parameter
B = 5.35
# CO2 params. C0 is the reference C02 level
C0 = 280

## ocean carbon pumps
# Solubility dependence on temperature (value from Fowler et al)
bP = 0.077
# Biopump dependence on temperature (Value from Fowler)
bB = 0.090
# Ocean carbon pump modulation parameter
cod = 0.54

## timescale and reference temperature (from Fowler)
# timescale 
tauc = 20
# Temperature reference
T0 = 288

## Coc0 ocean carbon depending on depth
coc0 = 73.78

## CO2 uptake by vegetation
wa = 0.015
vegcover = 0.4
Thigh = 305
Tlow = 275
Topt1 = 285
Topt2 = 295
acc = 5

## Volcanism
V = 0.028

## Anthropogenic carbon
# Switch to take anthropogenic emissions
sa = 1
# Anthropogenic emissions (zero or one)
Can = pd.read_csv("rcp45co2eqv3.csv")
#times2co2eq
#rcp85co2eq.csv
#Ca = Can[(Can["YEARS"] > 1899) & (Can["YEARS"] < 2201)]
#Ca = Can[(Can["YEARS"] > 1799) & (Can["YEARS"] < 2501)]
Ca = Can[(Can["YEARS"] > 1799) & (Can["YEARS"] < 2801)]
Ca1 = Can[(Can["YEARS"] > 1799) & (Can["YEARS"] < 2801)]
#Ca["YEARS"] = np.arange(start=0,stop=401,step=1)
#Ca = Ca.pd.DataFrame()
Ca = Ca["CO2EQ"]
#Ca = Ca - 286.76808
Ca = Ca - 281.69873
Ca = Ca.to_numpy()


#Ce = np.arange(401)
#Ce = np.arange(601)
Ce = np.arange(1001) * 1.0
#np.min(Ca)
for i in range(len(Ce)):
    if i == 0:
        Ce[i] = 0
    else:
        Ce[i] = Ca[i] - Ca[i-1] 

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


Ce = np.zeros(102)
Impulse = 1 # 1=with impulse 0= without impulse

if Impulse==1:
    Ce[0] = 100 / 2.13 # 100 GtC carbon impule, uncoment this -> with impulse


Cc = (870 - 580) / 2.13 + np.cumsum(Ce)
Cc 

(870 - 580) / 2.13

## Ocean albedo parameters
Talphaocean_low = 219
Talphaocean_high = 299
alphaocean_max = 0.85
alphaocean_min = 0.25

## Biopump parameters
Cbio_low = 150
Cbio_high = 1000


# ### FUNCTIONS

# Anthropogenic carbon fitting with cubic spline
#t_val = np.linspace(0, 600, 601)
#t_val = np.linspace(0, 700, 701)
# t_val = np.linspace(0, 1000, 1001)
t_val = np.linspace(0, 102, 102)

def Yem(t):
    t_points = t_val
    em_points = Ca
    
    tck = interpolate.splrep(t_points, em_points)
    return interpolate.splev(t,tck)

def Yam(t):
    t_points = t_val
    em_points = Ce
    
    tck = interpolate.splrep(t_points, em_points)
    return interpolate.splev(t,tck)

def Ycm(t):
    t_points = t_val
    em_points = Cc
    
    tck = interpolate.splrep(t_points, em_points)
    return interpolate.splev(t,tck)

plt.plot(t_val, Ca)
plt.plot(t_val, Yem(t_val))

plt.plot(t_val, Ce)
plt.plot(t_val, Yam(t_val))

plt.plot(t_val, Cc)
plt.plot(t_val, Ycm(t_val))


# Ocean albedo
def alphaocean(T):
    if T < Talphaocean_low:
        return alphaocean_max
    elif T < Talphaocean_high:
        return alphaocean_max + (alphaocean_min - alphaocean_max) / (Talphaocean_high - Talphaocean_low) * (T - Talphaocean_low)
    else: # so T is higher
        return alphaocean_min

T_values = np.linspace(200, 330, 201)
plt.plot(T_values, [alphaocean(val) for val in T_values])
plt.tick_params(axis='both', which='major', labelsize=13)
plt.grid(linestyle=':')
plt.xlabel('Temperature (K)',fontsize = 14);
plt.ylabel('Ocean albedo',fontsize = 14);

#Fraction of ocean covered by ice
def fracseaice(T):
    if T < Talphaocean_low:
        return 1
    elif T < Talphaocean_high:
        return 1 - 1 / (Talphaocean_high - Talphaocean_low) * (T - Talphaocean_low)
    else: # so T is higher
        return 0

plt.plot(T_values, [fracseaice(val) for val in T_values])
plt.tick_params(axis='both', which='major', labelsize=13)
plt.xlabel('Temperature (K)',fontsize = 14);
plt.ylabel('Fraction of sea ice',fontsize = 14);
plt.grid(linestyle=':')

# Biopump modulation

Cbio_low = 50
Cbio_high = 700

def biopump(Cc):
    if Cc < Cbio_low:
        return 1
    elif Cc < Cbio_high:
        return 1 - 1 / (Cbio_high - Cbio_low) * (Cc - Cbio_low)
        #return 1 - 2 / (Cbio_high - Cbio_low) * (Cc - Cbio_low)
    else: # so Cc is higher
        return 0
        #return -1

biopump = np.vectorize(biopump)

Cc_values = np.linspace(0, 1000, 201)
plt.plot(Cc_values, [biopump(val) for val in Cc_values])
plt.tick_params(axis='both', which='major', labelsize=13)
plt.xlabel('Cumulative anthropogenic emissions (ppm)',fontsize = 14);
plt.ylabel('Bio pump efficiency',fontsize = 14);
plt.grid(linestyle=':')

# Time varying biopump modulation

# +
# plt.plot(t_val, [biopump(val) for val in Ca])
# plt.tick_params(axis='both', which='major', labelsize=13)
# plt.xlabel('Time (years)',fontsize = 14);
# plt.ylabel('Bio pump efficiency',fontsize = 14);
# plt.grid(linestyle=':')
# -

# switch between Ca and Cc to allow for reversal of ocean acidity or locking it

biomodulation = [biopump(val) for val in Cc]
biomod = np.float_(biomodulation)

def bioefficiency(t):
    t_points = t_val
    em_points = biomod
    
    tck = interpolate.splrep(t_points, em_points)
    return interpolate.splev(t,tck)

t_val

plt.plot(t_val, bioefficiency(t_val))
plt.tick_params(axis='both', which='major', labelsize=18)
plt.xlabel('Time (years)',fontsize = 18);
plt.ylabel('Bio pump efficiency',fontsize = 18);
plt.grid(linestyle=':')


t_val_v2 = np.linspace(1800, 2800, 1001)
plt.plot(t_val_v2, bioefficiency(t_val))
plt.tick_params(axis='both', which='major', labelsize=18)
plt.xlabel('Time (years)',fontsize = 18);
plt.ylabel('Bio pump efficiency',fontsize = 18);
plt.grid(linestyle=':')


t_val_v2 = np.linspace(1800, 2800, 1001)

plt.plot(t_val_v2[0:600], bioefficiency(t_val)[0:600])
plt.tick_params(axis='both', which='major', labelsize=18)
plt.xlabel('Time (years)',fontsize = 18);
plt.ylabel('Bio pump efficiency',fontsize = 18);
plt.grid(linestyle=':')


t_val_v3 = np.linspace(1, 1, 1001)
plt.plot(t_val_v2[0:600], t_val_v3[0:600])
plt.tick_params(axis='both', which='major', labelsize=18)
plt.xlabel('Time (years)',fontsize = 18);
plt.ylabel('Bio pump efficiency',fontsize = 18);
plt.grid(linestyle=':')

# Vegetation growth function
def veggrowth(T):
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


Thigh = 299
Tlow = 282
Topt1 = 284
Topt2 = 290
acc = 2


plt.plot(T_values, [veggrowth(val) for val in T_values])
plt.tick_params(axis='both', which='major', labelsize=13)
plt.xlabel('Temperature (K)',fontsize = 14);
plt.ylabel('Vegetation growth',fontsize = 14);
plt.grid(linestyle=':')

#Incoming radiation modified by albedo
def Ri(T):
    return 1/cearth * (Q0 * (1 - p * alphaland - (1 - p) * alphaocean(T)))

plt.plot(T_values, [Ri(val) for val in T_values])
plt.xlabel('Temperature (K)')
plt.grid(linestyle=':')

Ri(T_values[0]), Ri(T_values[-1])


# Outgoing radiation modified by greenhouse effect
def Ro(T, C):
    return 1/cearth * (kappa * (T - Tkappa) -  B * np.log(C / C0))

#Solubility of atmospheric carbon into the oceans
# carbon pumps
def kappaP(T):
    np.exp(-bP * (T - T0))


# +
# def kappaB(T):
#     np.exp(bB * (T - T0))
# -

#Sum of two terms that reflect, respectively, the physical (or solubility) carbon pump in the ocean and Wally Broecker’s “biopump”, due to thermally enhanced bioproductivity (Fowler et al., 2013)
def oceanatmphysflux(T):
    return 1 / tauc * (coc0 * (np.exp(-bP * (T - T0))))

def oceanbioflux(T,t):
     return 1/tauc * (coc0 * (np.exp(bB * bioefficiency(t) * (T - T0))))


# +
# def oceanbioflux(T):
#      return 1/tauc * (coc0 * (np.exp(bB * (T - T0))))

# +
# def wallybroeker(T):
#     return biopump(T) * bB
# -

def oceanatmcorrflux(C):
    return 1 / tauc * (- cod * C)


# ### MODEL EQUATIONS

def dydt(t, y):
    T = y[0]
    C = y[1]

    dT = Ri(T) 
    dT -= Ro(T, C)
    
    dC = V
    dC += Yam(t) * sa                                  #  anthropogenic emissions from Ca spline                                                # volcanism 
    dC -= wa * C * vegcover * veggrowth(T)             # carbon uptake by vegetation
    dC += oceanatmphysflux(T) * (1 - fracseaice(T))    # physical solubility into ocean * fraction of ice-free ocean
    dC += oceanbioflux(T,t) * (1 - fracseaice(T))      # biological pump flux * fraction sea ice
#     dC += oceanbioflux(T) * (1 - fracseaice(T))      # biological pump flux * fraction sea ice
    dC += oceanatmcorrflux(C) * (1 - fracseaice(T))    # correction parameter

    return dT, dC


# Integrate the ODE

sa = 1
Ts = 286.7 + 0.56 # 282.9
Cs = 389 # 275.5

#wa = 0.05
#cod = 0.15
alphaland = 0.28
bP = 0.05
bB = 0.08
cod = 3.035
# cearth = 0.107
# tauc = 20
cearth = 35.
tauc = 6603.
coc0 =350
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

# +
#Thigh = 305
#Tlow = 282
#Topt1 = 295
#Topt2 = 282.5
#acc = 2
#Cs = C0
init = [Ts, Cs]
t_eval = np.linspace(0, 100, 100000)
sol = solve_ivp(dydt, t_eval[[0, -1]], init, t_eval=t_eval, method='RK45', max_step=0.1)

# sol = solve_ivp(dydt, t_eval[[0, -1]], init, t_eval=t_eval, method='BDF')
# -

#Extract values of temperature and C02
Tv = sol.y[0, :]
Cv = sol.y[1, :]
tv = sol.t

#Fixed points
print('Tp = {:.1f}'.format(Tv[-1]))
print('Cp = {:.1f}'.format(Cv[-1]))

Tvmid = Tv - 286.7 # 282.86880986118945

#Tvmid = Tv - 271.0298639974771
# np.mean(Tv) 

# Plot results
# Time series

# t_val_years = np.linspace(1800, 2800, 100000)
# tv2 = np.linspace(1397, 2399, 100000)

# plt.figure(figsize=(10, 5))
# plt.plot(tv, Tvmid)
# plt.plot(tv2[35229:99999], Tvmid[35229:99999])
# plt.tick_params(axis='both', which='major', labelsize=18)
# plt.xlabel('Time (year)',fontsize = 18);
# plt.ylabel('Temperature annomaly (K)',fontsize = 18);
# plt.grid(linestyle=':')

tv

bioefficiency([1,2,3])

plt.plot(tv, [bioefficiency(tt) for tt in tv])

plt.plot(tv, Tvmid)

Cv, Tvmid

# res = np.load(f"Baseline_{cearth}_{tauc}.npy")
# res.shape

# res[1], Tvmid

# tv[0:2]

figwidth = 10
fig, axs = plt.subplots(1, 2, sharex=True, figsize=(2 * figwidth, 0.5 * figwidth))
axs[0].plot(tv, Tvmid, label="With Pulse")
axs[0].plot(tv, res[1], label="Baseline")
axs[0].set_xlabel('Time (year)',fontsize = 16)
axs[0].set_ylabel('Temperature (K)',fontsize = 16)
axs[0].set_title('Temperature dynamics')
axs[0].grid(linestyle=':')
axs[0].legend()
axs[1].plot(tv, Cv, label="With Pulse")
axs[1].plot(tv, res[2], label="Baseline")
axs[1].set_xlabel('Time (year)')
axs[1].set_ylabel('Carbon (ppm)')
axs[1].set_title('Carbon dynamics');
axs[1].grid(linestyle=':')
axs[1].legend()
plt.tight_layout()
plt.savefig(f"Pulse_{cearth}.pdf")


# +
np.save(f"Pulse_{cearth}_{tauc}.npy", [tv, Tvmid, Cv])
# np.save(f"Baseline_{cearth}_{tauc}.npy", [tv, Tvmid, Cv])
# -


plt.figure(figsize=(8,5))
plt.plot(tv, (Tvmid - res[1]) / 100 * 1000)
plt.ylabel("Degree celsius")
plt.xlabel("Years")
plt.title("Impulse response per Teratonne of carbon")
# plt.savefig(f"Response_{cearth}.pdf")

# +
# np.save(f"Baseline_{cearth}.npy", [tv, Tvmid, Cv])
# -

(res[0] - Tvmid)[50]

(res[0] - Tvmid).mean() * 10

plt.plot(tv, (res[1] - Cv))

tv2 = np.linspace(1800, 2800, 100000)

plt.figure(figsize=(10, 5))
#plt.plot(tv, Tvmid)
plt.plot(tv2[0:59999], Tvmid[0:59999])
plt.tick_params(axis='both', which='major', labelsize=18)
plt.xlabel('Time (year)',fontsize = 18);
plt.ylabel('Temperature annomaly (K)',fontsize = 18);
plt.grid(linestyle=':')


# +
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
# -


# Plot results
# Time series

tv2 = np.linspace(1397, 2399, 100000)

figwidth = 10
fig, axs = plt.subplots(1, 2, sharex=True, figsize=(2 * figwidth, 0.5 * figwidth))
axs[0].plot(tv2[35000:75001], Tvmid[35000:75001])
axs[0].set_xlabel('Time (year)')
axs[0].set_ylabel('Temperature (K)')
axs[0].set_title('Temperature dynamics')
axs[0].grid(linestyle=':')
axs[1].plot(tv2[35000:65001], Cv[35000:65001])
axs[1].set_xlabel('Time (year)')
axs[1].set_ylabel('Carbon (ppm)')
axs[1].set_title('Carbon dynamics');
axs[1].grid(linestyle=':')


Tvs = np.linspace(0, 1000, 100000)

Tvs[0:2999] = Tvmid[0:2999]
for i in range(2999, 100000):
    Tvs[i] = np.mean(Tvmid[i-2999:i])

#Plot results
#Time series
figwidth = 10
fig, axs = plt.subplots(1, 2, sharex=True, figsize=(2 * figwidth, 0.5 * figwidth))
axs[0].plot(tv, Tvs)
axs[0].set_xlabel('Time (year)')
axs[0].set_ylabel('Temperature (K)')
axs[0].set_title('Temperature dynamics')
axs[0].grid(linestyle=':')
axs[1].plot(tv, Cv)
axs[1].set_xlabel('Time (year)')
axs[1].set_ylabel('Carbon (ppm)')
axs[1].set_title('Carbon dynamics');
axs[1].grid(linestyle=':')

tv2 = np.linspace(1397, 2399, 100000)

figwidth = 10
fig, axs = plt.subplots(1, 2, sharex=True, figsize=(2 * figwidth, 0.5 * figwidth))
axs[0].plot(tv2[35000:75001], Tvs[35000:75001])
axs[0].set_xlabel('Time (year)')
axs[0].set_ylabel('Temperature (K)')
axs[0].set_title('Temperature dynamics')
axs[0].grid(linestyle=':')
axs[1].plot(tv2[35000:65001], Cv[35000:65001])
axs[1].set_xlabel('Time (year)')
axs[1].set_ylabel('Carbon (ppm)')
axs[1].set_title('Carbon dynamics');
axs[1].grid(linestyle=':')


# +
# os.chdir('/Users/erik/Documents/Papers/Economic_Policy/C-T-dynamic-ODEs/results/')
# plt.savefig("figure_CT_"+str(sim_id)+"_index.png",  bbox_inches='tight',pad_inches = 1)
# plt.savefig("figure_CT_"+"no-carbon"+"_index.png",  bbox_inches='tight',pad_inches = 1)
# os.chdir('/Users/erik/Documents/Papers/Economic_Policy/C-T-dynamic-ODEs/')
# -


# Scatter plots

fig, ax = plt.subplots(1, 1, figsize=(0.5 * figwidth, 0.5 * figwidth))
ax.plot(Tvmid, Cv)
plt.tick_params(axis='both', which='major', labelsize=18)
ax.set_xlabel('Temperature (K)',fontsize = 16)
ax.set_ylabel('Carbon (ppm)',fontsize = 16)
ax.grid(linestyle=':')


#Nullclines
T_values = np.linspace(200, 330, 201)
C_values = np.linspace(1, 1000, 200)

z = np.zeros((len(C_values), len(T_values), 2))
for i, C_value in enumerate(C_values):
    for j, T_value in enumerate(T_values):
        z[i, j, :] = dydt(0, [T_value, C_value])

z = np.abs(z)**0.2 * np.sign(z)

plt.plot(T_values, [dydt(0, [val, C0])[0] for val in T_values], label=r'$C=C_{0}$')
plt.rcParams.update({'font.size': 16})
#ax.tick_params(axis='both', which='major', labelsize=12)
plt.axhline(0, color='gray')
plt.legend()
plt.xlabel('Temperature (K)', fontsize = 16)
plt.ylabel('dT/dt', fontsize = 16)
#plt.ylabel(r'$\dot{T}$')
plt.grid(linestyle=':')
# plt.ylim([-10, 10])

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(2 * figwidth, figwidth))
plt.rcParams.update({'font.size': 16})
plt.xticks(fontsize=16)
zmax = np.percentile(np.abs(z), 95, axis=(0, 1))
#axs[0].contourf(T_values, C_values, z[..., 0], 20, cmap='coolwarm', vmin=-zmax[0], vmax=zmax[0])
axs[0].contour(T_values, C_values, z[..., 0], levels=0, linestyles='-', colors='b')
axs[0].contour(T_values, C_values, z[..., 1], levels=0, linestyles='--', colors='r')
axs[0].quiver(T_values[::10], C_values[::10], z[::10, ::10, 0], z[::10, ::10, 1])
axs[0].axvline(Talphaocean_low, color='k', linestyle='--')
axs[0].axvline(Talphaocean_high, color='k', linestyle='--')
axs[0].axhline(C0, color='k', linestyle='--')
axs[0].text(Talphaocean_low, 1.01 * C_values[-1], r'$T_{\alpha,l}$', horizontalalignment='center', fontsize='x-large')
axs[0].text(Talphaocean_high, 1.01 * C_values[-1], r'$T_{\alpha,h}$', horizontalalignment='center', fontsize='x-large')
axs[0].text(1.0025 * T_values[-1], C0, r'$C_{0}$', verticalalignment='center', fontsize='x-large')
axs[0].plot(Tv, Cv, 'magenta', linewidth=2.5)
axs[0].set_xlabel('Temperature (K)', fontsize = 'x-large')
axs[0].set_ylabel('Carbon (ppm)', fontsize = 'x-large')
axs[0].set_title('dT/dt', fontsize = 'x-large')
#axs[0].set_title('$\dot T$')





# +
#Nullclines
T_values = np.linspace(278, 305, 201)
C_values = np.linspace(230, 350, 200)

z = np.zeros((len(C_values), len(T_values), 2))
for i, C_value in enumerate(C_values):
    for j, T_value in enumerate(T_values):
        z[i, j, :] = dydt(0, [T_value, C_value])

z = np.abs(z)**0.2 * np.sign(z)
# -

plt.plot(T_values, [dydt(0, [val, C0])[0] for val in T_values], label=r'$C=C_{0}$')
plt.rcParams.update({'font.size': 16})
#ax.tick_params(axis='both', which='major', labelsize=12)
plt.axhline(0, color='gray')
plt.legend()
plt.xlabel('Temperature (K)', fontsize = 16)
plt.ylabel('dT/dt', fontsize = 16)
#plt.ylabel(r'$\dot{T}$')
plt.grid(linestyle=':')
# plt.ylim([-10, 10])


#Nullclines
T_values = np.linspace(278, 305, 201)
C_values = np.linspace(250, 300, 200)

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(2 * figwidth, figwidth))
plt.rcParams.update({'font.size': 16})
plt.xticks(fontsize=16)
zmax = np.percentile(np.abs(z), 95, axis=(0, 1))
#axs[0].contourf(T_values, C_values, z[..., 0], 20, cmap='coolwarm', vmin=-zmax[0], vmax=zmax[0])
axs[0].contour(T_values, C_values, z[..., 0], levels=0, linestyles='-', colors='b')
axs[0].contour(T_values, C_values, z[..., 1], levels=0, linestyles='--', colors='r')
axs[0].quiver(T_values[::10], C_values[::10], z[::10, ::10, 0], z[::10, ::10, 1])
#axs[0].axvline(Talphaocean_low, color='k', linestyle='--')
#axs[0].axvline(Talphaocean_high, color='k', linestyle='--')
#axs[0].axhline(C0, color='k', linestyle='--')
#axs[0].text(Talphaocean_low, 1.01 * C_values[-1], r'$T_{\alpha,l}$', horizontalalignment='center', fontsize='x-large')
#axs[0].text(Talphaocean_high, 1.01 * C_values[-1], r'$T_{\alpha,h}$', horizontalalignment='center', fontsize='x-large')
axs[0].text(1.0025 * T_values[-1], C0, r'$C_{0}$', verticalalignment='center', fontsize='x-large')
axs[0].plot(Tv, Cv, 'magenta', linewidth=2.5)
axs[0].set_xlabel('Temperature (K)', fontsize = 'x-large')
axs[0].set_ylabel('Carbon (ppm)', fontsize = 'x-large')
axs[0].set_title('dT/dt', fontsize = 'x-large')
#axs[0].set_title('$\dot T$')

# # Assess oscillation

# # oscillation of full period

# load the data
data = Tv
# convert to numpy array
temp = np.array(data)
# convert the temp from Fahrenheit to Celsius
tempC = (temp - 32) * (5/9)
# we remove the mean of the signal to create a signal oscillating around 0
tempNorm = tempC - np.mean(tempC)
# this is sampling rate, 4 samples/day
#fs = 1600/400
fs = 1000/100000
# create timestamp that start with 1, this is means
# our day start with day 1. 
t = np.arange(len(temp))/float(fs) + 1
## use of Fourrier transform
# get the frequency and spectrum
f, Pxx = signal.periodogram(tempNorm, fs = fs, window='hanning', scaling='spectrum')
#plot the periodogram
plt.figure(figsize = (10, 8))
plt.plot(f, Pxx)
plt.xlim(0, 0.1)
plt.yscale('log')
plt.xlabel('Frequency (cycles/year)')
plt.ylabel('Spectrum Amplitude')
# print the top period in the signal
for amp_arg in np.argsort(np.abs(Pxx))[::-1][1:2]:
    year = 1 / f[amp_arg]
    print(year)

result = np.round(year,1)
if result > 400:
    result = -999

resultstr = str(result)

amplitr = str(np.max(data) - np.min(data))

# +
# os.chdir('/Users/erik/Documents/Papers/Economic_Policy/C-T-dynamic-ODEs/results/')
# plt.savefig("figure_"+str(sim_id)+"_index.png",  bbox_inches='tight',pad_inches = 1)
# plt.savefig("figure_"+"no-carbon"+"_index.png",  bbox_inches='tight',pad_inches = 1)
# os.chdir('/Users/erik/Documents/Papers/Economic_Policy/C-T-dynamic-ODEs/')
