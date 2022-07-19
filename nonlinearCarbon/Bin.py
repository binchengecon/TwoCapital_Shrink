
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

ImpulsePathNum = 10
CeMatrix = np.zeros(102,ImpulsePathNum)

CeMatrix = np.zeros(102)
Impulse = 1 # 1=with impulse 0= without impulse

if Impulse==1:
    Ce[0] = 100 / 2.13 # 100 GtC carbon impule, uncoment this -> with impulse


Cc = 1.34*12/44*1000/2.13 + np.cumsum(Ce)
Cc 


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


biomodulation = [biopump(val) for val in Cc]
biomod = np.float_(biomodulation)

def bioefficiency(t):
    t_points = t_val
    em_points = biomod
    
    tck = interpolate.splrep(t_points, em_points)
    return interpolate.splev(t,tck)

t_val


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



#Incoming radiation modified by albedo
def Ri(T):
    return 1/cearth * (Q0 * (1 - p * alphaland - (1 - p) * alphaocean(T)))



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


Tvmid = Tv - 286.7 # 282.86880986118945

# +
# np.save(f"Pulse_{cearth}_{tauc}.npy", [tv, Tvmid, Cv])
# np.save(f"Baseline_{cearth}_{tauc}.npy", [tv, Tvmid, Cv])

res = np.load(f"Baseline_{cearth}_{tauc}.npy")
res.shape

res[1], Tvmid

tv[0:2]

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


plt.figure(figsize=(8,5))
plt.plot(tv, (Tvmid - res[1]) / 100 * 1000)
plt.ylabel("Degree celsius")
plt.xlabel("Years")
plt.title("Impulse response per Teratonne of carbon")
plt.savefig(f"Response_{cearth}.pdf")