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
import matplotlib as mpl
import scipy.io as sio
import pandas as pd
import scipy.optimize as optim
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy import fft, arange, signal
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

mpl.rcParams["lines.linewidth"] = 1.5
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["figure.figsize"] = (16,10)
mpl.rcParams["font.size"] = 15
mpl.rcParams["legend.frameon"] = False
#### INPUT PARAMETERS


Figure_Dir="./nonlinearCarbon/figure/"

timearray = np.array((230,240,245,250))
# maxarray = np.array((10, 25, 50, 75, 100, 150, 200))
maxarray2 = np.array((10, 25, 50, 75, 100, 150, 200))
# maxarray2 = np.array((150, 200))
lengtharray = np.array((1,5,10))
# maxarray = np.array((10, 25))

# time =245


# code for displaying multiple images in one figure
  
#import libraries
import cv2
from matplotlib import pyplot as plt

file_name = "rcp45co2eqv3" 

# create figure
fig = plt.figure(figsize=(100, 25))
  
# setting values to rows and column variables
rows = 1
columns = len(maxarray2)
  
# reading images
time = 245

num = 0

for max in maxarray2:
    Image = cv2.imread(Figure_Dir+"Pulse="+file_name+",pulseyear="+str(time+1765)+",pulselength="+str(lengtharray)+",pulsesize="+str(max)+".png")
    Image = np.flip(Image, axis=-1) 
    fig.add_subplot(rows, columns, num+1)
    plt.imshow(Image)
    plt.axis('off')
    plt.title(f"Carbon Impulse={max}")
    num = num + 1

# Image = cv2.imread(Figure_Dir+"Pulse=rcp45co2eqv3,pulseyear=2010,pulselength=[ 1  5 10],pulsesize=200.jpg")
plt.savefig(Figure_Dir+"Pulse="+file_name+",pulselength="+str(lengtharray)+",back2back.png")
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
