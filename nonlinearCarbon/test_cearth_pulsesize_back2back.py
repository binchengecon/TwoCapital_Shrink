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

Figure_Dir = "./nonlinearCarbon/figure/cearth_pulsesize5/"

figwidth = 10

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(12, 2 * figwidth))


# ceartharray = np.array((15, 17, 20, 25))
ceartharray = np.array((0.3725, 0.3725))
# ceartharray = np.array((15, 15))
# ceartharray = np.array((.3725, 10))
# ceartharray = np.array((50, 500, 1000, 2000))
pulsearray = np.array((0, 10, 12, 14, 16, 18, 20, 22))
# pulsearray = np.array((0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110))
pathnum = 0

time = 1

file_name = "void_carbon"
# file_name = "rcp45co2eqv3"


for cearth in ceartharray:

    rows = 1
    columns = len(pulsearray)

    fig = plt.figure(figsize=(columns*12, 25))

    # # reading images

    num = 0
    # Cs = 389

    for pulse in pulsearray:
        Image = cv2.imread(Figure_Dir+"Baseline="+file_name+",cearth=" +
                           str(cearth)+",year"+str(time+1800)+",pulse="+str(pulse)+".png")
        Image = np.flip(Image, axis=-1)
        fig.add_subplot(rows, columns, num+1)
        plt.imshow(Image)
        plt.axis('off')
        plt.title(f"Carbon Impulse={pulse}")
        num = num + 1

    plt.savefig(Figure_Dir+"Baseline="+file_name+",cearth=" +
                           str(cearth)+",year"+str(time+1800)+",back2backslim.png")
    plt.savefig(Figure_Dir+"Baseline="+file_name+",cearth=" +
                str(cearth)+",year"+str(time+1800)+",back2backslim.pdf")
    plt.close()


figwidth = 10

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(12, 2 * figwidth))


# ceartharray = np.array((15, 17, 20, 25))
# ceartharray = np.array((0.3725, 0.3725))
ceartharray = np.array((15, 15))
# ceartharray = np.array((.3725, 10))
# ceartharray = np.array((50, 500, 1000, 2000))
# pulsearray = np.array((0, 10, 12, 14, 16, 18, 20, 22))
pulsearray = np.array((0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110))
pathnum = 0

time = 1

file_name = "void_carbon"
# file_name = "rcp45co2eqv3"


for cearth in ceartharray:

    rows = 1
    columns = len(pulsearray)

    fig = plt.figure(figsize=(columns*12, 25))

    # # reading images

    num = 0
    # Cs = 389

    for pulse in pulsearray:
        Image = cv2.imread(Figure_Dir+"Baseline="+file_name+",cearth=" +
                           str(cearth)+",year"+str(time+1800)+",pulse="+str(pulse)+".png")
        Image = np.flip(Image, axis=-1)
        fig.add_subplot(rows, columns, num+1)
        plt.imshow(Image)
        plt.axis('off')
        plt.title(f"Carbon Impulse={pulse}")
        num = num + 1

    plt.savefig(Figure_Dir+"Baseline="+file_name+",cearth=" +
                           str(cearth)+",year"+str(time+1800)+",back2backslim.png")
    plt.savefig(Figure_Dir+"Baseline="+file_name+",cearth=" +
                str(cearth)+",year"+str(time+1800)+",back2backslim.pdf")
    plt.close()


figwidth = 10

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(12, 2 * figwidth))


# ceartharray = np.array((15, 17, 20, 25))
ceartharray = np.array((0.3725, 0.3725))
# ceartharray = np.array((15, 15))
# ceartharray = np.array((.3725, 10))
# ceartharray = np.array((50, 500, 1000, 2000))
pulsearray = np.array((0, 10, 12, 14, 16, 18, 20, 22))
# pulsearray = np.array((0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110))
pathnum = 0

time = 1

file_name = "void_carbon"
# file_name = "rcp45co2eqv3"


for cearth in ceartharray:

    rows = 1
    columns = len(pulsearray)

    fig = plt.figure(figsize=(columns*12, 25))

    # # reading images

    num = 0
    # Cs = 389

    for pulse in pulsearray:
        Image = cv2.imread(Figure_Dir+"Baseline="+file_name+",cearth=" +
                           str(cearth)+",year"+str(time+1800)+",pulse="+str(pulse)+",prop.png")
        Image = np.flip(Image, axis=-1)
        fig.add_subplot(rows, columns, num+1)
        plt.imshow(Image)
        plt.axis('off')
        plt.title(f"Carbon Impulse={pulse}")
        num = num + 1

    plt.savefig(Figure_Dir+"Baseline="+file_name+",cearth=" +
                           str(cearth)+",year"+str(time+1800)+",prop,back2backslim.png")
    plt.savefig(Figure_Dir+"Baseline="+file_name+",cearth=" +
                str(cearth)+",year"+str(time+1800)+",prop,back2backslim.pdf")
    plt.close()


figwidth = 10

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(12, 2 * figwidth))


# ceartharray = np.array((15, 17, 20, 25))
# ceartharray = np.array((0.3725, 0.3725))
ceartharray = np.array((15, 15))
# ceartharray = np.array((.3725, 10))
# ceartharray = np.array((50, 500, 1000, 2000))
# pulsearray = np.array((0, 10, 12, 14, 16, 18, 20, 22))
pulsearray = np.array((0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110))
pathnum = 0

time = 1

file_name = "void_carbon"
# file_name = "rcp45co2eqv3"


for cearth in ceartharray:

    rows = 1
    columns = len(pulsearray)

    fig = plt.figure(figsize=(columns*12, 25))

    # # reading images

    num = 0
    # Cs = 389

    for pulse in pulsearray:
        Image = cv2.imread(Figure_Dir+"Baseline="+file_name+",cearth=" +
                           str(cearth)+",year"+str(time+1800)+",pulse="+str(pulse)+",prop.png")
        Image = np.flip(Image, axis=-1)
        fig.add_subplot(rows, columns, num+1)
        plt.imshow(Image)
        plt.axis('off')
        plt.title(f"Carbon Impulse={pulse}")
        num = num + 1

    plt.savefig(Figure_Dir+"Baseline="+file_name+",cearth=" +
                           str(cearth)+",year"+str(time+1800)+",prop,back2backslim.png")
    plt.savefig(Figure_Dir+"Baseline="+file_name+",cearth=" +
                           str(cearth)+",year"+str(time+1800)+",prop,back2backslim.pdf")
    plt.close()
