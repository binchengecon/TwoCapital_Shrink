from supportfunctions import finiteDiff
import SolveLinSys
from scipy.interpolate import RegularGridInterpolator
from scipy import fft, arange, signal
from scipy import interpolate
from scipy.optimize import curve_fit
import scipy.optimize as optim
import scipy.io as sio
import matplotlib.mlab
from matplotlib.colors import SymLogNorm
from matplotlib import rcParams
from matplotlib import pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.integrate import solve_ivp
import pickle
import pandas as pd
import numpy as np
import sys
sys.path.append("../src/")
rcParams["figure.figsize"] = (8, 5)
rcParams["savefig.bbox"] = 'tight'


def PDESolver(stateSpace, A, B1, B2, C1, C2, D, v0,
              ε=1, tol=-10):

    A = A.reshape(-1, 1, order='F')
    B = np.hstack([B1.reshape(-1, 1, order='F'), B2.reshape(-1, 1, order='F')])
    C = np.hstack([C1.reshape(-1, 1, order='F'), C2.reshape(-1, 1, order='F')])
    D = D.reshape(-1, 1, order='F')
    v0 = v0.reshape(-1, 1, order='F')
    out = SolveLinSys.solveFT(stateSpace, A, B, C, D, v0, ε, tol)

    return out


def model(pulse, year, cearth=0.3916, baseline="rcp60co2eqv3.csv"):

    #############################################
    ##########Climate Change Part################
    #############################################
    #############################################

    Q0 = 342.5

    # land fraction and albedo
    # Fraction of land on the planet
    p = 0.3
    # land albedo
    alphaland = 0.28

    # outgoing radiation linearized
    kappa = 1.74
    Tkappa = 154

    # Ocean albedo parameters
    Talphaocean_low = 219
    Talphaocean_high = 299
    alphaocean_max = 0.843
    alphaocean_min = 0.254

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
    coc0 = 280

    # CO2 uptake by vegetation

    # lower and upper G thresholds
    Cbio_low = 150
    Cbio_high = 750

    # vegetation carbon uptake temperatures
    Thigh = 307.15
    Tlow = 286.15
    Topt1 = 290.15
    Topt2 = 302.15
    acc = 8

    # lower and upper bounds of opt and viable temp
    Tbiopt1_low = Topt1
    Tbiopt1_high = Topt1 + 5
    Tbiolow_low = Tlow
    Tbiolow_high = Tlow + 5

    # vegetation growth parameters
    wa = 0.015
    vegcover = 0.4

    # Volcanism and atmospheric expansion (Hogg 2008 and LeQuere 2015)
    V = 2.028

    # Anthropogenic carbon
    # Switch to take anthropogenic emissions
    sa = 1
    # Anthropogenic emissions (zero or one)
    # csvname = baseline+'.csv'
    Can = pd.read_csv(baseline)
    #Can = pd.read_csv("Et-sim2.csv")
    # times2co2eq
    # rcp85co2eq.csv
    #Ca = Can[(Can["YEARS"] > 1899) & (Can["YEARS"] < 2201)]
    #Ca = Can[(Can["YEARS"] > 1799) & (Can["YEARS"] < 2501)]
    Ca = Can[(Can["YEARS"] > 1799) & (Can["YEARS"] < 2801)]
    # Ca1 = Can[(Can["YEARS"] > 1799) & (Can["YEARS"] < 2801)]
    #Ca["YEARS"] = np.arange(start=0,stop=401,step=1)
    #Ca = Ca.pd.DataFrame()
    Ca = Ca["CO2EQ"]
    #Ca = Ca - 286.76808
    Ca = Ca - 281.69873
    Ca = Ca.to_numpy()

    Ca[year-1800] += pulse

    tspan = len(Ca)

    #Ce = np.arange(401)
    #Ce = np.arange(601)
    Ce = np.arange(tspan) * 1.0
    # np.min(Ca)
    for i in range(len(Ce)):
        if i == 0:
            Ce[i] = 0
        else:
            Ce[i] = Ca[i] - Ca[i-1]

    Cebis = np.arange(tspan) * 1.0
    # np.min(Ca)
    for i in range(len(Cebis)):
        if i == 0:
            Cebis[i] = 0
        else:
            Cebis[i] = max(Ca[i] - Ca[i-1], 0)

    Cc = np.arange(tspan) * 1.0
    # np.min(Ca)
    for i in range(len(Cc)):
        if i == 0:
            Cc[i] = 0
        else:
            Cc[i] = sum(Cebis[0:i])

    # FUNCTIONS

    # Anthropogenic carbon fitting with cubic spline
    t_val = np.linspace(0, tspan-1, tspan)

    def Yem(t):
        t_points = t_val
        em_points = Ca

        tck = interpolate.splrep(t_points, em_points)
        return interpolate.splev(t, tck)

    def Yam(t):
        t_points = t_val
        em_points = Cebis

        tck = interpolate.splrep(t_points, em_points)
        return interpolate.splev(t, tck)

    def Ycm(t):
        t_points = t_val
        em_points = Cc

        tck = interpolate.splrep(t_points, em_points)
        return interpolate.splev(t, tck)

    # Ocean albedo

    # def alphaocean(T):
    #     if T < Talphaocean_low:
    #         return alphaocean_max
    #     elif T < Talphaocean_high:
    #         return alphaocean_max + (alphaocean_min - alphaocean_max) / (Talphaocean_high - Talphaocean_low) * (T - Talphaocean_low)
    #     else:  # so T is higher
    #         return alphaocean_min

    def alphaocean(T):
        """T, matrix, (nT, nC, nF)"""
        temp = np.zeros(T.shape)
        temp[T < Talphaocean_low] = alphaocean_max
        temp[(T >= Talphaocean_low) & (T < Talphaocean_high)] = alphaocean_max + (alphaocean_min - alphaocean_max) / \
            (Talphaocean_high - Talphaocean_low) * \
            (T[(T >= Talphaocean_low) & (T < Talphaocean_high)] - Talphaocean_low)
        temp[T >= Talphaocean_high] = alphaocean_min

        return temp

    # Fraction of ocean covered by ice

    # def fracseaice(T):
    #     if T < Talphaocean_low:
    #         return 1
    #     elif T < Talphaocean_high:
    #         return 1 - 1 / (Talphaocean_high - Talphaocean_low) * (T - Talphaocean_low)
    #     else:  # so T is higher
    #         return 0

    def fracseaice(T):

        temp = np.zeros(T.shape)
        temp[T < Talphaocean_low] = 1
        temp[(T >= Talphaocean_low) & (T < Talphaocean_high)] = 1 - 1 / (Talphaocean_high -
                                                                         Talphaocean_low) * (T[(T >= Talphaocean_low) & (T < Talphaocean_high)] - Talphaocean_low)
        temp[T >= Talphaocean_high] = 0

        return temp

    # Vegetation growth function

    # def veggrowth(T):
    #     if T < Tlow:
    #         return 0
    #     if (T >= Tlow) and (T < Topt1):
    #         return acc / (Topt1 - Tlow) * (T - Tlow)
    #     if (T >= Topt1) and (T <= Topt2):
    #         return acc
    #     if (T > Topt2) and (T < Thigh):
    #         # return acc
    #         return acc / (Topt2 - Thigh) * (T - Thigh)
    #     if T >= Thigh:
    #         # return acc
    #         return 0

    # ramp function of lower optimum temperature

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

    # ramp function of percentage of vegetation carbon lost
    Vecar_min = 0
    Vecar_max = 5/12

    def Bioloss(Cc):
        if Cc < Cbio_low:
            return Vecar_min
        elif Cc < Cbio_high:
            return Vecar_min + (Vecar_max - Vecar_min) / (Cbio_high - Cbio_low) * (Cc - Cbio_low)
        else:  # so Cc is higher
            return Vecar_max
            # return -1

    Bioloss = np.vectorize(Bioloss)

    # evolution of the percentage of the stock of vegetation carbon lost
    Coptmodulation = [Bioloss(val) for val in Cc]
    Coptmod = np.float_(Coptmodulation)

    def Cvegoptlow(t):
        t_points = t_val
        em_points = Coptmod

        tck = interpolate.splrep(t_points, em_points)
        return interpolate.splev(t, tck)

    # evolution of the vegetation carbon lost
    VCoptmodulation = [BioCloss(val) for val in Cc]
    VCoptmod = np.float_(VCoptmodulation)

    def VCvegoptlow(t):
        t_points = t_val
        em_points = VCoptmod

        tck = interpolate.splrep(t_points, em_points)
        return interpolate.splev(t, tck)

    # Tbiolow
    Tbiolow_low = Tlow
    Tbiolow_high = Tlow + 5

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

    # evolution of the lower viable temperature with cumulative emission scenario
    Tlowmodulation = [Tbiolow(val) for val in Cc]
    Tlowmod = np.float_(Tlowmodulation)

    # def Tveglow(t):
    #     t_points = t_val
    #     em_points = Tlowmod

    #     tck = interpolate.splrep(t_points, em_points)
    #     return interpolate.splev(t, tck)

    def Tveglow(G):

        temp = np.zeros(G.shape)

        temp[G <= Cbio_low] = Tbiolow_low
        temp[(G > Cbio_low) & (G < Cbio_high)] = Tbiolow_low + \
            (Tbiolow_high - Tbiolow_low) / \
            (Cbio_high - Cbio_low) * \
            (G[(G > Cbio_low) & (G < Cbio_high)] - Cbio_low)
        temp[G >= Cbio_high] = Tbiolow_high

        return temp

    # ramp function of vegetation carbon lost (ppm)
    C0v = 1000
    VC_min = 0
    VC_max = 5/12 * C0v

    def BioCloss(Cc):
        if Cc < Cbio_low:
            return VC_min
        elif Cc < Cbio_high:
            return VC_min + (VC_max - VC_min) / (Cbio_high - Cbio_low) * (Cc - Cbio_low)
        else:  # so Cc is higher
            return VC_max
            # return -1

    BioCloss = np.vectorize(BioCloss)

    # evolution of the lower optimum temperature with cumulative emission scenario
    Toptmodulation = [Tbioptlow(val) for val in Cc]
    Toptmod = np.float_(Toptmodulation)

    # def Tvegoptlow(t):
    #     t_points = t_val
    #     em_points = Toptmod

    #     tck = interpolate.splrep(t_points, em_points)
    #     return interpolate.splev(t, tck)

    def Tvegoptlow(G):

        temp = np.zeros(G.shape)

        temp[G <= Cbio_low] = VC_min
        temp[(G > Cbio_low) & (G < Cbio_high)] = VC_min + \
            (VC_max - VC_min) / \
            (Cbio_high - Cbio_low) * \
            (G[(G > Cbio_low) & (G < Cbio_high)] - Cbio_low)
        temp[G >= Cbio_high] = VC_max

        return temp

    # # Vegetation growth function
    # def veggrowthdyn(T, t):
    #     if T < Tveglow(t):
    #         return 0
    #     if (T >= Tveglow(t)) and (T < Tvegoptlow(t)):
    #         return acc / (Tvegoptlow(t) - Tveglow(t)) * (T - Tveglow(t))
    #     if (T >= Tvegoptlow(t)) and (T <= Topt2):
    #         return acc
    #     if (T > Topt2) and (T < Thigh):
    #         # return acc
    #         return acc / (Topt2 - Thigh) * (T - Thigh)
    #     if T > Thigh:
    #         # return acc
    #         return 0

    # Vegetation growth function: Vectorized

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

    # Incoming radiation modified by albedo

    def Ri(T):
        return 1/cearth * (Q0 * (1 - p * alphaland - (1 - p) * alphaocean(T)))

    # Outgoing radiation modified by greenhouse effect

    def Ro(T, C):
        return 1/cearth * (kappa * (T - Tkappa) - B * np.log(C / C0))

    # vegetation carbon flux

    def vegfluxdyn(T, C, t):
        return wa * C * vegcover * veggrowthdyn(T, t)

    # ocean carbon fluxes

    def oceanatmphysflux(T):
        return 1 / tauc * (coc0 * (np.exp(-bP * (T - T0))))

    def oceanbioflux(T):
        return 1/tauc * (coc0 * (np.exp(bB * (T - T0))))

    def oceanatmcorrflux(C):
        return 1 / tauc * (- cod * C)

    # MODEL EQUATIONS

    def dydt(t, y):
        T = y[0]
        C = y[1]
        #Cveg = y[3]

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
    Ts = 286.45
    Cs = 269

    #############################################
    ########Economic Model Part###################
    #############################################
    #############################################

    # Economic paramaters
    gamma_1 = 1.7675 / 10000.
    gamma_2 = 2 * 0.0022
    delta = 0.01
    eta = 0.032

    # State variable
    # Temperature anomaly, in celsius
    T_min = 1e-8
    T_max = 10.
    hT = 0.2
    T_grid = np.arange(T_min, T_max + hT, hT)

    # atmospheric carbon concentration, in ppm
    C_min = 200
    C_max = 400.
    hC = 4.
    C_grid = np.arange(C_min, C_max + hC, hC)

    # F, Sa in the notes, accumulative anthropogenic carbon, in gigaton, since 1800
    F_min = 1e-8  # 10. avoid
    F_max = 2000.  # 2500 x2.13 gm # # on hold -> 4000 / 2.13 ppm
    hF = 40.
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

    # cearth = 35.

    # tauc = 6603.

    # v0 = pickle.load(open("data_35.0_6603", "rb"))["v0"]
    v0 = - eta * T_mat -eta * C_mat- eta * F_mat
    # v0 =  delta * eta * np.log(delta /4 * (9000/2.13 - F_mat)) + (eta - 1) * gamma_2 * T_mat / cearth * (B * np.log(C_mat/ C0) + kappa * (T_mat + To - Tkappa))

    dG = gamma_1 + gamma_2 * T_mat
    epsilon = 0.1
    count = 0
    error = 1.
    tol = 1e-8
    max_iter = 5000
    fraction = 0.1

    while error > tol and count < max_iter:

        dvdT = finiteDiff(v0, 0, 1, hT)
        dvdTT = finiteDiff(v0, 0, 2, hT)
        dvdC = finiteDiff(v0, 1, 1, hC)
    #     dvdC[dvdC >= - 1e-16] = - 1e-16
        dvdCC = finiteDiff(v0, 1, 2, hC)
        dvdF = finiteDiff(v0, 2, 1, hF)
        dvdFF = finiteDiff(v0, 2, 2, hF)

        Ca = - eta * delta / (dvdC + dvdF)

        Ca[Ca <= 1e-32] = 1e-32

        if count >= 1:
            Ca = Ca * fraction + Ca_star * (1 - fraction)

    #     Ca = np.ones(T_mat.shape)
        A = - delta * np.ones(T_mat.shape)
        B1 = Ri(T_mat + To) - Ro(T_mat + To, C_mat)
        B2 = V
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

        out = PDESolver(stateSpace, A, B1, B2, B3, C1, C2, C3, D, v0, epsilon)
        v = out[2].reshape(v0.shape, order="F")

        rhs_error = A * v0 + B1 * dvdT + B2 * dvdC + B3 * \
            dvdF + C1 * dvdTT + C2 * dvdCC + C3 * dvdFF + D
        rhs_error = np.max(abs(rhs_error))
        lhs_error = np.max(abs((v - v0)/epsilon))

        error = lhs_error
        v0 = v
        Ca_star = Ca
        count += 1

        print("Iteration: %s;\t False Transient Error: %s;\t PDE Error: %s\t" %
              (count, lhs_error, rhs_error))

    print("Total iteration: %s;\t LHS Error: %s;\t RHS Error %s\t" %
          (count, lhs_error, rhs_error))


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(T_mat[:, :, 10], C_mat[:, :, 10], Ca[:, :, ii], 90, cmap='binary')
ax.set_xlabel('T')
ax.set_ylabel('C')
ax.set_zlabel('Ca')
ax.view_init(10, 10)
