import numpy as np
import pandas as pd
import sys
print(sys.path)

sys.path.append('./src')

import pickle
import plotly.graph_objects as go
import plotly.offline as pyo
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import CubicSpline
from matplotlib.backends.backend_pdf import PdfPages
import os


delta = 0.01
alpha = 0.115
kappa = 6.667
mu_k  = -0.043
sigma_k = 0.0095
beta_f = 1.86/1000
sigma_y = 1.2 * 1.86 / 1000
zeta = 0.0
# psi_0 = 0.00025
# psi_1 = 1/2
sigma_g = 0.016
gamma_1 = 1.7675 / 10000
gamma_2 = 0.0022 * 2

gamma_3_list = np.linspace(0,1/3,5)


y_bar = 2.
y_bar_lower = 1.5


Y_0  = 1.1

lower = 100

# y_limit = y_bar_lower
y_limit = y_bar
for i, y_limit in enumerate((y_bar,y_bar_lower)):
    
    fig, axes = plt.subplots(1,1,figsize = (14,8))
    Y = np.linspace(0,4, 100)
    gamma_3 = 0       
    LHS_ylimitlower = gamma_1 * Y + gamma_2/2 * Y**2 # y<y_limit
    # LHS_ylimitupper = gamma_1 * Y + (gamma_2)/2 * (y_bar + Y - y_limit)**2    + gamma_3/2 * (Y-y_limit)**2 - gamma_2/2 * y_bar**2 + gamma_2/2 * y_limit**2
    LHS_ylimitupper = gamma_1 * Y + gamma_2*y_bar*(Y-y_limit) + (gamma_2+gamma_3)/2 * (Y-y_limit)**2 + gamma_2/2 * y_limit**2
    # LHS = LHS_ylimitlower*(Y<y_limit) +  LHS_ylimitupper * (Y>=y_limit)
    LHS = LHS_ylimitupper 
    upper = np.exp(-LHS)
    print(upper.max())
    gamma_3 = 1/3      
    LHS_ylimitlower = gamma_1 * Y + gamma_2/2 * Y**2 # y<y_limit
    # LHS_ylimitupper = gamma_1 * Y + (gamma_2)/2 * (y_bar + Y - y_limit)**2    + gamma_3/2 * (Y-y_limit)**2 - gamma_2/2 * y_bar**2 + gamma_2/2 * y_limit**2
    LHS_ylimitupper = gamma_1 * Y + gamma_2*y_bar*(Y-y_limit) + (gamma_2+gamma_3)/2 * (Y-y_limit)**2 + gamma_2/2 * y_limit**2
    # LHS = LHS_ylimitlower*(Y<y_limit) +  LHS_ylimitupper * (Y>=y_limit)
    LHS = LHS_ylimitupper 
    lower = np.exp(-LHS)
    axes.plot(Y,(lower+upper)/2,label = 'mean',ls = "-",color = 'black')
    axes.vlines(y_limit,0,1,ls = '--',color = 'black')
    axes.fill_between(Y, lower, upper,  color='red', alpha=0.3)
    axes.set_xlabel("Temperature Anomaly")
    axes.set_ylabel("Proportional Reduction in Economic Output")
#     axes.set_title("exp(-logN)")
    # axes.set_ylim(0.65,1)
    axes.set_ylim(0.3,1.1)
    axes.legend(loc='lower left')  
    # plt.show()
    plt.savefig("./abatement_UD/pdf_2tech/Y_limit_new={}.png".format(y_limit))
    plt.close()


# plt.style.use('classic')
# plt.rcParams["savefig.bbox"] = "tight"
# plt.rcParams["figure.figsize"] = (10,5)
# plt.rcParams["figure.dpi"] = 400
# plt.rcParams["font.size"] = 15
# plt.rcParams["legend.frameon"] = True
# plt.rcParams["lines.linewidth"] = 2


# Y = np.linspace(1,2,100)

# r_1 = 1.5
# r_2 = 2.5
        
# Intensity = r_1 *( np.exp(r_2/2 *(Y-y_bar_lower)**2) - 1)
# Intensity = Intensity*(Y>y_bar_lower)
# # plt.plot(figsize=(10,5))
# plt.plot(Y, Intensity,color='black')
# plt.xlabel("Temperature Anomaly")
# plt.ylabel("Intensity")
# # plt.title("Intensity")
# plt.ylim(-0.03,0.6)
# # plt.legend(loc='lower left')  

# plt.savefig("./abatement_UD/pdf_2tech/Intensity1.png")
# plt.close()

# Y = np.linspace(1,2,100)

# r_1 = 1.5
# r_2 = 2.5
# Intensity = r_1 *( np.exp(r_2/2 *(Y-y_bar_lower)**2) - 1)
# Intensity = Intensity*(Y>y_bar_lower)

# plt.plot(Y, Intensity)
# plt.xlabel("Y")
# # plt.ylabel("Intensity")
# plt.title("Intensity")
# plt.ylim(0,0.6)
# # plt.legend(loc='lower left')  

# plt.savefig("./abatement_UD/pdf_2tech/Intensity2.png")
# plt.close()
