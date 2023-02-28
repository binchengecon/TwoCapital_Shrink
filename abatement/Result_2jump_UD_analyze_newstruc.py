import numpy as np
import pandas as pd
import sys
print(sys.path)

sys.path.append('./src')
sys.path.append('/home/bcheng4/TwoCapital_Shrink/abatement/')

import pickle
import plotly.graph_objects as go
import plotly.offline as pyo
import matplotlib.pyplot as plt
import SolveLinSys
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import CubicSpline
from matplotlib.backends.backend_pdf import PdfPages
from src.supportfunctions import finiteDiff_3D
import os
import argparse
import time
import petsc4py
from petsc4py import PETSc
import petsclinearsystem
from Result_support_analysis import *
sys.stdout.flush()




parser = argparse.ArgumentParser(description="xi_r values")
parser.add_argument("--dataname",type=str, default = "2jump_step_4.00,9.00_0.0,4.0_1.0,6.0_SS_0.2,0.2,0.2_LR_0.1" )
parser.add_argument("--pdfname",type=str, default= "mercury")

parser.add_argument("--xiaarr",nargs='+', type=float, default=np.array((0.0002)))
parser.add_argument("--xigarr",nargs='+', type=float, default=np.array((0.025)))

parser.add_argument("--psi0arr",nargs='+',type=float, default=np.array((0.105830)))
parser.add_argument("--psi1arr",nargs='+',type=float, default=np.array((0.5)))
parser.add_argument("--psi2arr",nargs='+',type=float, default=np.array((0.2)))
parser.add_argument("--num_gamma",type=int, default=4)

parser.add_argument("--hXarr",nargs='+',type=float, default = np.array((0.2, 0.2, 0.2)))
parser.add_argument("--Xminarr",nargs='+',type=float, default= np.array((4.00, 0.0, 1.0, 0.0)))
parser.add_argument("--Xmaxarr",nargs='+',type=float, default= np.array((9.00, 4.0, 6.0, 3.0)))

parser.add_argument("--auto",type=int, default=1)
parser.add_argument("--IntPeriod",type=int, default=26)

# parser.add_argument("--Update",type=int)


args = parser.parse_args()

dataname = args.dataname

# Update = args.Update
IntPeriod = args.IntPeriod
timespan = 1/12

psi0arr = args.psi0arr
psi1arr = args.psi1arr
psi2arr = args.psi2arr
xiaarr = args.xiaarr
xigarr = args.xigarr 


Xminarr = args.Xminarr
Xmaxarr = args.Xmaxarr
hXarr = args.hXarr
auto = args.auto

num_gamma = args.num_gamma
gamma_3_list = np.linspace(0,1./3.,num_gamma)


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
gamma_1 = 1.7675 / 1000
gamma_2 = 0.0022 * 2


y_bar = 2.
y_bar_lower = 1.5

# Tech
theta = 3
lambda_bar = 0.1206
vartheta_bar = 0.0453

lambda_bar_first = lambda_bar / 2.
vartheta_bar_first = vartheta_bar / 2.

lambda_bar_second = 1e-3
vartheta_bar_second = 0.

plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["figure.figsize"] = (16,10)
plt.rcParams["figure.dpi"] = 500
plt.rcParams["font.size"] = 15
plt.style.use('classic')
plt.rcParams["legend.frameon"] = True
plt.rcParams["lines.linewidth"] = 5

print("After, figure default size is: ", plt.rcParams["savefig.bbox"])
print("After, figure default size is: ", plt.rcParams["figure.figsize"])
print("After, figure default dpi is: ", plt.rcParams["figure.dpi"])
print("After, figure default size is: ", plt.rcParams["font.size"])
print("After, legend.frameon is: ", plt.rcParams["legend.frameon"])
print("After, lines.linewidth is: ", plt.rcParams["lines.linewidth"])


if os.path.exists("./abatement/pdf_2tech/"+args.dataname+"/")==False:
    os.makedirs("./abatement/pdf_2tech/"+args.dataname+"/", exist_ok=True)

id_xiag = 0
id_psi0 = 0
id_psi1 = 0
id_psi2 = 0


grid_info = (Xminarr, Xmaxarr, hXarr)
data_info = (dataname)
varying_argument_extraction = (xiaarr,xigarr,psi0arr,psi1arr,psi2arr, IntPeriod, timespan)
constant_argument_extraction = (delta, alpha, kappa, mu_k, sigma_k, beta_f, sigma_y, zeta, sigma_g, gamma_1, gamma_2, y_bar, y_bar_lower, theta, lambda_bar, vartheta_bar, lambda_bar_first, vartheta_bar_first, lambda_bar_second, vartheta_bar_second, num_gamma, gamma_3_list)
res_tpset_paru, res_tpset_nou, res_base_paru, res_base_nou, model_tech1_pre_damage, K, Y, L, Y_short = model_extract(grid_info, data_info, varying_argument_extraction, constant_argument_extraction)

## Checking Differences of no update and partial update for post damage pre tech: 0
for i in range(num_gamma):
    print((res_tpset_paru[i]["v0"]-res_tpset_nou[i]["v0"]).max(),(res_tpset_paru[i]["v0"]-res_tpset_nou[i]["v0"]).min())

## Checking Differences of no update and partial update for pre damage pre tech: 
print((res_base_paru["v0"]-res_base_nou["v0"]).min(),(res_base_paru["v0"]-res_base_nou["v0"]).max())

## Checking Differences of partial update and original version for pre damage pre tech: 

print((res_base_paru["v0"]-model_tech1_pre_damage["v0"]).min(),(res_base_paru["v0"]-model_tech1_pre_damage["v0"]).max())
print((res_base_paru["i_star"]-model_tech1_pre_damage["i_star"]).min(),(res_base_paru["i_star"]-model_tech1_pre_damage["i_star"]).max())
print((res_base_paru["e_star"]-model_tech1_pre_damage["e_star"]).min(),(res_base_paru["e_star"]-model_tech1_pre_damage["e_star"]).max())
print((res_base_paru["x_star"]-model_tech1_pre_damage["x_star"]).min(),(res_base_paru["x_star"]-model_tech1_pre_damage["x_star"]).max())

## Prepare Derivatives

dK_paru = finiteDiff_3D(res_base_paru["v0"],0,1,K[1]-K[0])
dY_paru = finiteDiff_3D(res_base_paru["v0"],1,1,Y[1]-Y[0])
dL_paru = finiteDiff_3D(res_base_paru["v0"],2,1,L[1]-L[0])

dK_orig = finiteDiff_3D(model_tech1_pre_damage["v0"],0,1,K[1]-K[0])
dY_orig = finiteDiff_3D(model_tech1_pre_damage["v0"],1,1,Y[1]-Y[0])
dL_orig = finiteDiff_3D(model_tech1_pre_damage["v0"],2,1,L[1]-L[0])

ddK_paru = finiteDiff_3D(res_base_paru["v0"],0,2,K[1]-K[0])
ddY_paru = finiteDiff_3D(res_base_paru["v0"],1,2,Y[1]-Y[0])
ddL_paru = finiteDiff_3D(res_base_paru["v0"],2,2,L[1]-L[0])

ddK_orig = finiteDiff_3D(model_tech1_pre_damage["v0"],0,2,K[1]-K[0])
ddY_orig = finiteDiff_3D(model_tech1_pre_damage["v0"],1,2,Y[1]-Y[0])
ddL_orig = finiteDiff_3D(model_tech1_pre_damage["v0"],2,2,L[1]-L[0])


home_dir = "/home/bcheng4/TwoCapital_Shrink/abatement/"

pic_home_dir = home_dir + "pdf_2tech/"

pic_subfolder = pic_home_dir+dataname +"/" +  "xi_a_{}_xi_g_{}_psi_0_{}_psi_1_{}/" .format(xiaarr,xigarr,psi0arr,psi1arr)

os.makedirs(pic_subfolder,exist_ok=True)



plt.plot(K,dK_paru[:,:,-1])
plt.savefig(pic_subfolder+"dK_paru.png")
plt.close()

plt.plot(K,dK_paru[:,0,-1])
plt.savefig(pic_subfolder+"dK_paru_Y[0].png")
plt.close()

plt.plot(K,dK_orig[:,:,-1])
plt.savefig(pic_subfolder+"dK_orig.png")
plt.close()

plt.plot(K,dK_orig[:,0,-1])
plt.savefig(pic_subfolder+"dK_orig_Y[0].png")
plt.close()


dK_max = max(dK_paru.max(),dK_orig.max())
dK_min = min(dK_paru.min(),dK_orig.min())

for index_L in range(len(L)):
    plt.plot(K,dK_paru[:,:,index_L])
    plt.ylim(dK_min,dK_max)
    plt.savefig(pic_subfolder+"dK_paru_L[{}].png".format(index_L))
    plt.close()
    plt.plot(K,dK_paru[:,0,index_L])
    plt.ylim(dK_min,dK_max)
    plt.savefig(pic_subfolder+"dK_paru_L[{}]_Y[0].png".format(index_L))
    plt.close()
    plt.plot(K,dK_orig[:,:,index_L])
    plt.ylim(dK_min,dK_max)    
    plt.savefig(pic_subfolder+"dK_orig_L[{}].png".format(index_L))
    plt.close()
    plt.plot(K,dK_orig[:,0,index_L])
    plt.ylim(dK_min,dK_max)
    plt.savefig(pic_subfolder+"dK_orig_L[{}]_Y[0].png".format(index_L))
    plt.close()
    plt.plot(K,dK_orig[:,:,index_L]-dK_paru[:,:,index_L])
    plt.ylim(dK_min,dK_max)
    plt.savefig(pic_subfolder+"dK_diff_L[{}].png".format(index_L))
    plt.close()
    plt.plot(K,dK_orig[:,0,index_L]-dK_orig[:,0,index_L])
    plt.ylim(dK_min,dK_max)
    plt.savefig(pic_subfolder+"dK_diff_L[{}]_Y[0].png".format(index_L))
    plt.close()

ddK_max = max(ddK_paru.max(),ddK_orig.max())
ddK_min = min(ddK_paru.min(),ddK_orig.min())

for index_L in range(len(L)):
    plt.plot(K,ddK_paru[:,:,index_L])
    plt.ylim(ddK_min,ddK_max)
    plt.savefig(pic_subfolder+"ddK_paru_L[{}].png".format(index_L))
    plt.close()
    plt.plot(K,ddK_paru[:,0,index_L])
    plt.ylim(ddK_min,ddK_max)
    plt.savefig(pic_subfolder+"ddK_paru_L[{}]_Y[0].png".format(index_L))
    plt.ylim(ddK_min,ddK_max)
    plt.close()
    plt.plot(K,ddK_orig[:,:,index_L])
    plt.ylim(ddK_min,ddK_max)
    plt.savefig(pic_subfolder+"ddK_orig_L[{}].png".format(index_L))
    plt.ylim(ddK_min,ddK_max)
    plt.close()
    plt.plot(K,ddK_orig[:,0,index_L])
    plt.ylim(ddK_min,ddK_max)
    plt.savefig(pic_subfolder+"ddK_orig_L[{}]_Y[0].png".format(index_L))
    plt.ylim(ddK_min,ddK_max)
    plt.close()
    plt.plot(K,ddK_orig[:,:,index_L]-ddK_paru[:,:,index_L])
    plt.ylim(ddK_min,ddK_max)
    plt.savefig(pic_subfolder+"ddK_diff_L[{}].png".format(index_L))
    plt.ylim(ddK_min,ddK_max)
    plt.close()
    plt.plot(K,ddK_orig[:,0,index_L]-ddK_orig[:,0,index_L])
    plt.ylim(ddK_min,ddK_max)
    plt.savefig(pic_subfolder+"ddK_diff_L[{}]_Y[0].png".format(index_L))
    plt.ylim(ddK_min,ddK_max)
    plt.close()


plt.plot(Y_short,dY_paru[:,:,-1].T)
plt.savefig(pic_subfolder+"dY_paru.png")
plt.close()

plt.plot(Y_short,dY_paru[0,:,-1].T)
plt.savefig(pic_subfolder+"dY_paru_K[0].png")
plt.close()

plt.plot(Y_short,dY_orig[:,:,-1].T)
plt.savefig(pic_subfolder+"dY_orig.png")
plt.close()

plt.plot(Y_short,dY_orig[0,:,-1].T)
plt.savefig(pic_subfolder+"dY_orig_K[0].png")
plt.close()

dY_max = max(dY_paru.max(),dY_orig.max())
dY_min = min(dY_paru.min(),dY_orig.min())


for index_L in range(len(L)):
    plt.plot(Y_short,dY_paru[:,:,index_L].T)
    plt.ylim(dY_min,dY_max)
    plt.savefig(pic_subfolder+"dY_paru_L[{}].png".format(index_L))
    plt.close()
    plt.plot(Y_short,dY_paru[0,:,index_L].T)
    plt.ylim(dY_min,dY_max)
    plt.savefig(pic_subfolder+"dY_paru_L[{}]_K[0].png".format(index_L))
    plt.ylim(dY_min,dY_max)
    plt.close()
    plt.plot(Y_short,dY_orig[:,:,index_L].T)
    plt.ylim(dY_min,dY_max)
    plt.savefig(pic_subfolder+"dY_orig_L[{}].png".format(index_L))
    plt.ylim(dY_min,dY_max)
    plt.close()
    plt.plot(Y_short,dY_orig[0,:,index_L].T)
    plt.ylim(dY_min,dY_max)
    plt.savefig(pic_subfolder+"dY_orig_L[{}]_K[0].png".format(index_L))
    plt.ylim(dY_min,dY_max)
    plt.close()
    plt.plot(Y_short,dY_orig[:,:,index_L].T-dY_paru[:,:,index_L].T)
    plt.ylim(dY_min,dY_max)
    plt.savefig(pic_subfolder+"dY_diff_L[{}].png".format(index_L))
    plt.ylim(dY_min,dY_max)
    plt.close()
    plt.plot(Y_short,dY_orig[0,:,index_L].T-dY_paru[0,:,index_L].T)
    plt.ylim(dY_min,dY_max)
    plt.savefig(pic_subfolder+"dY_diff_L[{}]_Y[0].png".format(index_L))
    plt.ylim(dY_min,dY_max)
    plt.close()

ddY_max = max(ddY_paru.max(),ddY_orig.max())
ddY_min = min(ddY_paru.min(),ddY_orig.min())


for index_L in range(len(L)):
    plt.plot(Y_short,ddY_paru[:,:,index_L].T)
    plt.ylim(ddY_min,ddY_max)
    plt.savefig(pic_subfolder+"ddY_paru_L[{}].png".format(index_L))
    plt.close()
    plt.plot(Y_short,ddY_paru[0,:,index_L].T)
    plt.ylim(ddY_min,ddY_max)
    plt.savefig(pic_subfolder+"ddY_paru_L[{}]_K[0].png".format(index_L))
    plt.ylim(ddY_min,ddY_max)
    plt.close()
    plt.plot(Y_short,ddY_orig[:,:,index_L].T)
    plt.ylim(ddY_min,ddY_max)
    plt.savefig(pic_subfolder+"ddY_orig_L[{}].png".format(index_L))
    plt.ylim(ddY_min,ddY_max)
    plt.close()
    plt.plot(Y_short,ddY_orig[0,:,index_L].T)
    plt.ylim(ddY_min,ddY_max)
    plt.savefig(pic_subfolder+"ddY_orig_L[{}]_K[0].png".format(index_L))
    plt.ylim(ddY_min,ddY_max)
    plt.close()
    plt.plot(Y_short,ddY_orig[:,:,index_L].T-ddY_paru[:,:,index_L].T)
    plt.ylim(ddY_min,ddY_max)
    plt.savefig(pic_subfolder+"ddY_diff_L[{}].png".format(index_L))
    plt.ylim(ddY_min,ddY_max)
    plt.close()
    plt.plot(Y_short,ddY_orig[0,:,index_L].T-ddY_paru[0,:,index_L].T)
    plt.ylim(ddY_min,ddY_max)
    plt.savefig(pic_subfolder+"ddY_diff_L[{}]_Y[0].png".format(index_L))
    plt.ylim(ddY_min,ddY_max)
    plt.close()


plt.plot(L,dL_paru[:,0,:])
plt.savefig(pic_subfolder+"dL_paru.png")
plt.close()

plt.plot(L,dL_paru[0,0,:])
plt.savefig(pic_subfolder+"dL_paru_K[0].png")
plt.close()

plt.plot(L,dL_orig[:,0,:])
plt.savefig(pic_subfolder+"dL_orig.png")
plt.close()

plt.plot(L,dL_orig[0,0,:])
plt.savefig(pic_subfolder+"dL_orig_K[0].png")
plt.close()



dL_max = max(dL_paru.max(),dL_orig.max())
dL_min = min(dL_paru.min(),dL_orig.min())


for index_Y in range(len(Y_short)):
    plt.plot(L,dL_paru[:,index_Y,:].T)
    plt.ylim(dL_min,dL_max)
    plt.savefig(pic_subfolder+"dL_paru_Y[{}].png".format(index_Y))
    plt.close()
    # plt.plot(L,dL_paru[0,:,index_L].T)
    # plt.ylim(dL_min,dL_max)
    # plt.savefig(pic_subfolder+"dL_paru_L[{}]_K[0].png".format(index_L))
    # plt.ylim(dL_min,dL_max)
    # plt.close()
    plt.plot(L,dL_orig[:,index_Y,:].T)
    plt.ylim(dL_min,dL_max)
    plt.savefig(pic_subfolder+"dL_orig_Y[{}].png".format(index_Y))
    plt.ylim(dL_min,dL_max)
    plt.close()
    # plt.plot(L,dL_orig[0,:,index_L].T)
    # plt.ylim(dL_min,dL_max)
    # plt.savefig(pic_subfolder+"dL_orig_L[{}]_K[0].png".format(index_L))
    # plt.ylim(dL_min,dL_max)
    # plt.close()
    plt.plot(L,dL_orig[:,index_Y,:].T-dL_paru[:,index_Y,:].T)
    plt.ylim(dL_min,dL_max)
    plt.savefig(pic_subfolder+"dL_diff_Y[{}].png".format(index_Y))
    plt.ylim(dL_min,dL_max)
    plt.close()
    # plt.plot(L,dL_orig[0,:,index_L].T-dL_paru[0,:,index_L].T)
    # plt.ylim(dL_min,dL_max)
    # plt.savefig(pic_subfolder+"dL_diff_L[{}]_Y[0].png".format(index_L))
    # plt.ylim(dL_min,dL_max)
    # plt.close()

ddL_max = max(ddL_paru.max(),ddL_orig.max())
ddL_min = min(ddL_paru.min(),ddL_orig.min())

for index_Y in range(len(Y_short)):
    plt.plot(L,ddL_paru[:,index_Y,:].T)
    plt.ylim(ddL_min,ddL_max)
    plt.savefig(pic_subfolder+"ddL_paru_Y[{}].png".format(index_Y))
    plt.close()
    # plt.plot(L,ddL_paru[0,:,index_L].T)
    # plt.ylim(ddL_min,ddL_max)
    # plt.savefig(pic_subfolder+"ddL_paru_L[{}]_K[0].png".format(index_L))
    # plt.ylim(ddL_min,ddL_max)
    # plt.close()
    plt.plot(L,ddL_orig[:,index_Y,:].T)
    plt.ylim(ddL_min,ddL_max)
    plt.savefig(pic_subfolder+"ddL_orig_Y[{}].png".format(index_Y))
    plt.ylim(ddL_min,ddL_max)
    plt.close()
    # plt.plot(L,ddL_orig[0,:,index_L].T)
    # plt.ylim(ddL_min,ddL_max)
    # plt.savefig(pic_subfolder+"ddL_orig_L[{}]_K[0].png".format(index_L))
    # plt.ylim(ddL_min,ddL_max)
    # plt.close()
    plt.plot(L,ddL_orig[:,index_Y,:].T-ddL_paru[:,index_Y,:].T)
    plt.ylim(ddL_min,ddL_max)
    plt.savefig(pic_subfolder+"ddL_diff_Y[{}].png".format(index_Y))
    plt.ylim(ddL_min,ddL_max)
    plt.close()
    # plt.plot(L,dL_orig[0,:,index_L].T-dL_paru[0,:,index_L].T)
    # plt.ylim(dL_min,dL_max)
    # plt.savefig(pic_subfolder+"dL_diff_L[{}]_Y[0].png".format(index_L))
    # plt.ylim(dL_min,dL_max)
    # plt.close()