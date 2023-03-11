import numpy as np
import pandas as pd
import sys
print(sys.path)

sys.path.append('./src')

import pickle
import plotly.graph_objects as go
import plotly.offline as pyo
import matplotlib.pyplot as plt
import SolveLinSys
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import CubicSpline
from matplotlib.backends.backend_pdf import PdfPages
# from src.supportfunctions import finiteDiff_3D
import os
import argparse
import time
import petsc4py
from petsc4py import PETSc
import petsclinearsystem
# from Result_support import *
sys.stdout.flush()


parser = argparse.ArgumentParser(description="xi_r values")
parser.add_argument("--dataname",type=str)
parser.add_argument("--pdfname",type=str)

parser.add_argument("--xiaarr",nargs='+', type=float)
parser.add_argument("--xigarr",nargs='+', type=float)

parser.add_argument("--psi0arr",nargs='+',type=float)
parser.add_argument("--psi1arr",nargs='+',type=float)
parser.add_argument("--psi2arr",nargs='+',type=float)
parser.add_argument("--num_gamma",type=int)

parser.add_argument("--hXarr",nargs='+',type=float)
parser.add_argument("--Xminarr",nargs='+',type=float)
parser.add_argument("--Xmaxarr",nargs='+',type=float)

parser.add_argument("--auto",type=int)
parser.add_argument("--IntPeriod",type=int)

parser.add_argument("--scheme",type=str)
parser.add_argument("--HJB_solution",type=str)

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

scheme = args.scheme
HJB_solution = args.HJB_solution


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


print(plt.rcParamsDefault)
# print("Before, figure default size is: ", plt.rcParams["figure.figsize"])
# print("Before, figure default dpi is: ", plt.rcParams["figure.dpi"])
# print("Before, figure default size is: ", plt.rcParams["font.size"])
# print("Before, legend.frameon is: ", plt.rcParams["legend.frameon"])
# print("Before, lines.linewidth is: ", plt.rcParams["lines.linewidth"])

plt.style.use('classic')
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams["figure.dpi"] = 500
plt.rcParams["font.size"] = 12
plt.rcParams["legend.frameon"] = True
plt.rcParams["lines.linewidth"] = 5

print("After, figure default size is: ", plt.rcParams["savefig.bbox"])
print("After, figure default size is: ", plt.rcParams["figure.figsize"])
print("After, figure default dpi is: ", plt.rcParams["figure.dpi"])
print("After, figure default size is: ", plt.rcParams["font.size"])
print("After, legend.frameon is: ", plt.rcParams["legend.frameon"])
print("After, lines.linewidth is: ", plt.rcParams["lines.linewidth"])


os.makedirs("./abatement_UD/pdf_2tech/"+args.dataname+"/"+scheme+"_"+HJB_solution+"/", exist_ok=True)

Plot_Dir = "./abatement_UD/pdf_2tech/"+args.dataname+"/"+scheme+"_"+HJB_solution+"/"

def model_simulation_generate(xi_a,xi_g,psi_0,psi_1,psi_2):

    Output_Dir = "/scratch/bincheng/"
    Data_Dir = Output_Dir+"abatement/data_2tech/"+args.dataname+"/"
    File_Dir = "xi_a_{}_xi_g_{}_psi_0_{}_psi_1_{}_" .format(xi_a,xi_g,psi_0,psi_1)


    with open(Data_Dir + File_Dir+"model_tech1_pre_damage"+"_UD_simul_{}".format(IntPeriod)+ scheme + "_" +HJB_solution, "rb") as f:
        res = pickle.load(f)


    
    return res

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_simulation_generate(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                if xiaarr[id_xiag]>10:
                    plt.plot(res["years"], (res["x"]/(alpha*np.exp(res["states"][:,0])))*100,label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"], (res["x"]/(alpha*np.exp(res["states"][:,0])))*100,label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                plt.xlabel('Years')
                plt.ylabel('$\%$ of GDP')
                plt.title('R&D investment as percentage of  GDP')
                if auto==0:   
                    plt.ylim(0,0.5)
                plt.xlim(0,IntPeriod)

                plt.legend(loc='upper left')        
print(res.keys())
plt.savefig(Plot_Dir+"/RD,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig(Plot_Dir+"/RD,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_simulation_generate(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], res["i"],label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"], res["i"],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                plt.xlabel('Years')
                plt.title("Capital investment")
                if auto==0:   
                    plt.ylim(65,110)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/CapI,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig(Plot_Dir+"/CapI,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_simulation_generate(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                if xiaarr[id_xiag]>10:
                    plt.plot(res["years"], res["e"],label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"], res["e"],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                # plt.plot(res2["years"][res2["states"][:, 1]<1.5], res2["e"][res2["states"][:, 1]<1.5],label=r'$\xi_p=\\xi_g=0.050$',linewidth=7.0)
                # plt.plot(res3["years"][res3["states"][:, 1]<1.5], res3["e"][res3["states"][:, 1]<1.5],label='baseline',linewidth=7.0)
                plt.xlabel('Years')
                plt.title("Carbon Emissions")
                if auto==0:   
                    plt.ylim(6.0,12.0)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/E,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig(Plot_Dir+"/E,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_simulation_generate(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], res["states"][:, 1],label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"], res["states"][:, 1],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                # plt.plot(res2["years"][res2["states"][:, 1]<1.5], res2["states"][:, 1][res2["states"][:, 1]<1.5],label=r'$\xi_p=\\xi_g=0.050$',linewidth=7.0)
                # plt.plot(res3["years"][res3["states"][:, 1]<1.5], res3["states"][:, 1][res3["states"][:, 1]<1.5],label='baseline',linewidth=7.0)
                plt.xlabel('Years')
                plt.title("Temperature anomaly")
                if auto==0:   
                    plt.ylim(1.1,1.5)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/TA,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig(Plot_Dir+"/TA,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_simulation_generate(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], np.exp(res["states"][:, 2]),label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"], np.exp(res["states"][:, 2]),label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                # plt.plot(res2["years"][res2["states"][:, 1]<1.5], np.exp(res2["states"][:, 2])[res2["states"][:, 1]<1.5],label=r'$\xi_p=\\xi_g=0.050$',linewidth=7.0)
                # plt.plot(res3["years"][res3["states"][:, 1]<1.5], np.exp(res3["states"][:, 2])[res3["states"][:, 1]<1.5],label='baseline',linewidth=7.0)
                plt.xlabel('Years')
                plt.title("Technology jump intensity $J_g$")
                if auto==0:   
                    plt.ylim(10.0,25.0)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper left')


plt.savefig(Plot_Dir+"/Ig,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig(Plot_Dir+"/Ig,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()



for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_simulation_generate(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], res["distorted_tech_prob"],label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"], res["distorted_tech_prob"],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                # plt.plot(res2["years"], res2["distorted_tech_prob"],label=r'$\xi_p=\\xi_g=0.050$',linewidth=7.0)
                # plt.plot(res3["years"], res3["distorted_tech_prob"],label='baseline',linewidth=7.0)
                plt.xlabel('Years')
                plt.title("Distorted probability of a technology jump")
                if auto==0:   
                    plt.ylim(0,1)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/PIgd,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig(Plot_Dir+"/PIgd,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_simulation_generate(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], res["distorted_damage_prob"],label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"], res["distorted_damage_prob"],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)
                # plt.plot(res2["years"], res2["distorted_damage_prob"],label=r'$\xi_p=\\xi_g=0.050$',linewidth=7.0)
                # plt.plot(res3["years"], res3["distorted_damage_prob"],label='baseline',linewidth=7.0)
                plt.xlabel('Years')
                plt.title("Distorted probability of damage changes")
                if auto==0:   
                    plt.ylim(0,1)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/PIdd,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig(Plot_Dir+"/PIdd,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_simulation_generate(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], res["true_tech_prob"],label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"], res["true_tech_prob"],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)

                plt.xlabel("Years")
                plt.title("True probability of a technology jump")
                if auto==0:   
                    plt.ylim(0.0,1.0)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/TPIg,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig(Plot_Dir+"/TPIg,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_simulation_generate(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], res["true_damage_prob"],label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"], res["true_damage_prob"],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)

                plt.xlabel("Years")
                plt.title("True probability of damage changes")
                if auto==0:   
                    plt.ylim(0,1)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/TPId,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig(Plot_Dir+"/TPId,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_simulation_generate(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], np.log(res["scc"]),label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"], np.log(res["scc"]),label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)

                plt.xlabel("Years")
                plt.title("Log of Social Cost of Carbon")
                if auto==0:   
                    plt.ylim(3.0,6.5)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/logSCC,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig(Plot_Dir+"/logSCC,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_simulation_generate(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                if xiaarr[id_xiag]>10:

                    plt.plot(res["years"], np.log(res["scrd"]),label='baseline'.format(psi2arr[id_psi2]) ,linewidth=5.0)
                else:
                    plt.plot(res["years"], np.log(res["scrd"]),label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]) ,linewidth=5.0)

                plt.xlabel("Years")
                plt.ticklabel_format(useOffset=False)

                plt.title("Log of Social Cost of R&D")
                if auto==0:   
                    plt.ylim(6.5,8.0)
                plt.xlim(0,IntPeriod)
                plt.legend(loc='upper right')

plt.savefig(Plot_Dir+"/logSCRD,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig(Plot_Dir+"/logSCRD,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):

                res = model_simulation_generate(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                if xigarr[id_xiag]>10:

                    plt.plot(res["years"], (res["Ambiguity_mean_dis"]-res["Ambiguity_mean_undis"])*1000,label='baseline'.format(psi2arr[id_psi2]))
                else:
                    plt.plot(res["years"], (res["Ambiguity_mean_dis"]-res["Ambiguity_mean_undis"])*1000,label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag]))

                plt.xlabel("Years")
                plt.title("Mean Difference")
                if auto==0:   
                    plt.ylim(0,0.8)   
                plt.legend()


plt.savefig(Plot_Dir+"/MeanDiff,xia={},xig={},psi0={},psi1={},psi2={}.pdf".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.savefig(Plot_Dir+"/MeanDiff,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):
                
                res = model_simulation_generate(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                if xigarr[id_xiag]>10:

                    plt.plot(res["years"], res["ME_total"],label='baseline'.format(psi2arr[id_psi2]))
                else:
                    plt.plot(res["years"], res["ME_total"],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag]))
                    
                plt.xlabel("Years")

                plt.title("ME_total")
                plt.xlim(0,IntPeriod)
                if auto==0:   
                    plt.ylim(0,0.000110)   
                plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/ME_total,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()

# for id_xiag in range(len(xiaarr)): 
#     for id_psi0 in range(len(psi0arr)):
#         for id_psi1 in range(len(psi1arr)):
#             for id_psi2 in range(len(psi2arr)):
#                     grid_info = (Xminarr, Xmaxarr, hXarr)
#                     data_info = (dataname)
#                     varying_argument_extraction = (xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2], IntPeriod, timespan)
#                     constant_argument_extraction = (delta, alpha, kappa, mu_k, sigma_k, beta_f, sigma_y, zeta, sigma_g, gamma_1, gamma_2, y_bar, y_bar_lower, theta, lambda_bar, vartheta_bar, lambda_bar_first, vartheta_bar_first, lambda_bar_second, vartheta_bar_second, num_gamma, gamma_3_list)
#                     res = model_simulation_graph(grid_info, data_info, varying_argument_extraction, constant_argument_extraction)

#                     if xigarr[id_xiag]>10:

#                         plt.plot(res["years"], res["ME_total2"],label='baseline'.format(psi2arr[id_psi2]))
#                     else:
#                         plt.plot(res["years"], res["ME_total2"],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag]))
                        
#                     plt.xlabel("Years")

#                     plt.title("ME_total2")
#                     plt.xlim(0,IntPeriod)
#                     plt.legend(loc='upper left')

# plt.savefig(Plot_Dir+"/ME_total2,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
# plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):
                    res = model_simulation_generate(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                    if xigarr[id_xiag]>10:

                        plt.plot(res["years"], res["ME_base"],label='baseline'.format(psi2arr[id_psi2]))
                    else:
                        plt.plot(res["years"], res["ME_base"],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag]))
                        
                    plt.xlabel("Years")

                    plt.title("ME_base")
                    plt.xlim(0,IntPeriod)
                    if auto==0:   
                        plt.ylim(0,0.000050)   
                    plt.legend(loc='upper left')

plt.savefig(Plot_Dir+"/ME_base,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()


# for id_xiag in range(len(xiaarr)): 
#     for id_psi0 in range(len(psi0arr)):
#         for id_psi1 in range(len(psi1arr)):
#             for id_psi2 in range(len(psi2arr)):
#                     grid_info = (Xminarr, Xmaxarr, hXarr)
#                     data_info = (dataname)
#                     varying_argument_extraction = (xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2], IntPeriod, timespan)
#                     constant_argument_extraction = (delta, alpha, kappa, mu_k, sigma_k, beta_f, sigma_y, zeta, sigma_g, gamma_1, gamma_2, y_bar, y_bar_lower, theta, lambda_bar, vartheta_bar, lambda_bar_first, vartheta_bar_first, lambda_bar_second, vartheta_bar_second, num_gamma, gamma_3_list)
#                     res = model_simulation_graph(grid_info, data_info, varying_argument_extraction, constant_argument_extraction)

#                     if xigarr[id_xiag]>10:

#                         plt.plot(res["years"], res["ME_SCC"],label='baseline'.format(psi2arr[id_psi2]))
#                     else:
#                         plt.plot(res["years"], res["ME_SCC"],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag]))
                        
#                     plt.xlabel("Years")

#                     plt.title("ME_SCC")
#                     plt.xlim(0,IntPeriod)
#                     plt.legend(loc='upper left')

# plt.savefig(Plot_Dir+"/ME_SCC,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
# plt.close()


# for id_xiag in range(len(xiaarr)): 
#     for id_psi0 in range(len(psi0arr)):
#         for id_psi1 in range(len(psi1arr)):
#             for id_psi2 in range(len(psi2arr)):
#                     grid_info = (Xminarr, Xmaxarr, hXarr)
#                     data_info = (dataname)
#                     varying_argument_extraction = (xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2], IntPeriod, timespan)
#                     constant_argument_extraction = (delta, alpha, kappa, mu_k, sigma_k, beta_f, sigma_y, zeta, sigma_g, gamma_1, gamma_2, y_bar, y_bar_lower, theta, lambda_bar, vartheta_bar, lambda_bar_first, vartheta_bar_first, lambda_bar_second, vartheta_bar_second, num_gamma, gamma_3_list)
#                     res = model_simulation_graph(grid_info, data_info, varying_argument_extraction, constant_argument_extraction)

#                     if xigarr[id_xiag]>10:

#                         plt.plot(res["years"], res["ME_consumption"],label='baseline'.format(psi2arr[id_psi2]))
#                     else:
#                         plt.plot(res["years"], res["ME_consumption"],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag]))
                        
#                     plt.xlabel("Years")

#                     plt.title("ME_consumption")
#                     plt.xlim(0,IntPeriod)
#                     plt.legend(loc='upper left')

# plt.savefig(Plot_Dir+"/ME_consumption,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
# plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):
                    res = model_simulation_generate(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                    if xigarr[id_xiag]>10:

                        plt.plot(res["years"], res["ME_total_base"],label='baseline'.format(psi2arr[id_psi2]))
                    else:
                        plt.plot(res["years"], res["ME_total_base"],label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],xigarr[id_xiag]))
                        
                    plt.xlabel("Years")

                    plt.title("ME_total_base")
                    plt.xlim(0,IntPeriod)
                    # if auto==0:   
                    #     plt.ylim(0,150)   
                    plt.legend()

plt.savefig(Plot_Dir+"/ME_total_base,xia={},xig={},psi0={},psi1={},psi2={}.png".format(xiaarr,xigarr,psi0arr,psi1arr,psi2arr))
plt.close()





plt.style.use('default')
plt.rcParams["lines.linewidth"] = 20
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["figure.figsize"] = (16,10)
plt.rcParams["font.size"] = 25
plt.rcParams["legend.frameon"] = True


print("After, figure default size is: ", plt.rcParams["savefig.bbox"])
print("After, figure default size is: ", plt.rcParams["figure.figsize"])
print("After, figure default dpi is: ", plt.rcParams["figure.dpi"])
print("After, figure default size is: ", plt.rcParams["font.size"])
print("After, legend.frameon is: ", plt.rcParams["legend.frameon"])
print("After, lines.linewidth is: ", plt.rcParams["lines.linewidth"])

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):
                
                res = model_simulation_generate(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                # theta_ell_new = res["theta_ell_new"][:,-1]
                # histogram of beta_f
                theta_ell = pd.read_csv("./data/model144_p.csv", header=None).to_numpy()[:, 0]
                # print("theta_ell")
                # print(theta_ell)
                # print("theta_ell_new")
                # print(theta_ell_new)
                pi_c_o = np.ones(len(theta_ell)) / len(theta_ell)
                # pi_c = np.load("πc_5.npy")
                time = 1/timespan
                pi_c = res["pic_t"][:, int(time)]


                # plt.figure(figsize=(16,10))

                print("mean of uncondition = {}" .format(np.average(theta_ell,weights = pi_c_o)))
                print("mean of condition = {}" .format(np.average(theta_ell,weights = pi_c)))
                    
                plt.hist(theta_ell, weights=pi_c_o, bins=np.linspace(0.8, 3., 16), density=True, 
                        alpha=0.5, ec="darkgrey", color="C3",label='baseline'.format(psi2arr[id_psi2]))
                plt.hist(theta_ell, weights=pi_c, bins=np.linspace(0.8, 3., 16), density=True, 
                        alpha=0.5, ec="darkgrey", color="C0",label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]))
                plt.legend(loc='upper left')
                plt.title("Distorted probability of Climate Models")

                plt.ylim(0, 3)
                plt.xlabel("Climate Sensitivity")
                
                plt.savefig(Plot_Dir+"/ClimateSensitivity_0,xia={:.5f},xig={:.3f},psi0={:.3f},psi1={:.3f},psi2={:.1f}.png".format(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2]))
                plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):
                
                res = model_simulation_generate(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                # theta_ell_new = res["theta_ell_new"][:,-1]
                # histogram of beta_f
                psi_2 = pd.read_csv("./data/psi2value_p.csv", header=None).to_numpy()[:, 0]
                # print("theta_ell")
                # print(theta_ell)
                # print("theta_ell_new")
                # print(theta_ell_new)
                pi_c_o = np.ones(len(psi_2)) / len(psi_2)
                # pi_c = np.load("πc_5.npy")
                time = 1/timespan
                pi_c = res["pic_t"][:, int(time)]


                # plt.figure(figsize=(16,10))

                print("mean of uncondition = {}" .format(np.average(psi_2,weights = pi_c_o)))
                print("mean of condition = {}" .format(np.average(psi_2,weights = pi_c)))
                    
                # plt.hist(psi_2, weights=pi_c_o, bins=np.linspace(0.8, 3., 16), density=True, 
                plt.hist(psi_2, weights=pi_c_o, density=True, 
                        alpha=0.5, ec="darkgrey", color="C3",label='baseline'.format(psi2arr[id_psi2]))
                # plt.hist(psi_2, weights=pi_c, bins=np.linspace(0.8, 3., 16), density=True, 
                plt.hist(psi_2, weights=pi_c, density=True, 
                        alpha=0.5, ec="darkgrey", color="C0",label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]))
                plt.legend(loc='upper left')
                plt.title("Distorted probability of R&D Parameters")

                plt.ylim(0, 24)
                plt.xlabel("R&D Parameter Sensitivity")
                
                plt.savefig(Plot_Dir+"/DRSSensitivity_0,xia={:.5f},xig={:.3f},psi0={:.3f},psi1={:.3f},psi2={:.1f}.png".format(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2]))
                plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):
                
                res = model_simulation_generate(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                # theta_ell_new = res["theta_ell_new"][:,-1]
                # histogram of beta_f
                theta_ell = pd.read_csv("./data/model144_p.csv", header=None).to_numpy()[:, 0]
                # print("theta_ell")
                # print(theta_ell)
                # print("theta_ell_new")
                # print(theta_ell_new)
                pi_c_o = np.ones(len(theta_ell)) / len(theta_ell)
                # pi_c = np.load("πc_5.npy")
                time = 1/timespan
                pi_c = res["pic_t"][:, -1]

                # plt.figure(figsize=(16,10))

                print("mean of uncondition = {}" .format(np.average(theta_ell,weights = pi_c_o)))
                print("mean of condition = {}" .format(np.average(theta_ell,weights = pi_c)))
                    
                plt.hist(theta_ell, weights=pi_c_o, bins=np.linspace(0.8, 3., 16), density=True, 
                        alpha=0.5, ec="darkgrey", color="C3",label='baseline'.format(psi2arr[id_psi2]))
                plt.hist(theta_ell, weights=pi_c, bins=np.linspace(0.8, 3., 16), density=True, 
                        alpha=0.5, ec="darkgrey", color="C0",label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]))
                plt.legend(loc='upper left')
                plt.title("Distorted probability of Climate Models")

                plt.ylim(0, 3)
                plt.xlabel("Climate Sensitivity")
                
                plt.savefig(Plot_Dir+"/ClimateSensitivity_25,xia={:.5f},xig={:.3f},psi0={:.3f},psi1={:.3f},psi2={:.1f}.png".format(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2]))
                plt.close()


for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):
                
                res = model_simulation_generate(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                # theta_ell_new = res["theta_ell_new"][:,-1]
                # histogram of beta_f
                psi_2 = pd.read_csv("./data/psi2value_p.csv", header=None).to_numpy()[:, 0]
                # print("theta_ell")
                # print(theta_ell)
                # print("theta_ell_new")
                # print(theta_ell_new)
                pi_c_o = np.ones(len(psi_2)) / len(psi_2)
                # pi_c = np.load("πc_5.npy")
                time = 1/timespan
                pi_c = res["pic_t"][:, -1]

                # plt.figure(figsize=(16,10))

                print("mean of uncondition = {}" .format(np.average(psi_2,weights = pi_c_o)))
                print("mean of condition = {}" .format(np.average(psi_2,weights = pi_c)))
                    
                # plt.hist(psi_2, weights=pi_c_o, bins=np.linspace(0.8, 3., 16), density=True, 
                plt.hist(psi_2, weights=pi_c_o, density=True, 
                        alpha=0.5, ec="darkgrey", color="C3",label='baseline'.format(psi2arr[id_psi2]))
                # plt.hist(psi_2, weights=pi_c, bins=np.linspace(0.8, 3., 16), density=True, 
                plt.hist(psi_2, weights=pi_c, density=True, 
                        alpha=0.5, ec="darkgrey", color="C0",label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]))
                plt.legend(loc='upper left')
                plt.title("Distorted probability of R&D Parameters")

                plt.ylim(0, 24)
                plt.xlabel("R&D Parameter Sensitivity")
                
                plt.savefig(Plot_Dir+"/DRSSensitivity_25,xia={:.5f},xig={:.3f},psi0={:.3f},psi1={:.3f},psi2={:.1f}.png".format(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2]))
                plt.close()

for id_xiag in range(len(xiaarr)): 
    for id_psi0 in range(len(psi0arr)):
        for id_psi1 in range(len(psi1arr)):
            for id_psi2 in range(len(psi2arr)):
                
                res = model_simulation_generate(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2])

                NUM_DAMAGE = res["gt_dmg"].shape[0]
                gamma_3_list = np.linspace(0., 1./3., NUM_DAMAGE)

                # γ3_distort = np.load("γ3_5.npy")

                γ3_distort = res["gt_dmg"][:, -1] 
                # plt.figure(figsize=(16,10))
                plt.hist(gamma_3_list, weights=np.ones(len(gamma_3_list)) / len(gamma_3_list), 
                        alpha=0.5, color="C3", ec="darkgray",label='baseline'.format(psi2arr[id_psi2]))
                plt.hist(gamma_3_list, weights= γ3_distort / np.sum(γ3_distort), 
                        alpha=0.5, color="C0", ec="darkgray",label='$\\xi_p={:.5f}$,$\\xi_m={:.3f}$' .format(xiaarr[id_xiag],xigarr[id_xiag],psi2arr[id_psi2]))
                plt.ylim(0, 1)
                plt.title("Distorted probability of Damage Models")
                plt.xlabel("Damage Curvature")
                plt.legend(loc='upper left')

                    
                plt.savefig(Plot_Dir+"/Gamma3,xia={:.5f},xig={:.3f},psi0={:.3f},psi1={:.3f},psi2={:.1f}.png".format(xiaarr[id_xiag],xigarr[id_xiag],psi0arr[id_psi0],psi1arr[id_psi1],psi2arr[id_psi2]))
                plt.close()

