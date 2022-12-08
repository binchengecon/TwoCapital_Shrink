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
import SolveLinSys
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import CubicSpline
from matplotlib.backends.backend_pdf import PdfPages
from src.supportfunctions import finiteDiff_3D
import os
import argparse


parser = argparse.ArgumentParser(description="xi_r values")
parser.add_argument("--dataname",type=str)
parser.add_argument("--pdfname",type=str)

parser.add_argument("--xiaarr",nargs='+', type=float)
parser.add_argument("--xigarr",nargs='+', type=float)

parser.add_argument("--psi0arr",nargs='+',type=float)
parser.add_argument("--psi1arr",nargs='+',type=float)

parser.add_argument("--hXarr",nargs='+',type=float)
parser.add_argument("--Xminarr",nargs='+',type=float)
parser.add_argument("--Xmaxarr",nargs='+',type=float)

# parser.add_argument("--Update",type=int)

# parser.add_argument("--year",type=int,default=60)
# parser.add_argument("--time",type=float,default=1/12.)
args = parser.parse_args()





mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["figure.figsize"] = (16,10)
mpl.rcParams["font.size"] = 12
mpl.rcParams["legend.frameon"] = False
# mpl.style.use('classic')
mpl.rcParams["lines.linewidth"] = 7.0


res1 = pickle.load(open("/scratch/bincheng/abatement/data_2tech/2jump_hterm_xiag/xi_a_0.0002_xi_g_0.025_psi_0_0.005_psi_1_0.5_model_tech1_pre_damage_simul_25", "rb"))
res2 = pickle.load(open("/scratch/bincheng/abatement/data_2tech/2jump_hterm_xiag/xi_a_0.0002_xi_g_0.05_psi_0_0.005_psi_1_0.5_model_tech1_pre_damage_simul_25", "rb"))
res3 = pickle.load(open("/scratch/bincheng/abatement/data_2tech/2jump_hterm_xiag/xi_a_1000.0_xi_g_1000.0_psi_0_0.005_psi_1_0.5_model_tech1_pre_damage_simul_25", "rb"))
res4 = pickle.load(open("/scratch/bincheng/abatement/data_2tech/2jump_hterm_xiag/xi_a_10000.0_xi_g_0.025_psi_0_0.005_psi_1_0.5_model_tech1_pre_damage_simul_25", "rb"))
res5 = pickle.load(open("/scratch/bincheng/abatement/data_2tech/2jump_hterm_xiag/xi_a_10000.0_xi_g_0.05_psi_0_0.005_psi_1_0.5_model_tech1_pre_damage_simul_25", "rb"))


# os.makedirs("./abatement/pdf_2tech/"+args.dataname+"/")

plt.plot(res3["years"], (res3["Ambiguity_mean_dis"]-res3["Ambiguity_mean_undis"])*1000,label='baseline'  )


plt.plot(res1["years"], (res1["Ambiguity_mean_dis"]-res1["Ambiguity_mean_undis"])*1000,label='$\\xi_a=0.0002$,$\\xi_g=\\xi_d=0.025$')



plt.plot(res2["years"], (res2["Ambiguity_mean_dis"]-res2["Ambiguity_mean_undis"])*1000,label='$\\xi_a=0.0002$,$\\xi_g=\\xi_d=0.050$')





plt.plot(res4["years"], (res4["Ambiguity_mean_dis_h"]-res4["Ambiguity_mean_undis"])*1000,label='$\\xi_a=10000.0$,$\\xi_g=\\xi_d=\\xi_m=0.025$')



plt.plot(res5["years"], (res5["Ambiguity_mean_dis_h"]-res5["Ambiguity_mean_undis"])*1000,label='$\\xi_a=10000.0$,$\\xi_g=\\xi_d=\\xi_m=0.050$')




plt.xlabel("Years")
plt.title("Mean Difference")
# plt.ylim(0,250)
plt.legend(loc='upper left')

plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/MeanDiff_25")
plt.savefig("./abatement/pdf_2tech/"+args.dataname+"/MeanDiff_25")
plt.close()


