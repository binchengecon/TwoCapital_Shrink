
from cProfile import label
from math import ceil
import matplotlib as mpl

from mpl_toolkits.mplot3d.axes3d import Axes3D
from numpy import array, loadtxt
import numpy as np

import sympy
import csv
import pandas as pd
import joblib
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


from param4 import *

#########################################


(gamma_3_list,eta_list) = np.meshgrid(gamma_3_list,eta_list,indexing='ij')

gamma_3_list = gamma_3_list.ravel(order='F')
eta_list = eta_list.ravel(order='F')

param_list = zip(gamma_3_list,eta_list)
#########################################





def print2pdf(epsilon):

    test_code = "_epsfrac_"+ str(epsilon)[0:len(str(epsilon))]+"_Paral"+"_Attemp_" +str(Attemp)

    pdf_pages = PdfPages(path_name+'gamma_eta_plot'+test_code+'.pdf')

    for gamma,eta in param_list:

        # gamma = param_list[num][0]
        # eta=param_list[num][1]

        file_name = "gamma_" + str(gamma)+"_"+"eta_"+str(eta)
        file = open(path_name+file_name+test_code+'.csv','r')
        reader = csv.reader(file,delimiter=',')
        file_header= next(reader)
        file_varnum = len(file_header)
        data = np.array(list(reader)).astype(float)
        file_length = len(data[:,1])

        figwidth = 10
        fig, axs = plt.subplots(int(np.ceil(file_varnum/2)), 2, sharex=True, figsize=(2  * figwidth, 2 *figwidth))  
        if file_length < max_iter:
            plt.suptitle("$\gamma$="+str(gamma)[0:5]+" , $\eta$=" + str(eta)+"_Success",fontsize=16)
        else :
            plt.suptitle("$\gamma$="+str(gamma)[0:5]+" , $\eta$=" + str(eta)+"_Failure",fontsize=16)
        for varnum in np.array(range(file_varnum)):
            axs[varnum//2,varnum%2].plot(np.array(range(file_length)),data[:,varnum],label='')
            axs[varnum//2,varnum%2].set_xlabel('epoch')
            axs[varnum//2,varnum%2].set_ylabel('%s' %file_header[varnum])
            axs[varnum//2,varnum%2].set_title('%s' %file_header[varnum])
            axs[varnum//2,varnum%2].grid(linestyle=':')
            axs[varnum//2,varnum%2].legend()
            # plt.savefig(f"tech4D/data/PostJump/gamma_{str(gamma)[:5]}_eta_{eta}.pdf")
        pdf_pages.savefig(fig)
        plt.close()


    pdf_pages.close()          

number_of_cpu = joblib.cpu_count()
delayed_funcs = [delayed(print2pdf)(epsilon) for epsilon in epsilon_list]
parallel_pool = Parallel(n_jobs=number_of_cpu)
res = parallel_pool(delayed_funcs)

