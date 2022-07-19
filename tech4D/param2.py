Attemp = 2

# Attemp = 1: add more output of variable like multi1,2 to diagnose where the problem is.  change starting multi* as 1.
# Attemp = 2:delete multi2[<1e-8]=1e-8 to see the explosion. This is for presentation to Lars.

import numpy as np


tol = 1e-7
epsilon  = 0.1
fraction = epsilon
max_iter = 35000
path_name = "./tech4D/data/PostJump/"
test_code = "_epsfrac_"+ str(epsilon)[0:len(str(epsilon))]+"_Paral"+"_Attemp_" +str(Attemp)

gamma_3_list = np.linspace(0., 1./3., 10)
# gamma_3_list = np.linspace(0., 1./3., 1)
eta_list     = np.array([0.1,0.05,0.01,0.001])
# eta_list     = np.array([0.001])


