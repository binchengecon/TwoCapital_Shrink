Attemp = 5

# Attemp = 1: add more output of variable like multi1,2 to diagnose where the problem is.  change starting multi* as 1.
# Attemp = 2: delete multi2[<1e-8]=1e-8 to see the explosion. This is for presentation to Lars.
# Attemp = 3: Maybe epsilon=0.1 worked, I need to check.
# Attemp = 4: delete multi2[<1e-8] and see the effect, previous results didnt get stored.
# Attemp = 5: more iteration steps to ensure convergence be done


import numpy as np


tol = 1e-7

max_iter = 60000
path_name = "./tech4D/data/PostJump/"

# test_code = "_epsfrac_"+ str(epsilon)[0:len(str(epsilon))]+"_Paral"+"_Attemp_" +str(Attemp)

gamma_3_list = np.linspace(0., 1./3., 10)
# gamma_3_list = np.linspace(0., 1./3., 1)
eta_list     = np.array([0.1,0.05,0.01,0.001])
# eta_list     = np.array([0.001])
epsilon_list = np.array([0.1, 0.025, 0.01, 0.005])

