import numpy as np

tol = 1e-7
epsilon  = 0.01
fraction = 0.01
max_iter = 20000
path_name = "./tech4D/data/PostJump/"
test_code = "_epsfrac_"+ str(epsilon)[0:len(str(epsilon))]

gamma_3_list = np.linspace(0., 1./3., 10)
# gamma_3_list = np.linspace(0., 1./3., 1)
eta_list     = np.array([0.1,0.05,0.01,0.001])
# eta_list     = np.array([0.001])
