
import numpy as np
import argparse
import pandas as pd


parser = argparse.ArgumentParser(description="xi_r values")

parser.add_argument("--theta_ell_type", type=str)

args = parser.parse_args()


theta_ell_array = pd.read_csv("./data/model144_p.csv", header=None).to_numpy()[:, 0]/1000.

n_temp = 16
n_carb = 9
n_RD = 3
    
theta_ell_reshape = theta_ell_array.reshape(n_temp, n_carb, n_RD)

if args.theta_ell_type=='full':
    theta_ell = theta_ell_array
elif args.theta_ell_type=='temp':
    theta_ell =np.mean(theta_ell_reshape, axis=(1,2))
    
print(theta_ell.shape)

K_mat = np.zeros((2,2,2))
theta_ell = np.array([temp * np.ones(K_mat.shape) for temp in theta_ell])

print(theta_ell.shape)
