import pandas as pd
import numpy as np

theta_ell = pd.read_csv('./data/model144.csv', header=None).to_numpy()[:, 0]
psi_2 = pd.read_csv('./data/psi2value2.csv', header=None).to_numpy()[:, 0]

print(theta_ell.shape, psi_2.shape)

theta_ell_m = np.matrix(theta_ell)
psi_2_m = np.matrix(psi_2).transpose()

print(theta_ell_m.shape, psi_2_m.shape)

psi_2_data = psi_2_m * np.ones_like(theta_ell_m)
theta_ell_data =np.ones_like(psi_2_m) *  theta_ell_m 

print(theta_ell_data.shape, psi_2_data.shape)


theta_ell_new = theta_ell_data.ravel(order='C').transpose()
psi_2_new = psi_2_data.ravel(order='C').transpose()

theta_ell_new_df = pd.DataFrame(theta_ell_new)
psi_2_new_df = pd.DataFrame(psi_2_new)

theta_ell_new_df.to_csv('./data/model1442_p.csv', sep='\t',header=None,index=None)
psi_2_new_df.to_csv('./data/psi2value2_p.csv', sep='\t',header=None,index=None)


theta_ell = pd.read_csv('./data/model144.csv', header=None).to_numpy()[:, 0]
psi_2 = pd.read_csv('./data/psi2value.csv', header=None).to_numpy()[:, 0]

print(theta_ell.shape, psi_2.shape)

theta_ell_m = np.matrix(theta_ell)
psi_2_m = np.matrix(psi_2).transpose()

print(theta_ell_m.shape, psi_2_m.shape)

psi_2_data = psi_2_m * np.ones_like(theta_ell_m)
theta_ell_data =np.ones_like(psi_2_m) *  theta_ell_m 

print(theta_ell_data.shape, psi_2_data.shape)


theta_ell_new = theta_ell_data.ravel(order='C').transpose()
psi_2_new = psi_2_data.ravel(order='C').transpose()

theta_ell_new_df = pd.DataFrame(theta_ell_new)
psi_2_new_df = pd.DataFrame(psi_2_new)

theta_ell_new_df.to_csv('./data/model144_p.csv', sep='\t',header=None,index=None)
psi_2_new_df.to_csv('./data/psi2value_p.csv', sep='\t',header=None,index=None)
