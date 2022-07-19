import os
import sys
from pathlib import Path
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="xi_r values")
parser.add_argument("--xi_a", type=float, default=1000.)
parser.add_argument("--xi_p", type=float, default=1000.)
parser.add_argument("--xi_g", type=float, default=1000.)
parser.add_argument("--psi_0", type=float, default=0.3)
parser.add_argument("--psi_1", type=float, default=0.5)
parser.add_argument("--name",type=str,default="ReplicateSuri")
args = parser.parse_args()


xi_a = args.xi_a  # Smooth ambiguity
xi_b = 1000. # Brownian misspecification
xi_g = args.xi_g  # Technology jump
xi_p = 1000. # Hold place for arguments, no real effects 

gamma_3_list = np.linspace(0., 1./3., 6)
id_damage = args.id
gamma_3_i = gamma_3_list[id_damage]

psi_0     = args.psi_0
psi_1     = args.psi_1


Data_Dir = "./abatement/data_2tech/"+args.name+"/"

File_Name = "xi_a_{}_xi_g_{}_psi_0_{}_psi_1_{}_" .format(xi_a,xi_g,psi_0,psi_1)

file = Data_Dir+File_Name+"model_tech2_pre_damage"

path = Path(file)

count = 0 

while count < 1:
    
    if path.is_file():
        count=count+1

print("wait pre-> graph is done for psi_0_{}_psi_1_{}_" .format(xi_a,xi_g,psi_0,psi_1))