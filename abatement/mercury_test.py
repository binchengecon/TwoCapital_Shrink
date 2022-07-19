import pickle
import os
Data_Dir = "./abatement/data_2tech/test/"

if not os.path.exists(Data_Dir):
    os.mkdir(Data_Dir)

a=1


with open(Data_Dir+"test", "wb") as f:
    pickle.dump(a, f)

print("python file done")
