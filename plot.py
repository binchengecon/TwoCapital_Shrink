# for post jump HJB
import os
import numpy as np
import pandas as pd
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["savefig.bbox"] = "tight"

# res = pd.read_csv("TestNorm.csv", header=0)
# plt.figure()
# plt.title("$\epsilon$ = 0.3, fraction = 1")
# plt.plot(res["epoch"], res["A_norm"])
# plt.xlabel("Episode")
# plt.ylabel("Norm of A")

# plt.figure()
# plt.title("$\epsilon$ = 0.3, fraction = 1")
# plt.plot(res["epoch"], res["b_norm"])
# plt.xlabel("Episode")
# plt.ylabel("Norm of b")

# plt.figure()
# plt.title("$\epsilon$ = 0.3, fraction = 1")
# plt.plot(res["epoch"], res["residual norm"])
# plt.xlabel("Episode")
# plt.ylabel("Residual norm")

# plt.figure()
# plt.title("$\epsilon$ = 0.3, fraction = 1")
# plt.plot(res["epoch"], res["iterations"])
# plt.xlabel("Episode")
# plt.ylabel("Number of iterations")

# res = pd.read_csv("eigen--1.0-False-14-2-3.csv", header=0)

# figuredir = "./figures/eigen-coarse-minushalf/"
# if not os.path.exists(figuredir):
    # os.makedirs(figuredir)

# with open("data/res-petsc-23-18-18", "rb") as file:
    # data = pickle.load(file)
# res = pd.read_csv("eigen--1.0-False-14-2-3.csv", header=0)

figuredir = "./figures/steady/"
if not os.path.exists(figuredir):
    os.makedirs(figuredir)

savePic = True

# plt.figure()
# plt.plot(res["epoch"], res["iterations"])
# plt.title("Iterations")
# plt.savefig(figuredir + "iterations.pdf")
# plt.show()

# plt.figure()
# plt.plot(res["epoch"], res["residual norm"])
# plt.title("Residual norm")
# plt.savefig(figuredir + "Res.pdf")
# plt.show()

# plt.figure()
# plt.plot(res["epoch"][:2000], res["PDE_Err"][:2000])
# plt.title("PDE error, first 2000 episodes")
# plt.savefig(figuredir + "PDE2000.pdf")
# plt.show()

# plt.figure()
# plt.plot(res["epoch"][2000:], res["PDE_Err"][2000:])
# plt.title("PDE error, 2001 to 20000 episode")
# plt.savefig(figuredir + "PDE2001.pdf")
# plt.show()

# plt.figure()
# plt.plot(res["epoch"][:2000], res["FC_Err"][:2000])
# plt.title("False transient error, first 2000 episodes")
# plt.savefig(figuredir + "FC2000.pdf")
# plt.show()

# plt.figure()
# plt.plot(res["epoch"][2000:], res["FC_Err"][2000:])
# plt.title("False transient error, 2001 to 20000 episode")
# plt.savefig(figuredir + "FC2001.pdf")
# plt.show()

# plt.figure()
# plt.plot(res["epoch"], res["A_norm"])
# plt.title("A norm")
# plt.savefig(figuredir + "Anorm.pdf")
# plt.show()

# plt.figure()
# plt.plot(res["epoch"], res["b_norm"])
# plt.title("b norm")
# plt.savefig(figuredir + "bnorm.pdf")
# plt.show()


with open("data/PostJump/Ag-0.15-gamma-0.3333333333333333-26-09:21", "rb") as file:
    data = pickle.load(file)

plt.plot(res["i_d"][:, :,10])
plt.show()

Kd = data["Kd"]
Kd_max = data["Kd_max"]
# Kd = Kd * Kd_max
Kg = data["Kg"]
Kg_max = data["Kg_max"]
# Kg = Kg * Kg_max
Y = data["Y"]
i_d = data["i_d"]
i_g = data["i_g"]
v = data["v0"]

print(v.shape)
# fig 1
plt.figure()
plt.title("value function - dirty capital")
plt.plot(Kd, v[:, 10,10], label="$K_g = {:d}, Y = {:.2f}$".format(int(Kg[10]), Y[10]))
# plt.plot(Kd, v[:, 10,10], label="$K_g = {:d}, Y = {:.2f}$".format(int(Kg[10]), Y[10]))
plt.xlabel("$K_d$")
plt.ylabel("v")
plt.legend()
# plt.ylim(0)
if savePic:
    plt.savefig(figuredir + "v-Kd.pdf")
plt.show()

# fig 2
plt.figure()
plt.title("value function - green capital")
plt.plot(Kg, v[10,:,10], label="$K_d = {:d}, Y = {:.2f}$".format(int(Kd[10]), Y[10]))
plt.xlabel("$K_g$")
plt.ylabel("v")
plt.legend()
# plt.ylim(0)
if savePic:
    plt.savefig(figuredir + "v-Kg.pdf")
plt.show()

# fig 3
plt.figure()
plt.title("value function - temperature")
plt.plot(Y, v[10,10,:], label="$K_d = {:d}, K_g = {:d}$".format(int(Kd[10]), int(Kg[10])))
plt.plot(Y, v[10,25,:], label="$K_d = {:d}, K_g = {:d}$".format(int(Kd[10]), int(Kg[25])))
plt.xlabel("$Y$")
plt.ylabel("v")
plt.legend()
plt.ylim(0)
if savePic:
    plt.savefig(figuredir + "v-Y.pdf")
plt.show()

# fig 1
plt.figure()
plt.title("dirty investment - dirty capital")
plt.plot(Kd, i_d[:,10,10], label="$K_g = {:d}, Y = {:.2f}$".format(int(Kg[10]), Y[10]))
plt.xlabel("$K_d$")
plt.ylabel("$i_d$")
plt.legend()
plt.ylim(0)
if savePic:
    plt.savefig(figuredir + "id-Kd.pdf")
plt.show()

# fig 2
plt.figure()
plt.title("dirty investment - green capital")
plt.plot(Kg, i_d[10,:,10], label="$K_d = {:d}, Y = {:.2f}$".format(int(Kd[10]), Y[10]))
plt.xlabel("$K_g$")
plt.ylabel("$i_d$")
plt.legend()
plt.ylim(0)
if savePic:
    plt.savefig(figuredir + "id-Kg.pdf")
plt.show()

# fig 3
plt.figure()
plt.title("dirty investment - temperature")
plt.plot(Y, i_d[10,10,:], label="$K_d = {:d}, K_g = {:d}$".format(int(Kd[10]), int(Kg[10])))
plt.xlabel("$Y$")
plt.ylabel("$i_d$")
plt.legend()
plt.ylim(0)
if savePic:
    plt.savefig(figuredir + "id-Y.pdf")
plt.show()
# print(i_g)

# fig 1
plt.figure()
plt.title("green investment - dirty capital")
plt.plot(Kd, i_g[:,10,10], label="$K_g = {:d}, Y = {:.2f}$".format(int(Kg[10]), Y[10]))
plt.xlabel("$K_d$")
plt.ylabel("$i_g$")
plt.legend()
plt.ylim(0)
if savePic:
    plt.savefig(figuredir + "ig-Kd.pdf")
plt.show()

# fig 2
plt.figure()
plt.title("green investment - green capital")
plt.plot(Kg, i_g[10,:,10], label="$K_d = {:d}, Y = {:.2f}$".format(int(Kd[10]), Y[10]))
plt.xlabel("$K_g$")
plt.ylabel("$i_g$")
plt.ylim(0)
plt.legend()
if savePic:
    plt.savefig(figuredir + "ig-Kg.pdf")
plt.show()

# fig 3
plt.figure()
plt.title("green investment - temperature")
plt.plot(Y, i_g[10,10,:], label="$K_d = {:d}, K_g = {:d}$".format(int(Kd[10]), int(Kg[10])))
plt.xlabel("$Y$")
plt.ylabel("$i_g$")
plt.legend()
if savePic:
    plt.savefig(figuredir + "ig-Y.pdf")
plt.show()
