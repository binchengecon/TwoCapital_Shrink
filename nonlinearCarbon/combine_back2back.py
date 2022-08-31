import cv2
from matplotlib import pyplot as plt
import numpy as np

Fig = plt.figure(figsize=(100, 25))

rows = 1
columns = 4
num = 0
for n in range(4):

    fig = plt.figure(figsize=(25,25))

    T = np.arange(1000)

    TT = T * 2

    Fig.add_subplot(rows, columns, num+1)

    plt.imshow(T,TT)
    
    num = num + 1

Figure_Dir="./nonlinearCarbon/figure/"

plt.savefig(Figure_Dir+"test.png")
plt.close()