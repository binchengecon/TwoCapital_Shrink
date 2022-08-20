from numpy import meshgrid
from sklearn.cluster import mean_shift
from param import *
from library import *
from function import *



## Looping

for ctpathnum in range(cearth_taucMatrixSize):
    figwidth = 10
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(2  * figwidth, 2 *figwidth))
    TvmidBase = np.zeros(10000)

    for pathnum in range(ImpulsePathSize):


        Ce = CeMatrix[pathnum,:]
        cearth, tauc = cearth_taucMatrix[ctpathnum]

        tv, Tvmid, Cv = model(Ts, Cs, cearth, tauc, Ce)

        plotnum = ImpulsePattern*pathnum
        
        
        Coef_Correction = 1
        
        if pathnum ==0:
            TvmidBase = Tvmid

        else :
            Coef_Correction = Carbon[-1]/Carbon[pathnum]

        axs[0].plot(tv, Tvmid, label=f"ImpulseValue_{CeMatrix[pathnum,plotnum]*2.13}")
        axs[0].set_xlabel('Time (year)',fontsize = 16)
        axs[0].set_ylabel('Temperature  (K)',fontsize = 16)
        axs[0].set_title('Temperature Anomaly Dynamics')
        axs[0].grid(linestyle=':')
        axs[0].legend()
        axs[1].plot(tv, Cv, label=f"ImpulseValue_{CeMatrix[pathnum,plotnum]*2.13}")
        axs[1].set_xlabel('Time (year)')
        axs[1].set_ylabel('Carbon (ppm)')
        axs[1].set_title('Carbon Concentration Dynamics')
        axs[1].grid(linestyle=':')
        axs[1].legend()
        axs[2].plot(tv, (Tvmid-TvmidBase)*Coef_Correction, label=f"ImpulseValue_{CeMatrix[pathnum,plotnum]*2.13}_Compared2_0")
        axs[2].set_xlabel('Time (year)',fontsize = 16)
        axs[2].set_ylabel('Degree Celsius',fontsize = 16)
        axs[2].set_title('Impulse Response per Teratonne of Carbon')
        axs[2].grid(linestyle=':')
        axs[2].legend()




    plt.tight_layout()
    plt.savefig(f"./nonlinearCarbon/Year_{t_span}_ImpulsePtn_{ImpulsePattern}_cearth_{cearth}_tauc_{tauc}.png")


