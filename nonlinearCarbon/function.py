
from param import *
from library import *

t_val = np.linspace(0, t_span-1, t_span)

def Yam(t,CcTemp):
    t_points = t_val
    em_points = CcTemp
    
    tck = interpolate.splrep(t_points, em_points)
    return interpolate.splev(t,tck)

    
def alphaocean(T):
    """"Ocean albedo"""
    if T < Talphaocean_low:
        return alphaocean_max
    elif T < Talphaocean_high:
        return alphaocean_max + (alphaocean_min - alphaocean_max) / (Talphaocean_high - Talphaocean_low) * (T - Talphaocean_low)
    else: # so T is higher
        return alphaocean_min


#
def fracseaice(T):
    """Fraction of ocean covered by ice"""
    if T < Talphaocean_low:
        return 1
    elif T < Talphaocean_high:
        return 1 - 1 / (Talphaocean_high - Talphaocean_low) * (T - Talphaocean_low)
    else: # so T is higher
        return 0


#
def Ri(T, cearth, tauc):
    """Incoming radiation modified by albedo"""
    return 1/cearth * (Q0 * (1 - p * alphaland - (1 - p) * alphaocean(T)))



# 
def Ro(T, C, cearth, tauc):
    """Outgoing radiation modified by greenhouse effect"""
    return 1/cearth * (kappa * (T - Tkappa) -  B * np.log(C / C0))


def kappaP(T):
    """Solubility of atmospheric carbon into the oceans: Carbon Pumps"""
    np.exp(-bP * (T - T0))





def biopump_vec(CcTemp):
    """Goal: vectorize Function biopump"""
    """Output: Grid Points for Interpolation"""
    def biopump(CcTempW1):
        if CcTempW1 < Cbio_low:
            return 1
        elif CcTempW1 < Cbio_high:
            return 1 - 1 / (Cbio_high - Cbio_low) * (CcTempW1 - Cbio_low)
        else: # so Cc is higher
            return 0
    
    biopump = np.vectorize(biopump)
    biomodulation = [biopump(val) for val in CcTemp]
    biomod = np.float_(biomodulation)
    return biomod



def bioefficiency(t,biomodTemp):
    t_points = t_val
    em_points = biomodTemp
    
    tck = interpolate.splrep(t_points, em_points)
    return interpolate.splev(t,tck)


def oceanatmphysflux(T):
    """Sum of two terms that reflect, respectively, the physical (or solubility) carbon pump in the ocean and Wally """
    """Broecker’s “biopump”, due to thermally enhanced bioproductivity (Fowler et al., 2013)"""
    return 1 / tauc * (coc0 * (np.exp(-bP * (T - T0))))

def oceanbioflux(T, t, biomodTemp):
     return 1/tauc * (coc0 * (np.exp(bB * bioefficiency(t, biomodTemp) * (T - T0))))




def oceanatmcorrflux(C):
    return 1 / tauc * (- cod * C)


def veggrowth(T):
    """Vegetation growth function"""
    if T < Tlow:
        return 0
    if (T >= Tlow) and (T < Topt1):
        return acc / (Topt1 - Tlow) * (T - Tlow)
    if (T >= Topt1) and (T <= Topt2):
        return acc
    if (T > Topt2) and (T < Thigh):
        #return acc
        return acc / (Topt2 - Thigh) * (T - Thigh)
    if T > Thigh:
        #return acc
        return 0





def model(Ts, Cs, cearth, tauc, Ce=np.zeros(t_span)):
    """Input: Starting Temp, CO2, Impulse """
    """Transition: Dynamic Equation"""
    """Output: Whole Array of Temp, Temp Anamoly, CO2"""
    Ce[Ce <0] = 0
    Cc = np.cumsum(Ce)
    # Cc = 1.34*12/44*1000/2.13 + np.cumsum(Ce) 
    biomod = biopump_vec(Cc)

    def dydt(t, y):
        """Transition: Dynamic Equation (3 Input)"""
        """t: time sequence"""
        """y: Starting Temp, CO2"""
        """Cc: Cumulated Carbon **Constant Array** """
        T = y[0]
        C = y[1]

        dT = Ri(T, cearth, tauc) 
        dT -= Ro(T, C, cearth, tauc)
    
        dC = V
        dC += Yam(t,Ce) * sa                                  #  anthropogenic emissions from Ca spline                                                # volcanism 
        dC -= wa * C * vegcover * veggrowth(T)             # carbon uptake by vegetation
        dC += oceanatmphysflux(T) * (1 - fracseaice(T))    # physical solubility into ocean * fraction of ice-free ocean
        
        
        dC +=  oceanbioflux(T, t, biomod) * (1 - fracseaice(T))
        
        # dC += oceanbioflux(T, t, Cc) * (1 - fracseaice(T))      # biological pump flux * fraction sea ice
        
        
        
        dC += oceanatmcorrflux(C) * (1 - fracseaice(T))    # correction parameter

        return dT, dC

    init = [Ts, Cs]

    t_eval = np.linspace(0, t_span, 10000)

    sol = solve_ivp(dydt, t_eval[[0, -1]], init, t_eval=t_eval, method='RK45', max_step=0.1)

    # sol = solve_ivp(dydt, t_eval[[0, -1]], init, t_eval=t_eval, method='BDF')
    # -

    #Extract values of temperature and CO2
    Tv = sol.y[0, :]
    Cv = sol.y[1, :]
    tv = sol.t


    Tvmid = Tv - 286.7 

    return tv, Tvmid, Cv