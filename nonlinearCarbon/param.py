from library import *


##################################################################
## Section 1.2: Parameter Initialization
##################################################################

## heat capacity, incoming radiation
Q0 = 342.5 #Incoming radiation


## land fraction and albedo
p = 0.3 #Fraction of land on the planet
alphaland = 0.28328 # land albedo

## outgoing radiation linearized
kappa = 1.74
Tkappa = 154

## CO2 radiative forcing
B = 5.35 # Greenhouse effect parameter
C0 = 280 # CO2 params. C0 is the reference C02 level


## ocean carbon pumps
bP = 0.077 # Solubility dependence on temperature (value from Fowler et al)
bB = 0.090 # Biopump dependence on temperature (Value from Fowler)
cod = 0.54 # Ocean carbon pump modulation parameter


## timescale and reference temperature (from Fowler)
tauc = 20 # timescale 
T0 = 288 # Temperature reference


## Coc0 ocean carbon depending on depth
coc0 = 73.78

## CO2 uptake by vegetation
wa = 0.015
vegcover = 0.4
Thigh = 305
Tlow = 275
Topt1 = 285
Topt2 = 295
acc = 5

## Volcanism
V = 0.028

## Anthropogenic carbon
sa = 1 # Switch to take anthropogenic emissions


##################################################################
## Section 3.1: Model Parameter
##################################################################

sa = 1
Ts = 286.7 + 0.56 # 282.9
Cs = 389 # 275.5

#wa = 0.05
#cod = 0.15
alphaland = 0.28
bP = 0.05
bB = 0.08
cod = 3.035

# cearth = 0.107
# tauc = 20
# cearth = 35.
# tauc = 6603.

coc0 =350
## Ocean albedo parameters
Talphaocean_low = 219
Talphaocean_high = 299
alphaocean_max = 0.84
alphaocean_min = 0.255


Cbio_low = 50
Cbio_high = 700

T0 = 298
C0 = 280

## CO2 uptake by vegetation
wa = 0.015
vegcover = 0.4

Thigh = 315
Tlow = 282
Topt1 = 295
Topt2 = 310
acc = 5


##################################################################
## Section 3.1: Function Parameter
##################################################################
t_span = 1000




##################################################################
## Impulse Path
ImpulsePattern = 0

## Pattern = 1
if ImpulsePattern == 0:
    """Equivalent Impulse Horizon with Hetero-Value"""
    # ImpulseMin = 0 
    # ImpulseMax = 1100
    # ImpulseStep = 100
    # ImpulsePathSize = int((ImpulseMax-ImpulseMin)/ImpulseStep )

    Carbon   = np.array([0, 100, 500, 1000])

    ImpulsePathSize = len(Carbon)
    CeMatrix = np.zeros((ImpulsePathSize,t_span))

    CeMatrix[:,0] =     Carbon[:] /2.13

elif ImpulsePattern ==1:
    """Heterogenous Impulse Horizon with Homo-value"""
    ImpulsePathSize = 10
    ImpulseValue = 100
    
    CeMatrix = ImpulseValue*np.eye(ImpulsePathSize, t_span)

elif ImpulsePattern ==2:
    """Fixed Impulse Response"""
    ImpulsePathSize = 2
    ImpulseValue = 10
    
    CeMatrix = np.zeros((ImpulsePathSize, t_span))
    CeMatrix[1,:] = ImpulseValue*np.ones((1,t_span))/2.13


## cearth, tauc Path

cearth_taucMatrix = [[35., 6603. ],
                     [10, 1886]    ]

cearth_taucMatrixSize = len(cearth_taucMatrix)

