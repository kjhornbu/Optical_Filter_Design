## Cost Functions Utilized in This work
# Analysis Imports
import pandas as pd
import numpy as np
import math
import warnings
import scipy
import os

# importing all the functions defined in Multi_Twist_Main.py 
from Multi_Twist_Main import *

def CIELABColorFunction(t):
    delta=6/29
    
    if (t > (delta**3)):
        colorFunction = t**(1/3)
    else:
        colorFunction = ((1/3)*t*(delta**-2))+(4/29)
    return colorFunction

# CIE 1964 10 degree observer shipped with this repo; the CIE URL is used as a fallback
DEFAULT_OBSERVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CIE_xyz_1964_10deg.csv")
CIE_OBSERVER_URL = "https://files.cie.co.at/CIE_xyz_1964_10deg.csv"

# Visible range used for color integration, 1 nm spacing (in nm)
CMF_WAVELENGTHS = np.arange(380,780,1)

_observer_cache = {}

def _trapezoid(y, x):
    #np.trapz was renamed np.trapezoid in NumPy 2.0
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return integrate(y, x)

def loadColorMatchingFunctions(observerPath=None):
    #Returns xBar, yBar, zBar sampled on CMF_WAVELENGTHS. Cached so the CSV is read only once per path.
    if observerPath is None:
        observerPath = DEFAULT_OBSERVER_PATH if os.path.exists(DEFAULT_OBSERVER_PATH) else CIE_OBSERVER_URL
    
    if observerPath not in _observer_cache:
        if os.path.exists(observerPath): 
            standardObservers_CIE1964_10deg = pd.read_csv(observerPath)
        else:
            #load standard observers: https://cie.co.at/datatable/cie-1964-colour-matching-functions-10-degree-observer
            colnames = ['wL', 'xbar', 'ybar','zbar']
            standardObservers_CIE1964_10deg = pd.read_csv(observerPath,names=colnames, header=None)
        
        cmfs = []
        for column in ['xbar','ybar','zbar']:
            bar = np.interp(CMF_WAVELENGTHS, standardObservers_CIE1964_10deg['wL'].values, standardObservers_CIE1964_10deg[column].values)
            bar[np.isnan(bar)] = 0
            cmfs.append(bar)
        _observer_cache[observerPath] = tuple(cmfs)
    
    return _observer_cache[observerPath]

def convertTransmissionToXYZ(wavelengths,transmission,observerPath=None):
    #wavelengths in micron, transmission 0-1. Returns X,Y,Z normalized so a perfect transmitter has Y = 1.
    xBar, yBar, zBar = loadColorMatchingFunctions(observerPath)
    
    # Assuming illumination is flat at this point across color range (CIE illuminant E)
    illumnationSourceProfile = 1
    
    transmission_adjustedRange = np.interp(CMF_WAVELENGTHS, np.asarray(wavelengths).reshape(-1)*1e3, transmission) # Need to convert to both being in nm hence x1e3
    transmission_adjustedRange[np.isnan(transmission_adjustedRange)] = 0
    
    # Math Following:  https://www.konicaminolta.com/instruments/knowledge/color/part4/02.html
    # Trapezoidal rule: integrate the spectrum (y) over wavelength (x)
    N = _trapezoid(yBar*illumnationSourceProfile, CMF_WAVELENGTHS)
    
    X = _trapezoid(xBar*illumnationSourceProfile*transmission_adjustedRange, CMF_WAVELENGTHS) / N
    Y = _trapezoid(yBar*illumnationSourceProfile*transmission_adjustedRange, CMF_WAVELENGTHS) / N
    Z = _trapezoid(zBar*illumnationSourceProfile*transmission_adjustedRange, CMF_WAVELENGTHS) / N
    
    return X,Y,Z

def convertMTRToXYZ(wavelengths,stokesData,observerPath=None):
    #Color of the MTR output seen through a crossed horizontal polarizer
    transmission = convertStokesToCrossedLinearHorzTransmission(stokesData)
    return convertTransmissionToXYZ(wavelengths,transmission,observerPath)

def whitePointXYZ(observerPath=None):
    #XYZ (scaled so Yn = 100) of a perfect transmitter under the same flat illuminant used in convertMTRToXYZ
    X, Y, Z = convertTransmissionToXYZ(CMF_WAVELENGTHS/1e3, np.ones(len(CMF_WAVELENGTHS)), observerPath)
    return 100*X, 100*Y, 100*Z

def convertStokesToCrossedCircularTransmission(stokesData):
    transmission = (1-stokesData[:,3])/2
    return transmission

def convertStokesToParallelCircularTransmission(stokesData):
    transmission = (1+stokesData[:,3])/2
    return transmission

def convertStokesToCrossedLinear45Transmission(stokesData):
    transmission = (1-stokesData[:,2])/2
    return transmission

def convertStokesToParallelLinear45Transmission(stokesData):
    transmission = (1+stokesData[:,2])/2
    return transmission

def convertStokesToCrossedLinearHorzTransmission(stokesData):
    transmission = (1-stokesData[:,1])/2
    return transmission

def convertStokesToParallelLinearHorzTransmission(stokesData):
    transmission = (1+stokesData[:,1])/2
    return transmission

def convertXYZToCIELAB(X,Y,Z,Xn=None,Yn=None,Zn=None):
    #X,Y,Z on a 0-1 scale (as returned by convertMTRToXYZ); white point on a 0-100 scale.
    #By default the white point matches the flat (illuminant E) source used in convertMTRToXYZ, so a perfect
    #transmitter maps to a* = b* = 0. Pass D65 (Xn=95.0489,Yn=100,Zn=108.8840) only if the spectra are D65 weighted.
    if Xn is None or Yn is None or Zn is None:
        Xn, Yn, Zn = whitePointXYZ()
    L = 116*CIELABColorFunction((100*Y)/Yn)-16
    a = 500*(CIELABColorFunction((100*X)/Xn) - CIELABColorFunction((100*Y)/Yn))
    b = 200*(CIELABColorFunction((100*Y)/Yn) - CIELABColorFunction((100*Z)/Zn))
     
    return L, a, b
