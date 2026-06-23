## Cost Functions Utilized in This work
# Analysis Imports
import pandas as pd
import numpy as np
import math
import warnings
import scipy
import os

# importing all the functions
# defined in Multi_Twist_Main.py
from Multi_Twist_Main import *
from cost_function import *

def convertMTRToXYZ(wavelengths,stokesData,observerPath="https://files.cie.co.at/CIE_xyz_1964_10deg.csv"):
    
    if os.path.exists(observerPath): 
        standardObservers_CIE1964_10deg = pd.read_csv(observerPath)
    else:
        #load standard observers: https://cie.co.at/datatable/cie-1964-colour-matching-functions-10-degree-observer
        colnames = ['wL', 'xbar', 'ybar','zbar']
        standardObservers_CIE1964_10deg = pd.read_csv(observerPath,names=colnames, header=None)      
    
        
    # Assuming illumination is flat at this point across color range (1)
    illumnationSourceProfile = 1
    # Using trapizoidal rule for discrete integration with 1 nm separations
    delWL = 1;
    
    xBar=np.interp(range(380,780,delWL), standardObservers_CIE1964_10deg['wL'].values.tolist(), standardObservers_CIE1964_10deg['xbar'].values.tolist())
    yBar=np.interp(range(380,780,delWL), standardObservers_CIE1964_10deg['wL'].values.tolist(), standardObservers_CIE1964_10deg['ybar'].values.tolist())
    zBar=np.interp(range(380,780,delWL), standardObservers_CIE1964_10deg['wL'].values.tolist(), standardObservers_CIE1964_10deg['zbar'].values.tolist())
    
    xBar[np.isnan(xBar)] = 0
    yBar[np.isnan(yBar)] = 0
    zBar[np.isnan(zBar)] = 0
    
    transmission = convertStokesToCrossedLinearHorzTransmission(stokesData)
    transmission_adjustedRange = np.interp(range(380,780,delWL), wavelengths*1e3, transmission) # Need to convert to both being in nm hence x1e3
    transmission_adjustedRange[np.isnan(transmission_adjustedRange)] = 0
    
    # Math Following:  https://www.konicaminolta.com/instruments/knowledge/color/part4/02.html
    N = np.trapz(range(380,780,delWL),yBar*illumnationSourceProfile)
    
    X = np.trapz(range(380,780,delWL),xBar*illumnationSourceProfile*transmission_adjustedRange)
    Y = np.trapz(range(380,780,delWL),yBar*illumnationSourceProfile*transmission_adjustedRange)
    Z = np.trapz(range(380,780,delWL),zBar*illumnationSourceProfile*transmission_adjustedRange)
    
    X = X / N
    Y = Y / N
    Z = Z / N
    
    return X,Y,Z

def convertXYZToCIELAB(X,Y,Z,Xn=95.0489,Yn=100,Zn=108.8840):
    #D65 of whitepoint in the XYZ Space Xn=95.0489,Yn=100,Zn=108.8840
    L = 116*CIELABColorFunction((100*Y)/Yn)-16
    a = 500*(CIELABColorFunction((100*X)/Xn) - CIELABColorFunction((100*Y)/Yn))
    b = 200*(CIELABColorFunction((100*Y)/Yn) - CIELABColorFunction((100*Z)/Zn))
     
    return L, a, b

def CIELABColorFunction(t):
    delta=6/29
    
    if (t > (delta**3)):
        colorFunction = t**(1/3)
    else:
        colorFunction = ((1/3)*t*(delta**-2))+(4/29)
    return colorFunction

def convertStokesToCrossedLinearHorzTransmission(stokesData):
    transmission = (1-stokesData[:,1])/2
    return transmission

def convertStokesToParallelLinearHorzTransmission(stokesData):
    transmission = (1+stokesData[:,1])/2
    return transmission

def convertStokesToCrossedLinear45Transmission(stokesData):
    transmission = (1-stokesData[:,2])/2
    return transmission

def convertStokesToParallelLinear45Transmission(stokesData):
    transmission = (1+stokesData[:,2])/2
    return transmission

def convertStokesToCrossedCircularTransmission(stokesData):
    transmission = (1-stokesData[:,3])/2
    return transmission

def convertStokesToParallelCircularTransmission(stokesData):
    transmission = (1+stokesData[:,3])/2
    return transmission