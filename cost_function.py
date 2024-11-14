## Cost Functions Utilized in This work
# Analysis Imports
import pandas as pd
import numpy as np
import math
import warnings

def cost_function_stokes(Target,Output,wavelengths):
    
    wavelengths_size=np.shape(wavelengths)
    Target_size=np.shape(Target)
    Output_size=np.shape(Output)
    
    if (np.ndim(Target)==1):
        #If Target is 1-D (achromatic condition)
        f=np.mean(1-np.dot(Output[:,1:4],Target[1:4]))
    elif ((np.ndim(Target)==2) and (wavelengths_size[0]==Target_size[0])) and ((np.ndim(Output)==2) and (wavelengths_size[0]==Output_size[0])):
        #If both Target and Output are 2D and have length of first dimension == number of wavelengths (chromatic condition)
        f=np.mean(1-((Output[:,1]*Target[:,1])+(Output[:,2]*Target[:,2])+(Output[:,3]*Target[:,3])))
    else:
            warnings.warn("Target and Output need to have same length as wavelengths or Target must be uniform across wavelength band")  
    return f


        