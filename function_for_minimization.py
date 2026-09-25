## The functions that are minimized to run the actual algorithm

# Analysis Imports
import pandas as pd
import numpy as np
import math
import warnings
from itertools import compress
import random

from cost_function import *
from color_manipulations import *

from function_for_minimization import *
def function_to_minimize_WideGamut(MTR_specification,numLayers,numPrimaries,wavelengths,target_stokes,input_stokes):
    output_stokes=[]
    
    for n in range(0,numPrimaries):
        MTR_specificationPrime=reform_WideGamut_seed_to_standard(MTR_specification,n,numLayers)
        output_muller_matrix, output_stokes_temp = full_matrix_specification_multi_wL(MTR_specificationPrime,wavelengths,input_stokes)
        output_stokes.append(output_stokes_temp)
    
    output_stokes=np.array(output_stokes)
    cost_function = cost_function_WideGamut(target_stokes,output_stokes,wavelengths)
    
    return cost_function

def function_to_minimize_WideGamutLAB(MTR_specification,numLayers,numPrimaries,wavelengths,target_stokes,input_stokes):
    output_stokes=[]
    
    for n in range(0,numPrimaries):
        MTR_specificationPrime=reform_WideGamut_seed_to_standard(MTR_specification,n,numLayers)
        output_muller_matrix, output_stokes_temp = full_matrix_specification_multi_wL(MTR_specificationPrime,wavelengths,input_stokes)
        output_stokes.append(output_stokes_temp)
    
    output_stokes=np.array(output_stokes)
    cost_function = cost_function_WideGamutCIELAB(target_stokes,output_stokes,wavelengths)
    
    return cost_function

def function_to_minimize(MTR_specification,wavelengths,target_stokes,input_stokes):
    output_muller_matrix, output_stokes = full_matrix_specification_multi_wL(MTR_specification,wavelengths,input_stokes)
    cost_function = cost_function_WideGamut(target_stokes,output_stokes,wavelengths)
    return cost_function
