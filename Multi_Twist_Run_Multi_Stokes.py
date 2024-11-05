#Testing if we can generate a vector of input and target with the current function

# Analysis 
import pandas as pd
import numpy as np
import math
from scipy.optimize import minimize
from matplotlib import pyplot as plt

# importing all the functions
# defined in Multi_Twist_Main.py
from Multi_Twist_Main import *
from cost_function_stokes import *

#Replicating the 3TR HW-B from R. Komanduri, K. Lawler, and M. Escuti, “Multi-twist retarders: broadband retardation control using self-aligning reactive liquid crystal layers,” Optics Express, vol. 21, no. 1, pp. 404–420, 2013.
MTR_specification = np.array([47.3*(math.pi/180), 76.4*(math.pi/180),1.1, 0*(math.pi/180), 2.27, -76.4*(math.pi/180), 1.1]) # in radians and micron
wavelengths = np.arange(0.400,0.800,0.001)# in micron
wavelengths_size = np.shape(wavelengths)

input_stokes = np.array([1,0,0,1]) #Right Circular Polarization
input_stokes = np.broadcast_to(input_stokes, (wavelengths_size[0], 4))

target_stokes = np.array([1,0,0,-1]) #Left Circular Polarization
target_stokes = np.broadcast_to(target_stokes, (wavelengths_size[0], 4))

output_muller_matrix, output_stokes = full_matrix_specification_multi_wL(MTR_specification,wavelengths,input_stokes)
cost_function = cost_function_stokes(target_stokes,output_stokes,wavelengths)

print(cost_function)