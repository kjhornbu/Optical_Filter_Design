#Making a cost function based on the prior work and creating a wrapper function for dealing with multi wavelengths

# Analysis 
import pandas as pd
import numpy as np
import math
from scipy.optimize import minimize
from matplotlib import pyplot as plt

# importing all the functions
# defined in Multi_Twist_Main.py
from Multi_Twist_Main import *
from cost_function import *

#Replicating the 3TR HW-B from R. Komanduri, K. Lawler, and M. Escuti, “Multi-twist retarders: broadband retardation control using self-aligning reactive liquid crystal layers,” Optics Express, vol. 21, no. 1, pp. 404–420, 2013.
MTR_specification = np.array([47.3*(math.pi/180), 76.4*(math.pi/180),1.1, 0*(math.pi/180), 2.27, -76.4*(math.pi/180), 1.1]) # in radians and micron
wavelengths = np.arange(0.400,0.800,0.001)# in micron

input_light = np.array([1,0,0,0]) #unpolarized input light
polarizer = np.array([[1, 0, 0, 1],[0, 0, 0, 0], [0, 0, 0, 0], [1,0, 0, 1]]) #Right Circular Polarizer (without attenuation)

input_stokes = np.matmul(polarizer,input_light)
target_stokes = np.array([1,0,0,-1]) #Left Circular Polarization

output_muller_matrix, output_stokes = full_matrix_specification_multi_wL(MTR_specification,wavelengths,input_stokes)
cost_function = cost_function_stokes(target_stokes,output_stokes,wavelengths)

print(cost_function)