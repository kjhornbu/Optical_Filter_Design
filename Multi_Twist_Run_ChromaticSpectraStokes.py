#Making a function for creating a shaped input and target spectra based on key wavelengths of interest-- This generates continous on the wavelengths defined -- step functions

# Analysis 
import pandas as pd
import numpy as np
import math
from scipy.optimize import minimize
from matplotlib import pyplot as plt
import random 

# importing all the functions
# defined in Multi_Twist_Main.py
from Multi_Twist_Main import *
from cost_function import *

#Replicating the 3TR HW-B from R. Komanduri, K. Lawler, and M. Escuti, “Multi-twist retarders: broadband retardation control using self-aligning reactive liquid crystal layers,” Optics Express, vol. 21, no. 1, pp. 404–420, 2013.
MTR_specification = np.array([47.3*(math.pi/180), 76.4*(math.pi/180),1.1, 0*(math.pi/180), 2.27, -76.4*(math.pi/180), 1.1]) # in radians and micron


(random.random( )*(2*math.pi))-math.pi,

wavelengths = np.arange(400,800,1)/1e3 #in micron
key_wavelengths = np.array([[0.4],[0.5],[0.6]])
key_stokes_input = np.array([[1,0,0,-1],[1,0,0,1],[1,0,0,-1]]) # LCP, RCP, LCP
key_stokes_target = np.array([[1,0,0,1],[1,0,0,-1],[1,0,0,1]]) # RCP, LCP, RCP

#HWP so flip response so this is exactly the same as an achromatic input!

input_stokes = define_chromatic_stokes(wavelengths,key_wavelengths,key_stokes_input)
target_stokes = define_chromatic_stokes(wavelengths,key_wavelengths,key_stokes_target)

output_muller_matrix, output_stokes = full_matrix_specification_multi_wL(MTR_specification,wavelengths,input_stokes)
cost_function = cost_function_stokes(target_stokes,output_stokes,wavelengths)

print(cost_function)
#Reproducing Graph 
plt.plot(wavelengths*1e3,output_stokes)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Stokes Output (-)')
plt.axis([400, 800, -1, 1])

plt.show()