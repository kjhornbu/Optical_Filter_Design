#Just generally reproducing the 3TR HW-B for a start for the Analysis

# Analysis 
import pandas as pd
import numpy as np
import math
from scipy.optimize import minimize
from matplotlib import pyplot as plt

# importing all the functions
# defined in Multi_Twist_Main.py
from Multi_Twist_Main import *

#Replicating the 3TR HW-B from R. Komanduri, K. Lawler, and M. Escuti, “Multi-twist retarders: broadband retardation control using self-aligning reactive liquid crystal layers,” Optics Express, vol. 21, no. 1, pp. 404–420, 2013.
MTR_specification = np.array([47.3*(math.pi/180), 76.4*(math.pi/180),1.1, 0*(math.pi/180), 2.27, -76.4*(math.pi/180), 1.1]) # in radians and micron
wavelengths = np.arange(0.400,0.800,0.001)# in micron

input_light = np.array([1,0,0,0]) #unpolarized input light
polarizer = np.array([[1, 0, 0, 1],[0, 0, 0, 0], [0, 0, 0, 0], [1,0, 0, 1]]) #Right Circular Polarizer (without attenuation)
input_stokes = np.matmul(polarizer,input_light)

output_stokes = []

for wavelength in wavelengths:
    
    output_muller_matrix = full_matrix_specification(MTR_specification,wavelength)
    stokes_output_temp=np.matmul(output_muller_matrix,input_stokes)
    output_stokes.append(stokes_output_temp[3])

#Reproducing Graph 
plt.plot(wavelengths*1e3,output_stokes)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Output $S_{3}$ (-)')
plt.axis([400, 800, -1, -0.95])

plt.show()

