#Applying a single seed to minimize a Green/Magenta Filter using the nelder-mead method in scipy

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

MTR_specification = random_seed_generator(3)

wavelengths = np.arange(400,800,1)/1e3 #in micron
key_wavelengths = np.array([[0.4],[0.495],[0.570]])
key_stokes_input = np.array([[1,1,0,0],[1,1,0,0],[1,1,0,0]]) # LH, LH, LH
key_stokes_target = np.array([[1,1,0,0],[1,-1,0,0],[1,1,0,0]]) #LH LV, LH

input_stokes = define_chromatic_stokes(wavelengths,key_wavelengths,key_stokes_input)
target_stokes = define_chromatic_stokes(wavelengths,key_wavelengths,key_stokes_target)

res = minimize(function_to_minimize,MTR_specification, method='nelder-mead', args=(wavelengths,target_stokes,input_stokes),options={'xatol': 1e-8, 'disp': True})

#Get resulting Output Spectra for the minimized design
output_muller_matrix, output_stokes = full_matrix_specification_multi_wL(MTR_specification,wavelengths,input_stokes)

#Plot Resulting Spectra with Target in Cross Configuration
plt.plot(wavelengths*1e3,100*(1-output_stokes[:,1])/2)
plt.plot(wavelengths*1e3,100*(1-target_stokes[:,1])/2)

plt.xlabel('Wavelength (nm)')
plt.ylabel('Output Transmission (%)')
plt.axis([400, 800, -2.5, 102.5])
plt.legend(['3TR Design', 'Target Spectra']) 

plt.show()