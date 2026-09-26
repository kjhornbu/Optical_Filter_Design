

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


wavelengths = np.arange(400,800,1)/1e3 #in micron
key_wavelengths = np.array([[0.4],[0.5],[0.6]])
key_stokes_input = np.array([[1,0,0,-1],[1,0,0,1],[1,0,0,-1]]) # LCP, RCP, LCP
key_stokes_target = np.array([[1,0,0,1],[1,0,0,-1],[1,0,0,1]]) # RCP, LCP, RCP

#HWP so flip response so this is exactly the same as an achromatic input!

input_stokes = define_chromatic_stokes(wavelengths,key_wavelengths,key_stokes_input)
target_stokes = define_chromatic_stokes(wavelengths,key_wavelengths,key_stokes_target)


res = minimize(function_to_minimize,MTR_specification, method='nelder-mead', args=(wavelengths,target_stokes,input_stokes),options={'xatol': 1e-8, 'disp': True})

print(res)
