#Applying a multi seed to minimize a "Wide Gamut" using Simulated Dual Annealing to find "Global" Solutions
import time
start_time = time.time()

# Analysis 
import pandas as pd
import numpy as np
import math
from scipy.optimize import minimize, dual_annealing
from matplotlib import pyplot as plt

# importing all the functions
# defined in Multi_Twist_Main.py
from Multi_Twist_Main import *
from cost_function import *
from color_manipulations import *

# Get Input and Target Setup for the spectra desired
wavelengths = np.arange(380,800,1)/1e3 #in micron

key_wavelengths = np.array([[0.38],[0.425],[0.600]])
key_stokes_input = np.array([[1,1,0,0],[1,1,0,0],[1,1,0,0]]) # LH, LH, LH
input_stokes = define_chromatic_stokes(wavelengths,key_wavelengths,key_stokes_input)

target_stokes = [];

#Red
key_wavelengths = np.array([[0.38],[0.620],[0.75]])
key_stokes_target= np.array([[1,1,0,0],[1,-1,0,0],[1,1,0,0]]) # LH, LH, LH
target_stokes_temp = define_chromatic_stokes(wavelengths,key_wavelengths,key_stokes_target)
target_stokes.append(target_stokes_temp)

#Green
key_wavelengths = np.array([[0.38],[0.495],[0.570]])
key_stokes_target = np.array([[1,1,0,0],[1,-1,0,0],[1,1,0,0]]) # LH, LH, LH
target_stokes_temp = define_chromatic_stokes(wavelengths,key_wavelengths,key_stokes_target)
target_stokes.append(target_stokes_temp)

# Blue
key_wavelengths = np.array([[0.38],[0.5]])
key_stokes_target = np.array([[1,-1,0,0],[1,1,0,0]]) # LH, LH, LH
target_stokes_temp = define_chromatic_stokes(wavelengths,key_wavelengths,key_stokes_target)
target_stokes.append(target_stokes_temp)

target_stokes = np.array(target_stokes)

#Preliminaries: Number of Seeds to Try and How Many Layers
num_seeds = 2**0
M=3
primaries=3;
bounds = bound_generator_WideGamut(M,primaries)

MTR_seed = []
output_cost=[]
output_MTR=[]

# Find Set of Random Seeds and Iterate over number of seeds
for n in range(0,num_seeds):
    MTR_seed.append(random_seed_generator_WideGamut(M,primaries))
    
    res = dual_annealing(function_to_minimize_WideGamut, bounds=bounds, args=(M,primaries,wavelengths,target_stokes,input_stokes),x0=MTR_seed[n])
    
    output_cost.append(res.fun)
    output_MTR.append(res.x)

MTR_seed=np.array(MTR_seed)
output_cost = np.array(output_cost)
output_MTR = np.array(output_MTR)

#find Minimal Cost function from run
min_idx=np.argmin(output_cost)

#Get resulting Output Spectra for the minimized design
#output_muller_matrix, output_stokes = full_matrix_specification_multi_wL(output_MTR[min_idx],wavelengths,input_stokes)

output_stokes=[]

for n in range(0,primaries):
    MTR_specificationPrime=reform_seed_WideGamut_to_Standard(output_MTR[min_idx],n,M)
    output_muller_matrix, output_stokes_temp = full_matrix_specification_multi_wL(MTR_specificationPrime,wavelengths,input_stokes)
    output_stokes.append(output_stokes_temp)

output_stokes=np.array(output_stokes)
    
print(str(output_MTR[min_idx])+":"+str(output_cost[min_idx]))

np.savetxt('output_MTRDesigns.csv', output_MTR, delimiter=',') 
np.savetxt('output_costFunction_perMTRDesigns.csv', output_cost, delimiter=',') 

#Plot Resulting Spectra with Target in Cross Configuration

for n in range(0,primaries):
    plt.plot(wavelengths*1e3,100*(1-output_stokes[n,:,1])/2)
    plt.plot(wavelengths*1e3,100*(1-target_stokes[n,:,1])/2)

plt.xlabel('Wavelength (nm)')
plt.ylabel('Output Transmission (%)')
plt.axis([400, 800, -2.5, 102.5])
plt.legend([str(M)+"TR Design", 'Target Spectra']) 
plt.title(str(output_MTR[min_idx])+":"+str(output_cost[min_idx]))

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time to find solution: {elapsed_time} seconds")

plt.show()
