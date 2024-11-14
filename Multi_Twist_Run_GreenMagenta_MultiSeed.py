#Applying a multi seed to minimize a Green/Magenta Filter using the nelder-mead method in scipy
import time
start_time = time.time()

# Analysis 
import pandas as pd
import numpy as np
import math
from scipy.optimize import minimize, differential_evolution
from matplotlib import pyplot as plt



# importing all the functions
# defined in Multi_Twist_Main.py
from Multi_Twist_Main import *
from cost_function import *

# Get Input and Target Setup for the spectra desired
wavelengths = np.arange(400,800,1)/1e3 #in micron
key_wavelengths = np.array([[0.4],[0.495],[0.570]])
key_stokes_input = np.array([[1,1,0,0],[1,1,0,0],[1,1,0,0]]) # LH, LH, LH
key_stokes_target = np.array([[1,1,0,0],[1,-1,0,0],[1,1,0,0]]) #LH LV, LH

input_stokes = define_chromatic_stokes(wavelengths,key_wavelengths,key_stokes_input)
target_stokes = define_chromatic_stokes(wavelengths,key_wavelengths,key_stokes_target)

#Preliminaries: Number of Seeds to Try and How Many Layers
num_seeds = 2**3
M=3

MTR_seed = []
output_cost=[]
output_MTR=[]

# Find Set of Random Seeds and Iterate over number of seeds
for n in range(0,num_seeds):
    MTR_seed.append(random_seed_generator(M))
    
    res = minimize(function_to_minimize,MTR_seed[n], method='nelder-mead', args=(wavelengths,target_stokes,input_stokes),options={'xatol': 1e-8, 'disp': False})
    
    output_cost.append(res.fun)
    output_MTR.append(res.x)

MTR_seed=np.array(MTR_seed)
output_cost = np.array(output_cost)
output_MTR = np.array(output_MTR)

#find Minimal Cost function from run
min_idx=np.argmin(output_cost)

#Get resulting Output Spectra for the minimized design
output_muller_matrix, output_stokes = full_matrix_specification_multi_wL(output_MTR[min_idx],wavelengths,input_stokes)

print(str(output_MTR[min_idx])+":"+str(output_cost[min_idx]))

np.savetxt('output_MTR_designs.csv', output_MTR, delimiter=',') 
np.savetxt('output_costFunction_perMTRDesigns.csv', output_cost, delimiter=',') 

#Plot Resulting Spectra with Target in Cross Configuration
plt.plot(wavelengths*1e3,100*(1-output_stokes[:,1])/2)
plt.plot(wavelengths*1e3,100*(1-target_stokes[:,1])/2)

plt.xlabel('Wavelength (nm)')
plt.ylabel('Output Transmission (%)')
plt.axis([400, 800, -2.5, 102.5])
plt.legend([str(M)+"TR Design", 'Target Spectra']) 
plt.title(str(output_MTR[min_idx])+":"+str(output_cost[min_idx]))

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Script execution time: {elapsed_time} seconds")

plt.show()