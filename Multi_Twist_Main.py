# Analysis Imports
import pandas as pd
import numpy as np
import math
import warnings
from itertools import compress
import random
from cost_function_stokes import *

def function_to_minimize(MTR_specification,wavelengths,target_stokes,input_stokes):
    output_muller_matrix, output_stokes = full_matrix_specification_multi_wL(MTR_specification,wavelengths,input_stokes)
    cost_function = cost_function_stokes(target_stokes,output_stokes,wavelengths)
    
    return cost_function

def random_seed_generator(numLayers):
    array_length=(numLayers*2)+1
    x0 = []
    
    for n in range(0,array_length):
        if (n==0):
            x0.append((random.random( )*(2*math.pi))-math.pi)
        elif( n % 2 == 1): 
            x0.append((random.random( )*(2*math.pi))-math.pi)
        elif( n % 2 == 0): 
            x0.append(random.random( )*10)
    
    x0=np.array(x0)
    
    return x0
def define_chromatic_stokes(wavelengths,key_wavelengths,key_stokes):
    wavelengths_size = np.shape(wavelengths)
    key_wavelengths_size = np.shape(key_wavelengths)
    key_stokes_size = np.shape(key_stokes)
    
    chromatic_stokes = []
    
    if (key_wavelengths[0] != wavelengths[0]): 
        warnings.warn("You must have key_wavelength[0] equal to the first wavelength in the series")
        
    if (key_stokes_size[0] == key_wavelengths_size[0]):
        for key_length in range(0,key_wavelengths_size[0]):
            
            if (key_length <  key_wavelengths_size[0]-1):
                #For any key position besides last one figure out the number of entries between itself and the next entry
                t1 = wavelengths == key_wavelengths[key_length]
                t2 = wavelengths == key_wavelengths[key_length+1]
                
                t1_index = list(compress(range(len(t1)), t1))
                t2_index = list(compress(range(len(t2)), t2))
                
            elif (key_length ==  key_wavelengths_size[0]-1):
                #for last key position figure out the distance between itself and the full wavelength size (the last entry in the array)
                t1 = wavelengths == key_wavelengths[key_length]
                t1_index = list(compress(range(len(t1)), t1))
                
                t2_index[0]= wavelengths_size[0]
            
            temp_key_stokes = np.broadcast_to(key_stokes[key_length,:], (t2_index[0]-t1_index[0], 4))
            chromatic_stokes.append(temp_key_stokes)
    else:
        warnings.warn("Define the # of entries in the Key Wavelengths to be the same as Key Stokes. The second dimension of Key Stokes should be 4, not the first.")
    
    chromatic_stokes = np.vstack((chromatic_stokes[:]))
    chromatic_stokes_size = np.shape(chromatic_stokes)
    
    if (chromatic_stokes_size[0] != wavelengths_size[0]):
        warnings.warn("Hey the chromatic stokes generated is not the same size as the wavelengths you are defining for! Double check your key wavelength definitions")
    
    return chromatic_stokes

def full_matrix_specification_multi_wL(MTR_specification,wavelengths,input_stokes):
    #Does the wavelength handling of full matrix specification adds in stokes output along with muller matrix
    wavelengths_size=np.shape(wavelengths)
    input_stokes_size=np.shape(input_stokes)
    
    output_muller_matrix=[];
    output_stokes=[];
    count = 0
    
    for wavelength in wavelengths:
        temp_muller_matrix=full_matrix_specification(MTR_specification,wavelength)
        output_muller_matrix.append(temp_muller_matrix)
        
        if np.ndim(input_stokes)==1:
            temp_stokes=np.matmul(temp_muller_matrix,input_stokes)
            output_stokes.append(temp_stokes)
        elif (np.ndim(input_stokes)== 2) and (wavelengths_size[0]==input_stokes_size[0]):
            temp_stokes=np.matmul(temp_muller_matrix,input_stokes[count,:])
            output_stokes.append(temp_stokes)
            count = count+1
        else:
            warnings.warn("input_stokes needs to be 1 entry or multiple -- the length of wavelength input")    
    
    output_muller_matrix=np.array(output_muller_matrix)
    output_stokes=np.array(output_stokes)
    
    return output_muller_matrix, output_stokes

def full_matrix_specification(MTR_specification,wavelength):
    #Takes the MTR specification and builds it into the output muller matrix
    array_length = len(MTR_specification)
    num_layer = (array_length-1)/2
    num_layer = int(num_layer) #Makes sure we are in int type for doing math
    
    initial_orientation = MTR_specification[0]
    twist_array = MTR_specification[range(1,array_length,2)]
    thickness_array = MTR_specification[range(2,array_length,2)]
    
    output_muller_matrix = np.array([[1, 0, 0, 0],[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])     #Identity matrix to start
    
    for m in range(0, num_layer):
        #for each layer set the twist and thickness rotate the full matrix to line up as we go through.
        twist = twist_array[m]
        thickness = thickness_array[m] 
        MM_layer = twisted_nematic_cell_formula(twist,thickness,wavelength)
        
        if (m == 0):
            output_muller_matrix = np.matmul(rotate_muller_matrix(MM_layer,initial_orientation),output_muller_matrix)
        else:
            total_twist = sum(twist_array[0:m])+initial_orientation #Initial orientation plus the sum of all the angles throughout the system.
            output_muller_matrix = np.matmul(rotate_muller_matrix(MM_layer,total_twist),output_muller_matrix)
        
    return output_muller_matrix

def rotate_muller_matrix(matrix,rotation):
    # Rotation should be in radians
    pos_rotation_matrix=np.array([[1, 0, 0, 0],[0,math.cos(2*rotation),math.sin(2*rotation),0],[0, -math.sin(2*rotation), math.cos(2*rotation), 0],[0, 0, 0, 1]]);
    neg_rotation_matrix=np.array([[1, 0, 0, 0],[0,math.cos(-2*rotation),math.sin(-2*rotation),0],[0, -math.sin(-2*rotation), math.cos(-2*rotation), 0],[0, 0, 0, 1]]);
    
    #you rotate into the frame and then back out
    rotated_matrix = np.matmul(matrix,pos_rotation_matrix)
    rotated_matrix = np.matmul(neg_rotation_matrix,rotated_matrix)
    
    return rotated_matrix

def retardance_from_birefringence(wavelength,thickness):
    #Returns the retardance in radians
    retardance = (2*math.pi*birefringence_formula(wavelength)*thickness)/wavelength
    return retardance

def twisted_nematic_cell_formula(twist,thickness,wavelength):
    #Twist should be in radians
    
    #Doped twist model by Tang and Kwok:
    #S. T. Tang and H. S. Kwok, “Mueller calculus and perfect polarization conversion modes in liquid crystal displays,” Journal of Applied Physics, vol. 89, no. 10, pp. 5288–5294, 2001.
    #Expressed for MTRs:
    # R. Komanduri, K. Lawler, and M. Escuti, “Multi-twist retarders: broadband retardation control using self-aligning reactive liquid crystal layers,” Optics Express, vol. 21, no. 1, pp. 404–420, 2013.
    # Used Extenstively in Dissertation:
    # A Study of Aspherical Geometric-Phase Lenses and Retarders Formed by Liquid Crystal Polymer Films. by Kathryn Hornburg 
    # http://www.lib.ncsu.edu/resolver/1840.20/36277
    
    Norm_Ret = retardance_from_birefringence(wavelength,thickness)/2.0 #normalized retardance is regular retardation by 2
    X = math.sqrt((Norm_Ret**2)+(twist**2))
    # Just modelnig single layer and rotating the whole matrix to the prior orientation rather than keeping track of the orientation throughout.
    Biased_Mean_singleLayer = twist + math.pi
    
    a = (math.cos(X)*math.cos(twist))+(twist*math.sin(twist)*np.sinc(X/np.pi))
    b = -Norm_Ret*math.cos((2*1*Biased_Mean_singleLayer)-twist)*np.sinc(X/np.pi) # in formula setting m==1 because "single" layer
    c = (math.cos(X)*math.sin(twist))-(twist*math.cos(twist)*np.sinc(X/np.pi))
    d = -Norm_Ret*math.sin((2*1*Biased_Mean_singleLayer)-twist)*np.sinc(X/np.pi) # in formula setting m==1 because "single" layer
    
    M11 = 1-2*(c**2+d**2)
    M12 = 2*((b*d)-(a*c))
    M13 = -2*((a*d)+(b*c))
    M21 = 2*((a*c)+(b*d))
    M22 = 1-2*(b**2 + c**2)
    M23 = 2*((a*b)-(c*d))
    M31 = 2*((a*d)-(b*c))
    M32 = -2*((a*b)+(c*d))
    M33 = 1-2*(b**2 + d**2)
    
    single_wl_single_layer_MM=np.array([[1, 0, 0, 0],[0, M11, M12, M13],[0, M21, M22, M23],[0, M31, M32, M33]])
    
    return  single_wl_single_layer_MM

def birefringence_formula(wavelength):

    #Nominally the birefringence profile of RMS10-025 used in R. Komanduri, K. Lawler, and M. Escuti, “Multi-twist retarders: broadband retardation control using self-aligning reactive liquid crystal layers,” Optics Express, vol. 21, no. 1, pp. 404–420, 2013.
    wavelength = wavelength * (1e-6/1e-9) #Converting from micron to nm
    
    ne = 1.629 + 18350/wavelength**2
    no = 1.501 + 10010/wavelength**2
    
    #Following 5CB Cauchy Dispersion Coeffs From Tkachenko, V., Marino, A., & Abbate, G. (2010). Study of Nematic Liquid Crystals by Spectroscopic Ellipsometry. Molecular Crystals and Liquid Crystals, 527(1), 80/[236]-91/[247]. https://doi.org/10.1080/15421406.2010.486366
    #wavelength in micron
    #ne = 1.64499 + (0.01545 / wavelength) + (0.001019 / wavelength**2)
    #no = 1.50945 + (0.00934 / wavelength) + (0.00017 / wavelength**2)
    
    birefringence_single_wl = (ne - no)
    
    return birefringence_single_wl

    