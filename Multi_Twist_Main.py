## All General User Defined Functions for Working Specifically with Multi-Twist Retarders

# Analysis Imports
import pandas as pd
import numpy as np
import math
from itertools import compress
import random

from color_manipulations import *
 
def birefringence_formula(wavelength):

    #Nominally the birefringence profile of RMS10-025 used in R. Komanduri, K. Lawler, and M. Escuti, “Multi-twist retarders: broadband retardation control using self-aligning reactive liquid crystal layers,” Optics Express, vol. 21, no. 1, pp. 404–420, 2013.
    # Input wavelength is in microns; the Cauchy coefficients below expect nm
    wavelength = wavelength * (1e-6/1e-9) #Converting from micron to nm (x1000)
    
    ne = 1.629 + 18350/wavelength**2
    no = 1.501 + 10010/wavelength**2
    
    #Following 5CB Cauchy Dispersion Coeffs From Tkachenko, V., Marino, A., & Abbate, G. (2010). Study of Nematic Liquid Crystals by Spectroscopic Ellipsometry. Molecular Crystals and Liquid Crystals, 527(1), 80/[236]-91/[247]. https://doi.org/10.1080/15421406.2010.486366
    #wavelength in micron
    #ne = 1.64499 + (0.01545 / wavelength) + (0.001019 / wavelength**2)
    #no = 1.50945 + (0.00934 / wavelength) + (0.00017 / wavelength**2)
    
    birefringence_single_wl = (ne - no)
    
    return birefringence_single_wl

def bound_generator(numLayers):
    array_length=(numLayers*2)+1
    bound = [] #Needs to be a list not a nparray.
    
    for n in range(0,array_length):
        if (n==0):
            bound.append((-math.pi/2,math.pi/2)) # -90 to +90
        elif( n % 2 == 1): 
            bound.append((-2*math.pi,2*math.pi)) # -360 to +360
        elif( n % 2 == 0): 
            bound.append((0,10))
    
    return bound

def bound_generator_WideGamut(numLayers,numPrimaries):
    array_length=(numLayers*2)+numPrimaries
    bound = [] #Needs to be a list not a nparray.
    
    for n in range(0,array_length):
        if (n >= (numLayers*2)):
            bound.append((-math.pi/2,math.pi/2)) # -90 to +90
        elif( n % 2 == 0): 
            bound.append((-2*math.pi,2*math.pi)) # -360 to +360
        elif( n % 2 == 1): 
            bound.append((0,10))
    
    return bound

def define_chromatic_stokes(wavelengths,key_wavelengths,key_stokes):
    #Builds an N x 4 stokes array that is piecewise constant: key_stokes[k] applies from
    #key_wavelengths[k] up to (not including) key_wavelengths[k+1]; the last key runs to the end.
    wavelengths = np.asarray(wavelengths).reshape(-1)
    key_wavelengths = np.asarray(key_wavelengths).reshape(-1)
    key_stokes = np.asarray(key_stokes)
    
    if (key_stokes.ndim != 2) or (key_stokes.shape[1] != 4):
        raise ValueError(f"key_stokes must have shape (number of keys, 4), got {key_stokes.shape}. "
                         "Each row is one stokes vector [S0, S1, S2, S3].")
    if key_stokes.shape[0] != key_wavelengths.shape[0]:
        raise ValueError(f"Got {key_wavelengths.shape[0]} key wavelengths but {key_stokes.shape[0]} key stokes rows; "
                         "they need one row of key_stokes per key wavelength.")
    if not np.isclose(key_wavelengths[0], wavelengths[0]):
        raise ValueError(f"key_wavelengths[0] ({key_wavelengths[0]}) must equal the first wavelength in the series "
                         f"({wavelengths[0]}), otherwise the start of the band has no stokes defined.")
    if np.any(np.diff(key_wavelengths) <= 0):
        raise ValueError(f"key_wavelengths must be strictly increasing, got {key_wavelengths.tolist()}")
    
    #Index of each key wavelength in the series (isclose so float rounding like 0.1+0.2 doesn't matter)
    key_indices = []
    for key_wavelength in key_wavelengths:
        matches = np.flatnonzero(np.isclose(wavelengths, key_wavelength, rtol=0, atol=1e-9))
        if matches.size == 0:
            raise ValueError(f"Key wavelength {key_wavelength} is not in the wavelength series "
                             f"({wavelengths[0]} to {wavelengths[-1]}, {wavelengths.size} points). "
                             "Key wavelengths must land exactly on the wavelength grid.")
        key_indices.append(matches[0])
    key_indices.append(wavelengths.size) #last key runs to the end of the series
    
    chromatic_stokes = []
    for key_length in range(0,key_wavelengths.shape[0]):
        count = key_indices[key_length+1]-key_indices[key_length]
        chromatic_stokes.append(np.broadcast_to(key_stokes[key_length,:], (count, 4)))
    
    return np.vstack(chromatic_stokes)

def full_matrix_specification(MTR_specification,wavelength):
    #Takes the MTR specification and builds it into the output muller matrix
    #wavelength may be a scalar (returns a 4x4 matrix) or a 1-D array (returns an N x 4 x 4 stack)
    MTR_specification = np.asarray(MTR_specification)
    array_length = len(MTR_specification)
    num_layer = (array_length-1)/2
    num_layer = int(num_layer) #Makes sure we are in int type for doing math
    
    initial_orientation = MTR_specification[0]
    twist_array = MTR_specification[range(1,array_length,2)]
    thickness_array = MTR_specification[range(2,array_length,2)]
    
    output_muller_matrix = np.broadcast_to(np.eye(4), np.shape(wavelength) + (4, 4))     #Identity matrix to start
    
    for m in range(0, num_layer):
        #for each layer set the twist and thickness rotate the full matrix to line up as we go through.
        twist = twist_array[m]
        thickness = thickness_array[m] 
        MM_layer = twisted_nematic_cell_formula(twist,thickness,wavelength)
        
        total_twist = sum(twist_array[0:m])+initial_orientation #Initial orientation plus the sum of all the prior twists in the system.
        output_muller_matrix = np.matmul(rotate_muller_matrix(MM_layer,total_twist),output_muller_matrix)
        
    return output_muller_matrix

def full_matrix_specification_multi_wL(MTR_specification,wavelengths,input_stokes):
    #Does the wavelength handling of full matrix specification adds in stokes output along with muller matrix
    #All wavelengths are computed at once as an N x 4 x 4 stack of Mueller matrices
    wavelengths = np.asarray(wavelengths).reshape(-1)
    input_stokes = np.asarray(input_stokes)
    
    output_muller_matrix = full_matrix_specification(MTR_specification,wavelengths)
    
    if np.ndim(input_stokes)==1:
        output_stokes = output_muller_matrix @ input_stokes
    elif (np.ndim(input_stokes)== 2) and (wavelengths.shape[0]==input_stokes.shape[0]):
        output_stokes = np.einsum('nij,nj->ni', output_muller_matrix, input_stokes)
    else:
        raise ValueError("input_stokes needs to be 1 entry or multiple -- the length of wavelength input")
    
    return output_muller_matrix, output_stokes

def random_seed_generator(numLayers):
    array_length=(numLayers*2)+1
    x0 = []
    
    for n in range(0,array_length):
        if (n==0):
            x0.append((random.random( )*(math.pi))-math.pi/2) # -90 to +90
        elif( n % 2 == 1): 
            x0.append((random.random( )*(4*math.pi))-2*math.pi) # -360 to +360
        elif( n % 2 == 0): 
            x0.append(random.random( )*10)  
    
    x0=np.array(x0)
    
    return x0

def random_seed_generator_WideGamut(numLayers,numPrimaries):
    array_length=(numLayers*2)+numPrimaries
    x0 = []
    
    for n in range(0,array_length):
        if (n >= (numLayers*2)):
            x0.append((random.random( )*(math.pi))-math.pi/2) # -90 to +90
        elif( n % 2 == 0): 
            x0.append((random.random( )*(4*math.pi))-2*math.pi) # -360 to +360
        elif( n % 2 == 1): 
            x0.append(random.random( )*10)  
    
    x0=np.array(x0)
    
    return x0

def reform_WideGamut_seed_to_standard(x0,keyPrimary,numLayers):
    #keyPrimary: index 0-(N-1) where N == length of primaries in data
    keyPrimaryIndex=(numLayers*2)+keyPrimary 
    array_length=(numLayers*2)
    x0Prime=[]
    
    x0Prime.append(x0[keyPrimaryIndex])
    for n in range(0,array_length):
        x0Prime.append(x0[n])
    
    x0Prime=np.array(x0Prime)
    
    return x0Prime

def retardance_from_birefringence(wavelength,thickness):
    #Returns the retardance in radians
    retardance = (2*math.pi*birefringence_formula(wavelength)*thickness)/wavelength
    return retardance

def rotate_muller_matrix(matrix,rotation):
    # Rotation should be in radians
    pos_rotation_matrix=np.array([[1, 0, 0, 0],[0,math.cos(2*rotation),math.sin(2*rotation),0],[0, -math.sin(2*rotation), math.cos(2*rotation), 0],[0, 0, 0, 1]]);
    neg_rotation_matrix=np.array([[1, 0, 0, 0],[0,math.cos(-2*rotation),math.sin(-2*rotation),0],[0, -math.sin(-2*rotation), math.cos(-2*rotation), 0],[0, 0, 0, 1]]);
    
    #you rotate into the frame and then back out
    rotated_matrix = np.matmul(matrix,pos_rotation_matrix)
    rotated_matrix = np.matmul(neg_rotation_matrix,rotated_matrix)
    
    return rotated_matrix

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
    X = np.sqrt((Norm_Ret**2)+(twist**2))
    # Just modelnig single layer and rotating the whole matrix to the prior orientation rather than keeping track of the orientation throughout.
    Biased_Mean_singleLayer = twist + math.pi
    
    a = (np.cos(X)*np.cos(twist))+(twist*np.sin(twist)*np.sinc(X/np.pi))
    b = -Norm_Ret*np.cos((2*1*Biased_Mean_singleLayer)-twist)*np.sinc(X/np.pi) # in formula setting m==1 because "single" layer
    c = (np.cos(X)*np.sin(twist))-(twist*np.cos(twist)*np.sinc(X/np.pi))
    d = -Norm_Ret*np.sin((2*1*Biased_Mean_singleLayer)-twist)*np.sinc(X/np.pi) # in formula setting m==1 because "single" layer
    
    # Works for a scalar wavelength (4x4) or an array of wavelengths (N x 4 x 4)
    single_wl_single_layer_MM = np.zeros(np.shape(X) + (4, 4))
    single_wl_single_layer_MM[..., 0, 0] = 1
    single_wl_single_layer_MM[..., 1, 1] = 1-2*(c**2+d**2)
    single_wl_single_layer_MM[..., 1, 2] = 2*((b*d)-(a*c))
    single_wl_single_layer_MM[..., 1, 3] = -2*((a*d)+(b*c))
    single_wl_single_layer_MM[..., 2, 1] = 2*((a*c)+(b*d))
    single_wl_single_layer_MM[..., 2, 2] = 1-2*(b**2 + c**2)
    single_wl_single_layer_MM[..., 2, 3] = 2*((a*b)-(c*d))
    single_wl_single_layer_MM[..., 3, 1] = 2*((a*d)-(b*c))
    single_wl_single_layer_MM[..., 3, 2] = -2*((a*b)+(c*d))
    single_wl_single_layer_MM[..., 3, 3] = 1-2*(b**2 + d**2)
    
    return  single_wl_single_layer_MM


## Two-film (sandwich) elements
# Two MTR films, each coated on its own piece of glass, sandwiched film-side to film-side and
# rotated independently between fixed polarizers. Light passes through film 1 then film 2.
# The glass is treated as isotropic and reflections are ignored, so the sandwich is just the
# product of the two films' Mueller matrices.

def flip_MTR_specification(MTR_specification):
    #Spec of a film flipped over (rotated 180 degrees about the in-plane x axis), as seen by light
    #arriving from the original exit side. Layers are met in reverse order, azimuths are mirrored
    #(angle -> -angle), and each layer keeps its own twist and thickness (chirality is unchanged
    #by a flip). The new entrance azimuth is the mirrored exit azimuth of the original film.
    MTR_specification = np.asarray(MTR_specification, dtype=float)
    twist_array = MTR_specification[1::2]
    thickness_array = MTR_specification[2::2]
    
    flipped = [-(MTR_specification[0] + np.sum(twist_array))]
    for twist, thickness in zip(twist_array[::-1], thickness_array[::-1]):
        flipped.extend([twist, thickness])
    
    return np.array(flipped)

def rotate_MTR_specification(MTR_specification, rotation):
    #Spec of the same film rotated in its own plane by `rotation` radians
    rotated = np.array(MTR_specification, dtype=float)
    rotated[0] = rotated[0] + rotation
    return rotated

def rotation_muller_matrices(rotations):
    #Stack of Mueller rotation matrices (A x 4 x 4) for an array of A angles, same convention as rotate_muller_matrix
    rotations = np.asarray(rotations, dtype=float).reshape(-1)
    R = np.zeros((rotations.size, 4, 4))
    R[:, 0, 0] = 1
    R[:, 3, 3] = 1
    R[:, 1, 1] = np.cos(2*rotations)
    R[:, 1, 2] = np.sin(2*rotations)
    R[:, 2, 1] = -np.sin(2*rotations)
    R[:, 2, 2] = np.cos(2*rotations)
    return R

def rotate_muller_stack(muller_matrices, rotations):
    #Rotates an N x 4 x 4 stack (one matrix per wavelength) to each of A angles -> A x N x 4 x 4
    #Equivalent to rotate_muller_matrix(M, angle) for every angle, but computed all at once
    R_pos = rotation_muller_matrices(rotations)[:, None, :, :]
    R_neg = rotation_muller_matrices(-np.asarray(rotations, dtype=float))[:, None, :, :]
    return R_neg @ muller_matrices[None, :, :, :] @ R_pos

def two_film_muller_matrix(MTR_specification_1, MTR_specification_2, wavelengths, rotation_1=0.0, rotation_2=0.0, flip_second=True):
    #N x 4 x 4 Mueller matrices of the sandwich: film 1 rotated by rotation_1, then film 2 rotated by rotation_2.
    #flip_second=True is the film-to-film sandwich (film 2 is face down); False stacks both films facing the same way.
    #Rotations are applied in the lab frame, after any flip.
    wavelengths = np.asarray(wavelengths).reshape(-1)
    film_2 = flip_MTR_specification(MTR_specification_2) if flip_second else np.asarray(MTR_specification_2, dtype=float)
    
    M1 = full_matrix_specification(rotate_MTR_specification(MTR_specification_1, rotation_1), wavelengths)
    M2 = full_matrix_specification(rotate_MTR_specification(film_2, rotation_2), wavelengths)
    
    return M2 @ M1

def two_film_rotation_sweep(MTR_specification_1, MTR_specification_2, wavelengths, rotations_1, rotations_2, input_stokes, flip_second=True):
    #Output stokes for every combination of film rotations: shape (A1, A2, N, 4).
    #Each film's Mueller matrices are computed once and then only rotated, so this is fast for large grids.
    wavelengths = np.asarray(wavelengths).reshape(-1)
    input_stokes = np.asarray(input_stokes, dtype=float)
    if np.ndim(input_stokes) == 1:
        input_stokes = np.broadcast_to(input_stokes, (wavelengths.size, 4))
    elif input_stokes.shape != (wavelengths.size, 4):
        raise ValueError(f"input_stokes must have shape (4,) or ({wavelengths.size}, 4), got {input_stokes.shape}")
    
    film_2 = flip_MTR_specification(MTR_specification_2) if flip_second else np.asarray(MTR_specification_2, dtype=float)
    M1 = full_matrix_specification(MTR_specification_1, wavelengths)
    M2 = full_matrix_specification(film_2, wavelengths)
    
    M1_rotated = rotate_muller_stack(M1, rotations_1)               # A1 x N x 4 x 4
    M2_rotated = rotate_muller_stack(M2, rotations_2)               # A2 x N x 4 x 4
    
    stokes_after_1 = np.einsum('anij,nj->ani', M1_rotated, input_stokes)        # A1 x N x 4
    return np.einsum('bnij,anj->abni', M2_rotated, stokes_after_1)            # A1 x A2 x N x 4

# Which rotations give distinct colors: turning BOTH films by 90 degrees never changes the color (it is the same as
# rotating the whole setup by 90 degrees, which only swaps which polarizer is horizontal; lossless retarders have
# M01 = M10 = 0, so that swap leaves the transmission unchanged). Turning one film alone by 90 degrees generally
# does change it. So every distinct color appears once with film 1 in [0, 90) and film 2 in [0, 180) degrees.
ROTATION_1_RANGE = (0.0, math.pi/2)
ROTATION_2_RANGE = (0.0, math.pi)

def two_film_rotation_grid(step):
    #Rotations (radians) covering every distinct setting exactly once: film 1 over [0, 90), film 2 over [0, 180)
    return np.arange(ROTATION_1_RANGE[0], ROTATION_1_RANGE[1] - 1e-12, step), np.arange(ROTATION_2_RANGE[0], ROTATION_2_RANGE[1] - 1e-12, step)

def canonical_two_film_rotations(rotation_1, rotation_2):
    #The equivalent setting (same color) with film 1 in [0, 90) and film 2 in [0, 180) degrees, in radians
    quarter_turns = np.floor(rotation_1/(math.pi/2))
    return rotation_1 - quarter_turns*math.pi/2, (rotation_2 - quarter_turns*math.pi/2) % math.pi
