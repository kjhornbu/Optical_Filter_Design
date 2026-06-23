## Cost Functions Utilized in This work
# Analysis Imports
import pandas as pd
import numpy as np
import math
import warnings

from color_manipulations import *

def cost_function_stokes(Target,Output,wavelengths):
    
    wavelengths_size=np.shape(wavelengths)
    Target_size=np.shape(Target)
    Output_size=np.shape(Output)
    
    if (np.ndim(Target)==1):
        #If Target is 1-D (achromatic condition)
        f=np.mean(1-np.dot(Output[:,1:4],Target[1:4]))
    elif (((np.ndim(Target)==2) and (wavelengths_size[0]==Target_size[0])) and ((np.ndim(Output)==2) and (wavelengths_size[0]==Output_size[0]))):
        #If both Target and Output are 2D and have length of first dimension == number of wavelengths (chromatic condition)
        f=np.mean(1-((Output[:,1]*Target[:,1])+(Output[:,2]*Target[:,2])+(Output[:,3]*Target[:,3])))
    else:
        warnings.warn("Target and Output need to have same length as wavelengths or Target must be uniform across wavelength band")  
    return f

def cost_function_WideGamut(Target,Output,wavelengths):
    #Using the same setup as the stokes but for multiple "primaries"
    wavelengths_size=np.shape(wavelengths)
    Target_size=np.shape(Target)
    Output_size=np.shape(Output)
    
    if (Target_size[1]==1):
        #If Target is 1-D (achromatic condition)
        temp = 0;
        for n in range(0,Output_size[0]):
            temp=np.dot(Output[n,:,1:4],Target[1:4]) + temp 
        
        f = np.mean(Output_size[0]-temp)
        
    elif (((np.ndim(Target)==3) and (wavelengths_size[0]==Target_size[1])) and ((np.ndim(Output)==3) and (wavelengths_size[0]==Output_size[1]))):
        #If both Target and Output are 2D and have length of first dimension == number of wavelengths (chromatic condition)
        
        temp = 0;
        for n in range(0,Output_size[0]):
            temp = ((Output[n,:,1]*Target[n,:,1])+(Output[n,:,2]*Target[n,:,2])+(Output[n,:,3]*Target[n,:,3])) + temp
            
        f = np.mean(Output_size[0]-temp)
    else:
        warnings.warn("Target and Output need to have same length as wavelengths or Target must be uniform across wavelength band")  
    return f

def cost_function_WideGamutCIELAB(Target,Output,wavelengths):
    # Convert Target and Output to Colors
    wavelengths_size=np.shape(wavelengths)
    Target_size=np.shape(Target)
    Output_size=np.shape(Output)

    #https://books.google.com/books?id=OxlBqY67rl0C&q=jnd+gaurav+sharma&pg=PA31#v=snippet&q=jnd%20gaurav%20sharma&f=false
    justNoticeableDifference = 2.3
    f = 0
    
    if(Target_size[1]==1):
        Xtar, Ytar, Ztar = convertMTRToXYZ(wavelengths,Target,"/Users/kjh60/Documents/CIE_xyz_1964_10deg.csv")
        Ltar, Atar, Btar = convertXYZToCIELAB(Xtar,Ytar,Ztar)
        for n in range(0,Output_size[0]):
                       
            Xout, Yout, Zout = convertMTRToXYZ(wavelengths,Output[n,:,:],"/Users/kjh60/Documents/CIE_xyz_1964_10deg.csv")
            Lout, Aout, Bout = convertXYZToCIELAB(Xout,Yout,Zout)

            f = np.sqrt((Lout-Ltar)**2 + (Aout-Atar)**2 + (Bout-Btar)**2) + f

        f = (f/Output_size[0])/justNoticeableDifference
        #How many "shades" out of just noticable                       
    
    elif (((np.ndim(Target)==3) and (wavelengths_size[0]==Target_size[1])) and ((np.ndim(Output)==3) and (wavelengths_size[0]==Output_size[1]))):
        
        for n in range(0,Output_size[0]):
            Xtar, Ytar, Ztar = convertMTRToXYZ(wavelengths,Target[n,:,:],"/Users/kjh60/Documents/CIE_xyz_1964_10deg.csv")
            Ltar, Atar, Btar = convertXYZToCIELAB(Xtar,Ytar,Ztar)

            Xout, Yout, Zout = convertMTRToXYZ(wavelengths,Output[n,:,:],"/Users/kjh60/Documents/CIE_xyz_1964_10deg.csv")
            Lout, Aout, Bout = convertXYZToCIELAB(Xout,Yout,Zout)

            f = np.sqrt((Lout-Ltar)**2 + (Aout-Atar)**2 + (Bout-Btar)**2) + f

        f = (f/Output_size[0])/justNoticeableDifference  
        #How many "shades" out of just noticable  
                   
    else:
         warnings.warn("Target and Output need to have same length as wavelengths or Target must be uniform across wavelength band")  
                       
    return f

        