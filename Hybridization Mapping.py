# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 15:15:16 2024

@author: Alex Sheardy

Description: This code works to open and process a file created in Digital Micrograph/GMS.  Specifically, this code is intended for performing
mapping of the %sp2 is carbonaceous materials.  The current method for opening assumes that you have saved the data by saving a workspace.
If you run this file in the folder with the saved workspace, and select the file corresponding to the workspace, the program will then 
select the folder of the same name, and open the data within.  In this implementation any number of files can be opened at one time, allowing 
for rapid data analysis.  If you prefer another method of saving/opening data, The first two functions below can be either ignored or 
modified as desired. A sample data set has been included in the repository. As a chemist who happens to do some coding, I will be the
first to admit this code may not be optimized, but it is functional. 
Please feel free to comment with any suggestions or questions.
"""

import numpy as np #Required
import hyperspy.api as hs #Required, needs additional installation
import easygui #This package is optional and requires additional installation, but is currently used to open the desired files
import os #Only needed to open files with easygui
import matplotlib.pyplot as plt #Required for plotting
from scipy import optimize #Required for curve fitting
from scipy.fft import fft, ifft #Required for Fourier-Ratio Deconvolution

os.chdir("./Hybridization Mapping")

def select_file(): #Function to open file
    _, _, filenames = next(os.walk('./'), (None, None, []))
    msg ="Please select a sample file"
    title = "List of files"
    choices = filenames
    data_name  = easygui.multchoicebox(msg, title, choices)
    return data_name

def power_law(x, a, b): #Function to define the power law background
    y = a*x**-b
    return y

def gauss(x, a, b, c):
    y = a*np.exp(-np.power(x - b, 2)/(2*np.power(c, 2)))
    return y

def pow_sub(x, y, ene_min, ene_max): #Function for fitting and subtracting the power law background
    ene_min = np.abs(x - ene_min).argmin()
    ene_max = np.abs(x - ene_max).argmin()
    alpha = optimize.curve_fit(power_law, xdata=x[ene_min:ene_max], ydata=y[ene_min:ene_max], maxfev=5000)[0]
    # data_sub = y - power_law(x, alpha[0], alpha[1])
    return alpha

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    from math import factorial
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid') 

def integrate(low, high, data_x, data_y): #This function performs the integral of the region from "low" to "high"
    integral = 0
    for x in range(len(data_x)):
        if data_x[x] > low and data_x[x] < high:
            integral += ((data_y[x] + data_y[x+1])/2)*(data_x[x+1]-data_x[x])
    return integral 

def open_file(name):
    name = name[:-4]
    os.chdir("./" + name)
    sp = hs.load('STEM SI.dm4')
    os.chdir("../")
    return sp


def deconvolve(sp): #This function is the primary function, as it performs the binning, the Fourier-ratio decovonvolution, and calculates the pi*/(pi*+sigma*) ratio
     
    LL = np.array(sp[-2])
    HL = np.array(sp[-1])
    x, y, ene = np.shape(HL)

    bins_x = int(x/n)
    bins_y = int(y/n)
    LL_bin = np.zeros((bins_x, bins_y, ene))
    HL_bin = np.zeros((bins_x, bins_y, ene))
    for i in range(bins_x):
        for j in range(bins_y):
            LL_bin[i,j,:] = np.sum(LL[n*i:n*(i+1), n*j:n*(j+1), :], axis = (0,1))
            HL_bin[i,j,:] = np.sum(HL[n*i:n*(i+1), n*j:n*(j+1), :], axis = (0,1))

    scale = sp[-1].axes_manager[-1].scale
    LL_offset = sp[-2].axes_manager[-1].offset
    HL_offset = sp[-1].axes_manager[-1].offset

    LL_ene = np.arange(LL_offset, LL_offset+scale*np.shape(LL)[2], scale)
    HL_ene = np.arange(HL_offset, HL_offset+scale*np.shape(HL)[2], scale)
    
    HL_decon = np.zeros((bins_x, bins_y, ene))
    ratios = np.zeros((bins_x, bins_y))
    ratios_no_zeros = []
    ML_data = np.zeros((bins_x, bins_y)) #Note that this array was only used to generate data for the machine learning.

    for i in range(bins_x):
        for j in range(bins_y):
            LL_temp = LL_bin[i,j,:]
            HL_temp = HL_bin[i,j,:]
            LL_smooth = savitzky_golay(LL_temp, 20, 3)
            HL_smooth = savitzky_golay(HL_temp, 20, 3)         
            LL_fft = fft(LL_smooth)
            HL_fft = fft(HL_smooth)
            
            min_range = [280, 290]
            m_r_ind = [np.abs(min_range[0]-HL_ene).argmin(), np.abs(min_range[1]-HL_ene).argmin()]
            beta = optimize.curve_fit(gauss, LL_ene[:250], LL_smooth[:250], [25000, 0, 5], maxfev=5000)[0]
            gaus = gauss(LL_ene, beta[0], beta[1], beta[2])
            gaus_fft = fft(gaus)
            decon_gauss = ifft(gaus_fft*(HL_fft/LL_fft))
            decon_gauss = np.real(decon_gauss)
            decon_gauss = decon_gauss - np.min(decon_gauss[m_r_ind[0]:m_r_ind[1]])
            HL_decon[i,j,:] = decon_gauss
            
            pi = integrate(int_lim[0], int_lim[1], HL_ene, decon_gauss)
            sigma = integrate(int_lim[2], int_lim[3], HL_ene, decon_gauss)
            ratios[i,j] = (pi/(pi+sigma))
            if ratios[i,j]/ratio_ref > 0.05 and ratios[i,j]/ratio_ref < 0.6:
                ML_data[i,j] = 1 #ND Defined as 1
            else: ML_data[i,j] = 2 #Lacey defined as 2, #MWCNT defined as 3
            if np.max(decon_gauss) < noise_cutoff:
                ratios[i,j] = 0
                ratios_no_zeros.append(pi/(pi+sigma))
                ML_data[i,j] = 0 #Noise defined as 0

    return LL_ene, HL_ene, LL_bin, HL_decon, ratios, ML_data

n = 4 #This determines the number of pixels used for binning. 1 means no binning, 2 is 2x2, 3 is 3x3, etc. When adjust, be sure to change noise cutoff
#This parameter determines the noise cutoff, and depends on the binning. Higher binning needs a higher cutoff.  This formula works well for most bins,
#but a specific value can be set,  If you see spots in vacuum, increase the cutoff.  If you lose signal from your sample, reduce the cutoff.
noise_cutoff = n**2+1   

data_name = select_file()

int_lim = [282, 288, 288, 308] #This sets the integration limits in eV. The first two values are for the pi* peak, while the last two are the sigma*
ratio_ref = 0.1610 #This value, the reference ratio, was specific to this work, and may need to be changed for a different system
all_data = []
all_ratios = []
legend = []
enes = []
for name in data_name:
    sp = open_file(name)
    LL_ene, HL_ene, LL, HL_decon, ratios, ML_data = deconvolve(sp)
    
    percent = ratios/ratio_ref
    
    x, y, ene = np.shape(np.array(sp[-1]))
    plt.imshow(percent*100, extent = [0, sp[-1].axes_manager[0].scale*x, 0, sp[-1].axes_manager[1].scale*y], vmin = 0, vmax = 100)
    plt.colorbar(label = "%sp\u00B2")
    plt.show()  
    all_data.append(HL_decon)
    enes.append(HL_ene)
    all_ratios.append(ratios)
    legend.append(name)

    #The following lines were used to save the data for the ML algorithm. They are provided if desired, but not required for regular analysis.    
    # save_name_y = name[:-4] + "_y.npy"
    # save_name_x = name[:-4] + "_x.npy"
    # np.save(save_name_x, HL_decon[:,:,:1024])
    # np.save(save_name_y,ML_data)