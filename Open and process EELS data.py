# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 14:12:21 2024

@author: Alex Sheardy

Description: This code works to open and process a file created in Digital Micrograph/GMS.  Specifically, this code is intended for bulk processing
of the raw data to find the ratio of the pi*/(pi*+sigma*) peaks, for determing MACs.  The current method for opening assumes
that you have saved the data by saving a workspace.  If you run this file in the folder with the saved workspace, and select the file
corresponding to the workspace, the program will then select the folder of the same name, and open the data within.  In this implementation
any number of files can be opened at one time, allowing for rapid data analysis.  If you prefer another method of saving/opening data,
The first two functions below can be either ignored or modified as desired. A sample data set (titled: "Collection ..." has been included in the repository.
As a chemist who happens to do some coding, I will be the first to admit this code may not be optimized, but it is functional. 
Please feel free to comment with any suggestions or questions.
"""

import numpy as np #Required
import hyperspy.api as hs #Required, needs additional installation
import easygui #This package is optional and requires additional installation, but is currently used to open the desired files
import os #Only needed to open files with easygui
import matplotlib.pyplot as plt #Required for plotting
from scipy import optimize #Required for curve fitting
from scipy.fft import fft, ifft #Required for Fourier-Ratio Deconvolution

os.chdir("./Open and process EELS data")

def select_file(): #Function to open files, this code allows you to open as many files as desired
    _, _, filenames = next(os.walk('./'), (None, None, []))
    msg ="Please select a sample file"
    title = "List of files"
    choices = filenames
    data_name  = easygui.multchoicebox(msg, title, choices)
    return data_name

def open_file(name):
    name = name[:-4]
    os.chdir("./" + name)
    sp = hs.load('STEM SI.dm4')
    os.chdir("../")
    return sp

def power_law(x, a, b): #Function to define the power law background
    y = a*x**-b
    return y

def gauss(x, a, b, c): #Function to define Gaussian function for curve fitting
    y = a*np.exp(-np.power(x - b, 2)/(2*np.power(c, 2)))
    return y

def pow_sub(x, y, ene_min, ene_max): #Function for fitting and subtracting the power law background. The range for subtraction is given by ene_min and ene_max
    ene_min = np.abs(x - ene_min).argmin()
    ene_max = np.abs(x - ene_max).argmin()
    alpha = optimize.curve_fit(power_law, xdata=x[ene_min:ene_max], ydata=y[ene_min:ene_max], maxfev=5000)[0]
    data_sub = y - power_law(x, alpha[0], alpha[1])
    return data_sub, alpha

def savitzky_golay(y, window_size, order, deriv=0, rate=1): #Function used for data smoothing, not currently used
    from math import factorial
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid') 

def deconvolve(sp): #This function perfroms the Fourier-ratio deconvolution
    #First the data for the low-loss (LL) and high-loss (HL) data is loaded and summed across the x and y axes.
    LL = np.sum(np.array(sp[-2]), axis = (0,1))
    HL = np.sum(np.array(sp[-1]), axis = (0,1))
    #The meta data from the file is read to load the energy scale and offset
    scale = sp[-1].axes_manager[-1].scale
    LL_offset = sp[-2].axes_manager[-1].offset
    HL_offset = sp[-1].axes_manager[-1].offset
    #The scale and offset are used to create the x-range for the LL and HL spectra
    LL_ene = np.arange(LL_offset, LL_offset+scale*np.shape(LL)[0], scale)
    HL_ene = np.arange(HL_offset, HL_offset+scale*np.shape(HL)[0], scale)
    #A power low subtraction is performed on the HL data
    HL, alpha = pow_sub(HL_ene, HL, 270, 280)
    #The FFT of both LL and HL spectra is calculated    
    LL_fft = fft(LL)
    HL_fft = fft(HL)
    
    #Finally, some fittings are performed, and the actual deconvolution is calculated
    min_range = [280, 290]
    m_r_ind = [np.abs(min_range[0]-HL_ene).argmin(), np.abs(min_range[1]-HL_ene).argmin()]
    beta = optimize.curve_fit(gauss, LL_ene[:250], LL[:250], [25000, 0, 5], maxfev=5000)[0] #Note that the [25000, 0, 5] represent initial guesses for a, b, and c. If you get poor fitting/errors, you may need to adjust these
    gaus = gauss(LL_ene, beta[0], beta[1], beta[2])
    gaus_fft = fft(gaus)
    decon_gauss = ifft(gaus_fft*(HL_fft/LL_fft))
    decon_gauss = np.real(decon_gauss)
    decon_gauss = decon_gauss - np.min(decon_gauss[m_r_ind[0]:m_r_ind[1]])
    decon_gauss = decon_gauss / np.max(decon_gauss)
    
    return HL_ene, decon_gauss, HL, LL, LL_ene

def integrate(low, high, data_x, data_y): #This function performs the integral of the region from "low" to "high"
    integral = 0
    for x in range(len(data_x)):
        if data_x[x] > low and data_x[x] < high:
            integral += ((data_y[x] + data_y[x+1])/2)*(data_x[x+1]-data_x[x])
    return integral 

data_name = select_file()
int_lim = [283.5, 288.5, 289.5, 309.5] #This sets the integration limits in eV. The first two values are for the pi* peak, while the last two are the sigma*

pi_to_sigma_ratio = [] #List of pi*/sigma* for all data
pi_to_pi_sigma_ratio = [] #List of pi*/(pi*+sigma*)

legend = [] #List of file names for plotting'
HL_data = [] #All HL spectral intensity (y) data
LL_data = [] #All LL spectral intensity (y) data
HL_enes = [] #All HL spectral energy (x) data
LL_enes = [] #All LL spectral energy (x) data

HL_raw_all = []

for name in data_name:
    sp = open_file(name) #Open the file
    ene, HL_processed, HL_raw, LL, LL_ene = deconvolve(sp) #Perform the deconvolution. This returns both the processed and raw HL data.
    
    #Integrals of pi and sigma peaks are calculated
    pi = integrate(int_lim[0], int_lim[1], ene, HL_processed)
    sigma = integrate(int_lim[2], int_lim[3], ene, HL_processed)
    
    #Two methods of data analysis are performed, first is just pi*/sigma*, while the second is pi*/(pi*+sigma*)      
    pi_to_sigma_ratio.append(pi/sigma)
    pi_to_pi_sigma_ratio.append(pi/(pi+sigma))
    
    HL_raw_all.append(HL_raw)
    HL_data.append(HL_processed)
    LL_data.append(LL/np.max(LL)) #Normalizes the LL data, but normalization is not required.
    legend.append(name[:-4]) #The [:-4] is to remove the file extension
    HL_enes.append(ene)
    LL_enes.append(LL_ene)
   
    print(name[:-4] + " pi*/(pi*+sigma*): " + str(round(pi/(pi+sigma), 3))) #This will print the file name and pi*/(pi*+sigma*) ratio

# colors = ['r','b', 'green', 'purple', 'tab:brown', 'black'] #This is optional, just to change the default colors for plotting

#Plot HL data
plt.figure(figsize=(6.5,3))    
for i in range(len(data_name)):
    # plt.plot(HL_enes[i], HL_data[i], color = colors[i]) #Use this line to use the alternate colors above
    plt.plot(HL_enes[i], HL_data[i])  #Use this line for default colors
plt.xlim(275, 345)
plt.ylim(-0.2, 1.1)
plt.yticks([0])
plt.legend((legend))
plt.xlabel("Energy (eV)")
plt.ylabel("Relative Intensity")
plt.title("High-loss Spectra")
#These next lines plot the integration limints, but can be removed
plt.axvline(int_lim[0], color = 'black', linestyle='dashed')
plt.axvline(int_lim[1], color = 'black', linestyle='dashed')
plt.axvline(int_lim[2], color = 'black')
plt.axvline(int_lim[3], color = 'black') 
plt.show()   

#Plot LL data
plt.figure(figsize=(6.5,3))
for i in range(len(data_name)):
    # plt.plot(LL_enes[i], LL_data[i], color = colors[i]) #Use this line to use the alternate colors above
    plt.plot(LL_enes[i], LL_data[i]) #Use this line for default colors
plt.xlim(-10, 110)
plt.ylim(0, 1)
plt.yticks([])
plt.legend((legend))
plt.xlabel("Energy (eV)")
plt.ylabel("Relative Intensity")
plt.title("Low-loss Spectra")
plt.show()

