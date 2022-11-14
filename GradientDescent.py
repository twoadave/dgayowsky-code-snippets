# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 15:50:05 2022

@author: David
"""

#Fit some curve to data using gradient descent.

#######################################################################

import numpy as np
import math 
import torch
import matplotlib.pyplot as plt
import pandas as pds
import pathlib 
import cmath
from scipy import stats
from scipy.constants import k

#######################################################################

#1. choose initial starting point.

#Calculate next point by:
#1. using gradient at current position,
#2. scale it using learning rate Delta,
#3. and subtracts obtained value from current position (taking step in opposite direction from gradient)
#Then: p_n+1 = p_n - Delta*gradient(f(p))

#Repeat until step size is smaller than some tolerance, due to scaling or a small gradient.

#######################################################################

def func_1(x):
    return x**2 - 2*x + 1

def deriv_func_1(x):
    return 2*x - 2

def func_2(x):
    return 3*math.sin(x)

def deriv_func_2(x):
    return 3*math.cos(x)

def func_3(x):
    return x**3 - 5*x

def deriv_func_3(x):
    return 3*x**2 - 5

#######################################################################
    
#Define a function to find the minima, given some graduent function.
def find_minima(x_initial, function, grad_function, learning_rate, numsteps):
    
    stepvals = []
    funcvals = []
    
    x_val = x_initial
    
    stepvals.append(x_initial)
    funcvals.append(function(x_initial))
    
    for i in range(numsteps):
        
        x_val = x_val - learning_rate*grad_function(x_val)
        
        funcvals.append(function(x_val))
        stepvals.append(x_val)
        
    return stepvals, funcvals

def plot_graddescent(x_initial, function, grad_function, learning_rate, numsteps):
    
    stepvals, grad_func_vals = find_minima(x_initial, function, grad_function, learning_rate, numsteps)
    
    x_vals = np.arange(min(stepvals)-2, x_initial+3.5, 0.01)
    full_func_vals = []
    
    for i in range(len(x_vals)):
        full_func_vals.append(function(x_vals[i]))
    
    plt.plot(x_vals, full_func_vals, marker ='o', markersize = 0.5, linewidth = 0.5)
    plt.plot(stepvals, grad_func_vals, marker ='o', markersize = 6, linewidth = 2, color = 'red')
    plt.show()
    
#######################################################################
    
plot_graddescent(-0.8, func_3, deriv_func_3, 0.2, 15)