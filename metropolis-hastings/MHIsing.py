# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 11:00:17 2022

@author: David
"""

#######################################################################

#Import all of the wonderful things we like and need to manage data and make plots.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pds
import pathlib 
import math
import cmath
from scipy import stats
from scipy.constants import k

#######################################################################

#Recall: ising energy for a configuration j is:
#E_j = -J * (sum(i->N)s_i s_i+1)
#for s in pm1

#1. Start with arbitrary configuration for an NxN lattice.

#2. Generate trial configuration by flipping spin at some (i,j) on the lattice.

#3. Calculate difference in energy as result of flipping the one spin.

#4. Test to see if new configuration is ok.
#a. If new E < old E, ie. DeltaE <= 0, say we are good.
#b. If new E > old E, ie. DeltaE > 0, discard new configuration, keep old config, and redo 2-4.

#######################################################################

def calc_DeltaE(J, config, i, j, N):
    #Want to calculate the change in energy based on spin flip.
    #Note energy depends on NNs.
    #E_j at NNs only:
    #Calc Old:
    #-J* (s_ij * s(i+1,j) + s_ij * s(i,j+1) + s_ij * s(i-1,j) + s_ij * s(i,j-1))
    #Flip spin, ie. s_ij = -s_ij
    #Calc New:
    #-J* (s_ij * s(i+1,j) + s_ij * s(i,j+1) + s_ij * s(i-1,j) + s_ij * s(i,j-1))
    
    #Need to implement periodic conditions so we don't break the code at the edges...
    if i == 0:
        s_i_min = N-1
    else:
        s_i_min = i-1
    if j == 0:
        s_j_min = N-1
    else:
        s_j_min = j-1
    if i == N-1:
        s_i_max = 0
    else:
        s_i_max = i+1
    if j == N-1:
        s_j_max = 0
    else:
        s_j_max = j+1

    E_old = -J*(config[i,j]*config[s_i_max,j] + config[i,j]*config[i,s_j_max] + config[i,j]*config[s_i_min,j] + config[i,j]*config[i,s_j_min])
    
    s_ij_f = -1*config[i,j]
    
    E_new = -J*(s_ij_f*config[s_i_max,j] + s_ij_f*config[i,s_j_max] + s_ij_f*config[s_i_min,j] + s_ij_f*config[i,s_j_min])
    
    DeltaE = E_new - E_old
    #print(DeltaE)
    
    return DeltaE

def calc_magnetization(config, N):
    #Here we want to plot the average magnetization as 1/beta scales from 0.5 to 5
    #beta = 1/kT --> 1/beta = kT
    #Average magnetization = spin per particle(?), should be between 0 and 1.
    
    #Sum of all magnetizations for the configuration, divided by number of particles.
    #Recall we have N^2 particles.
    
    mag = np.sum(config)
    avgmag = (1/(N**2))*mag
    
    return avgmag
    

def calc_relativeprob(J, DeltaE, T):
    
    #P = math.exp((-1*DeltaE)/(k*T))
    P = math.exp((-1*DeltaE)/(T))
    #print(P)
    return P
    

def ising_MH(T, J, N, numsteps):
    
    #Initialize random NxN array of spins.
    #This actually makes random array of 0s and 1s, so let 0s be spin downs.
    config = np.random.randint(0, 2, (N,N))
    
    #Convert our zeros to -1s...
    config[config == 0] = -1
    
    #For the number of steps we want to take, flip a spin each time.
    
    for n in range(numsteps):
        
        #Generate the random spot we want to flip a spin at.
        i = np.random.randint(0,N)
        j = np.random.randint(0,N)
        
        DeltaE = calc_DeltaE(J, config, i, j, N)
        
        if DeltaE <= 0:
            config[i,j] = -1*config[i,j]
        
        else: 
            P = calc_relativeprob(J, DeltaE, T)
            u = np.random.uniform()
            
            if u <= P:
                config[i,j] = -1*config[i,j]
            else:
                pass
        
        if n % 100 == 0:
            plt.imshow(config, cmap='gray')
            plt.savefig(str(n) + '.png')
        else:
            pass
        

def ising_MH_coldstart(T, J, N, numsteps):         
    
    #Initialize NxN array of spin ups.
    #This actually makes random array of 0s and 1s, so let 0s be spin downs.
    config = np.random.randint(0, 2, (N,N))
    
    #Convert our zeros to 1s...
    config[config == 0] = 1
    
    #print(config)
    
    #For the number of steps we want to take, flip a spin each time.
    
    for n in range(numsteps):
        
        #Generate the random spot we want to flip a spin at.
        i = np.random.randint(0,N)
        j = np.random.randint(0,N)
        
        DeltaE = calc_DeltaE(J, config, i, j, N)
        
        if DeltaE <= 0:
            config[i,j] = -1*config[i,j]
        
        else: 
            P = calc_relativeprob(J, DeltaE, T)
            u = np.random.uniform()
            
            if u <= P:
                config[i,j] = -1*config[i,j]
            else:
                pass
        
        if n % 100 == 0:
            plt.imshow(config, cmap='gray')
            plt.savefig(str(n) + '.png')
        else:
            pass
 
             
def average_magnetization_beta(J, N, minbeta, maxbeta, numsteps):
    #Here we want to plot the average magnetization as 1/beta scales from 0.5 to 5
    #beta = 1/kT --> 1/beta = kT
    #Average magnetization = spin per particle(?), should be between 0 and 1.
    
    #Do we want to take multiple configurations and average the magnetization for all?
    #Or just vary beta --> take one config --> continue?
    #I guess we'll try the second way first, sounds easier...
    
    avgmagvals = []
    
    stepsize = (maxbeta-minbeta)/numsteps
    
    betasteps = np.arange(minbeta, maxbeta, stepsize)
    
    #Initialize random NxN array of spins.
    #This actually makes random array of 0s and 1s, so let 0s be spin downs.
    config = np.random.randint(0, 2, (N,N))
    
    #Convert our zeros to -1s...
    config[config == 0] = 1
    
    #For the number of steps we want to take, flip a spin each time.
    
    for n in range(len(betasteps)):
        
        #Generate the random spot we want to flip a spin at.
        i = np.random.randint(0,N)
        j = np.random.randint(0,N)
        
        DeltaE = calc_DeltaE(J, config, i, j, N)
        
        if DeltaE <= 0:
            config[i,j] = -1*config[i,j]
            
        
        else: 
            P = calc_relativeprob(J, DeltaE, betasteps[n])
            u = np.random.uniform()
            
            if u <= P:
                config[i,j] = -1*config[i,j]
            else:
                pass
        
        avg_mag = calc_magnetization(config, N)
        avgmagvals.append(avg_mag)
        
    '''plt.plot(betasteps, avgmagvals, marker ='o', markersize = 1, linestyle = 'none')
    plt.xlabel(r'$\beta^{-1}$')
    plt.ylabel(r'|$M$|')
    plt.title('Average Magnetization as a Function of Temperature \n')
    plt.show()'''
    
    return avgmagvals
        

       
def average_magnetization_mean(J, N, minbeta, maxbeta, numsteps, numavg):
    
    #why do you not collapse
    
    #Here we're going to run average_magnetization, and take the average for multiple values. 
    #See if we can get something more accurate.
    
    avg_mag_all = []
    
    stepsize = (maxbeta-minbeta)/numsteps
    
    betasteps = np.arange(minbeta, maxbeta, stepsize)
    
    for i in range(numavg):
        
        avg_mag_func_beta = average_magnetization_beta(J, N, minbeta, maxbeta, numsteps)
        
        if i == 0:
            avg_mag_all = avg_mag_func_beta.copy()
        else:
            avg_mag_all = np.vstack((avg_mag_all, avg_mag_func_beta))
        
    avg_mag_mean_val = np.sum(avg_mag_all, axis = 0)  
    
    avg_mag_mean_val = avg_mag_mean_val/numavg
    
    plt.plot(betasteps, avg_mag_mean_val, marker ='o', markersize = 1, linestyle = 'none')
    plt.xlabel(r'$\beta^{-1}$')
    plt.ylabel(r'|$M$|')
    plt.title('Average Magnetization as a Function of Temperature \n')
    plt.show()
    
#######################################################################
    
#ising_MH(2.5, 1, 50, 10000)
    
#average_magnetization_beta(1, 16, 0.1, 6, 15000)
    
average_magnetization_mean(1, 16, 0.1, 6, 15000, 2)