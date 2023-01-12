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

#Write function to calculate our change in energy.
def calc_DeltaE(J, B, mu, config, i, j, N):
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
    
    #Now calculate our old energy... only need the spin we're flipping plus neighboring spins, and B term.
    E_old = -J*(config[i,j]*config[s_i_max,j] + config[i,j]*config[i,s_j_max] + config[i,j]*config[s_i_min,j] + config[i,j]*config[i,s_j_min]) - mu*B*config[i,j]
    
    #Flip our spin...
    s_ij_f = -1*config[i,j]
    
    #Now calculate our new energy... only need the spin we're flipping plus neighboring spins, and B term.
    E_new = -J*(s_ij_f*config[s_i_max,j] + s_ij_f*config[i,s_j_max] + s_ij_f*config[s_i_min,j] + s_ij_f*config[i,s_j_min]) - mu*B*s_ij_f
    
    #Subtract to get the change...
    DeltaE = E_new - E_old
    
    return DeltaE

#Write function to calculate the magnetization.
def calc_magnetization(config, N):
    #Here we want to plot the average magnetization as 1/beta scales.
    #beta = 1/kT --> 1/beta = kT
    #Average magnetization = spin per particle, should be between 0 and 1.
    
    #Sum of all magnetizations for the configuration, divided by number of particles.
    #Recall we have N^2 particles.
    
    mag = np.sum(config)
    avgmag = (1/(N**2))*mag
    
    return avgmag

#Write function to calculate our internal energy.  
def calc_internalE(J, mu, B, config, N):
    
    #Find the B term - just multiply this sum right away.
    B_term = mu*B*np.sum(config)
    
    #Initialize J term...
    J_term = 0
    
    #Iterate over all [i,j] lattice sites.
    for i in range(N):
        for j in range(N):
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
            
            #Find J term at our specific lattice site.
            J_add = -J*(config[i,j]*config[s_i_max,j] + config[i,j]*config[i,s_j_max] + config[i,j]*config[s_i_min,j] + config[i,j]*config[i,s_j_min])
            
            #Add it.
            J_term = J_term + J_add
    
    #Then sum these together to get total E of our configuration.
    total_internalE = -1*B_term + J_term
    
    return total_internalE

#Write function to calculate our relative probability.
def calc_relativeprob(J, DeltaE, inv_Beta):
    
    #What it says on the tin. We take the exponent.
    P = math.exp((-1*DeltaE)/(inv_Beta))

    return P    

#Write function to find and plot Ising lattice configurations.
def ising_MH(inv_Beta, J, N, B, mu, numsteps):
    
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
        
        #Calculate our change in energy.
        DeltaE = calc_DeltaE(J, B, mu, config, i, j, N)
        
        #If our change in energy is negative, accept right away.
        if DeltaE <= 0:
            config[i,j] = -1*config[i,j]
        
        #If our change in energy is positive, generate random uniform variable and compare.
        else: 
            P = calc_relativeprob(J, DeltaE, inv_Beta)
            u = np.random.uniform()
            
            #Accept/reject condition:
            if u <= P:
                config[i,j] = -1*config[i,j]
            else:
                pass
        
        #Now we take a snapshot at every 200 steps... these are automatically saved to file.
        #if n % 200 == 0:
        if (n == 10) or (n == 15) or (n == 20):
            plt.imshow(config, cmap='gray')
            plt.xlabel('Lattice Index')
            plt.ylabel('Lattice Index')
            plt.title('Behaviour of ' + str(N) + ' by ' + str(N) + ' Ising Lattice \n B = ' + str(B) + ' at kT = ' + str(inv_Beta))
            #plt.savefig(str(n) + '.png')
            plt.show()
        else:
            pass

#Write function to find our average magnetization.
def average_magnetization(J, N, B, mu, minbeta, maxbeta, numsteps):
    #Average magnetization = avg spin per particle, should be between 0 and 1.
    #Note here, beta is used as an abbreviation for inverse beta.
    
    #Initialize values array.
    avgmagvals = []
    
    #Calculate step size in kT and which values we're taking.
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
        
        #Calculate change in energy.
        DeltaE = calc_DeltaE(J, B, mu, config, i, j, N)
        
        #If negative, accept new configuration.
        if DeltaE <= 0:
            config[i,j] = -1*config[i,j]
            
        #If positive, compare to random uniform variable.
        else: 
            P = calc_relativeprob(J, DeltaE, betasteps[n])
            u = np.random.uniform()
            
            if u <= P:
                config[i,j] = -1*config[i,j]
            else:
                pass
        
        #Calculate the magnetization and append to array of values.
        avg_mag = calc_magnetization(config, N)
        avgmagvals.append(avg_mag)
    
    return avgmagvals

#Write a function to find average magnetization and plot.
def average_magnetization_mean(J, N, B, mu, minbeta, maxbeta, numsteps, numavg):
    
    #Here we're going to run average_magnetization, and take the average for multiple values. 
    #See if we can get something more accurate.
    
    #Initialize values array.
    avg_mag_all = []
    
    #Generate beta steps and step size.
    stepsize = (maxbeta-minbeta)/numsteps
    betasteps = np.arange(minbeta, maxbeta, stepsize)
    
    #Take values for the number of samples we want for each average.
    for i in range(numavg):
        
        #Calculate average magnetization...
        avg_mag_func_beta = average_magnetization(J, N, B, mu, minbeta, maxbeta, numsteps)
        
        #Do do dooo some average stuff.
        if i == 0:
            avg_mag_all = avg_mag_func_beta.copy()
        else:
            avg_mag_all = np.vstack((avg_mag_all, avg_mag_func_beta))
    
    #Taking the average...
    avg_mag_mean_val = np.sum(avg_mag_all, axis = 0)  
    avg_mag_mean_val = avg_mag_mean_val/numavg
    
    #Now we plot!
    plt.plot(betasteps, avg_mag_mean_val, marker ='o', markersize = 1, linestyle = 'none')
    plt.xlabel(r'$\beta^{-1} = kT$')
    plt.ylabel(r'$\langle M \rangle$')
    plt.title('Average Magnetization as a Function of Temperature \n' + str(N) + ' by ' + str(N) + ' Lattice, B = ' + str(B) + '\n')
    plt.show()

#Write a function to find our average internal energy and plot.
def average_internalE(J, N, B, mu, minbeta, maxbeta, numsteps, numsamples):
    
    #Initialize array.
    avg_internal_E_vals = []
    
    #Generate step size and kT steps.
    stepsize = (maxbeta-minbeta)/numsteps
    betasteps = np.arange(minbeta, maxbeta, stepsize)
    
    #Initialize random NxN array of spins.
    #This actually makes random array of 0s and 1s, so let 0s be spin downs.
    config = np.random.randint(0, 2, (N,N))
    
    #Convert our zeros to -1s...
    config[config == 0] = 1
    
    #For the number of steps over kT we are taking...
    for p in range(len(betasteps)):
        
        #Initialize values array...
        internal_E_vals = []
        
        #For the number of samples we want to take...
        for n in range(numsamples):
            
            #Generate the random spot we want to flip a spin at.
            i = np.random.randint(0,N)
            j = np.random.randint(0,N)
            
            #Calculate change in energy...
            DeltaE = calc_DeltaE(J, B, mu, config, i, j, N)
            
            #Again here's our Metropolis algorithm conditions...
            if DeltaE <= 0:
                config[i,j] = -1*config[i,j]
            else: 
                P = calc_relativeprob(J, DeltaE, betasteps[p])
                u = np.random.uniform()
                if u <= P:
                    config[i,j] = -1*config[i,j]
                else:
                    pass
            
            #Calculate internal E.
            internal_E = calc_internalE(J, mu, B, config, N)
            internal_E_vals.append(internal_E)
        
        #Now take the average.
        avg_internal_E = (1/numsamples)*np.sum(internal_E_vals)
        avg_internal_E_vals.append(avg_internal_E)
    
    #Now plot all our average values.
    plt.plot(betasteps, avg_internal_E_vals, marker ='o', markersize = 1, linestyle = 'none')
    plt.xlabel(r'$\beta^{-1} = kT$')
    plt.ylabel(r'$\langle E \rangle$')
    plt.title('Average Internal Energy as a Function of Temperature \n' + str(N) + ' by ' + str(N) + ' Lattice, B = ' + str(B) + '\n')
    plt.show()

#Write a function to find our specific heat and plot.            
def specific_heat(J, N, B, mu, minbeta, maxbeta, numsteps, numsamples):
    
    #beta = kT --> T = beta/k 
    
    #Initialize C_V vals array.
    C_V_vals = []
    
    #Do our step thing
    stepsize = (maxbeta-minbeta)/numsteps
    betasteps = np.arange(minbeta, maxbeta, stepsize)
    
    #Initialize random NxN array of spins.
    #This actually makes random array of 0s and 1s, so let 0s be spin downs.
    config = np.random.randint(0, 2, (N,N))
    
    #Convert our zeros to -1s...
    config[config == 0] = 1
    
    #Again in the range of steps over kT we want to take
    for p in range(len(betasteps)):
        internal_E_vals = []
        internal_E_sq_vals = []
        
        #Find T at the kT value we have.
        T_val = betasteps[p]/k
        
        #For each sample we want to take...
        for n in range(numsamples):
            
            #Generate the random spot we want to flip a spin at.
            i = np.random.randint(0,N)
            j = np.random.randint(0,N)
            
            #Find change in energy...
            DeltaE = calc_DeltaE(J, B, mu, config, i, j, N)
            
            #Again do our little accept reject method...
            if DeltaE <= 0:
                config[i,j] = -1*config[i,j]
            else: 
                P = calc_relativeprob(J, DeltaE, betasteps[p])
                u = np.random.uniform()
                
                if u <= P:
                    config[i,j] = -1*config[i,j]
                else:
                    pass
            
            #Find our internal E and internal E squared as required.
            internal_E = calc_internalE(J, mu, B, config, N)
            internal_E_vals.append(internal_E)
            internal_E_sq_vals.append(internal_E**2)
        
        #Sum before 1/N multiplication
        avg_internal_E_sq = np.sum(internal_E_sq_vals)
        avg_internal_E_sq = (1/numsamples)*avg_internal_E_sq
        
        #Sum before 1/N^2 multiplication
        avg_internal_E = np.sum(internal_E_vals)
        avg_internal_E = (1/(numsamples**2))*(avg_internal_E**2)
        
        #Calculate C_V and append to our array.
        C_V = (avg_internal_E_sq - avg_internal_E)/((N**2)*betasteps[p]*T_val)
        C_V_vals.append(C_V)
    
    #Now we plot!
    plt.plot(betasteps, C_V_vals, marker ='o', markersize = 1, linestyle = 'none')
    plt.xlabel(r'$\beta^{-1} = kT$')
    plt.ylabel(r'$C_V$')
    plt.title('Specific Heat Capacity as a Function of Temperature \n' + str(N) + ' by ' + str(N) + ' Lattice, B = ' + str(B) + '\n')
    plt.show()    

#Write a function to find our magnetic susceptibility and plot.
def zero_field_mag_suc(J, N, B, mu, minbeta, maxbeta, numsteps, numsamples):
    
    #Initialize array of values.
    mag_suc_vals = []
    
    #Do our little step thing do do do do do 
    stepsize = (maxbeta-minbeta)/numsteps
    betasteps = np.arange(minbeta, maxbeta, stepsize)
    
    #Initialize random NxN array of spins.
    #This actually makes random array of 0s and 1s, so let 0s be spin downs.
    config = np.random.randint(0, 2, (N,N))
    
    #Convert our zeros to -1s...
    config[config == 0] = 1
    
    #Again for our range of steps...
    for p in range(len(betasteps)):
        #Initialize arrays to store values we want...
        total_mag_vals = []
        total_mag_sq_vals = []
        
        #And our range of samples...
        for n in range(numsamples):
            
            #Generate the random spot we want to flip a spin at.
            i = np.random.randint(0,N)
            j = np.random.randint(0,N)
            
            #Calculate change in energy.
            DeltaE = calc_DeltaE(J, B, mu, config, i, j, N)
            
            #And do our little acception or rejection...
            if DeltaE <= 0:
                config[i,j] = -1*config[i,j]
            else: 
                P = calc_relativeprob(J, DeltaE, betasteps[p])
                u = np.random.uniform()
                if u <= P:
                    config[i,j] = -1*config[i,j]
                else:
                    pass
                
            #Then calculate various magnetic values we need...
            total_mag = np.sum(config)
            total_mag = (1/(N**2))*total_mag
            total_mag_vals.append(total_mag)
            total_mag_sq_vals.append(total_mag**2)
            
        #Sum before 1/N multiplication
        avg_mag_sq = np.sum(total_mag_sq_vals)
        avg_mag_sq = (1/numsamples)*avg_mag_sq
        
        #Sum before 1/N^2 multiplication
        avg_mag = np.sum(total_mag_vals)
        avg_mag = (1/(numsamples**2))*(avg_mag**2)
        
        #Calulate magnetic susceptibility value and append to array.
        mag_suc = ((avg_mag_sq - avg_mag)*mu)/(betasteps[p])
        mag_suc_vals.append(mag_suc)
    
    #Now plot!
    plt.plot(betasteps, mag_suc_vals, marker ='o', markersize = 1, linestyle = 'none')
    plt.xlabel(r'$\beta^{-1} = kT$')
    plt.ylabel(r'$\chi$')
    plt.title('Magnetic Susceptibility as a Function of Temperature \n' + str(N) + ' by ' + str(N) + ' Lattice, B = ' + str(B) + '\n')
    plt.show()

#######################################################################
    
#Main: Let's run some functions!

#average_internalE(1, 2, 0, 0, 0, 8, 250, 5000)
#average_internalE(1, 4, 0, 0, 0, 8, 250, 5000)
#average_internalE(1, 6, 0, 0, 0, 8, 250, 5000)

#average_magnetization_mean(1, 2, 0, 0, 0, 8, 250, 5000)
#average_magnetization_mean(1, 4, 0, 0, 0, 8, 1000, 500)
#average_magnetization_mean(1, 6, 0, 0, 0, 8, 2000, 500)

#specific_heat(1, 2, 0, 0, 0.1, 8, 3000, 500)
#specific_heat(1, 4, 0, 0, 0.1, 8, 2000, 500)
#specific_heat(1, 6, 0, 0, 0.1, 8, 2000, 250)
    
#zero_field_mag_suc(1, 4, 0, 1, 0.5, 8, 1000, 10000)
#zero_field_mag_suc(1, 2, 0, 1, 0.5, 8, 1000, 10000)
#zero_field_mag_suc(1, 6, 0, 1, 0.5, 8, 1000, 10000)
    
#######################################################################
    
#average_internalE(1, 2, 1, 1, 0, 8, 250, 5000)
#average_internalE(1, 4, 1, 1, 0, 8, 250, 5000)
#average_internalE(1, 6, 1, 1, 0, 8, 250, 5000)
    
#average_magnetization_mean(1, 2, 1, 1, 0, 8, 250, 5000)
#average_magnetization_mean(1, 4, 1, 1, 0, 8, 1000, 500)
#average_magnetization_mean(1, 6, 1, 1, 0, 8, 2000, 500)
    
#specific_heat(1, 2, 1, 1, 0.1, 8, 3000, 500)
#specific_heat(1, 4, 1, 1, 0.1, 8, 2000, 500)
#specific_heat(1, 6, 1, 1, 0.1, 8, 2000, 250)
    
#zero_field_mag_suc(1, 4, 1, 1, 0.5, 8, 1000, 10000)
#zero_field_mag_suc(1, 2, 1, 1, 0.5, 8, 1000, 10000)
#zero_field_mag_suc(1, 6, 1, 1, 0.5, 8, 1000, 10000)

#Okay this function is only called once because I forgot to save all my calls oops
#But you get the idea :^)
ising_MH(1, 1, 4, 0, 0, 51)