"""
Created on Fri May 26

@author: David Gayowsky

Simple growth Monte Carlo learning algorithm.

Based on work by Viktor Selin, modified and re-written by David Gayowsky.
"""

#######################################################################

import matplotlib.pyplot as plt
import numpy as np
import os
import scipy as sp
from scipy.constants import c
import copy
import math
import itertools

#######################################################################

'''1. Start with two 2D arrays, one containing liquid (li = 1, vi = 0) or 
vapour (vi = 1, li = 0), the other containing nanoparticle sites (ni = 0, 1). 
Nanoparticle sites exclude liquid or gas, just nanoparticle.

2. Choose a selection of N random sites in the liquid array to attempt to change state. 

3. Calculate change in energy ΔE of state change, taking into account NNN interactions.

4. Calculate probability of energy change, and compare to x in U(0,1). If prob > x, 
accept. If prob < x, deny. 

5. Attempt to move a nanoparticle by one step in a randomized direction (random walk). 

6. Accept move IFF the lattice spots the nanoparticle is moving to are completely 
filled with liquid, and accept only with metropolis probability above.

7. If move is accepted, fill displaced spots with fluid.

8. Repeat steps 5-7 P number of times per each liquid cycle, and do so for a number 
of random nanoparticles M each cycle (i.e. 30 nanoparticle cycles per solvent cycle).

9. Repeat until we have completed a set number of cycles. '''

#######################################################################

'''Args:
kT: temperature * Kb
mu: chemical potential
e_nn: nano-nano attraction
e_nl: nano-solvent attraction
e_ll: solvent-solvent attraction -- All other attractions and mu are given in relation to this, leave as 1'''

#######################################################################

'''Default arguments to (mostly) match paper: 
y_dim = 1000
x_dim = 1000
frac = 0.3
mu = -2.5
kT = 0.2
e_n = 2
e_nl = 1.5
e_l = 1
nano_steps = 30 (Per Solvent Step)
nano_size = 3'''

#######################################################################

#Define function to initialize water/vapour and nanoparticle arrays.
def init_arrays(x_dim, y_dim, nano_size, num_nano_attempts, n_nano):

    #Initial liquid array should be all liquid.
    liquid_arr = np.ones((x_dim, y_dim))

    #Here, we want to generate a list of nanoparticle top left indices,
    #as well as populate the spots in the nano array which are filled
    #with nanoparticles.
    nano_arr = np.zeros((x_dim, y_dim))

    nano_list_indices = []

    for j in range(n_nano):

        #Now attempt to populate nanoparticle array with nanoparticles.
        for i in range(num_nano_attempts):

            #Generate random index:
            indices = (np.random.randint(0, x_dim-nano_size-1), np.random.randint(0, y_dim-nano_size-1))

            #Check and see if this index is already occupied by a nanoparticle:
            nano_placement = nano_arr[indices[1]:indices[1]+nano_size, indices[0]:indices[0]+nano_size]
            
            #If any value in our nano placements are 1, we don't add this particle.
            if any(1 in x for x in nano_placement):
                pass
            else:
                #Now if those spots are empty, we accept the nanoparticle and add it to our arrays.
                #Add to list of nano indices:
                nano_list_indices.append(indices)
                #Add to nanoparticle placement:
                nano_arr[indices[1]:indices[1]+nano_size, indices[0]:indices[0]+nano_size] = 1
                #Remove from liquid array:
                liquid_arr[indices[1]:indices[1]+nano_size, indices[0]:indices[0]+nano_size] = 0
                #Now we've placed a nanoparticle, so break the for loop.
                break

    #Convert our zeros to -1s...
    nanopart_copy = copy.deepcopy(nano_arr)
    nanopart_copy[nanopart_copy == 1] = 2

    config = liquid_arr + nanopart_copy

    plt.imshow(config, cmap='gray')
    plt.xlabel('Lattice Index')
    plt.ylabel('Lattice Index')
    plt.title('Initial Nanoparticle Placements in Liquid')
    plt.show()

    return liquid_arr, nano_arr, nano_list_indices

#Define function to calculate energy change when performing liquid step.
def delta_E_liquid(flip_index, liquid_arr, nano_arr, e_l, e_nl, mu):
    
    norm = 1/(1+1/math.sqrt(2))
    x_i = flip_index[0]
    y_i = flip_index[1]

    delta_e = -(1-2*liquid_arr[x_i,y_i])*((norm)*(e_l*(liquid_arr[(x_i-1),y_i]+
                                                             liquid_arr[(x_i+1),y_i]+
                                                             liquid_arr[x_i,(y_i-1)]+
                                                             liquid_arr[x_i,(y_i+1)]+
                                                             (1/math.sqrt(2))*
                                                             (liquid_arr[(x_i-1),(y_i-1)]+
                                                             liquid_arr[(x_i+1),(y_i-1)]+
                                                             liquid_arr[(x_i-1),(y_i+1)]+
                                                             liquid_arr[(x_i+1),(y_i+1)]))
                                                 +e_nl*(nano_arr[(x_i-1),y_i]+
                                                             nano_arr[(x_i+1),y_i]+
                                                             nano_arr[x_i,(y_i-1)]+
                                                             nano_arr[x_i,(y_i+1)]+
                                                             (1/math.sqrt(2))*
                                                             (nano_arr[(x_i-1),(y_i-1)]+
                                                             nano_arr[(x_i+1),(y_i-1)]+
                                                             nano_arr[(x_i-1),(y_i+1)]+
                                                             nano_arr[(x_i+1),(y_i+1)])))
                                                 +mu)
    return delta_e

#Define function to calculate bond energy change in nanoparticle step.
def bond_ch(index, ch_indices, liquid_arr, nano_arr, e_l, e_nl, e_n):
    
    x_i = index[0]
    y_i = index[1]
    x_i_neigh = (index[0]+ch_indices[0])
    y_i_neigh = (index[1]+ch_indices[1])
    
    if (x_i_neigh >= len(liquid_arr)) or (y_i_neigh >= len(liquid_arr)) or (y_i >= len(liquid_arr)) or (x_i >= len(liquid_arr)):
        de = 0
    else:
        #Calculate bond energy change
        de = (1-2*liquid_arr[x_i,y_i])*\
                    (e_l*(liquid_arr[x_i_neigh,y_i_neigh])+\
                    e_nl*(nano_arr[x_i_neigh,y_i_neigh]))+\
                (1-2*nano_arr[x_i,y_i])*\
                    (e_n*(nano_arr[x_i_neigh,y_i_neigh])+\
                    e_nl*(liquid_arr[x_i_neigh,y_i_neigh]))
    return -de

#Define function to calculate energy change when performing nanoparticle step.
def delta_E_nano(nano_move, liquid_arr, nano_arr, e_l, e_nl, e_n, ch_indices, wake_offset, nano_size, offset):
    
    delta_e = 0

    #Get cell and wake cell indices
    x = (nano_move[0] + offset[0])
    y = (nano_move[1] + offset[1]) 
    x_wake = (nano_move[0] + wake_offset[0])
    y_wake = (nano_move[1] + wake_offset[1])         

    for i in range(nano_size):
        #Get indices of cells
        x_i = (x+i*abs(ch_indices[1]))
        y_i = (y+i*abs(ch_indices[0]))
        x_i_wake = (x_wake+i*abs(ch_indices[1]))
        y_i_wake = (y_wake+i*abs(ch_indices[0]))
                
        #Add Needed bond energy contributions
        #Add needed bond contributions - nearest neighbours
        delta_e += bond_ch((x_i,y_i), ch_indices, liquid_arr, nano_arr, e_l, e_nl, e_n)
        delta_e += bond_ch((x_i_wake,y_i_wake),(-ch_indices[0],-ch_indices[1]), liquid_arr, nano_arr, e_l, e_nl, e_n)

        #Second nearest neighbours
        delta_e += (1/math.sqrt(2))*bond_ch((x_i,y_i),(ch_indices[0]+ch_indices[1],ch_indices[1]+ch_indices[0]), liquid_arr, nano_arr, e_l, e_nl, e_n)
        delta_e += (1/math.sqrt(2))*bond_ch((x_i,y_i),(ch_indices[0]-ch_indices[1],ch_indices[1]-ch_indices[0]), liquid_arr, nano_arr, e_l, e_nl, e_n)
        delta_e += (1/math.sqrt(2))*bond_ch((x_i_wake,y_i_wake),(-ch_indices[0]+ch_indices[1],-ch_indices[1]+ch_indices[0]), liquid_arr, nano_arr, e_l, e_nl, e_n)                
        delta_e += (1/math.sqrt(2))*bond_ch((x_i_wake,y_i_wake),(-ch_indices[0]-ch_indices[1],-ch_indices[1]-ch_indices[0]), liquid_arr, nano_arr, e_l, e_nl, e_n)

        #Extra contributions needed if nanoparticle cells at end
        if i == 0:
            delta_e += bond_ch((x_i,y_i),(-abs(ch_indices[1]),-abs(ch_indices[0])), liquid_arr, nano_arr, e_l, e_nl, e_n)
            delta_e += bond_ch((x_i_wake,y_i_wake),(-abs(ch_indices[1]),-abs(ch_indices[0])), liquid_arr, nano_arr, e_l, e_nl, e_n)
                    
            delta_e += (1/math.sqrt(2))*bond_ch((x_i,y_i),(-ch_indices[0]-abs(ch_indices[1]),-ch_indices[1]-abs(ch_indices[0])), liquid_arr, nano_arr, e_l, e_nl, e_n)
            delta_e += (1/math.sqrt(2))*bond_ch((x_i_wake,y_i_wake),(ch_indices[0]-abs(ch_indices[1]),ch_indices[1]-abs(ch_indices[0])), liquid_arr, nano_arr, e_l, e_nl, e_n)   

        elif i == (nano_size-1):
            delta_e += bond_ch((x_i,y_i),(abs(ch_indices[1]),abs(ch_indices[0])), liquid_arr, nano_arr, e_l, e_nl, e_n)
            delta_e += bond_ch((x_i_wake,y_i_wake),(abs(ch_indices[1]),abs(ch_indices[0])), liquid_arr, nano_arr, e_l, e_nl, e_n)
                    
            delta_e += (1/math.sqrt(2))*bond_ch((x_i,y_i),(-ch_indices[0]+abs(ch_indices[1]),-ch_indices[1]+abs(ch_indices[0])), liquid_arr, nano_arr, e_l, e_nl, e_n)
            delta_e += (1/math.sqrt(2))*bond_ch((x_i_wake,y_i_wake),(ch_indices[0]+abs(ch_indices[1]),ch_indices[1]+abs(ch_indices[0])), liquid_arr, nano_arr, e_l, e_nl, e_n)

    delta_e *= (1/(1+1/math.sqrt(2)))
    return delta_e

#Define function to perform liquid step.
def liquid_step(x_dim, y_dim, liquid_arr, nano_arr, kT, e_l, e_nl, mu):

    #Generate random flip index:
    flip_index = (np.random.randint(0, x_dim-2), np.random.randint(0, y_dim-2))

    #Check if we can perform that flip:
    if liquid_arr[flip_index[0], flip_index[1]] == 0:
        pass
    else:
        #Calculate energy change:
        DeltaE = delta_E_liquid(flip_index, liquid_arr, nano_arr, e_l, e_nl, mu)
        #print(DeltaE)
        #Compare with metropolis probability:
        P = min(1, np.exp(-DeltaE/kT))
        rand_var = np.random.uniform(0,1)
        #print(P)

        #If our prob is less than randomly generated uniform variable, do not flip.
        if P < rand_var:
            pass
        #Otherwise, accept flip.
        else:
            liquid_arr[flip_index[0], flip_index[1]] = 0
    
    return liquid_arr

#Define function to perform nanoparticle step.
def nano_step(x_dim, y_dim, liquid_arr, nano_arr, nano_list_indices, kT, e_l, e_nl, e_n, nano_size):

    #Randomly pick which nanoparticle we'd like to move.
    nano_move = np.random.randint(0, len(nano_list_indices))
    x_i = nano_list_indices[nano_move][0]
    y_i = nano_list_indices[nano_move][1]

    #print(x_i, y_i)

    #Now randomly pick a direction we'd like to try and move it in.
    #Note: 0, 1, 2, 3 = N, S, E, W, respectively.
    move_dir = np.random.randint(0, 4)

    #Now see whether we can actually move...
    #If we hit the boundary, pass.
    '''if ((move_dir == 0) and (y_i <= 0)) or ((move_dir == 1) and ((y_i + nano_size) >= y_dim)) or ((move_dir == 2) and ((x_i + nano_size) >= x_dim)) or ((move_dir == 3) and (x_i <= 0)):
        pass'''
    
    boundary_hit = 0
    
    if move_dir == 0:
        if y_i <= 0:
            boundary_hit = 1
        else:
            pass

    elif move_dir == 1:
        if y_i >= (y_dim - 2 - nano_size):
            boundary_hit = 1
        else:
            pass
    elif move_dir == 2:
        if x_i >= (x_dim - 2 - nano_size):
            boundary_hit = 1
        else:
            pass
    else:
        if x_i <= 0:
            boundary_hit = 1
        else:
            pass

    if boundary_hit == 0:
        if move_dir == 0:
            #print(move_dir, x_i, y_i)
            liquid_move = liquid_arr[y_i-1, x_i:x_i+nano_size]
            liquid_move = liquid_move.astype(int)
        elif move_dir == 1:
            #print(move_dir, x_i, y_i)
            liquid_move = liquid_arr[y_i+nano_size, x_i:x_i+nano_size]
            liquid_move = liquid_move.astype(int)
        elif move_dir == 2:
            #print(move_dir, x_i, y_i)
            liquid_move = liquid_arr[y_i:y_i+nano_size, x_i+nano_size]
            liquid_move = liquid_move.astype(int)
        else:
            #print(move_dir, x_i, y_i)
            liquid_move = liquid_arr[y_i:y_i+nano_size, x_i-1]
            liquid_move = liquid_move.astype(int)

        #If the bit we're moving into do not have water in them, pass.
        if 0 in liquid_move:
            pass
        #Now if we have all water and we're not on a boundary, we can try to move.
        else:
            #print(liquid_move)

            if move_dir == 0:
                ch_indices = (0, -1)
                offset = (0,-1)
                wake_offset = (0, nano_size-1)
            elif move_dir == 1:
                ch_indices = (0, 1)
                offset = (0, nano_size)
                wake_offset = (0,0)
            elif move_dir == 2:
                ch_indices = (1, 0)
                offset = (nano_size,0)
                wake_offset = (0,0)
            else:
                ch_indices = (-1, 0)
                offset = (-1,0)
                wake_offset = (nano_size-1,0)

            #Calculate change in energy as consequence of move:
            DeltaE = delta_E_nano((nano_list_indices[nano_move][0], nano_list_indices[nano_move][1]), liquid_arr, nano_arr, e_l, e_nl, e_n, ch_indices, wake_offset, nano_size, offset)
            
            #Compare to probability:
            P = min(1, np.exp(-1*DeltaE/kT))
            rand_var = np.random.uniform(0,1)

            #If our prob is less than randomly generated uniform variable, do not flip.
            if P < rand_var:
                pass
            #Otherwise, accept flip.
            else:
                #Remove nanoparticle and fill spot with liquid:
                nano_arr[y_i:y_i+nano_size, x_i:x_i+nano_size] = 0
                liquid_arr[y_i:y_i+nano_size, x_i:x_i+nano_size] = 1
                #Change our index in our list of indices...
                new_nano = (x_i + ch_indices[0], y_i + ch_indices[1])
                nano_list_indices[nano_move] = new_nano
                #Remove from liquid array:
                liquid_arr[new_nano[1]:new_nano[1]+nano_size, new_nano[0]:new_nano[0]+nano_size] = 0
                #Add to nanoparticle placement:
                nano_arr[new_nano[1]:new_nano[1]+nano_size, new_nano[0]:new_nano[0]+nano_size] = 1


                '''if move_dir == 0:
                    new_nano = (x_i, y_i-1)
                    nano_list_indices[nano_move] = new_nano
                    liquid_arr[y_i-1, x_i:x_i+nano_size] = 0
                    nano_arr[y_i-1, x_i:x_i+nano_size] = 1
                    nano_arr[y_i+nano_size, x_i:x_i+nano_size] = 0
                    liquid_arr[y_i+nano_size, x_i:x_i+nano_size] = 1
                    
                elif move_dir == 1:
                    new_nano = (x_i, y_i+1)
                    nano_list_indices[nano_move] = new_nano
                    liquid_arr[y_i+nano_size, x_i:x_i+nano_size] = 0
                    nano_arr[y_i+nano_size, x_i:x_i+nano_size] = 1
                    nano_arr[y_i-1, x_i:x_i+nano_size] = 0
                    liquid_arr[y_i-1, x_i:x_i+nano_size] = 1
                    
                elif move_dir == 2:
                    new_nano = (x_i+1, y_i)
                    nano_list_indices[nano_move] = new_nano
                    liquid_arr[y_i:y_i+nano_size, x_i+nano_size] = 0
                    nano_arr[y_i:y_i+nano_size, x_i+nano_size] = 1
                    nano_arr[y_i:y_i+nano_size, x_i-1] = 0
                    liquid_arr[y_i:y_i+nano_size, x_i-1] = 1
                    
                else:
                    new_nano = (x_i-1, y_i)
                    nano_list_indices[nano_move] = new_nano
                    liquid_arr[y_i:y_i+nano_size, x_i-1] = 0
                    nano_arr[y_i:y_i+nano_size, x_i-1] = 1
                    nano_arr[y_i:y_i+nano_size, x_i+nano_size] = 0
                    liquid_arr[y_i:y_i+nano_size, x_i+nano_size] = 1'''

    else: 
        pass

    return liquid_arr, nano_arr, nano_list_indices
            
#Define function to perform our cycles and simulation.
def growth_sim(x_dim, y_dim, kT, e_l, e_nl, e_n, mu, nano_size, num_cycles, num_nano_per_cycle, num_nano_attempts, num_epochs, seed):

     #Initialize RNG seed
    np.random.seed(seed)

    #How many nanoparticles we actually want, based on coverage.
    n_nano = int(frac*(x_dim*y_dim)/(nano_size*nano_size))

    #Initialize arrays.
    liquid_arr, nano_arr, nano_list_indices = init_arrays(x_dim, y_dim, nano_size, num_nano_attempts, n_nano)

    #For a number of epochs.
    for m in range(num_epochs):

        #For a number of steps.
        for i in range(num_cycles):
            #Perform a liquid step.
            liquid_arr = liquid_step(x_dim, y_dim, liquid_arr, nano_arr, kT, e_l, e_nl, mu)

        #For the number of required nanoparticle cycles:
        for j in range(num_nano_per_cycle):

            for p in range(len(nano_list_indices)):

                liquid_arr, nano_arr, nano_list_indices = nano_step(x_dim, y_dim, liquid_arr, nano_arr, nano_list_indices, kT, e_l, e_nl, e_n, nano_size)

                '''if p == 0 or p % 100 == 0:

                    #Convert our zeros to -1s...
                    nanopart_copy = copy.deepcopy(nano_arr)
                    nanopart_copy[nanopart_copy == 1] = 2

                    config = liquid_arr + nanopart_copy

                    #save_results_to = 'C:/Users/David/Documents/University/THM8999 PhD Thesis/Inverse Design/Growth_Sim_Plots'

                    script_dir = os.path.dirname(__file__)
                    results_dir = os.path.join(script_dir, 'Results/')

                    if not os.path.isdir(results_dir):
                        os.makedirs(results_dir)


                    #plt.imshow(config, cmap='gray')
                    plt.imshow(config)
                    plt.xlabel('Lattice Index')
                    plt.ylabel('Lattice Index')
                    plt.title('Nanoparticle Placements in Liquid')
                    #plt.show()
                    plt.savefig(results_dir + str(plot_count) + '.png')

                    plot_count = plot_count+1'''
            
    #Convert our zeros to -1s...
    nanopart_copy = copy.deepcopy(nano_arr)
    nanopart_copy[nanopart_copy == 1] = 2

    config = liquid_arr + nanopart_copy

    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'Results/')

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    plt.imshow(config)
    plt.xlabel('Lattice Index')
    plt.ylabel('Lattice Index')
    plt.title('Nanoparticle Placements in Liquid \n kbT = ' + str(kbT) + ', Fraction = ' + str(frac) + ', ' + str(num_epochs) + ' Epochs')
    plt.savefig(results_dir + 'kbt_02_frac_02_1000epochs_fin.png')
    plt.show()

#######################################################################

frac = 0.2
x_dim = 1000
y_dim = 1000
nano_size = 3
num_nano_attempts = 100

nano_steps = 30
solv_iter = x_dim*y_dim

kbT = 0.2

num_epochs = 1000

seed = 18

growth_sim(x_dim, y_dim, kbT, 1, 1.5, 2, -2.5, nano_size, solv_iter, 30, num_nano_attempts, num_epochs, seed)



