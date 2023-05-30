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

#Define function to initialize water/vapour and nanoparticle arrays.
def init_arrays(x_dim, y_dim, nano_size, num_nano_attempts):

    #Initial liquid array should be all liquid.
    liquid_arr = np.ones((x_dim, y_dim))

    #Here, we want to generate a list of nanoparticle top left indices,
    #as well as populate the spots in the nano array which are filled
    #with nanoparticles.
    nano_arr = np.zeros((x_dim, y_dim))

    nano_list_indices = []

    #Now attempt to populate nanoparticle array with nanoparticles.
    for i in range(num_nano_attempts):

        #Generate random index:
        indices = (np.random.randint(0, x_dim-(nano_size-1)), np.random.randint(0, y_dim-(nano_size-1)))

        #Check and see if this index is already occupied by a nanoparticle:
        nano_placement = nano_arr[indices[0]:indices[0]+nano_size, indices[1]:indices[1]+nano_size]
        
        #If any value in our nano placements are 1, we don't add this particle.
        if any(1 in x for x in nano_placement):
            pass
        else:
            #Now if those spots are empty, we accept the nanoparticle and add it to our arrays.
            #Add to list of nano indices:
            nano_list_indices.append(indices)
            #Add to nanoparticle placement:
            nano_arr[indices[0]:indices[0]+nano_size, indices[1]:indices[1]+nano_size] = 1
            #Remove from liquid array:
            liquid_arr[indices[0]:indices[0]+nano_size, indices[1]:indices[1]+nano_size] = 1

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
            
#######################################################################

#liquid_arr, nano_arr, nano_list_indices = init_arrays(30, 30, 3, 5)
        



