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
from copy import deepcopy

#######################################################################

#Question 1 functions.

#Define a function to generate our list of N city indices in an MxM lattice.
#Note that these cities are already in a random order - we can save some computational time
#by not having to generate a permutation! 
def generate_cities(N, M):
    
    #Initialize array.
    cities = []
    
    #For the number of cities we want N:
    while len(cities) < N:
        #Generate x and y coordinates.
        city_x_index = np.random.randint(0, M-1)
        city_y_index = np.random.randint(0, M-1)
        #Throw these in a tuple.
        city_indices = (city_x_index, city_y_index)
        #Throw away our city if we've already generated it, we don't want two on top of each other:
        if city_indices in cities:
            pass
        else:
            cities.append(city_indices)
    return cities

#Define a function to caluclate the route length:
def calc_route_length(config):
    
    route_length = 0
    
    #For each city, calculate the distance between it and previous.
    for i in range(len(config)):
        d = np.sqrt((config[i][0] - config[i-1][0])**2 + (config[i][1] - config[i-1][1])**2)
        route_length = route_length + d
        
    return route_length

#Define a function to generate a trial configuration where we swap the order of visiting two cities.
def generate_trial_config(config):
    
    #Make a deep copy so we don't accidentally mess up our config:
    trial_config = deepcopy(config)
    
    #Generate one random city to switch:
    first_swap = np.random.randint(0, len(config))
    #Assign this value to our second swap:
    second_swap = first_swap
    
    #Generate a second random city to switch, making sure it's different from our first:
    while second_swap == first_swap:
        second_swap = np.random.randint(0, len(config))
    
    #Now swap our indices:
    hold_value = deepcopy(trial_config[first_swap])
    trial_config[first_swap] = trial_config[second_swap]
    trial_config[second_swap] = hold_value
    
    #print(trial_config)
    
    return trial_config, first_swap, second_swap

#Define a function to calculate the change in distance from our configuration change:
def calc_distance_change(config, trial_config, first_swap, second_swap):
    
    #Calculate original distance:
    x_i = config[first_swap][0]
    y_i = config[first_swap][1]
    #Distance between first swap -1:
    x_1_min = config[first_swap-1][0]
    y_1_min = config[first_swap-1][1]
    #Pythagorean theorem:
    d_one_min = np.sqrt((x_i - x_1_min)**2 + (y_i - y_1_min)**2)
    #Distance between first swap +1, set our bounds on our array of values:
    if (first_swap + 1) > (len(config)-1):
        swap_max = 0
    else:
        swap_max = first_swap +1
    x_1_max = config[swap_max][0]
    y_1_max = config[swap_max][1]
    #Pythagoran theorem:
    d_one_max = np.sqrt((x_1_max - x_i)**2 + (y_1_max - y_i)**2)
    #Distance between second swap -1:
    x_j = config[second_swap][0]
    y_j = config[second_swap][1]
    x_2_min = config[second_swap-1][0]
    y_2_min = config[second_swap-1][1]
    #Pythagorean theorem:
    d_two_min = np.sqrt((x_j - x_2_min)**2 + (y_j - y_2_min)**2)
    #Distance between second swap +1, set our bounds on array of values:
    if (second_swap + 1) > (len(config)-1):
        swap_max = 0
    else:
        swap_max = second_swap +1
    x_2_max = config[swap_max][0]
    y_2_max = config[swap_max][1]
    #Pythagorean theorem:
    d_two_max = np.sqrt((x_2_max - x_j)**2 + (y_2_max - y_j)**2)
    
    #Now we have our original distance that navigating to these points took:
    original_distance = d_one_min + d_one_max + d_two_min + d_two_max
    #print(original_distance)
    
    #Now repeat for the trial configuration...
    #Calculate original distance:
    x_i = trial_config[first_swap][0]
    y_i = trial_config[first_swap][1]
    #Distance between first swap -1:
    x_1_min = trial_config[first_swap-1][0]
    y_1_min = trial_config[first_swap-1][1]
    #Pythagorean theorem:
    d_one_min = np.sqrt((x_i - x_1_min)**2 + (y_i - y_1_min)**2)
    #Distance between first swap +1, set our bounds on our array of values:
    if (first_swap + 1) > (len(trial_config)-1):
        swap_max = 0
    else:
        swap_max = first_swap +1
    x_1_max = trial_config[swap_max][0]
    y_1_max = trial_config[swap_max][1]
    #Pythagoran theorem:
    d_one_max = np.sqrt((x_1_max - x_i)**2 + (y_1_max - y_i)**2)
    #Distance between second swap -1:
    x_j = trial_config[second_swap][0]
    y_j = trial_config[second_swap][1]
    x_2_min = trial_config[second_swap-1][0]
    y_2_min = trial_config[second_swap-1][1]
    #Pythagorean theorem:
    d_two_min = np.sqrt((x_j - x_2_min)**2 + (y_j - y_2_min)**2)
    #Distance between second swap +1, set our bounds on array of values:
    if (second_swap + 1) > (len(config)-1):
        swap_max = 0
    else:
        swap_max = second_swap +1
    x_2_max = trial_config[swap_max][0]
    y_2_max = trial_config[swap_max][1]
    #Pythagorean theorem:
    d_two_max = np.sqrt((x_2_max - x_j)**2 + (y_2_max - y_j)**2)
    
    #Now we have our new distance that navigating to these points took:
    new_distance = d_one_min + d_one_max + d_two_min + d_two_max
    #print(new_distance)
    
    #Calculate our change in distance:
    delta_d = new_distance - original_distance
    #print(delta_d)
    
    return delta_d

#Define a function to calculate our probability and evaluate it:
def evaluate_route_change(delta_d, T):
    
    #Calculate our probability:
    P = math.exp((-1*delta_d)/(T))
    #Geneate uniform random variable:
    u = np.random.uniform()
    #Estimate whether we're keeping this or not:
    if u <= P:
        accept = True
    else:
        accept = False
    
    return accept
    
#Define a function to minimize our route:
def minimize_route(config, T, roc_tol, mean_samples):
    
    #Create counter for our number of steps, and initial large value for rate of change:
    numsteps = 0
    roc = 10*roc_tol
    
    #Find our initial route length - we'll need this later...
    route_length = calc_route_length(config)
    #Create array to store new route lengths:
    route_lengths = []
    
    #Want rate of change of our config over some number of steps to be small...
    while roc > roc_tol:
        
        #Generate trial configuration and calulate change in distance:
        trial_config, first_swap, second_swap = generate_trial_config(config)
        delta_d = calc_distance_change(config, trial_config, first_swap, second_swap)
        
        #Evaluate if we are keeping our new configuration or not...
        if delta_d <= 0:
            config = deepcopy(trial_config)
            route_length = route_length + delta_d
            route_lengths.append(route_length)
            #Increment numsteps counter:
            numsteps = numsteps +1
        else:
            accept = evaluate_route_change(delta_d, T)
            if accept == True:
                config = deepcopy(trial_config)
                route_length = route_length + delta_d
                route_lengths.append(route_length)
                #Increment numsteps counter:
                numsteps = numsteps +1
            else: 
                pass
        
        #If we have more than 100 steps, evaluate our mean route length over the last 50 steps,
        #compare to the 50 before, and see what our rate of change is.
        if numsteps > (mean_samples*2):
            new_mean = np.mean(route_lengths[numsteps-mean_samples:])
            old_mean = np.mean(route_lengths[numsteps-(mean_samples*2):numsteps-mean_samples])
            roc = abs(new_mean - old_mean)
            
    return config, route_lengths

#Define a function to compute our smallest route overall:
def travelling_salesman(N, M, initial_T, delta_T, roc_tol_at_T, roc_tol, mean_samples, failsafe):
    
    #Generate initial configuration:
    initial_config = generate_cities(N, M)
    
    #Create equivalent of do-while loop:
    while True:
        #Do some math with our original route, including calculating route lengths etc.
        config = initial_config
        old_route_length = calc_route_length(config)
        old_route_lengths = []
        old_route_lengths.append(old_route_length)
        #Create set of unique routes.
        unique_routes = set(np.around(old_route_lengths,3))
        #Set an initial roc we can evaluate against.
        roc = 10*roc_tol
        #Set our initial T.
        T = initial_T
        #Plot our initial configuration:
        x_coords = []
        y_coords = []
                
        for i in range(len(config)):
            x_coords.append(config[i][0])
            y_coords.append(config[i][1])
            
                
        final_plot_xs = []
        final_plot_xs.append(x_coords[0])
        final_plot_xs.append(x_coords[-1])
        final_plot_ys = []
        final_plot_ys.append(y_coords[0])
        final_plot_ys.append(y_coords[-1])
                    
        #Now we plot!
        plt.plot(x_coords, y_coords, marker ='o', color = 'tab:blue', linestyle = 'solid', label = 'Route')
        plt.plot(final_plot_xs, final_plot_ys, color = 'tab:blue', linestyle = '--', label = 'Path from Final City \n Back to Start')
        plt.xlabel('Lattice Index')
        plt.ylabel('Lattice Index')
        plt.title('Minimal Travelling Salesman Route \n For ' + str(N) + ' Cities on an ' + str(M) + ' by ' + str(M) + ' Lattice \n Current T = ' + str(T) + ' \n')
        plt.legend(loc=(1.05, 0), markerscale=1)
        plt.show()
        
        #Run while our roc is greater than our tol, and our temperature is above 0:
        while (roc > roc_tol) and (T > 0):
            #Minimize route at current T:
            config, route_lengths = minimize_route(config, T, roc_tol_at_T, mean_samples)
            #If we've come up with a shorter route:
            if len(route_lengths) != 0:
                #Calculate change in mean:
                old_mean = np.mean(old_route_lengths)
                new_mean = np.mean(route_lengths)
                roc = abs(new_mean - old_mean)
                #If our roc is still greater:
                if roc > roc_tol:
                    #Swap over and update our recent routes...
                    old_route_lengths = deepcopy(route_lengths)
                    temp_set_routes = set(np.around(old_route_lengths,3))
                    unique_routes = unique_routes.union(temp_set_routes)
                else:
                    pass
            else:
                pass
            
            #Now we show some plots at various T so we can show our evolution...
            if T in [20, 15, 10, 5]:
                x_coords = []
                y_coords = []
                
                for i in range(len(config)):
                    x_coords.append(config[i][0])
                    y_coords.append(config[i][1])
            
                
                final_plot_xs = []
                final_plot_xs.append(x_coords[0])
                final_plot_xs.append(x_coords[-1])
                final_plot_ys = []
                final_plot_ys.append(y_coords[0])
                final_plot_ys.append(y_coords[-1])
                    
                #Now we plot!
                plt.plot(x_coords, y_coords, marker ='o', color = 'tab:blue', linestyle = 'solid', label = 'Route')
                plt.plot(final_plot_xs, final_plot_ys, color = 'tab:blue', linestyle = '--', label = 'Path from Final City \n Back to Start')
                plt.xlabel('Lattice Index')
                plt.ylabel('Lattice Index')
                plt.title('Minimal Travelling Salesman Route \n For ' + str(N) + ' Cities on an ' + str(M) + ' by ' + str(M) + ' Lattice \n Current T = ' + str(T) + ' \n')
                plt.legend(loc=(1.05, 0), markerscale=1)
                plt.show()
            else: 
                pass
            
            #Increment T downwards...
            T = T - delta_T
            print(T)
            
            #If T goes below zero...
            if T <= 0:
                route_lengths.append(old_route_lengths[-1])
            else:
                pass
        
        #Sort our set of routes so we can use these in failsafe.
        sorted_routes = sorted(unique_routes)
        #print(sorted_routes)
            
        #If we're not using the failsafe, or if our route is as short as the shortest possible route after hitting roc tolerance, break the loop.
        if (failsafe == False) or (round(route_lengths[-1],3) <= (round(sorted_routes[0],3)+0.001)):
            break
    
    #Now we plot our final route...
    x_coords = []
    y_coords = []
    
    for i in range(len(config)):
        x_coords.append(config[i][0])
        y_coords.append(config[i][1])

    
    final_plot_xs = []
    final_plot_xs.append(x_coords[0])
    final_plot_xs.append(x_coords[-1])
    final_plot_ys = []
    final_plot_ys.append(y_coords[0])
    final_plot_ys.append(y_coords[-1])
        
    #Now we plot!
    plt.plot(x_coords, y_coords, marker ='o', color = 'tab:blue', linestyle = 'solid', label = 'Route')
    plt.plot(final_plot_xs, final_plot_ys, color = 'tab:blue', linestyle = '--', label = 'Path from Final City \n Back to Start')
    plt.xlabel('Lattice Index')
    plt.ylabel('Lattice Index')
    plt.title('Minimal Travelling Salesman Route \n For ' + str(N) + ' Cities on an ' + str(M) + ' by ' + str(M) + ' Lattice \n Current T = ' + str(T) + ' \n')
    plt.legend(loc=(1.05, 0), markerscale=1)
    plt.show()

    return config, unique_routes, route_lengths[-1]

#######################################################################

#Question 2 functions.
    
#Write function to generate configuration:
def generate_configuration(N):
    #Initialize random NxN array of spins.
    #This actually makes random array of 0s and 1s, so let 0s be spin downs.
    config = np.random.randint(0, 2, (N,N))
    
    #Convert our zeros to -1s...
    config[config == 0] = -1

    return config

#Write function to generate our glass spins:
def generate_glass_spins(N):
    #Initialize random Nx2N array of spins.
    #This actually makes random array of 0s and 1s, so let 0s be spin downs.
    config = np.random.randint(0, 2, (2*N,N))
    
    #Convert our zeros to -1s...
    config[config == 0] = -1
    
    return config
    
#Write function to calculate our internal energy.  
def calc_energy(J, config, N):
    
    #Initialize J term...
    total_E = 0
    
    #Iterate over all [i,j] lattice sites and all possible J values.
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
                J_i_one = 2*i
                J_i_max_one = 2*i + 1
                J_i_max_two = 0
            else:
                s_i_max = i+1
                J_i_one = 2*i
                J_i_max_one = 2*i + 1
                J_i_max_two = 2*i + 2
            if j == N-1:
                s_j_max = 0
                J_j_max = 0
            else:
                s_j_max = j+1
                J_j_max = j+1
            
            #Find J term at our specific lattice site.
            #DOWN
            #RIGHT
            #UP
            #LEFT
            J_add = J[J_i_max_two, j]*(config[i,j]*config[s_i_max,j]) + J[J_i_max_one, J_j_max]*(config[i,j]*config[i,s_j_max]) + J[J_i_one, j]*(config[i,j]*config[s_i_min,j]) + J[J_i_max_one, j]*(config[i,j]*config[i,s_j_min])
            
            #Add it.
            total_E = total_E + J_add
    
    return total_E

#Write function to generate offspring.
def make_offspring(parent_1, parent_2, N):
    
    #Initialize offspring arrays:
    offspring_1 = np.zeros((N, N))
    offspring_2 = np.zeros((N, N))
    
    #Iterate over parent spins:
    for i in range(N):
        for j in range(N):
            
            #If parent spins are the same at a given index, pop them in offspring.
            if parent_1[i][j] == parent_2[i][j]:
                offspring_1[i][j] = parent_1[i][j]
                offspring_2[i][j] = parent_1[i][j]
            else: 
                #If not, randomly give offspring a spin (0, 1)
                offspring_1[i][j] = np.random.randint(0,2)
                offspring_2[i][j] = np.random.randint(0,2)
            
    #Convert our zeros to -1s...
    offspring_1[offspring_1 == 0] = -1
    offspring_2[offspring_2 == 0] = -1
    
    return offspring_1, offspring_2

#Write function to make mutations.
def make_mutations(offspring, N, R):
    
    #For all spin indices:
    for i in range(N):
        for j in range(N):
            #Generate uniform random variable:
            u = np.random.uniform()
            #If u is less than our mutation rate R:
            if u <= R:
                #Swap the spin at this spot.
                offspring[i][j] = -1*offspring[i][j]
            else:
                pass
            
    return offspring
    
#Write function to perform our genetic algorithm.
def genetic_algorithm(N, R, M, tol):
    
    #Generate our J values, which will be the same for all configurations.
    J = generate_glass_spins(N)
    
    #Generate initial population configurations and their energies:
    all_configs = []
    all_energies = []
    
    for i in range(M):
        #Generate config and calculate its energy.
        append_config = generate_configuration(N)
        append_energy = calc_energy(J, append_config, N)
        #Append to arrays to store - note index of config in all_configs will be
        #the same as index in all_energies.
        all_configs.append(append_config)
        all_energies.append(append_energy)
    
    #Generate list of parent one indices:
    parent_indices = np.arange(0, M, 2)
    
    mean_diff = 0
    
    num_iterations = 0
    
    #For the number of generations we want to perform:
    while mean_diff < tol:
        #print(w)
        for i in parent_indices:
            #Make list of energies to compare and configurations we can easily grab.
            energies_compare = []
            pick_configs = []
            #Put parent 1 energy and config into list:
            energies_compare.append(all_energies[i])
            pick_configs.append(all_configs[i])
            #Put parent 2 energy into list:
            energies_compare.append(all_energies[i+1])
            pick_configs.append(all_configs[i+1])
            
            #Generate the two offspring:
            offspring_1, offspring_2 = make_offspring(all_configs[i], all_configs[i+1], N)
            #Create mutations in the offspring:
            offspring_1 = make_mutations(offspring_1, N, R)
            offspring_2 = make_mutations(offspring_2, N, R)
            #Calculate offspring energies:
            offspring_1_energy = calc_energy(J, offspring_1, N)
            offspring_2_energy = calc_energy(J, offspring_1, N)
            #Pop these in to the arrays to compare energies and pick configs:
            energies_compare.append(offspring_1_energy)
            pick_configs.append(offspring_1)
            energies_compare.append(offspring_2_energy)
            pick_configs.append(offspring_2)
            
            #Initialize our number of minima, and prepare to store energies and configs we're keeping:
            num_minima = 0
            keep_energies = []
            keep_configs = []
            
            while num_minima < 2:
                
                where_min = []
                
                #Now find our minimum energy and the indices.
                min_energy = min(energies_compare)
                
                for m in range(len(energies_compare)):
                    if energies_compare[m] == min_energy:
                        where_min.append(m)
                
                #Either we can have one, two, three, or four minimum values (i.e. if they are all the same).
                #Go through each case.
                
                #If we have one minima:
                if len(where_min) == 1:
                    #Grab this energy and config.
                    where_1 = where_min[0]
                    energy_1 = energies_compare[where_1]
                    config_1 = pick_configs[where_1]
                    
                    #Pop them into our arrays to store.
                    keep_energies.append(energy_1)
                    keep_configs.append(config_1)
                    
                    #Now remove this energy and config from the lists:
                    energies_compare.pop(where_1)
                    pick_configs.pop(where_1)
                    
                    num_minima = num_minima+1
                    
                else:
                    
                    #Pick one random index from the number of possible minima we have to keep.
                    index_keep = np.random.randint(0,len(where_min))
                    where_1 = where_min[index_keep]
                    #Grab this energy and config.
                    energy_1 = energies_compare[where_1]
                    config_1 = pick_configs[where_1]
                    
                    #Pop them into our arrays to store.
                    keep_energies.append(energy_1)
                    keep_configs.append(config_1)
                    
                    #Now remove this energy and config from the lists:
                    energies_compare.pop(where_1)
                    pick_configs.pop(where_1)
                    
                    num_minima = num_minima+1
                    
            #Now replace our energy and config in our big parent list for next iteration:
            all_energies[i] = keep_energies[0]
            all_configs[i] = keep_configs[0]
            all_energies[i+1] = keep_energies[1]
            all_configs[i+1] = keep_configs[1]
            
            
        vals, count_vals = np.unique(all_energies, return_counts=True)
        
        mean_diff = max(count_vals)/M
        print(mean_diff)
        min_val = min(vals)
        print(min_val)
        
        num_iterations = num_iterations + 1
        
    #Then once we've finished our generations, once again find minimum energy and 
    #corresponding configuration(s).
    gs_energy = min(all_energies)
    
    gs_index = all_energies.index(gs_energy)
    
    ground_state = all_configs[gs_index]
    
    plt.imshow(ground_state, cmap='gray')
    plt.xlabel('\n Lattice Index')
    plt.yticks(np.arange(8), ['0', '1', '2', '3', '4', '5', '6', '7', '8']) 
    plt.ylabel('Lattice Index \n')
    plt.title('Ground State of ' + str(N) + ' by ' + str(N) + ' Spin-Glass Ising Lattice \n At B = 0, After ' + str(num_iterations) + ' Iterations \n')
    
#######################################################################
#Main: let's run some functions:
    
#config, unique_routes, final_route_length = travelling_salesman(6, 10, 20, 0.05, 0.001, 0.001, 100, True)
#print('Set of Unique Route Lengths: ' + str(unique_routes))
#print('Final Route Length: ' + str(final_route_length))

#config, unique_routes, final_route_length = travelling_salesman(80, 500, 25, 0.5, 0.1, 0.1, 100, False)
#print('Final Route Length: ' + str(final_route_length))

genetic_algorithm(4, 0.1, 100, 0.9)