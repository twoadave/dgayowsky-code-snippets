"""
Created on Fri May 26

@author: David Gayowsky

Simple growth Monte Carlo learning algorithm.

Written by Viktor Selin, and modified by David Gayowsky. Base code for 
Neural Network integration to growth algorithms.
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

import numba as nb
from numba.experimental import jitclass
from numba import jit
from numba.types import UniTuple
from typing import List
from numba.typed import List as NumbaList

import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import Linear, ReLU, Sigmoid, Module, BCELoss, Softmax
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#######################################################################

'''Args:
kT: temperature * Kb
mu: chemical potential
e_nn: nano-nano attraction
e_nl: nano-solvent attraction
e_ll: solvent-solvent attraction -- All other attractions and mu are given in relation to this, leave as 1'''

#######################################################################

'''
--------------------------------------------------------
Main Growth class 
--------------------------------------------------------
Args:
x_dim: x dimension of lattice
y_dim: y dimension of lattice
n_nano: number of nanoparticles to place
KbT: temperature * Kb
mu: chemical potential
e_nn: nano-nano attraction
e_nl: nano-solvent attraction
e_ll: solvent-solvent attraction -- All other attractions and mu are given in relation to this, leave as 1
seed: random seed for rng generator
nano_mob: number of nanoparticle cycles to perform per solvent cycle
nano_size: size of nanoparticles (nano_size x nano_size)
seed: random seed to use
--------------------------------------------------------
Use example:
growth = Growth(args)
growth.initialize_nano()
for range(n_epochs):
  growth.step_simulation()

You can the extract the growth by simply using:
nano_array = growth.nano
solv_array = growth.fluid
--------------------------------------------------------
Note: mostly used for loops to take advantage of Cuda jitclass
--------------------------------------------------------
Default arguments to (mostly) match paper: 
y_dim = 1000
x_dim = 1000
frac = 0.3
mu = -2.5
KbT = 0.2
e_nn = 2
e_nl = 1.5
e_ll = 1
nano_mob = 30
nano_size = 3
n_nano = int(frac*(x_dim*y_dim)/(nano_size*nano_size))
--------------------------------------------------------
'''
@jitclass
class Growth_NonPeriodic:
    #Type annotation for Numba
    x_dim: int
    y_dim: int
    n_nano: int
    n_nano_placed: int
    nano_size: int
    fluid: nb.int64[:,:]
    nano: nb.int64[:,:]
    nano_list: List[UniTuple(nb.int64,2)]
    total_energy: float
    e_ll: float
    e_nl: float
    e_nn: float
    mu: float
    nano_mob: int
    solv_iter: int
    KbT: float
    seed: int
        
    def __init__(self, x_dim, y_dim, n_nano, KbT, mu, e_nn, e_nl, e_ll, nano_mob, nano_size, seed):
        #Keep parameters
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_nano = n_nano
        self.n_nano_placed = 0
        self.seed = seed
        self.nano_size = nano_size
        
        #Initialize RNG seed
        np.random.seed(self.seed)
        
        #Initialize lattices
        self.fluid = np.ones((self.x_dim,self.y_dim),dtype=np.int64)
        self.nano = np.zeros((self.x_dim,self.y_dim),dtype=np.int64)
        #Initialize list of nanoparticles
        self.nano_list = NumbaList([(1,2) for x in range(0)])
        
        #Initialize prev energy
        self.total_energy = 0
        
        #Initialize constants
        #Liquid - liquid attraction
        self.e_ll = e_ll
        #Nano - nano attraction
        self.e_nn = e_nn*self.e_ll
        #Nano - liquid attraction
        self.e_nl = e_nl*self.e_ll
        #Chemical potential
        self.mu = mu*self.e_ll
        
        #Nano mobility and per step solvent iterations
        self.nano_mob = nano_mob
        self.solv_iter = x_dim*y_dim
        
        #Boltzmann*Temperature
        self.KbT = self.e_ll*KbT
        
    '''
    Old function to calculate energy
    Calculates total energy on entire lattice
    Left here for comparison purposes
    '''
    #Function that multiplies all elements with neighbours and sums
    def sum_neighbour_energy(self,A,B):
        return np.sum(A*(np.roll(B,1,0) + np.roll(B,-1,0) + np.roll(B,1,1) + np.roll(B,-1,1)))

    '''
    As above
    '''
    #Slow way of calcuating total energy
    def calculate_total_energy(self,nano,fluid):
        total_energy = 0

        #Liquid - liquid contribution
        total_energy -= self.e_ll * self.sum_neighbour_energy(fluid,fluid) / 2

        #Nano - nano contribution
        total_energy -= self.e_nn * self.sum_neighbour_energy(nano,nano) / 2

        #Nano - liquid contribution
        total_energy -= self.e_nl * self.sum_neighbour_energy(nano,fluid)

        #Liquid phase contribution
        total_energy -= self.mu * np.sum(fluid)

        return total_energy
         
    '''
    Function that performs a single solvent step
    Chooses a random lattice site and attempts to change phase
    '''
    def step_fluid(self):
        #Choose random lattice cell
        x_i = np.random.randint(1,self.x_dim-1)
        y_i = np.random.randint(1,self.y_dim-1)

        #Only proceed if no nano in cell
        delta_e = 0
        #Scaling for next nearest and nearest neighbour Energy contribution
        norm = 1/(1+1/math.sqrt(2))
        if self.nano[x_i,y_i] == 0:
            #Calculate change in energy
            #Takes into account nearest and next nearest
            delta_e = -(1-2*self.fluid[x_i,y_i])*((norm)*(self.e_ll*(self.fluid[(x_i-1),y_i]+
                                                             self.fluid[(x_i+1),y_i]+
                                                             self.fluid[x_i,(y_i-1)]+
                                                             self.fluid[x_i,(y_i+1)]+
                                                             (1/math.sqrt(2))*
                                                             (self.fluid[(x_i-1),(y_i-1)]+
                                                             self.fluid[(x_i+1),(y_i-1)]+
                                                             self.fluid[(x_i-1),(y_i+1)]+
                                                             self.fluid[(x_i+1),(y_i+1)]))
                                                 +self.e_nl*(self.nano[(x_i-1),y_i]+
                                                             self.nano[(x_i+1),y_i]+
                                                             self.nano[x_i,(y_i-1)]+
                                                             self.nano[x_i,(y_i+1)]+
                                                             (1/math.sqrt(2))*
                                                             (self.nano[(x_i-1),(y_i-1)]+
                                                             self.nano[(x_i+1),(y_i-1)]+
                                                             self.nano[(x_i-1),(y_i+1)]+
                                                             self.nano[(x_i+1),(y_i+1)])))
                                                 +self.mu)



            #Calculate probability of accepting move
            Pacc = min(1,np.exp(-delta_e/self.KbT))

            if np.random.random() <= Pacc:
                #Accept move - update total energy
                self.total_energy += delta_e
                
                #Change solvent phase
                self.fluid[x_i,y_i] = 1-self.fluid[x_i,y_i]

        return delta_e

    '''
    Function that calculates the change in bond energy for a nanoparticle move
    '''
    def neigh_de(self,cell,dis):
        x_i = cell[0]
        y_i = cell[1]
        x_i_neigh = (cell[0]+dis[0])
        y_i_neigh = (cell[1]+dis[1])
        
        #Calculate bond energy change
        de = (1-2*self.fluid[x_i,y_i])*\
                (self.e_ll*(self.fluid[x_i_neigh,y_i_neigh])+\
                 self.e_nl*(self.nano[x_i_neigh,y_i_neigh]))+\
             (1-2*self.nano[x_i,y_i])*\
                (self.e_nn*(self.nano[x_i_neigh,y_i_neigh])+\
                 self.e_nl*(self.fluid[x_i_neigh,y_i_neigh]))
        return -de
    
    '''
    Function that calculates total energy using neighbour bonds
    '''
    def calculate_total_energy_neigh(self):
        total_e = 0
        #Nearest neighors
        diss = [(1,0),(-1,0),(0,1),(0,-1)]
        #Next-nearest neighbors
        nneigh = [(1,1),(-1,1),(1,-1),(-1,-1)]
        #Exclude the boundaries -> non-periodic
        for i in range(1,self.x_dim-1):
            for j in range(1,self.y_dim-1):
                for k in range(4):
                    dis = diss[k]
                    neigh = nneigh[k]
                    x = (i+dis[0])
                    y = (j+dis[1])
                    x_n = (i+neigh[0])
                    y_n = (j+neigh[1])
                    #Add current cell energy contribution
                    #Everything divided by 2 due to calculating each bond twice
                    total_e -= self.fluid[i,j]*\
                                ((1/(1+1/math.sqrt(2)))*(self.e_ll*(self.fluid[x,y]+(1/math.sqrt(2))*self.fluid[x_n,y_n])/2 +\
                                 self.e_nl*(self.nano[x,y]+(1/math.sqrt(2))*self.nano[x_n,y_n])) +\
                                 self.mu/4)+\
                               self.nano[i,j]*\
                                 ((1/(1+1/math.sqrt(2)))*self.e_nn*(self.nano[x,y]+(1/math.sqrt(2))*self.nano[x_n,y_n])/2)
        return total_e
                
    '''
    Function that performs a single nanoparticle step
    Chooses a random nanoparticle and attempts to move it in a random direction
    '''
    def step_nano(self):
        #Select nano particle to move
        i_nano = np.random.randint(0,len(self.nano_list))
        
        x_nano = self.nano_list[i_nano][0]
        y_nano = self.nano_list[i_nano][1]
        
        #Select displacement direction
        dir_nano = np.random.randint(0,4)
        hitBoundary = False
        #Determine what direction the nanoparticle moves in
        #dis: displacement vector
        #offset: offset for the cells in front of nanoparticles
        #wake_offset: offset for the wake
        #hitBoundary: whether the particle is trying to leave the simulation box
        match dir_nano:
            case 0: #+y
                dis = (0,1)
                offset = (0,self.nano_size)
                wake_offset = (0,0)
                hitBoundary = y_nano >= (self.y_dim-self.nano_size-1)		
            case 1: #-y
                dis = (0,-1)
                offset = (0,-1)
                wake_offset = (0,self.nano_size-1)
                hitBoundary = y_nano <= 1
            case 2: #+x
                dis = (1,0)
                offset = (self.nano_size,0)
                wake_offset = (0,0)
                hitBoundary = x_nano >= (self.x_dim-self.nano_size-1)
            case 3: #-x
                dis = (-1,0)
                offset = (-1,0)
                wake_offset = (self.nano_size-1,0)
                hitBoundary = x_nano <= 1

        #Check if all cells in the movement direction have fluid and no nanoparticles present
        fluid_sum = 0
        nano_sum = 0
        for i in range(self.nano_size):
            fluid_sum += self.fluid[(x_nano+offset[0]+i*abs(dis[1])),(y_nano+offset[1]+i*abs(dis[0]))]
            nano_sum += self.nano[(x_nano+offset[0]+i*abs(dis[1])),(y_nano+offset[1]+i*abs(dis[0]))]
        
        delta_e = 0
        #Move only if no nanoparticles blocking and all cells are occupied by fluid
        if not hitBoundary and fluid_sum == self.nano_size and nano_sum == 0:

            #Get cell and wake cell indices
            x = (x_nano + offset[0])
            y = (y_nano + offset[1]) 
            x_wake = (x_nano + wake_offset[0])
            y_wake = (y_nano + wake_offset[1])         

            for i in range(self.nano_size):
                #Get indices of cells
                x_i = (x+i*abs(dis[1]))
                y_i = (y+i*abs(dis[0]))
                x_i_wake = (x_wake+i*abs(dis[1]))
                y_i_wake = (y_wake+i*abs(dis[0]))
                
                #Add Needed bond energy contributions
                #Add needed bond contributions - nearest neighbours
                delta_e += self.neigh_de((x_i,y_i),dis)
                delta_e += self.neigh_de((x_i_wake,y_i_wake),(-dis[0],-dis[1]))
                
                #Second nearest neighbours
                delta_e += (1/math.sqrt(2))*self.neigh_de((x_i,y_i),(dis[0]+dis[1],dis[1]+dis[0]))
                delta_e += (1/math.sqrt(2))*self.neigh_de((x_i,y_i),(dis[0]-dis[1],dis[1]-dis[0]))
                
                delta_e += (1/math.sqrt(2))*self.neigh_de((x_i_wake,y_i_wake),(-dis[0]+dis[1],-dis[1]+dis[0]))                
                delta_e += (1/math.sqrt(2))*self.neigh_de((x_i_wake,y_i_wake),(-dis[0]-dis[1],-dis[1]-dis[0]))
                
                #Extra contributions needed if nanoparticle cells at end
                if i == 0:
                    delta_e += self.neigh_de((x_i,y_i),(-abs(dis[1]),-abs(dis[0])))
                    delta_e += self.neigh_de((x_i_wake,y_i_wake),(-abs(dis[1]),-abs(dis[0])))
                    
                    delta_e += (1/math.sqrt(2))*self.neigh_de((x_i,y_i),(-dis[0]-abs(dis[1]),-dis[1]-abs(dis[0])))
                    delta_e += (1/math.sqrt(2))*self.neigh_de((x_i_wake,y_i_wake),(dis[0]-abs(dis[1]),dis[1]-abs(dis[0])))                    
                elif i == (self.nano_size-1):
                    delta_e += self.neigh_de((x_i,y_i),(abs(dis[1]),abs(dis[0])))
                    delta_e += self.neigh_de((x_i_wake,y_i_wake),(abs(dis[1]),abs(dis[0])))
                    
                    delta_e += (1/math.sqrt(2))*self.neigh_de((x_i,y_i),(-dis[0]+abs(dis[1]),-dis[1]+abs(dis[0])))
                    delta_e += (1/math.sqrt(2))*self.neigh_de((x_i_wake,y_i_wake),(dis[0]+abs(dis[1]),dis[1]+abs(dis[0])))
            delta_e *= (1/(1+1/math.sqrt(2)))
            #Calculate probability of accepting move
            Pacc = min(1,np.exp(-delta_e/self.KbT))
            if np.random.random() <= Pacc:
                #Accept 
                self.total_energy += delta_e
                #Move nanoparticle
                for i in range(self.nano_size):
                    x_i = (x_nano + offset[0] + i*abs(dis[1]))
                    y_i = (y_nano + offset[1] + i*abs(dis[0]))
                    x_i_wake = (x_nano + wake_offset[0] + i*abs(dis[1]))
                    y_i_wake = (y_nano + wake_offset[1] + i*abs(dis[0]))
                    self.fluid[x_i,y_i] = (1-self.fluid[x_i,y_i])
                    self.fluid[x_i_wake,y_i_wake] = (1-self.fluid[x_i_wake,y_i_wake])
                    self.nano[x_i,y_i] = (1-self.nano[x_i,y_i])
                    self.nano[x_i_wake,y_i_wake] = (1-self.nano[x_i_wake,y_i_wake])

                self.nano_list[i_nano] = ((x_nano+dis[0]),(y_nano+dis[1]))
        return delta_e

    '''
    Function that populates nanoparticle lattice with a number of nanoparticles
    Only attempts 100 random placements - can result in a lower fraction
    '''
    #Randomly populate nanoparticles
    def initialize_nano(self):
        for i in range(self.n_nano):
            tries = 0
            isdone = False
            while not isdone:
                tries += 1
                
                x_i = np.random.randint(1,self.x_dim-(self.nano_size))
                y_i = np.random.randint(1,self.y_dim-(self.nano_size))
                
                nano_sum = 0
                
                for i in range(self.nano_size):
                    for j in range(self.nano_size):
                        nano_sum += self.nano[(x_i+i),(y_i+j)]
                #Only place if no intersection
                if nano_sum == 0:
                    for i in range(self.nano_size):
                        for j in range(self.nano_size):
                            self.nano[(x_i+i),(y_i+j)] = 1
                            self.fluid[(x_i+i),(y_i+j)] = 0
                    self.nano_list.append((x_i,y_i))
                    self.n_nano_placed += 1
                    isdone = True
                elif tries > 100:
                    isdone = True
        self.total_energy = self.calculate_total_energy_neigh()
    
    '''
    Function that performs a single epoch
    '''
    def step_simulation(self):
        for i in range(self.solv_iter):
            self.step_fluid()
        for j in range(self.nano_mob):
            for i in range(self.n_nano_placed):
                self.step_nano()

#######################################################################

#Function to actually run our code:
def growth_sim(num_epochs):

    #Declare our values for simulation,
    x_dim = 1000
    y_dim = 1000
    frac = 0.2
    nano_size = 3
    KbT = 0.2
    mu = -2.5
    e_nn = 2
    e_nl = 1.5
    e_ll = 1
    nano_mob = 30
    n_nano = int(frac*(x_dim*y_dim)/(nano_size*nano_size))
    seed = np.random.randint(1,100)

    #Pass to class (Growth_NonPeriodic) object (growth_run)
    growth_run = Growth_NonPeriodic(x_dim, y_dim, n_nano, KbT, mu, e_nn, e_nl, e_ll, nano_mob, nano_size, seed)

    #Initialize nanoparticles:
    growth_run.initialize_nano()

    #Now actually run the simulation:
    for i in range(num_epochs):
        growth_run.step_simulation()

    #Now grab our arrays and make a picture.
    nano_array = copy.deepcopy(growth_run.nano)
    nano_array[nano_array == 1] = 2

    config = growth_run.fluid + nano_array

    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'Results/')

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    plt.imshow(config)
    plt.xlabel('Lattice Index')
    plt.ylabel('Lattice Index')
    plt.title('Nanoparticle Placements in Liquid \n kbT = ' + str(KbT) + ', Fraction = ' + str(frac) + ', ' + str(num_epochs) + ' Epochs')
    plt.savefig(results_dir + 'kbt_' + str(int(KbT*10)) + '_frac_' + str(int(frac*10)) + '_' + str(num_epochs) + 'epochs_fin.png')
    plt.show()

    return growth_run.fluid, growth_run.nano

#######################################################################

#Function that scores growth based on mean nano cluster size     
def Score_Growth(nano_array):
    
    target_size = 100

    nano_array[nano_array == 0] = 2
    nano_array[nano_array == 1] = 0 
    nano_array[nano_array == 2] = 1
    
    label, n = sp.ndimage.label(nano_array)

    print(n)

    plt.imshow(label)
    plt.xlabel('Lattice Index')
    plt.ylabel('Lattice Index')
    plt.title('Hole Labelling for Nanoparticle Growth Simulation')
    plt.colorbar(label)
    plt.show()

    for xy in range(label.shape[0]):
        if label[xy,0] > 0 and label[xy,-1] > 0:
            label[label == label[xy,-1]] = label[xy,0]
        if label[0,xy] > 0 and label[-1,xy] > 0:
            label[label == label[-1,xy]] = label[0,xy]
    
    summed_labels = sp.ndimage.sum_labels(nano_array,label,range(1,n+1))

    print(summed_labels)
    
    return -abs(target_size-np.mean(summed_labels,where=summed_labels>0))

#######################################################################

#Create class for our NN:
'''class NeuralNetwork(nn.Module):
    
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # Input 16 spins, then increase number of nodes, then decrease to number
        #of classes ie. number of possible energy states.
        self.layer_1 = nn.Linear(16, 64) 
        self.layer_2 = nn.Linear(64, 32)
        self.layer_out = nn.Linear(32, 15) 
        
        self.relu = nn.ReLU()
        #self.soft = nn.Softmax(dim = 0)
        #self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(32)'''

#######################################################################

#Main: Let's run some code:
fluid_arr, nano_arr = growth_sim(1000)
scores = Score_Growth(nano_arr)
