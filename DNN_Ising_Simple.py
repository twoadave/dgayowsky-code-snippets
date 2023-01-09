# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 11:32:26 2022

@author: David
"""

#######################################################################

import numpy as np
import math 
import torch
import matplotlib.pyplot as plt
import os
import torch.optim as optim
import torch.nn as nn
from torch.nn import Linear, ReLU, Sigmoid, Module, BCELoss, Softmax
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import itertools
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

#######################################################################

#Here, we want to:
#Sample half the configuration space.
#Test on the other half of the space.

#We do this by:
#1. Generating the full set of possible 4x4 Ising configurations.
#2. Sample from this set by taking samples across full range of energy spectra.
#3. Set aside the other half to use as testing data.

#######################################################################

#Firstly, write functions to generate configurations and calculate energies from the Ising model.

#Define function to add elements on to the end of a list.
def merger(seglist):
  total_list = []
  for seg in seglist:
    #Add given elements of input list to the end.
    total_list.extend(seg)
  return total_list
    
#Define a function to create our sample Ising states.
def create_sample_states(N):
    #Here we consider each site in the grid to have an index in {0, 15}.
    #So we take combinations without repetition to declare which sites are spin up (or down, however
    #we choose to initialize).
    
    #Create range of indices from 1 to N.
    indices = range(N*N)
    
    #Initialize array of possible configurations.
    poss_configs = []
    
    #Create full list of combinations without repetition to decide which sites will be spin up or spin down.
    for i in range((N*N)+1):
        spin_indices = list(itertools.combinations(indices, i))
        #Convert to array.
        spin_indices = np.asarray(spin_indices)
        #Append array to full array of all possible configurations.
        poss_configs.append(spin_indices)
    
    #Note that len(poss_configs) = N*N, so we can access each set of indices.
    poss_configs = np.asarray(poss_configs)
    
    return poss_configs

#Define a function to convert site index to array index.
def convert_sites(N, config):
    #Site index 0->N*N modulo N gives the column.
    #Floor of site index divided by N gives the row.
    
    #Initialize array to store values.
    config_converted = []
    
    for i in range(len(config)):
        #Convert index using arithmetic and append to new array of converted indices.
        converted_index = [math.floor(config[i]/N), config[i]%N]
        config_converted.append(converted_index)
        
    #print(config_converted)
    return config_converted

#Define a function to create our explicit ising lattice of 1s and -1s.
def create_explicit_config(N, config):
    
    #Create a lattice and fill it with zeros.
    config_explicit = np.zeros((N,N))
    
    #If our index is stored in our list of indices, set the value to 1 (spin up) in our lattice.
    for i in range(len(config)):
        index = config[i]
        config_explicit[index[0], index[1]] = 1
        
    #Convert our zeros to -1s...
    config_explicit[config_explicit == 0] = -1

    return config_explicit

#Define a function to explicitly calculate the energy of our configuration.
def calc_energy_config(N, config, J):
    
    #Given some configuration, convert it and get our explicit configuration in terms of -1s and 1s.
    config_conv = convert_sites(N, config)
    config_conv = create_explicit_config(N, config_conv)
    
    #Initialize total energy value.
    E_tot = 0
    
    #For all sites in the lattice:
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
            
            #Explicitly calculate energy at a given site.
            E_site = -J*(config_conv[i,j]*config_conv[s_i_max,j] + config_conv[i,j]*config_conv[i,s_j_max] + config_conv[i,j]*config_conv[s_i_min,j] + config_conv[i,j]*config_conv[i,s_j_min])
            
            #Add site value to total energy value.
            E_tot = E_tot + E_site
    
    return E_tot, config_conv

#Define a function to explicitly calculate all energies of set of possible configurations.
def calculate_all_energies(N, J):
    
    #Generate set of possible configurations:
    poss_configs = create_sample_states(N)
    #print(poss_configs)
    
    #Initialize arrays to store all energies and configurations.
    all_energies = []
    all_configs = []
    
    #For all possible ising configurations:
    for i in range(len(poss_configs)):
        
        #Grab the subsets we want to find the energy of (where these subsets are for 0 spins up, 1 spin up, etc..)
        calc_E_configs = poss_configs[i]
        
        #Now for each configuration with a given number of spins up:
        for j in range(len(calc_E_configs)):
            #Calculate the explicit energy and lattice configuration of a given ising state.
            E_config, explicit_config = calc_energy_config(N, calc_E_configs[j], J)
            #Append to arrays.
            all_energies.append(E_config)
            all_configs.append(explicit_config.flatten('C'))
    
    #Now create a second list of all possible unique energies:
    poss_energies = np.unique(all_energies)
    
    print(poss_energies)
    return all_energies, all_configs, poss_energies
        
#######################################################################
    
#Now we create our training and testing data sets:
    
#Make our tensor out of our ising configurations:
def create_tensor(N, J):
    #Find all energies and corresponding lattice configurations, and list of all possible energies:
    all_energies, all_ising_configs, poss_energies = calculate_all_energies(N,J)
    #Pop these into a torch tensor:
    all_ising_tensor = torch.tensor(all_ising_configs)
    all_energies_tensor = torch.tensor(all_energies)
    
    return all_ising_tensor, all_energies_tensor, poss_energies

#Split into training and testing data sets:
#Note we just do this randomly and not probabilistically for the moment...
#We can try and fix this later.
def create_training_testing_data(N, J):
    
    #Create tensors from our given data:
    all_ising_tensor, all_energies_tensor, poss_energies = create_tensor(N,J)
    
    #print(poss_energies)
    
    #print(all_ising_tensor, all_energies_tensor)
    
    #Shuffle indices, take half our data to be training, and half our data to be testing.
    index_ids = np.arange(len(all_ising_tensor))
    #Shuffle all the IDs so we get a random set of training and testing data.
    np.random.shuffle(index_ids)
    
    #Generate training data tensor, configurations and corresponding energies:
    #Grab half of these indices to be our training data.
    train_indices = index_ids[:math.floor(0.5*len(all_ising_tensor))]
    #print(train_indices)
    ising_training_tensor = all_ising_tensor[train_indices]
    #print(ising_training_tensor)
    
    #Grab our training energies:
    energies_training_tensor_i = all_energies_tensor[train_indices]
    #Now convert these to a list, temporarily:
    energies_training_tensor_i = energies_training_tensor_i.tolist()
    #print(energies_training_tensor)
    
    #Initialize our training tensor:
    energies_training_tensor = []
    
    #Create a list for our possible energies so we can search for indices:
    poss_energies_list = poss_energies.tolist()
    
    #Now create our training data list as a list of the indices of our possible energies, rather than 
    #the actual values.
    for i in range(len(energies_training_tensor_i)):
        index_val = poss_energies_list.index(energies_training_tensor_i[i])
        energies_training_tensor.append(index_val)
    
    #Now pop these back into tensor form.
    energies_training_tensor = torch.tensor(energies_training_tensor)
    
    #Make our training dataset into a tensor:
    training_dataset = TensorDataset(ising_training_tensor, energies_training_tensor)
    #print(training_dataset)
    
    #Generate testing data tensor, configurations and corresponding energies:
    test_indices = index_ids[math.floor(0.5*len(all_ising_tensor)):]
    ising_testing_tensor = all_ising_tensor[test_indices]
    
    energies_testing_tensor_i = all_energies_tensor[test_indices]
    #Now convert these to a list, temporarily:
    energies_testing_tensor_i = energies_testing_tensor_i.tolist()
    
    #Initialize our testing tensor:
    energies_testing_tensor = []
    #Now create our testing data list as a list of the indices of our possible energies, rather than 
    #the actual values.
    for i in range(len(energies_testing_tensor_i)):
        index_val = poss_energies_list.index(energies_testing_tensor_i[i])
        energies_testing_tensor.append(index_val)
        
    #Now pop these back into tensor form.
    energies_testing_tensor = torch.tensor(energies_testing_tensor)
    
    #Make our testing dataset into a tensor:
    testing_dataset = TensorDataset(ising_testing_tensor, energies_testing_tensor)
    
    return training_dataset, testing_dataset, poss_energies

#######################################################################
    
#Use torch.nn.CrossEntropyLoss(...)
#Need to count how many unique energy elements (classes) we have

#Create class for our NN:
class NeuralNetwork(nn.Module):
    
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
        self.batchnorm2 = nn.BatchNorm1d(32)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        #x = self.dropout(x)
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        #x = self.dropout(x)
        x = self.batchnorm2(x)
        #print("Softmax in", x)
        #x = self.soft(self.layer_out(x))
        #print("Softmax out", x)
        x = self.layer_out(x)
        return x

#Define a function which we can call to train our neural network:
def train_NN(model, loss_fn, optimizer, train_dataloader, training_pass_loss):
    
    #Put it into training mode:
    model.train()
    
    #For each batch in our training data:
    for configs_batch, energies_batch in train_dataloader:
        #Send a batch of configurations to device.
        configs_batch = configs_batch.to(device)
        energies_batch = energies_batch.type(torch.LongTensor)
        #Send a batch of energies to device.
        energies_batch = energies_batch.to(device)
        optimizer.zero_grad()
        #Compute our predicted energies given our configurations:
        energy_pred = model(configs_batch.float())
        #Energies_batch needs to be list of indices, not values!
        #Compute loss between our energy values and our predicted energies:
        train_loss = loss_fn(energy_pred, energies_batch)
        #Make our backwards pass:
        train_loss.backward()
        optimizer.step()
        training_pass_loss += train_loss.item()
        
        #Record the loss per epoch in training loop and plot it, we want to see this 
        #number going down.
        #Also plot validation data (?) so we can see loss going down and show accuracy increasing.
        #Want to evaluate our neural net as we go.
        #Write log files into weights and biases so we can show our losses etc.
        #Make plots of validation curves etc.
        
        #Follow wandb tutorial/coding styles.
        
        #Log loss on training set
        #Log loss on validation set
        #Compute metrics
        
        #Ask Gavin and Chris!
        
    return training_pass_loss

#Define a function which we can call to test our neural network:
def test_NN(model, test_dataloader):
    
    #Initialize list of testing and predicted energies:
    energies_test_list = []
    energies_pred_list = []
    
    with torch.no_grad():
      #Put our model into testing mode:
      model.eval()
      
      #For all of our configurations and energies:
      for configs_batch, energies_batch in test_dataloader:
        #Send our configurations to device:
        configs_batch = configs_batch.to(device)
        #Generate our predicted energies:
        energies_test_pred = model(configs_batch.float())
        #??? (x-files theme music plays)
        _, energies_pred_tags = torch.max(energies_test_pred, dim = 1)
        energies_pred_list.append(energies_pred_tags.cpu().numpy())
        energies_test_list.append(energies_batch)
    
    #More x-files theme music???
    energies_pred_list = [a.squeeze().tolist() for a in energies_pred_list]
    energies_test_list = [a.squeeze().tolist() for a in energies_test_list]
    
    #Well we throw all these lists together here.
    energies_test_list = merger(energies_test_list)
    energies_pred_list = merger(energies_pred_list)
    
    return energies_test_list, energies_pred_list

#Define a function that we use to call our neural network et al.
def pass_to_NN(network_passes, batch_sze, learning_rate, N, J):
    
    model = NeuralNetwork().to(device)
    
    #Epochs = number of passes over network 
    
    #Declare our loss function and optimizer.
    loss_fn = nn.CrossEntropyLoss()
    #loss_fn = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), learning_rate)
        
    #Create data loaders
    #Generally we want to split our training and testing data sets into 
    #batches and normalize across them.
    training_dataset, testing_dataset, poss_energies = create_training_testing_data(N, J)
    
    #print(type(poss_energies))
    
    #Here each batch is some set of configurations and energies.
    train_dataloader = DataLoader(training_dataset, batch_sze, drop_last = True, shuffle = True)
    test_dataloader = DataLoader(testing_dataset, batch_sze, drop_last = True, shuffle = True)
    
    #Initialize loss value.
    training_pass_loss = 0
    #Initialize array to store our loss values.
    training_losses = []
    
    #Make our training passes over the network.
    for i in range(1, network_passes+1):
        #Call our training NN.
        training_step_loss = train_NN(model, loss_fn, optimizer, train_dataloader, training_pass_loss)
        #Print out our training step loss.
        print(training_step_loss)
        #And append it to an array to store it.
        training_losses.append(training_step_loss)
    
    #Test our neural network.
    energies_test_list, energies_pred_list = test_NN(model, test_dataloader)
    #print(energies_test_list, energies_pred_list)
    
    #Generate a confusion matrix (??) of our predicted and resulting energies.
    results = confusion_matrix(energies_test_list, energies_pred_list)
    
    #Read the number of correct values off of this matrix.
    diagonal_sum = 0
    for i in range(results.shape[0]):
      diagonal_sum += results[i][i]
      
    #And then calculate our correct percentage of values.
    percentage_correct = diagonal_sum/len(energies_test_list) * 100
    
    return percentage_correct, training_losses
    
#Define a functon to plot our training losses across each step.
def plot_training_losses(training_losses, network_passes):
    
    step_vals = np.arange(1, network_passes+1, 1)

    plt.plot(step_vals, training_losses, marker ='o', markersize = 2)
    plt.xlabel('Epoch (Network Pass) Number')
    plt.ylabel('Log Loss (Training Loss)')
    plt.title('Log Loss (Training Loss) of Simple Ising DNN per Epoch \n')
    plt.show()


#16 input nodes ie -1, 1s, and increase nodes as we go through layers, then decrease
#And then number of classes is the number of possible energies
'''self.layer_1 = nn.Linear(16, 64) 
        self.layer_2 = nn.Linear(64, 32)
        self.layer_out = nn.Linear(32, len(Classes))

Normalizing data within each batch/layer rather than across whole NN
self.batchnorm1 = nn.BatchNorm1d(64)

Hey computer, hold on to these weights for me while I work with them:
model = NeuralNetwork().to(device)'''

#Recreate some of the plots in Kyle's paper, see if we can remake those.
#Move over to vscode and run on wsl.
#Print out, log loss function etc. so we can keep track of it.
#Tensorboard or weights and biases.

#######################################################################

#create_training_testing_data(4, 1)

network_passes = 10
percentage_correct, training_losses = pass_to_NN(network_passes, 25, 0.0003, 4, 1)
print('Percent of correctly predicted energies: ' + str(percentage_correct))
#plot_training_losses(training_losses, network_passes)