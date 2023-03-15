"""
Created on Mon Mar 6 2023

@author: David Gayowsky

Create a little artificial neural network from scratch.
"""

#######################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import itertools
import math

#######################################################################

#1.1 Make ReLu Function: ReLu essentially returns only the positive component
#of a given function. If the function is <0, returns negative for those values.

#Define our ReLu function:
def relu(x):
    if x > 0:
        return x
    else:
        return 0

#Define a function to plot the response of ReLu(x):
def plot_relu_response(min_x, max_x, numsteps):
    #Create x values:
    x_vals = np.linspace(min_x, max_x, numsteps)
    #Create array to store response:
    response_vals = []
    #Generate response:
    for i in range(len(x_vals)):
        response_val = relu(x_vals[i])
        response_vals.append(response_val)
    #Plot response over domain:
    plt.plot(x_vals, response_vals, linewidth = 1)
    plt.xlabel('Input Value')
    plt.ylabel('Response Value')
    plt.title('ReLu Response Function Over Domain ['+ str(min_x) + ', ' + str(max_x) + ']\n')
    plt.show()  

#######################################################################

#1.2 Fully Connected Layer
#Write a neural network with:
#A single input x,
#N hidden nodes (neurons) on a single hidden layer,
#A single output y.
#Use ReLu as the activation function for all hidden layer neurons.
#Each neuron connected to output of all other neurons in *preceding* layer.

#Define a function to act as a single layer DNN:
def simple_ann(input_x, N_hidden_neurons, weights, bias):

    #Neuron activation:
    #a = sum(w_i * x_i) where x_i is neuron input, and w_i is bias of input x_i
    #Activation function:
    #y(a) = ReLu(a)
    #So essentially we take the input x_i from each previous layer neuron --> get a --> feed to y 

    #Okay we're just gonna explicitly build this for one hidden layer because I'm lazy lol

    #Single neuron input:
    a_init = weights[0]*input_x + bias
    output_y = relu(a_init)

    #Now to the hidden layer:
    hidden_outputs = []
    #Calculate the output of each hidden neuron:
    for i in range(N_hidden_neurons):
        a_val = weights[i+1]*output_y + bias
        hidden_output = relu(a_val)
        hidden_outputs.append(hidden_output)

    #Now to the final layer:
    a_val = 0
    #Calculate a by summing output of each neuron in previous layer:
    for i in range(N_hidden_neurons):
        a_val = a_val + weights[i+1]*hidden_outputs[i]
    #Add the bias...
    a_val = a_val + bias
    #Pop into relu activation function:
    total_output = relu(a_val)
    #print(total_output)

    return total_output

#Define a function to plot the response of our neural network:
def plot_neural_net_response(min_x, max_x, numsteps, N_hidden_neurons, weights, bias):
    #Create x values:
    x_vals = np.linspace(min_x, max_x, numsteps)
    #Create array to store response:
    response_vals = []
    #Generate response:
    for i in range(len(x_vals)):
        response_val = simple_ann(x_vals[i], N_hidden_neurons, weights, bias)
        response_vals.append(response_val)
    #Plot response over domain:
    plt.plot(x_vals, response_vals, linewidth = 1)
    plt.xlabel('Input Value')
    plt.ylabel('Response Value')
    plt.title('Simple ANN Response Over Domain ['+ str(min_x) + ', ' + str(max_x) + ']\n')
    plt.show() 

#######################################################################

#1.3 Random Initialization

#Define a function to plot the response of our neural network with random initialization:
def plot_nn_response_rand(min_x, max_x, numsteps, N_hidden_neurons):
    #Create x values:
    x_vals = np.linspace(min_x, max_x, numsteps)
    #Create array to store response:
    response_vals = []
    #Randomly initialize parameters: 
    weights = np.random.normal(0,np.sqrt(0.1), N_hidden_neurons+1)
    bias = np.random.normal(0,np.sqrt(0.1))
    #Generate response:
    for i in range(len(x_vals)):
        response_val = simple_ann(x_vals[i], N_hidden_neurons, weights, bias)
        response_vals.append(response_val)
    #Plot response over domain:
    plt.plot(x_vals, response_vals, linewidth = 1)
    plt.xlabel('Input Value')
    plt.ylabel('Response Value')
    plt.title('Simple ANN Response Over Domain ['+ str(min_x) + ', ' + str(max_x) + ']\n Randomly Initialized Parameters')
    plt.show()

#######################################################################

#1.4 Visualize

#Define a function to repeat the random initialization many times, for a single input:
def visualize_ann(x, N_hidden_neurons, num_trials, num_bins):
    #Create array to store response:
    response_vals = []
    #Generate response:
    for i in range(num_trials):
        #Randomly initialize parameters, assuming we generate new parameters each time:
        weights = np.random.normal(0,np.sqrt(0.5), N_hidden_neurons+1)
        bias = np.random.normal(0,np.sqrt(0.5))
        #Pass to neural net:
        response_val = simple_ann(x, N_hidden_neurons, weights, bias)
        response_vals.append(response_val)
    #Plot the histogram:
    plt.hist(response_vals, num_bins, density=True)
    plt.xlabel('Response Value')
    plt.ylabel('Frequency')
    plt.title('Simple ANN Response for Input x = ' + str(x) + '\n Randomly Initialized Parameters')
    plt.show() 

#######################################################################

#1.5 Change Initialization
#See above code...

#######################################################################

#Main: Let's run some functions!
#plot_relu_response(-1, 1, 100)

#weights_vec = np.full(17, 0.1)
#plot_neural_net_response(-1, 1, 100, 16, weights_vec, 0.1)

#plot_nn_response_rand(-2*math.pi, 2*math.pi, 100, 16)

#visualize_ann(1, 16, 1000, 20)
