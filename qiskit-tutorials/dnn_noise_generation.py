# -*- coding: utf-8 -*-
"""
Created on Wed Jan 2 2023

@author: David

Generating state counts with different types of noise for the purpose of
running on a DNN.
"""

#######################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import itertools

#The thing we use to create an sql database!
from sqlalchemy import create_engine, text
import sqlalchemy

from qiskit import IBMQ, transpile
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.tools.visualization import plot_histogram
from qiskit.quantum_info import Kraus, SuperOp

# Import from Qiskit Aer noise module
from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,
    pauli_error, depolarizing_error, thermal_relaxation_error)

#######################################################################

#Write a function to generate all possible qubit states based on our number of qubits:
def generate_all_poss_states(no_qubits):
    #Initialize array of possible configurations.
    poss_configs = []
    for j in range(no_qubits+1):
        spin_indices = list(itertools.combinations(range(no_qubits), j))
        #Convert to array.
        spin_indices = np.asarray(spin_indices)
        #print(spin_indices)
        for k in range(len(spin_indices)):
            poss_configs.append(spin_indices[k])

    #If our index is stored in our list of indices, set the value to 1 (spin up) in our lattice.
    all_configs_list = []

    for k in range(len(poss_configs)):
        individual_config = poss_configs[k]

        initial_config = np.zeros(no_qubits)
        if len(individual_config) == 0:
            str_config = ''.join(str(int(x)) for x in initial_config)
            all_configs_list += [str_config]
        else:
            for m in range(len(individual_config)):
                index = individual_config[m]
                initial_config[index] = 1
            str_config = ''.join(str(int(x)) for x in initial_config)
            all_configs_list += [str_config]
                    
    #Okay, at this point in the code we have a list of strings which are all our states. Yay! 
    return(all_configs_list)

#Write a function to generate one set of error data, with single probabilities:
def generate_error_data(meas_prob, gate_prob, num_shots, num_tests, no_qubits):
    
    p_meas = meas_prob
    p_gate = gate_prob
    error_meas = pauli_error([('X',p_meas), ('I', 1 - p_meas)])
    error_gate1 = pauli_error([('X',p_gate), ('I', 1 - p_gate)])
    error_gate2 = error_gate1.tensor(error_gate1)

    # Add errors to noise model
    noise_class = NoiseModel()
    noise_class.add_all_qubit_quantum_error(error_meas, "measure")
    noise_class.add_all_qubit_quantum_error(error_gate1, ["ch"])
    noise_class.add_all_qubit_quantum_error(error_gate2, ["cx"])

    # Construct quantum circuit
    circ = QuantumCircuit(no_qubits, 3)
    circ.h(0)
    circ.cx(0, 1)
    circ.cx(1, 2)
    circ.measure([0, 1, 2], [0, 1, 2])

    # Create noisy simulator backend
    #sim_noise = AerSimulator(noise_model=noise_test)
    sim_noise = AerSimulator(noise_model=noise_class)

    # Transpile circuit for noisy basis gates
    circ_tnoise = transpile(circ, sim_noise)

    #Generate set of all possible qubit configurations:
    all_configs_list = generate_all_poss_states(no_qubits)

    #Array of... stuff...
    all_shots_counts = []

    for i in range(num_tests):
        # Run and get counts
        result_test = sim_noise.run(circ_tnoise, shots=num_shots).result()
        counts_test = result_test.get_counts(0)

        counts_test = dict(sorted(counts_test.items()))
  
        #Now need to add zero counts possibilities...
        num_combinations = len(all_configs_list)

        if num_combinations != len(counts_test):
            #Check to see which states are missing from our counts_test...
            for j in range(len(all_configs_list)):
                if all_configs_list[j] in counts_test:
                    pass
                else:
                    counts_test.update({all_configs_list[j]: 0})
        else:
            pass
        
        counts_test = dict(sorted(counts_test.items()))
        #Dictionary, key:data

        counts_values = list(counts_test.values())
        counts_values = np.asarray(counts_values)

        #Here are our state labels for each column:
        counts_labels = list(counts_test.keys())

        if i==0:
            all_shots_counts.append(counts_values)
        else:
            all_shots_counts = np.vstack([all_shots_counts, counts_values])
        
    #Now we want to take the average and standard deviation of each column.
    #Gives 1D array of mean values by column:
    state_mean_counts = np.mean(all_shots_counts, axis=0)
    state_std_devs = np.std(all_shots_counts, axis=0)

    return state_mean_counts, state_std_devs, counts_labels

#Write a function to generate all of our error data across multiple probabilities:   
def genetate_all_error_data(min_meas_prob, max_meas_prob, min_gate_prob, max_gate_prob, num_vals, num_shots, num_tests, no_qubits):

    #Generate error probability values:
    meas_prob_vals = np.linspace(min_meas_prob, max_meas_prob, num_vals)
    gate_prob_vals = np.linspace(min_gate_prob, max_gate_prob, num_vals)

    tot_means = []
    tot_stdevs = []
    tot_labels = []
    tot_meas_prob = []
    tot_gate_prob = []

    #Generate data for each combination of measurement and gate errors:
    for i in range(num_vals):
        for j in range(num_vals):
            
            #Generate this data set:
            state_mean_counts, state_std_devs, counts_labels = generate_error_data(meas_prob_vals[i], gate_prob_vals[j], num_shots, num_tests, no_qubits)
            all_meas_prob_vals = [meas_prob_vals[i]] * len(counts_labels)
            all_gate_prob_vals = [gate_prob_vals[j]] * len(counts_labels)
            #print(state_mean_counts, state_std_devs, counts_labels, all_meas_prob_vals, all_gate_prob_vals)

            tot_means.extend(state_mean_counts)
            tot_stdevs.extend(state_std_devs)
            tot_labels.extend(counts_labels)
            tot_meas_prob.extend(all_meas_prob_vals)
            tot_gate_prob.extend(all_gate_prob_vals)

    #Now we want to pop all this stuff into a data frame...
    noise_data = pd.DataFrame()
    noise_data['State']=pd.Series(tot_labels)
    noise_data['Meas. Err. Prob']=pd.Series(tot_meas_prob)
    noise_data['Gate Err. Prob']=pd.Series(tot_gate_prob)
    noise_data['Count Mean']=pd.Series(tot_means)
    noise_data['Count Std. Dev.']=pd.Series(tot_stdevs)

    #print(noise_data)

    return noise_data

#Write a function to create an sql database and send data to database:
def noise_data_to_database(min_meas_prob, max_meas_prob, min_gate_prob, max_gate_prob, num_vals, num_shots, num_tests, no_qubits):
    #Use our previous function to create our noise data:
    noise_data = genetate_all_error_data(min_meas_prob, max_meas_prob, min_gate_prob, max_gate_prob, num_vals, num_shots, num_tests, no_qubits)

    #Create in-memory sql database:
    sql_engine = create_engine('sqlite://', echo=False)
    #Make sure we're connected to the database...
    connection = sql_engine.raw_connection()
    #Send dataframe to sql:
    noise_data.to_sql('quantum_noise_data', connection, if_exists='replace')

    #Grab our raw results...
    results = connection.execute("SELECT * FROM quantum_noise_data").fetchall()
    #print(results)

    #Or we can goback into pandas from our database...
    results_df = pd.read_sql_query("SELECT * FROM quantum_noise_data", connection)
    print(results_df)

    return results, results_df

#def plot_noise_data(min_meas_prob, max_meas_prob, min_gate_prob, max_gate_prob, num_vals, num_shots, num_tests, no_qubits):

    #Create state frequency plot as p_meas and p_gate increase. Want to use t-SNE reduction to plot.
#######################################################################

#Main: Let's run some functions!

#genetate_all_error_data(0, 0.2, 0, 0.2, 3, 100, 10, 3)

noise_data_to_database(0, 0.2, 0, 0.2, 3, 100, 10, 3)
