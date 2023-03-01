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

from qiskit import IBMQ, transpile
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.tools.visualization import plot_histogram
from qiskit.quantum_info import Kraus, SuperOp

# Import from Qiskit Aer noise module
from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,
    pauli_error, depolarizing_error, thermal_relaxation_error)

#######################################################################

#Write a function to generate one set of error data, with single probabilities:
def generate_error_data(meas_prob, gate_prob, num_shots, num_tests):
    
    p_meas = meas_prob
    p_gate = gate_prob
    error_meas = pauli_error([('X',p_meas), ('I', 1 - p_meas)])
    error_gate1 = pauli_error([('X',p_gate), ('I', 1 - p_gate)])
    error_gate2 = error_gate1.tensor(error_gate1)

    # Add errors to noise model
    noise_class = NoiseModel()
    noise_class.add_all_qubit_quantum_error(error_meas, "measure")
    noise_bit_flip.add_all_qubit_quantum_error(error_gate1, ["ch"])
    noise_bit_flip.add_all_qubit_quantum_error(error_gate2, ["cx"])

    # Construct quantum circuit
    circ = QuantumCircuit(3, 3)
    circ.h(0)
    circ.cx(0, 1)
    circ.cx(1, 2)
    circ.measure([0, 1, 2], [0, 1, 2])

    # Create noisy simulator backend
    #sim_noise = AerSimulator(noise_model=noise_test)
    sim_noise = AerSimulator(noise_model=noise_class)

    # Transpile circuit for noisy basis gates
    circ_tnoise = transpile(circ, sim_noise)

    all_shots_counts = []

    for i in range(num_tests):
        # Run and get counts
        result_test = sim_noise.run(circ_tnoise, shots=num_shots).result()
        counts_test = result_test.get_counts(0)
        counts_test = dict(sorted(counts_test.items()))
        #print(counts_test)
        #Outputs something like: {'001': 63, '110': 79, '101': 85, '000': 254, '010': 91, '111': 271, '011': 80, '100': 77}
        #So how do we pull out this data?
        #Dictionary, key:data
        #Can we just pass a dictionary to our DNN? Or do we need to pull out counts?
        #We'll have Noise Type which corresponds to a distribution... so think we need to just pull out counts.
        counts_values = list(counts_test.values())
        counts_values = np.asarray(counts_values)

        #Here are our state labels for each column:
        counts_labels = list(counts_test.keys())

        if i==0:
            all_shots_counts.append(counts_values)
        else:
            all_shots_counts = np.vstack([all_shots_counts, counts_values])
    
    #print(all_shots_counts)
        
    #Now we want to take the average and standard deviation of each column.
    #Gives 1D array of mean values by column:
    state_mean_counts = np.mean(all_shots_counts, axis=0)
    state_std_devs = np.std(all_shots_counts, axis=0)

    return state_mean_counts, state_std_devs, counts_labels

#Write a function to generate all of our error data across multiple probabilities:   
def genetate_all_error_data(min_meas_prob, max_meas_prob, min_gate_prob, max_gate_prob, num_vals, num_shots, num_tests):

    #Generate error probability values:
    meas_prob_vals = np.linspace(min_meas_prob, max_meas_prob, num_vals)
    gate_prob_vals = np.linspace(min_gate_prob, max_gate_prob, num_vals)

    #Generate data for each combination of measurement and gate errors:
    for i in range(num_vals):
        for j in range(num_vals):
            
            #Generate this data set:
            state_mean_counts, state_std_devs, counts_labels = generate_error_data(meas_prob_vals[i], gate_prob_vals[j], num_shots, num_tests)
            


#######################################################################

#Main: Let's run some functions!


'''p_gate1 = 0

# QuantumError objects
error_gate1 = pauli_error([('X',p_gate1), ('I', 1 - p_gate1)])
error_gate2 = error_gate1.tensor(error_gate1)

# Add errors to noise model
noise_class = NoiseModel()
noise_class.add_all_qubit_quantum_error(error_gate1, ["ch"])
noise_class.add_all_qubit_quantum_error(error_gate2, ["cx"])

# Construct quantum circuit
circ = QuantumCircuit(3, 3)
circ.h(0)
circ.cx(0, 1)
circ.cx(1, 2)
circ.measure([0, 1, 2], [0, 1, 2])

# Create noisy simulator backend
#sim_noise = AerSimulator(noise_model=noise_test)
sim_noise = AerSimulator(noise_model=noise_class)

# Transpile circuit for noisy basis gates
circ_tnoise = transpile(circ, sim_noise)

# Run and get counts
result_test = sim_noise.run(circ_tnoise).result()
counts_test = result_test.get_counts(0)'''

