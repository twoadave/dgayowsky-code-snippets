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

def generate_measurement_error_data(meas_prob, num_shots, num_tests):
    
    p_meas = meas_prob
    error_meas = pauli_error([('X',p_meas), ('I', 1 - p_meas)])

    # Add errors to noise model
    noise_class = NoiseModel()
    noise_class.add_all_qubit_quantum_error(error_meas, "measure")

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

    return state_mean_counts, state_std_devs
    

#######################################################################

#Main: Let's run some functions!
generate_measurement_error_data(0.2, 1000, 2)

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

