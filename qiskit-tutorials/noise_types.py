# -*- coding: utf-8 -*-
"""
Created on Wed Jan 2 2023

@author: David

Qiskit tutorial: types of noise and custom noise.
"""

#######################################################################

import numpy as np
import matplotlib.pyplot as plt
from qiskit import IBMQ, transpile
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.tools.visualization import plot_histogram
from qiskit.quantum_info import Kraus, SuperOp

# Import from Qiskit Aer noise module
from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,
    pauli_error, depolarizing_error, thermal_relaxation_error)

#######################################################################

'''When adding a quantum error to a noise model we must specify the type of 
instruction that it acts on, and what qubits to apply it to. There are two 
cases for Quantum Errors:
1. All-qubit quantum error: This applies the same error to any occurrence of 
    an instruction, regardless of which qubits it acts on.
2. Specific qubit quantum error: This applies the error to any occurrence of 
    an instruction acting on a specified list of qubits. Note that the order 
    of the qubit matters: For a 2-qubit gate an error applied to qubits [0, 1] 
    is different to one applied to qubits [1, 0] for example.'''

# Create an empty noise model
noise_test = NoiseModel()

# Add depolarizing error to all single qubit u1, u2, u3 gates
error = depolarizing_error(0.05, 1)
noise_test.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3'])

# Example error probabilities
p_reset = 0
p_meas = 0
p_gate1 = 0.2

# QuantumError objects
error_reset = pauli_error([('X', p_reset), ('I', 1 - p_reset)])
error_meas = pauli_error([('X',p_meas), ('I', 1 - p_meas)])
error_gate1 = pauli_error([('X',p_gate1), ('I', 1 - p_gate1)])
error_gate2 = error_gate1.tensor(error_gate1)

# Add errors to noise model
noise_bit_flip = NoiseModel()
noise_bit_flip.add_all_qubit_quantum_error(error_reset, "reset")
noise_bit_flip.add_all_qubit_quantum_error(error_meas, "measure")
noise_bit_flip.add_all_qubit_quantum_error(error_gate1, ["ch"])
noise_bit_flip.add_all_qubit_quantum_error(error_gate2, ["cx"])
#Add our other previous error:
#noise_bit_flip.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3'])


# Print noise model info
#print(noise_test)

# Construct quantum circuit
circ = QuantumCircuit(3, 3)
circ.h(0)
circ.cx(0, 1)
circ.cx(1, 2)
circ.measure([0, 1, 2], [0, 1, 2])

# Create noisy simulator backend
#sim_noise = AerSimulator(noise_model=noise_test)
sim_noise = AerSimulator(noise_model=noise_bit_flip)

# Transpile circuit for noisy basis gates
circ_tnoise = transpile(circ, sim_noise)

# Run and get counts
result_test = sim_noise.run(circ_tnoise).result()
counts_test = result_test.get_counts(0)

# Plot noisy output
plot_histogram(counts_test)
plt.show()