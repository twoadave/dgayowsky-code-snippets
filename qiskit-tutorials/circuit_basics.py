# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 11:32:26 2022

@author: David

Attempt #2 at creating qiskit tutorial...
"""

#######################################################################

import numpy as np
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector

# Create a Quantum Circuit acting on a quantum register (circuit) of three qubits.
circ = QuantumCircuit(3)

# Set the intial state of the simulator to the ground state using from_int
state = Statevector.from_int(0, 2**3)
print(state)

# Add a H gate on qubit 0, putting this qubit in superposition.
circ.h(0)
#circ.draw(output='mpl')
#plt.show()
state = state.evolve(circ)
state.draw('qsphere')
plt.show()

# Add a CX (CNOT) gate on control qubit 0 and target qubit 1, putting
# the qubits in a Bell state.
circ.cx(0, 1)
#circ.draw(output='mpl')
#plt.show()
#state = state.evolve(circ)
#state.draw('qsphere')
#plt.show()

# Add a CX (CNOT) gate on control qubit 0 and target qubit 2, putting
# the qubits in a GHZ state.
circ.cx(0, 2)
#circ.draw(output='mpl')
#plt.show()
#state = state.evolve(circ)
#state.draw('qsphere')
#plt.show()