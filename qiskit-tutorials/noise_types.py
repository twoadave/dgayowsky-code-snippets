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

#######################################################################