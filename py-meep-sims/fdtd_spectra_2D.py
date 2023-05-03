"""
Created on Fri Mar 31 2023

@author: David Gayowsky

FDTD transmission spectra, tiny3d -> pymeep.
"""

#######################################################################

import meep as mp
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Video 
import os
import scipy as sp
from scipy.constants import c

#######################################################################

#Initial simulations:

resolution = 10 # pixels/um

sx = 32  # size of cell in X direction
sy = 16  # size of cell in Y direction
cell = mp.Vector3(sx,sy,0)

dpml = 1.0
pml_layers = [mp.PML(dpml)]

geometry = [mp.Block(size=mp.Vector3(1.5,mp.inf,mp.inf),
                     center=mp.Vector3(0,0,0),
                     material=mp.Medium(epsilon=12))]

#The source is a GaussianSource instead of a ContinuousSource, parameterized by 
# a center frequency and a frequency width (the width of the Gaussian spectrum).

fcen = 0.09  # pulse center frequency
df = 0.1     # pulse width (in frequency)
sources = [mp.Source(mp.GaussianSource(fcen,fwidth=df),
                     component=mp.Ez,
                     center=mp.Vector3(-9,0,0),
                     size=mp.Vector3(0,9,0))]

sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution)

plt.figure(dpi=100)
sim.plot2D()
plt.show()