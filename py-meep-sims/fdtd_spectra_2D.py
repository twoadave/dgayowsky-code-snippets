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

fcen = 0.15  # pulse center frequency
df = 0.1     # pulse width (in frequency)
sources = [mp.Source(mp.GaussianSource(fcen,fwidth=df),
                     component=mp.Ez,
                     center=mp.Vector3(-9,0,0),
                     size=mp.Vector3(0,9,0))]

sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    #geometry=geometry,
                    sources=sources,
                    resolution=resolution)

plt.figure(dpi=100)
sim.plot2D()
plt.show()

sim.run(until=200)

plt.figure(dpi=100)
sim.plot2D(fields=mp.Ez)
plt.show()

sim.reset_meep()
f = plt.figure(dpi=100)
Animate = mp.Animate2D(fields=mp.Ez, f=f, realtime=False, normalize=True)
plt.close()

sim.run(mp.at_every(1, Animate), until=120)
plt.close()

file_path = os.path.realpath(__file__)
filename = file_path + "fdtd_2d_init.mp4"
Animate.to_mp4(10, filename)

Video(filename)

#######################################################################

#Now we go to our actual simulation...
sim.reset_meep()


geometry = [mp.Block(size=mp.Vector3(1.5,mp.inf,mp.inf),
                     center=mp.Vector3(0,0,0),
                     material=mp.Medium(epsilon=12))]


sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution)

plt.figure(dpi=100)
sim.plot2D()
plt.show()

sim.run(until=250)

plt.figure(dpi=100)
sim.plot2D(fields=mp.Ez)
plt.show()

sim.reset_meep()
f = plt.figure(dpi=100)
Animate = mp.Animate2D(fields=mp.Ez, f=f, realtime=False, normalize=True)
plt.close()

sim.run(mp.at_every(1, Animate), until=120)
plt.close()

file_path = os.path.realpath(__file__)
filename = file_path + "fdtd_2d.mp4"
Animate.to_mp4(10, filename)

Video(filename)