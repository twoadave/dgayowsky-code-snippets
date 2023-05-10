"""
Created on Mon May 8 2023

@author: David Gayowsky

Simple PyMeep Monte-Carlo "learning" simulation. Replicated from Viktor's work.
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

'''1. Create simulation outline, i.e. cell, PMLs, wave source, and determine time we will run for.
2. Using the target shaped waveguide, take data of the electric field at all possible times (i.e. each 1s). This is our target result.
3. Remove the target shaped waveguide. 
4. Re-run simulation, record data, and compute error. (Error = difference between target.)
5. Add a small shape, i.e. a small sphere.
6. Re-run simulation, record data, and compute error. 
7. Use Monte-Carlo method to accept or deny change, i.e. compute probability using change in error from previous iteration. If error is less, keep. If error is more, and P > u in U(0,1), keep, if P < u in U(0,1), deny.
8. Repeat steps 5-7 until target error is reached.
'''

#######################################################################

#1. Create simulation outline.
def init_sim():

    resolution = 10 # pixels/um

    sx = 32  # size of cell in X direction
    sy = 32  # size of cell in Y direction
    cell = mp.Vector3(sx,sy,0)

    dpml = 1.0
    pml_layers = [mp.PML(dpml)]

    fcen = 0.15  # pulse center frequency
    df = 0.1     # pulse width (in frequency)

    #Create a simple point source
    sources = [mp.Source(mp.GaussianSource(fcen,fwidth=df),
                        component=mp.Ez,
                        center=mp.Vector3(-1*(0.5*sx)+dpml,0,0))]

    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        #geometry=geometry,
                        sources=sources,
                        resolution=resolution)

    #plt.figure(dpi=100)
    #sim.plot2D()
    #plt.show()

    sim.run(until=100)

    #plt.figure(dpi=100)
    #sim.plot2D(fields=mp.Ez)
    #plt.show()

    return sim, cell, pml_layers, sources, resolution

#######################################################################

#2. Using the target shaped waveguide, take data of the electric field 
# at all possible times (i.e. each 1s). This is our target result.

def with_target(run_time):

    sim, cell, pml_layers, sources, resolution = init_sim()

    sim.reset_meep()

    geometry = [mp.Block(size=mp.Vector3(1.5,mp.inf,mp.inf),
                     center=mp.Vector3(0,0,0),
                     material=mp.Medium(epsilon=12))]


    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=sources,
                        resolution=resolution)

    #plt.figure(dpi=100)
    #sim.plot2D()
    #plt.show()

    sim.run(until=run_time)

    #plt.figure(dpi=100)
    #sim.plot2D(fields=mp.Ez)
    #plt.show()

    #Now grab our data at time = 200...
    #We can add in other data points later, but let's try and make a waveguide which echoes this.
    ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)

    return ez_data

#######################################################################

#Main: Let's run some functions!
target_data = with_target(100)
