"""
Created on Mon May 15 2023

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
import copy

#######################################################################

#Create function to calculate intensity given all three E components:
def intensity_threevec(ex_data, ey_data, ez_data):
    ez_sq = np.square(ez_data)
    ey_sq = np.square(ey_data)
    ex_sq = np.square(ex_data)
    
    in_temp = np.add(ez_sq, ey_sq)
    in_fin = np.add(in_temp, ex_sq)
    return in_fin

#Create simulation outline.
def init_sim(run_time):

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
                        center=mp.Vector3(-1*(0.5*sx)+dpml,0,0)),
                mp.Source(mp.GaussianSource(fcen,fwidth=df),
                        component=mp.Ey,
                        center=mp.Vector3(-1*(0.5*sx)+dpml,0,0)),
                mp.Source(mp.GaussianSource(fcen,fwidth=df),
                        component=mp.Ex,
                        center=mp.Vector3(-1*(0.5*sx)+dpml,0,0))]

    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        #geometry=geometry,
                        sources=sources,
                        resolution=resolution)

    #plt.figure(dpi=100)
    #sim.plot2D()
    #plt.show()

    sim.run(until=run_time)

    ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)
    ey_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ey)
    ex_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ex)

    '''plt.figure(dpi=100)
    sim.plot2D(fields=mp.Ez)
    plt.show()

    plt.figure(dpi=100)
    sim.plot2D(fields=mp.Ey)
    plt.show()

    plt.figure(dpi=100)
    sim.plot2D(fields=mp.Ex)
    plt.show()'''

    return sim, cell, pml_layers, sources, resolution, ez_data, ey_data, ex_data

#Create intensity plot.
def plot_intensity(in_data, run_time):

    in_data = np.rot90(in_data)

    plt.imshow(in_data, aspect = 'equal', interpolation='none', origin='upper', norm="logit")
    plt.colorbar(label="\n Electric Field Intensity")
    plt.title("Electric Field Intensity at Time " + str(run_time) + " [Meep Time] \n")
    plt.show()


#######################################################################

#Main: Let's run some functions!

#run_time = 75
#sim, cell, pml_layers, sources, resolution, ez_data, ey_data, ex_data = init_sim(run_time)
#in_data = intensity_threevec(ex_data, ey_data, ez_data)
#plot_intensity(in_data, run_time)

#print(ez_data)
#print(ey_data)
#print(ex_data)