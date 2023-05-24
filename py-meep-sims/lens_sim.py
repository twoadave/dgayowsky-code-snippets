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
import math

#######################################################################

#Create function to calculate intensity given all three E components:
def intensity_threevec(ex_data, ey_data, ez_data):
    ez_sq = np.square(ez_data)
    ey_sq = np.square(ey_data)
    ex_sq = np.square(ex_data)
    
    in_temp = np.add(ez_sq, ey_sq)
    in_fin = np.add(in_temp, ex_sq)
    #in_fin = ez_sq
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

    return ez_data, ey_data, ex_data

#Create intensity plot.
def plot_intensity(in_data, run_time):

    in_data = np.rot90(in_data)

    major_ticks = np.arange(0, 601, 100)
    minor_ticks = np.arange(0, 601, 5)

    major_yticks = np.arange(0, 321, 100)
    minor_yticks = np.arange(0, 321, 5)


    plt.imshow(in_data, aspect = 'equal', interpolation='none', origin='upper', norm="logit")

    plt.xticks(major_ticks)
    plt.xticks(minor_ticks, minor=True)
    plt.yticks(major_yticks)
    plt.yticks(minor_yticks, minor=True)

    plt.colorbar(label="\n Electric Field Intensity")
    plt.title("Electric Field Intensity at Time " + str(run_time) + " [Meep Time] \n")
    plt.grid(which='both')
    plt.show()

#Create some basic test shapes to see if we can get an idea of what we want our lens to look like:
def basic_shapes_test_sim(run_time):
    resolution = 10 # pixels/um

    sx = 60  # size of cell in X direction
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
    
    #Create a basic geometric shape... see if we can figure out what we want in a lens.
    '''geometry = [mp.Wedge(radius = 10, 
                     wedge_angle=math.pi,
                     wedge_start=mp.Vector3(0,-1,0),
                     material=mp.Medium(epsilon=12))]'''
    
    '''geometry = [mp.Block(size=mp.Vector3(4,mp.inf,mp.inf),
                     center=mp.Vector3(0,0,0),
                     material=mp.Medium(epsilon=25)),
                mp.Prism(vertices=[mp.Vector3(2,sy*0.5,0), mp.Vector3(8,sy*0.5,0), mp.Vector3(2,sy*0.5 - 8,0)],
                         height=mp.inf,
                         material=mp.Medium(epsilon=25)),
                mp.Prism(vertices=[mp.Vector3(2,-1*sy*0.5,0), mp.Vector3(8,-1*sy*0.5,0), mp.Vector3(2,-1*sy*0.5 + 8,0)],
                         height=mp.inf,
                         material=mp.Medium(epsilon=25))]'''
    
    '''geometry = [mp.Block(size=mp.Vector3(4,mp.inf,mp.inf),
                     center=mp.Vector3(0,0,0),
                     material=mp.Medium(epsilon=12)),
                mp.Prism(vertices=[mp.Vector3(-2,sy*0.5,0), mp.Vector3(-8,sy*0.5,0), mp.Vector3(-2,sy*0.5 - 8,0)],
                         height=mp.inf,
                         material=mp.Medium(epsilon=12)),
                mp.Prism(vertices=[mp.Vector3(-2,-1*sy*0.5,0), mp.Vector3(-8,-1*sy*0.5,0), mp.Vector3(-2,-1*sy*0.5 + 8,0)],
                         height=mp.inf,
                         material=mp.Medium(epsilon=12))]'''
    
    geometry = [mp.Ellipsoid(size=mp.Vector3(10,sy,mp.inf),
                     center=mp.Vector3(0,0,0),
                     material=mp.Medium(epsilon=15))]

    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=sources,
                        resolution=resolution)

    plt.figure(dpi=100)
    sim.plot2D()
    plt.show()

    sim.run(until=run_time)

    ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)
    ey_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ey)
    ex_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ex)

    return ez_data, ey_data, ex_data

#Try and recreate our shape on a basic grid.
def grid_shapes_test(run_time):

    resolution = 10 # pixels/um

    sx = 60  # size of cell in X direction
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
    
    geometry = [mp.Ellipsoid(size=mp.Vector3(10,sy,mp.inf),
                     center=mp.Vector3(0,0,0),
                     material=mp.Medium(epsilon=15))]

    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=sources,
                        resolution=resolution)

    plt.figure(dpi=100)
    sim.plot2D()
    plt.show()

    sim.run(until=run_time)

    ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)
    ey_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ey)
    ex_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ex)

    return ez_data, ey_data, ex_data

#######################################################################

#Main: Let's run some functions!

run_time = 75
ez_data, ey_data, ex_data = basic_shapes_test_sim(run_time)
in_data = intensity_threevec(ex_data, ey_data, ez_data)
plot_intensity(in_data, run_time)
