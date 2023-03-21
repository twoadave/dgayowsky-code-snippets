"""
Created on Mon Mar 6 2023

@author: David Gayowsky

PyMeep tutorials.
"""

#######################################################################

import meep as mp
import matplotlib.pyplot as plt
import numpy as np

#######################################################################

#Tutorial 1: A straight wave guide.

#We can begin specifying each of the simulation objects starting with the 
# computational cell. We're going to put a source at one end and watch the 
# fields propagate down the waveguide in the x direction, so let's use a 
# cell of length 16 μm in the x direction to give it some distance to propagate. 
# In the y direction, we just need enough room so that the boundaries do not 
# affect the waveguide mode; let's give it a size of 8 μm.

#The Vector3 object stores the size of the cell in each of the three coordinate 
# directions. This is a 2d cell in x and y where the z direction has size 0.
cell = mp.Vector3(16,8,0)

#Next we add the waveguide. Most commonly, the device structure is specified by 
# a set of GeometricObjects stored in the geometry object.

geometry = [mp.Block(mp.Vector3(mp.inf,1,mp.inf),
                     center=mp.Vector3(),
                     material=mp.Medium(epsilon=12))]

#We have the structure and need to specify the current sources using the sources 
# object. The simplest thing is to add a single point source Jz:

sources = [mp.Source(mp.ContinuousSource(frequency=0.15),
                     component=mp.Ez,
                     center=mp.Vector3(-7,0))]

#As for boundary conditions, we want to add absorbing boundaries around our cell. 
# Absorbing boundaries in Meep are handled by perfectly matched layers (PML) 
# — which aren't really a boundary condition at all, but rather a fictitious 
# absorbing material added around the edges of the cell. To add an absorbing layer 
# of thickness 1 μm around all sides of the cell, we do:

pml_layers = [mp.PML(1.0)]

#Meep will discretize this structure in space and time, and that is specified by 
# a single variable, resolution, that gives the number of pixels per distance unit. 
# We'll set this resolution to 10 pixels/μm, which corresponds to around 67 
# pixels/wavelength, or around 20 pixels/wavelength in the high-index material. In 
# general, at least 8 pixels/wavelength in the highest dielectric is a good idea. 
# This will give us a 160×80 cell.

resolution = 10

#The final object to specify is Simulation which is based on all the previously 
# defined objects.

sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution)

#We are ready to run the simulation. We time step the fields until a time of 200:

sim.run(until=200)

#We can analyze and visualize the fields with the NumPy and Matplotlib libraries.
#We will first create an image of the dielectric function ε. This involves obtaining 
# a slice of the data using the get_array routine which outputs to a NumPy array 
# and then display the results.

eps_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)
plt.figure()
plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
plt.axis('off')
plt.show()

#Next, we create an image of the scalar electric field Ez by overlaying the dielectric 
# function. We use the "RdBu" colormap which goes from dark red (negative) to white 
# (zero) to dark blue (positive).

ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)
plt.figure()
plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
plt.imshow(ez_data.transpose(), interpolation='spline36', cmap='RdBu', alpha=0.9)
plt.axis('off')
plt.show()

