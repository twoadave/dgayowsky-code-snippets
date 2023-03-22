"""
Created on Mon Mar 6 2023

@author: David Gayowsky

PyMeep tutorials.
"""

#######################################################################

import meep as mp
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Video 
import os

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

plt.figure(dpi=100)
sim.plot2D()
plt.show()

sim.run(until=200)

plt.figure(dpi=100)
sim.plot2D(fields=mp.Ez)
plt.show()

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

#Often, we want to track the evolution of the fields as a function of time. This helps us 
# ensure the fields are propogating as we would expect. We can easily accomplish this 
# using a run function. Run functions are passed to the sim.run() method and can be called 
# every time step. The Animate2D() run function can be used to generate an animation object 
# by grabbing frames from an arbitrary number of time steps. We need to pass the sim object 
# we created, specify which fields component we are interested in tracking, specify how often 
# we want to record the fields, and whether to plot everything in real time. For this 
# simulation, let's look at the Ez fields and take a snapshot every 1 time units.

sim.reset_meep()
f = plt.figure(dpi=100)
Animate = mp.Animate2D(fields=mp.Ez, f=f, realtime=False, normalize=True)
plt.close()

sim.run(mp.at_every(1, Animate), until=100)
plt.close()

#Now that we've run the simulation, we can postprocess the animation and export it to an mp4 video 
# using the to_mp4() method. We'll specify a filename and 10 frames-per-second (fps).

file_path = os.path.realpath(__file__)
filename = file_path + "straight_waveguide.mp4"
Animate.to_mp4(10, filename)

#Finally, we can use some iPython tools to visualize the animation natively.
Video(filename)

#######################################################################

#Tutorial 2: Ninety-Degree Bend
#We'll start a new simulation where we look at the fields propagating through a waveguide 
# bend, and we'll do a couple of other things differently as well. 

#Then let's set up the bent waveguide in a slightly larger cell:
cell = mp.Vector3(16,16,0)

geometry = [mp.Block(mp.Vector3(12,1,mp.inf),
                     center=mp.Vector3(-2.5,-3.5),
                     material=mp.Medium(epsilon=12)),
            mp.Block(mp.Vector3(1,12,mp.inf),
                     center=mp.Vector3(3.5,2),
                     material=mp.Medium(epsilon=12))]

pml_layers = [mp.PML(1.0)]

resolution = 10

#There are a couple of items to note. First, a point source does not couple very efficiently 
# to the waveguide mode, so we'll expand this into a line source, centered at (-7,-3.5), with 
# the same width as the waveguide by adding a size property to the source. This is shown in 
# green in the figure above. An eigenmode source can also be used which is described in 
# Tutorial/Optical Forces. Second, instead of turning the source on suddenly at t=0 which excites 
# many other frequencies because of the discontinuity, we will ramp it on slowly. Meep uses a 
# hyperbolic tangent (tanh) turn-on function over a time proportional to the width of 20 time 
# units which is a little over three periods. Finally, just for variety, we'll specify the vacuum 
# wavelength instead of the frequency; again, we'll use a wavelength such that the waveguide is 
# half a wavelength wide.

sources = [mp.Source(mp.ContinuousSource(wavelength=2*(11**0.5), width=20),
                     component=mp.Ez,
                     center=mp.Vector3(-7,-3.5),
                     size=mp.Vector3(0,1))]

#Finally, we'll run the simulation. The first set of arguments to the run routine specify fields 
# to output or other kinds of analyses at each time step.

sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution)

#sim.run(mp.at_beginning(mp.output_epsilon),
#        mp.to_appended("ez", mp.at_every(0.6, mp.output_efield_z)),
#        until=200)

plt.figure(dpi=100)
sim.plot2D()
plt.show()

#Instead of running output_efield_z only at the end of the simulation, however, we run it at 
# every 0.6 time units (about 10 times per period) via mp.at_every(0.6, mp.output_efield_z). 
# By itself, this would output a separate file for every different output time, but instead we'll 
# use another feature to output to a single 3d HDF5 file, where the third dimension is time. "ez" 
# determines the name of the output file, which will be called ez.h5 if you are running interactively 
# or will be prefixed with the name of the file name for a Python file (e.g. tutorial-ez.h5 for 
# tutorial.py).

#Let's create an animation of the fields as a function of time. First, we have to create images 
# for all of the time slices:

#Instead of doing an animation, another interesting possibility is to make an image from a x×t slice. 
# To get the y=−3.5 slice, which gives us an image of the fields in the first waveguide branch as a 
# function of time, we can use get_array in a step function to collect a slice for each time step:

vals = []

def get_slice(sim):
    vals.append(sim.get_array(center=mp.Vector3(-2.5,-3.5), size=mp.Vector3(16,0), component=mp.Ez))

sim.run(mp.at_beginning(mp.output_epsilon),
        mp.at_every(0.6, get_slice),
        until=200)

plt.figure()
plt.imshow(vals, interpolation='spline36', cmap='RdBu')
plt.axis('off')
plt.show()

