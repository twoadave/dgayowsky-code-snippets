"""
Created on Fri Mar 31 2023

@author: David Gayowsky

PyMeep tutorials, transmittance and reflectance spectra.
"""

#######################################################################

import meep as mp
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Video 
import os

#######################################################################

#Tutorial: transmittance spectra:

resolution = 10 # pixels/um

sx = 16  # size of cell in X direction
sy = 32  # size of cell in Y direction
cell = mp.Vector3(sx,sy,0)

dpml = 1.0
pml_layers = [mp.PML(dpml)]

#We'll also define a couple of parameters to set the width of the waveguide 
# and the "padding" between it and the edge of the cell:
pad = 4  # padding distance between waveguide and cell edge
w = 1    # width of waveguide

#In order to define the waveguide positions, we will have to use arithmetic 
# to define the horizontal and vertical waveguide centers as:

wvg_xcen =  0.5*(sx-w-2*pad)  # x center of horiz. wvg
wvg_ycen = -0.5*(sy-w-2*pad)  # y center of vert. wvg

#We proceed to define the geometry. We have to do two simulations with different 
# geometries: the bend, and also a straight waveguide for normalization. We will 
# first set up the straight waveguide.

geometry = [mp.Block(size=mp.Vector3(mp.inf,w,mp.inf),
                     center=mp.Vector3(0,wvg_ycen,0),
                     material=mp.Medium(epsilon=12))]

#The source is a GaussianSource instead of a ContinuousSource, parameterized by 
# a center frequency and a frequency width (the width of the Gaussian spectrum).

fcen = 0.15  # pulse center frequency
df = 0.1     # pulse width (in frequency)
sources = [mp.Source(mp.GaussianSource(fcen,fwidth=df),
                     component=mp.Ez,
                     center=mp.Vector3(-0.5*sx+dpml,wvg_ycen,0),
                     size=mp.Vector3(0,w,0))]

sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution)

plt.figure(dpi=100)
sim.plot2D()
plt.show()

#Finally, we have to specify where we want Meep to compute the flux spectra, and 
# at what frequencies. As described in Introduction/Transmittance/Reflectance 
# Spectra, the flux is the integral of the Poynting vector over the specified 
# FluxRegion. It only integrates one component of the Poynting vector and the 
# direction property specifies which component. In this example, since the FluxRegion 
# is a line, the direction is its normal by default which therefore does not need 
# to be explicitly defined.

nfreq = 100  # number of frequencies at which to compute flux

# reflected flux
refl_fr = mp.FluxRegion(center=mp.Vector3(-0.5*sx+dpml+0.5,wvg_ycen,0), size=mp.Vector3(0,2*w,0))
refl = sim.add_flux(fcen, df, nfreq, refl_fr)

# transmitted flux
tran_fr = mp.FluxRegion(center=mp.Vector3(0.5*sx-dpml,wvg_ycen,0), size=mp.Vector3(0,2*w,0))
tran = sim.add_flux(fcen, df, nfreq, tran_fr)

#computing the reflection spectra requires some care because we need to separate the 
# incident and reflected fields. We do this by first saving the Fourier-transformed 
# fields from the normalization run. We run the first simulation as follows:

pt = mp.Vector3(0.5*sx-dpml-0.5,wvg_ycen)

sim.run(until_after_sources=mp.stop_when_fields_decayed(50,mp.Ez,pt,1e-3))

# for normalization run, save flux fields data for reflection plane
straight_refl_data = sim.get_flux_data(refl)

plt.figure(dpi=100)
sim.plot2D(fields=mp.Ez)
plt.show()

# save incident power for transmission plane
straight_tran_flux = mp.get_fluxes(tran)

#We need to run the second simulation which involves the waveguide bend. 
# We reset the structure and fields using reset_meep() and redefine the geometry, 
# Simulation, and flux objects. At the end of the simulation, we save the 
# reflected and transmitted fluxes.

sim.reset_meep()

geometry = [mp.Block(mp.Vector3(sx-pad,w,mp.inf), center=mp.Vector3(-0.5*pad,wvg_ycen), material=mp.Medium(epsilon=12)),
            mp.Block(mp.Vector3(w,sy-pad,mp.inf), center=mp.Vector3(wvg_xcen,0.5*pad), material=mp.Medium(epsilon=12))]

sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution)

plt.figure(dpi=100)
sim.plot2D()
plt.show()

# reflected flux
refl = sim.add_flux(fcen, df, nfreq, refl_fr)

tran_fr = mp.FluxRegion(center=mp.Vector3(wvg_xcen,0.5*sy-dpml-0.5,0), size=mp.Vector3(2*w,0,0))
tran = sim.add_flux(fcen, df, nfreq, tran_fr)

# for normal run, load negated fields to subtract incident from refl. fields
sim.load_minus_flux_data(refl, straight_refl_data)

pt = mp.Vector3(wvg_xcen,0.5*sy-dpml-0.5)

sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, pt, 1e-3))

plt.figure(dpi=100)
sim.plot2D(fields=mp.Ez)
plt.show()

bend_refl_flux = mp.get_fluxes(refl)
bend_tran_flux = mp.get_fluxes(tran)

flux_freqs = mp.get_flux_freqs(refl)

#With the flux data, we are ready to compute and plot the reflectance and 
# transmittance. The reflectance is the reflected flux divided by the incident 
# flux. We also have to multiply by -1 because all fluxes in Meep are computed 
# in the positive-coordinate direction by default, and we want the flux in the −x 
# direction. The transmittance is the transmitted flux divided by the incident 
# flux. Finally, the scattered loss is simply 1−transmittance−reflectance.

wl = []
Rs = []
Ts = []
for i in range(nfreq):
    wl = np.append(wl, 1/flux_freqs[i])
    Rs = np.append(Rs,-bend_refl_flux[i]/straight_tran_flux[i])
    Ts = np.append(Ts,bend_tran_flux[i]/straight_tran_flux[i])

if mp.am_master():
    plt.figure()
    plt.plot(wl,Rs,'bo-',label='reflectance')
    plt.plot(wl,Ts,'ro-',label='transmittance')
    plt.plot(wl,1-Rs-Ts,'go-',label='loss')
    plt.axis([5.0, 10.0, 0, 1])
    plt.xlabel("wavelength (μm)")
    plt.legend(loc="upper right")
    plt.show()