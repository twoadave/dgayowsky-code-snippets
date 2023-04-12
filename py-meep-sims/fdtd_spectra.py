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

#Initial calculation: no waveguide:

resolution = 10 # pixels/um
cell = mp.Vector3(2,2,4)
geometry = [mp.Block(mp.Vector3(mp.inf,mp.inf, 0.5),
                     center=mp.Vector3(0,0,0),
                     material=mp.Medium(epsilon=1))]

nfreq = 25
fcen = 1
df = 0.75

pml_layers = [mp.PML(thickness=0.1,direction=mp.Z)]

sources = [mp.Source(mp.GaussianSource(fcen,fwidth=df),
                     component=mp.Ez,
                     center=mp.Vector3(0,0,-1),
                     size=mp.Vector3(2,2,0))]

sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution)

refl_fr = mp.FluxRegion(center=mp.Vector3(0,0,-1), size=mp.Vector3(2,2,0))
refl = sim.add_flux(fcen, df, nfreq, refl_fr)

tran_fr = mp.FluxRegion(center=mp.Vector3(0,0,1), size=mp.Vector3(2,2,0))
tran = sim.add_flux(fcen, df, nfreq, tran_fr)

sim.run(until=500)

# for normalization run, save flux fields data for reflection plane
straight_refl_data = sim.get_flux_data(refl)

# save incident power for transmission plane
straight_tran_flux = mp.get_fluxes(tran)

#######################################################################

#Now we go to our actual simulation...
sim.reset_meep()

#Next we add the waveguide. Most commonly, the device structure is specified by 
# a set of GeometricObjects stored in the geometry object.

geometry = [mp.Block(mp.Vector3(mp.inf,mp.inf, 0.5),
                     center=mp.Vector3(0,0,0),
                     material=mp.Medium(epsilon=12))]

sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution)

nfreq = 25  # number of frequencies at which to compute flux

#######################################################################

# Wavelength and frequency range
#freq_range = (200e12, 400e12)
#freq0 = np.sum(freq_range)/2
#lambda0 = c/freq0
#print(lambda0)

#fcen = 1/lambda0 # pulse center frequency
#print(fcen)

#######################################################################

#plt.figure(dpi=100)
#sim.plot3D()
#plt.show()

#Now want to put in our flux calculation...

# reflected flux
refl = sim.add_flux(fcen, df, nfreq, refl_fr)

# transmitted flux
tran_fr = mp.FluxRegion(center=mp.Vector3(0,0,1), size=mp.Vector3(2,2,0))
tran = sim.add_flux(fcen, df, nfreq, tran_fr)

# for normal run, load negated fields to subtract incident from refl. fields
sim.load_minus_flux_data(refl, straight_refl_data)

sim.run(until=500)

bend_refl_flux = mp.get_fluxes(refl)
bend_tran_flux = mp.get_fluxes(tran)

flux_freqs = mp.get_flux_freqs(refl)

wl = []
Rs = []
Ts = []
for i in range(nfreq):
    wl = np.append(wl, 1/flux_freqs[i])
    Rs = np.append(Rs,-bend_refl_flux[i]/straight_tran_flux[i])
    Ts = np.append(Ts,bend_tran_flux[i]/straight_tran_flux[i])

#print(Ts)
#print(wl)

if mp.am_master():
    plt.figure()
    plt.plot(wl,Rs,'bo-',label='reflectance')
    plt.plot(wl,Ts,'ro-',label='transmittance')
    plt.plot(wl,1-Rs-Ts,'go-',label='loss')
    plt.axis([0.7, 1.6, 0, 1])
    plt.xlabel("wavelength (μm)")
    plt.legend(loc="upper right")
    plt.show()
