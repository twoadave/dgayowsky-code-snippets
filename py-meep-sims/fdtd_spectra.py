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

resolution = 10 # pixels/um

cell = mp.Vector3(2,2,4)

#Next we add the waveguide. Most commonly, the device structure is specified by 
# a set of GeometricObjects stored in the geometry object.

geometry = [mp.Block(mp.Vector3(mp.inf,mp.inf, 0.5),
                     center=mp.Vector3(0,0,0),
                     material=mp.Medium(epsilon=12))]

nfreq = 25  # number of frequencies at which to compute flux

###############################################################

# Wavelength and frequency range
freq_range = (200e12, 400e12)
freq0 = np.sum(freq_range)/2
lambda0 = c/freq0
print(lambda0)

#fcen = 1/lambda0 # pulse center frequency
fcen = 1 #Corresponds to 300THz
#print(fcen)

###############################################################

df = 0.75   # pulse width (in frequency)

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

#plt.figure(dpi=100)
#sim.plot3D()
#plt.show()

#Now want to put in our flux calculation...

# reflected flux
refl_fr = mp.FluxRegion(center=mp.Vector3(0,0,-1), size=mp.Vector3(2,2,0))
refl = sim.add_flux(fcen, df, nfreq, refl_fr)

# transmitted flux
tran_fr = mp.FluxRegion(center=mp.Vector3(0,0,1), size=mp.Vector3(2,2,0))
tran = sim.add_flux(fcen, df, nfreq, tran_fr)

pt = mp.Vector3(0,0,1)

sim.run(until=500)

# save incident power for transmission plane
straight_tran_flux = mp.get_fluxes(tran)
flux_freqs = mp.get_flux_freqs(tran)
# for normalization run, save flux fields data for reflection plane
straight_refl_data = sim.get_flux_data(refl)

wl = []
Ts = []
for i in range(nfreq):
    wl = np.append(wl, 1/flux_freqs[i])
    Ts = np.append(Ts,straight_tran_flux[i])

print(Ts)
print(wl)

if mp.am_master():
    plt.figure()
    plt.plot(wl,Ts,'ro-',label='transmittance')
    #plt.axis([5.0, 10.0, 0, 1])
    plt.xlabel("wavelength (μm)")
    plt.legend(loc="upper right")
    plt.show()
