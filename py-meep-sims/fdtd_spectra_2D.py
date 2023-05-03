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

'''
sim.reset_meep()
f = plt.figure(dpi=100)
Animate = mp.Animate2D(fields=mp.Ez, f=f, realtime=False, normalize=True)
plt.close()

sim.run(mp.at_every(1, Animate), until=120)
plt.close()

file_path = os.path.realpath(__file__)
filename = file_path + "fdtd_2d_init.mp4"
Animate.to_mp4(10, filename)

Video(filename)'''

#Great, we have our initial thing - now let's see if we can do this with flux. 

sim.reset_meep()

nfreq = 100  # number of frequencies at which to compute flux

# reflected flux
refl_fr = mp.FluxRegion(center=mp.Vector3(-0.5*sx+dpml+0.5,0,0), size=mp.Vector3(0,sy-2*dpml,0))
refl = sim.add_flux(fcen, df, nfreq, refl_fr)

# transmitted flux
tran_fr = mp.FluxRegion(center=mp.Vector3(0.5*sx-dpml-0.5,0,0), size=mp.Vector3(0,sy-2*dpml,0))
tran = sim.add_flux(fcen, df, nfreq, tran_fr)

pt = mp.Vector3(-0.5*sx+dpml+0.5,0,0)

sim.run(until_after_sources=mp.stop_when_fields_decayed(50,mp.Ez,pt,1e-3))

plt.figure(dpi=100)
sim.plot2D(fields=mp.Ez)
plt.show()

# for normalization run, save flux fields data for reflection plane
straight_refl_data = sim.get_flux_data(refl)
# save incident power for transmission plane
straight_tran_flux = mp.get_fluxes(tran)

flux_freqs = mp.get_flux_freqs(tran)

wl = []
Ts = []
for i in range(nfreq):
    wl = np.append(wl, flux_freqs[i])
    Ts = np.append(Ts, straight_tran_flux[i])

if mp.am_master():
    plt.figure()
    plt.plot(wl,Ts,'ro-',label='transmittance')
    #plt.axis([0.7, 1.6, 0, 1])
    plt.xlabel("wavelength (μm)")
    plt.legend(loc="upper right")
    plt.show()

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

# reflected and transmitted flux
refl = sim.add_flux(fcen, df, nfreq, refl_fr)
tran = sim.add_flux(fcen, df, nfreq, tran_fr)

# for normal run, load negated fields to subtract incident from refl. fields
sim.load_minus_flux_data(refl, straight_refl_data)

sim.run(until_after_sources=mp.stop_when_fields_decayed(50,mp.Ez,pt,1e-3))

plt.figure(dpi=100)
sim.plot2D(fields=mp.Ez)
plt.show()

bend_refl_flux = mp.get_fluxes(refl)
bend_tran_flux = mp.get_fluxes(tran)

flux_freqs = mp.get_flux_freqs(refl)

wl = []
Rs = []
Ts = []
for i in range(nfreq):
    wl = np.append(wl, flux_freqs[i])
    Rs = np.append(Rs,-bend_refl_flux[i]/straight_tran_flux[i])
    Ts = np.append(Ts,bend_tran_flux[i]/straight_tran_flux[i])

if mp.am_master():
    plt.figure()
    plt.plot(wl,Rs,'bo-',label='reflectance')
    plt.plot(wl,Ts,'ro-',label='transmittance')
    plt.plot(wl,1-Rs-Ts,'go-',label='loss')
    #plt.axis([5.0, 10.0, 0, 1])
    plt.xlabel("wavelength (μm)")
    plt.legend(loc="upper right")
    plt.show()

'''
sim.reset_meep()
f = plt.figure(dpi=100)
Animate = mp.Animate2D(fields=mp.Ez, f=f, realtime=False, normalize=True)
plt.close()

sim.run(mp.at_every(1, Animate), until=120)
plt.close()

file_path = os.path.realpath(__file__)
filename = file_path + "fdtd_2d.mp4"
Animate.to_mp4(10, filename)

Video(filename)'''