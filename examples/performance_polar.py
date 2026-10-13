# Basic function to create polar plots of variables from CHARM *perf.dat file
import numpy as np
import sys
import matplotlib.pyplot as plt
# add my share folder to path to use my utility module functions
sys.path.append("/projectnb/turbomac/arozman/share")
from charm_functions import read_perf_header, read_perf, plot_polar

perf_file_path="jobyperf.dat"

# column indices to read from perf.dat. [8,14] returns Fz and AoA
var_ixs = [8,14]

# CW or CCW spin of each rotor
nrotor, npsi, nx, mrev = read_perf_header(perf_file_path)

polar_data, psis, xs = read_perf(perf_file_path, nrotor, npsi, nx, mrev, var_ixs)
#######################################################
# polar_data is shape [nrotor, MREV, npsi, nx, nvars]
#######################################################

# Fz polar plot
# Average Fz over revolutions. You could also plot an individual rev.
Fz = np.average(polar_data[...,0],axis=1)

plot_polar(psis, xs, Fz, title='Fz',
               fname='polar_Fz.png', nrotor=nrotor)

# Aoa polar plot
# Average AoA over revolutions.
AoA = np.average(polar_data[...,1],axis=1)

plot_polar(psis, xs, AoA, title='AoA',
               fname='polar_AoA.png', nrotor=nrotor)

# dFz/dAz polar plot
# Pad Fz psi dimension to size npsi+1 so that the diff over psi produces npsi values 
dFdaz = np.diff(np.concatenate((polar_data[...,0], polar_data[:,:,0:1,:,0]),axis=2), axis=2)
# Average over revolutions.
dFdaz = np.average(dFdaz,axis=1)

plot_polar(psis, xs, dFdaz, title='dFz/dAz',
               fname='polar_dFzdAz.png', nrotor=nrotor)

# Uncomment below if you have graphical forwarding and want plots to show to screen.
#plt.show()
