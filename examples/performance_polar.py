# Basic function to create polar plots of variables from CHARM *perf.dat file

import sys
# add my share folder to path to use my utility module functions
sys.path.append("/projectnb/turbomac/arozman/share")
from charm_functions import read_perf_header, read_perf, plot_polar

perf_file_path="jobyperf.dat"

# read case info from perf file header
nrotor, npsi, nx, mrev = read_perf_header(perf_file_path)

# dCt / dx polar
# for var_ix: read perf header or doc of /projectnb/turbomac/arozman/share/charm_functions.py read_perf function
var_ix = 3
polar_aoa, psis, xs = read_perf(perf_file_path, nrotor, npsi, nx, mrev, var_ix)
plot_polar(psis, xs, polar_aoa, title='dCt/dx',
               fname='polar_dctdx.png', nrotor=nrotor)

# Fz polar
var_ix = 8
polar_aoa, psis, xs = read_perf(perf_file_path, nrotor, npsi, nx, mrev, var_ix)
plot_polar(psis, xs, polar_aoa, title='Fz',
               fname='polar_Fz.png', nrotor=nrotor)
