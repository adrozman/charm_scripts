# -*- coding: utf-8 -*-
"""
How to plot experimental data directivity contours

"""

# import numpy as np
# import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py
from scipy.interpolate import griddata

expdir="/projectnb/turbomac/arozman/projects/ULI/experimental_data/16degPitch_90degYaw_10mps_AutoSpec.mat"
micsdir="/projectnb/turbomac/arozman/projects/ULI/experimental_data/Xc_U10.txt"

# load data
file =  h5py.File(expdir)

print(file.keys())

fq = np.squeeze(file['freq'][:])
bbn = file['broadband_spectra'][:]
rpms = np.squeeze(file['rpm'][:])
tonal = file['tonal_spectra'][:]
total = file['total_spectra'][:]

xyz = np.loadtxt(micsdir,delimiter=",")

def format_fig(fig, ax, CS, levels, title):
    ax.set_xlabel('X (m)', fontsize=14)
    ax.set_ylabel('Y (m)', fontsize=14)
    ax.tick_params(axis='both', labelsize=10)
    plt.axis('equal')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    
    ticks = np.linspace(levels[0], levels[-1], 10).tolist()
    
    cbar = fig.colorbar(CS, cax=cax, ticks=ticks)
    cbar.ax.set_ylabel(title, fontsize=14)
    cbar.ax.set_yticklabels([f'{x:.2f}' for x in ticks])

    ax.set_ylim((-1.,1.))
    ax.set_xlim((-2.,2.))
# %% Filter by BPF, remove NaNs
rpm_ix=-1   # 4000 RPM
BPF = rpms[rpm_ix]/60*5

# index in fq array of BPF frequency
ix = np.argmin(abs(fq-BPF))

# where there are not NaNs at the BPF 
mask = ~np.isnan(tonal[-1,:,ix])

# also remove the explicitly bad mics
bad_mics = [17, 155, 219]

mask[bad_mics]=False

# remove those values from xyz and noise arrays
xyz = xyz[mask,:]
tonal = tonal[:,mask,:]
total = total[:,mask,:]
bbn = bbn[:,mask,:]


# %% Interp PDD on grid & acoustic processing
X, Y, = np.meshgrid(np.linspace(-2,2,100), np.linspace(-1,1,50))
Z = griddata((xyz[:,0], xyz[:,1]), tonal[-1,:,ix], (X,Y), method='linear')

df = fq[1] - fq[0]

# SPL (dB) is multiplied by bandwidth
Z_spectrum = 10*np.log10( Z * df / (2e-5)**2 )

# PSD (dB/Hz) is not.
Z_PSD = 10*np.log10( Z / (2e-5)**2 )

# values for contour colorbar
levels = np.linspace(65,75,75)

# %% plotting
# plot SPL
fig, ax = plt.subplots(layout='constrained')
CS = ax.contourf(X, Y, Z_spectrum, levels, extend='both', cmap="jet")
format_fig(fig, ax, CS, levels, 'SPL (dB)')
ax.set_title(f"Experiment 4000RPM Hover SPL, df={fq[1]}", fontsize=14)

# %%
# plot PSD
fig, ax = plt.subplots(layout='constrained')
CS = ax.contourf(X, Y, Z_PSD, levels, extend='both', cmap="jet")
format_fig(fig, ax, CS, levels, 'PSD (dB/sqrt(Hz))')
ax.set_title(f"Experiment 4000RPM Hover PSD, df={fq[1]}", fontsize=14)

# %%
# plot PSD averaged within +/- 70Hz of BPF
freq_mask = (fq >= (BPF - 70)) & (fq <= (BPF + 70))
tonal_avg_70Hz = np.mean(tonal[-1][:, freq_mask], axis=1)
Z_avg_70Hz = griddata((xyz[:, 0], xyz[:, 1]), tonal_avg_70Hz, (X, Y), method='linear')
Z_PSD_70Hz = 10 * np.log10(Z_avg_70Hz / (2e-5)**2)

fig, ax = plt.subplots(layout='constrained')
CS = ax.contourf(X, Y, Z_PSD_70Hz, levels, extend='both', cmap="jet")
format_fig(fig, ax, CS, levels, 'PSD (dB/Hz)')
ax.set_title(f"Experiment 4000RPM Hover PSD (Avg +/- 70Hz), df={fq[1]}", fontsize=14)

# %%
# plot PSD averaged within +/- 10Hz of BPF
freq_mask = (fq >= (BPF - 10)) & (fq <= (BPF + 10))
tonal_avg_70Hz = np.mean(tonal[-1][:, freq_mask], axis=1)
Z_avg_70Hz = griddata((xyz[:, 0], xyz[:, 1]), tonal_avg_70Hz, (X, Y), method='linear')
Z_PSD_70Hz = 10 * np.log10(Z_avg_70Hz / (2e-5)**2)

fig, ax = plt.subplots(layout='constrained')
CS = ax.contourf(X, Y, Z_PSD_70Hz, levels, extend='both', cmap="jet")
format_fig(fig, ax, CS, levels, 'PSD (dB/Hz)')
ax.set_title(f"Experiment 4000RPM Hover PSD (Avg +/- 10Hz), df={fq[1]}", fontsize=14)

# %%
# plot broadband noise (BBN) integrated from 500 to 20000 Hz
bbn_mask = (fq >= 500) & (fq <= 20000)
bbn_int = np.sum(bbn[-1][:, bbn_mask] * df, axis=1)
Z_bbn = griddata((xyz[:, 0], xyz[:, 1]), bbn_int, (X, Y), method='linear')
Z_bbn_SPL = 10 * np.log10(Z_bbn / (2e-5)**2)

levels_bbn = np.linspace(90, 100, 75)
fig, ax = plt.subplots(layout='constrained')
CS = ax.contourf(X, Y, Z_bbn_SPL, levels_bbn, extend='both', cmap="jet")
format_fig(fig, ax, CS, levels_bbn, 'BBN SPL (dB)')
ax.set_title("Experiment 4000RPM Broadband Noise SPL (500-20000 Hz)", fontsize=14)

plt.show()

