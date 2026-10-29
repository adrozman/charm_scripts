"""
 Parses through folders of CHARM runs that have already run PSU-WOPWOP for VT 
 Joby experimental microphones. Reads pressure file and generates a directivity
 map of the BPF. Takes input of the colorbar limits [MIN,MAX] of the maps
 @author Adam Rozman
"""
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import h5py

# add my share folder to path to use my utility module functions
sys.path.append("/projectnb/turbomac/arozman/share")
from charm_acoustics import readXfile, readFnfile, p_to_SPL, generate_directivity_map
from charm_functions import read_perf_header

plt.rcParams['font.size']=16

##########################
# INPUTS
##########################
run_name = "joby" # *.inp file

# XYZ location in experimental frame to compare spectra at.
target_location = [-1,0,1.56]

# expliclity list folders you want to compare
run_folders=["JOBY_DIRECTIVITY_COLL/charm_runs/U32_81_tilt-20_00_rpm3806_00_coll16_0/","JOBY_DIRECTIVITY_COLL_80/charm_runs/U32_81_tilt-20_00_rpm3806_00_coll16_0"]
# or us os to grab a whole directory of runs...
#run_folders = os.listdir(base_run_dir)

legend=["NSPI 40","NPSI 80","EXP"] # add to this if you add run folders above ^

# experimental spectra to compare to
exp_path = "/projectnb/turbomac/arozman/projects/ULI/experimental_data/16degPitch_70degYaw_10mps_AutoSpec.mat"
# experimental rpm to plot
rpm = 3806

##########################
# RUN LOGIC
#########################

print("folders found: ", run_folders)

# load experimental data acoustics
file =  h5py.File(exp_path)
rpms_e = np.squeeze(file['rpm'][:])
rpm_ix=np.argmin(np.abs(rpms_e-rpm))

f_exp = np.squeeze(file['freq'][:])
tonal_exp = file['tonal_spectra'][rpm_ix] 
total_exp = file['total_spectra'][rpm_ix] 


# fig 1 is pressure in time domain. fig 2 is fq-domain SPL spectrum/
fig1,ax1=plt.subplots(layout='constrained')
fig2,ax2=plt.subplots(layout='constrained')

for run_folder in run_folders:

    # get rpm value from folder name
    rpm_val = float(run_folder.split("rpm")[-1].split('coll')[0][:-1].replace("_","."))
    period = 60/rpm_val
    BPF = rpm_val/60 * 5
    
    # get information for this run (if necessary)
    #perf_file_path = os.path.join(run_folder, run_name+"perf.dat") 
    #nrotor, npsi, nx, mrev = read_perf_header(perf_file_path)

    wopwop_folder = os.path.join(run_folder, run_name+"PSU-WOPWOP")
    # read mic coords
    xyz = readXfile(os.path.join(wopwop_folder, "pressure.x"))[:,0,:] # mic is stationary, use first timestep location

    # read mic pressure
    data = readFnfile(os.path.join(wopwop_folder, "pressure.fn"))

    t = data[0,0,:,0] - data[0,0,0,0]  # make t start at 0
    p = np.squeeze(data[:,:,:,-1])  # [NI,NJ,NT,NVAR]. VAR [-1] is total noise

    nmics,nt = p.shape
    
    # calculate the mean perrev time-domain pressure signal.
    nrev = int(np.round( (t[-1] / period) ))
    tperrev = int(nt / nrev)
    p_mean = np.mean( np.reshape(p, (nmics,nrev,tperrev)) , axis=1)

    # determine the index of the microphone closest to a particular coordinate.
    target_ix = np.argmin(np.linalg.norm(np.abs(xyz - target_location),axis=1))

    ax1.plot(np.linspace(0,1,tperrev),p_mean[target_ix])
    
    # convert p to SPL using imported function
    f, SPL = p_to_SPL(p, t)

    # plot SPL at the target index.
    ax2.semilogx(f/BPF,SPL[target_ix])

# plot the experiment tonal noise peaks.
ax2.semilogx(f_exp/BPF,10*np.log10(tonal_exp[target_ix]*f_exp[1]/(2e-5)**2),color='k',linewidth=2)

ax1.legend(legend[:-1]) # we don't have time-domain experiment.
ax1.set_xlabel("Period")
ax1.set_ylabel("Pressure (Pa)")

ax2.set_xlabel("BPF Harmonic")
ax2.set_ylabel("SPL (dB)")
ax2.legend(legend)
ax2.set_xlim([0.5,10])
ax2.set_ylim([0,80])

#save the plots
fig1.savefig("pressure.png", dpi=300)
fig2.savefig("spl.png", dpi=300)
