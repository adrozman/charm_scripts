"""
 Parses through folders of CHARM runs that have already run PSU-WOPWOP for VT 
 Joby experimental microphones. Reads pressure file and generates a directivity
 map of the BPF. Takes input of the colorbar limits [MIN,MAX] of the maps
 @author Adam Rozman
"""
import os
import numpy as np
import sys

# add my share folder to path to use my utility module functions
sys.path.append("/projectnb/turbomac/arozman/share")
from charm_acoustics import readXfile, readFnfile, p_to_SPL, generate_directivity_map

##########################
# INPUTS
##########################
base_run_dir = "charm_runs" # folder name with the outputs
run_name = "joby" # *.inp file

colormap_limits = [0,90] # (SPL (dB) limits of the map colorbar 

##########################
# RUN LOGIC
#########################

# list folders in base run dir
run_folders = os.listdir(base_run_dir)
print("folders found: ", run_folders)

# create folder for the plots
os.makedirs("directivity_plots", exist_ok=True)

# iterate through run folders
for run_folder in run_folders:

    # get rpm value from folder name
    rpm_val = float(run_folder.split("rpm")[-1].replace("_","."))

    wopwop_folder = os.path.join(base_run_dir, run_folder, run_name+"PSU-WOPWOP")
    # read mic coords
    xyz = readXfile(os.path.join(wopwop_folder, "pressure.x"))[:,0,:]

    # read mic pressure
    data = readFnfile(os.path.join(wopwop_folder, "pressure.fn"))

    t = data[0,0,:,0] - data[0,0,0,0]  # make t start at 0
    p = np.squeeze(data[:,:,:,-1])  # [NI,NJ,NT,NVAR]. var -1 is total noise
    
    # convert p to SPL usin
    f, SPL = p_to_SPL(p, t)

    BPF = rpm_val / 60 * 5
    # find fq index of the BPF 
    idx_freq = np.argmin(np.abs(f - BPF))

    #plotting utility
    generate_directivity_map(SPL[:,idx_freq], xyz, levels=np.linspace(colormap_limits[0], colormap_limits[1], 41), filename=f"directivity_plots/directivity" + run_folder + ".png")
