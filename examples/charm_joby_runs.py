"""
 Sets up and runs CHARM cases of the Joby prop in the wind tunnel configuration with
 input of RPM, YAW, and U_INF for each run.
 Then modifies PSU-WOPWOP acoustics input deck and observer input to the experimental
 microphones. Writes forces and moments from each case to a text file.
 @author Adam Rozman
"""
import subprocess
import os
import numpy as np
import sys

# add my share folder to path to use my utility module functions
sys.path.append("/projectnb/turbomac/arozman/share")
from charm_acoustics import get_input_files, acoustics_no_processing
from charm_automation import edit_charm_input, get_performance_mrev

#################################
# INPUTS 
################################

# variables lists for the cases we want to run. All must be the same length.
# we can use list repetition for repeated cases. [0]*2 = [0,0]
# adding lists combines them. [0,1] + [3] = [1,2,3]
rpms = [4000]*4 
tilts = [0]*2 + [-90]*2 # degrees 
Uinfs = [0,-32.81]*2 # ft /s, in CHARM frame.  we want negative since SFRAME 1 is vehicle, not air velocity
# 0 degree tilt in CHARM has -Z as the propeller rotation direction. This is the experiment edgewise case (90 yaw)
# -90 degree tilt in CHARM has has +X as the propeller rotation direction. This is the experiment axial case (0 yaw)

# path definitions
templatedir = "INPUT_TEMPLATE/Hover"
rwfilename = "Prop_000_000rw.inp"
casename = "joby" # *.inp file

CHARM_CALL = "runv8rlm_nowop . joby"
PROCESS_MREV_CALL = '/projectnb/turbomac/arozman/share/processmrev'
WOPWOP_CALL = "/project/turbomac/WOPWOP/wopwop3_linux_v3.5.1_af4377a"

# Microphone locations (meters) as a list of lists, in the CHARM frame.
# *Note: please put more than 1 mic or this script won't work! (it will write pressure.tec instead of pressure.fn)
MICS = np.loadtxt("VT_mics_hover.txt")

# unit to multiply Charm values to get in meters. If CHARM is set to feet, this should be 0.3048 (feet in meters)
charm_unit_in_meters = 0.3048

##################################
# RUN LOGIC
##################################

base_path = os.getcwd()

# write output data text file
outputfile=os.path.join(base_path, "charm_force_moments.txt")
output_header = "U (ft/s) Tilt (deg) RPM Fx Fy Fz (lb) Mx My Mz (lb*ft)"
with open(outputfile,'w') as f:
    f.write(output_header+"\n") # add newline

# iterate through cases
for rpm, tilt, Uinf in zip(rpms, tilts, Uinfs):

    omega = rpm * 2 * np.pi / 60

    # ensure we return to home directory every loop
    os.chdir(base_path)
    # make new directory for charm run
    new_run_directory = f"charm_runs/U{Uinf:.2f}_tilt{tilt:.2f}_rpm{rpm:.2f}".replace(".","_")

    os.makedirs(new_run_directory, exist_ok=True)

    # copy input templates into new directory
    os.system(f"cp {templatedir}/* " + new_run_directory)

    os.chdir(new_run_directory)

    # modify input deck values
    edit_charm_input(casename+".inp", ["U"], [Uinf])
    edit_charm_input(rwfilename, ["OMEGA", "X,Y,Z"], [omega, tilt], [0,3])
    # offset [0,3] is a clunky workaround for CHARM's clunky input system.
    # X,Y,Y tilt needs offset of 3 to get the rotate about Y axis since
    # there are multiple values for the params in this line (XROTOR & TILT)

    # run charm
    print("CHARM running...")
    os.system(CHARM_CALL)

    # extract forces and moments from hubacr
    force_moments = get_performance_mrev(PROCESS_MREV_CALL, casename)
    print(force_moments)
    force_moments_str = list(map(lambda x: f"{float(x):>13.6e}", force_moments))

    print("Case forces (lb) and moments (ft*lb):\n" + ' '.join(force_moments_str))

    # write forces and moments to output file
    with open(outputfile, "a") as f:
        f.write(f"{Uinf} {tilt} {rpm} " + ' '.join(force_moments_str) + "\n")

    # run wopwop
    os.chdir(casename+"PSU-WOPWOP")

    # Locate input, RW, BG files for processing script
    INPUT_PATH, N_PROPS, RW_PATH, BG_PATH = get_input_files()

    # edir inputs for new mics, run wopwop
    acoustics_no_processing(
            WOPWOP_CALL, RW_PATH, BG_PATH, INPUT_PATH, MICS,
            padding=1.5, charm_unit_in_meters=charm_unit_in_meters)

    # acoustics processed in next script
