# -*- coding: utf-8 -*-
"""
Useful functions for automating the setup and running of CHARM cases

@author: Adam Rozman
"""

import os
import glob
import sys
import numpy as np

#########################################################
# %% UTILITY FUNCTIONS
########################################################

# edit settings in CHARM input files
def edit_charm_input(file_path, parameter_strings, new_values, offsets=None):
    """
    inputs: "parameter_strings" a list of settings to change e.g. [OMEGA, U, YAW]
    to "new_values" a list of new values of the same length e.g. [314.2, -5, 15]
    at "file_path"
    """

    # new: add offset for multi-value parameters like tilt
    if offsets is None:
        offsets = [0] * len(parameter_strings)

    with open(file_path) as file:
        lines = file.readlines()

        for parameter_string, new_value, offset in zip(parameter_strings, new_values, offsets):
            for i, line in enumerate(lines):
                if (parameter_string + ' ') in line or (parameter_string + '\n') in line:
                    # split namelist into list, find index matching "parameter_string"
                    var_index = line.split().index(parameter_string)

                    # update var_index of data line (startline+1) with "new_value"
                    parameter_vals = lines[i+1].split()
                    # new: offset here
                    parameter_vals[var_index+offset] = str(new_value)

                    # join "inputs" list back into a string
                    updatedline = ' ' + '\t'.join(parameter_vals) + '\n'

                    # put updated line back into "lines" list
                    lines[i+1] = updatedline
                    break

    # re-write the file
    with open(file_path, 'w') as file:
        file.writelines(lines)

# get performance values from last line of *hubacr.dat
def get_performance_nrev(file_path):
    """
    outputs: array of 6 floats: [Fx, Fy, Fz, Mx, My, Mz]
    reading file last line code from stackoverflow
    """
    with open(file_path, 'rb') as f:    # binary red to use OS functions
        try:                        # catch OSError in case of a one line file
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
        except OSError:
            f.seek(0)
        last_line = f.readline().decode()

    force_moments = np.atleast_2d(np.loadtxt('fort.98', dtype=float))[-1,:]

    return force_moments

# call Dan W.'s processmrev fortran scrpit. Call this from CHARM run folder directory
def get_performance_mrev(PROCESS_MREV_CALL, CASE_NAME):
    """
    outputs: array of 6 floats: [Fx, Fy, Fz, Mx, My, Mz]
    """
    os.system(f"cp {CASE_NAME}hubacr.dat fort.99")
    os.system(PROCESS_MREV_CALL)
    os.system('rm fort.99')
    
    force_moments = np.atleast_2d(np.loadtxt('fort.98', dtype=float))[-1,:]
    
    return force_moments

def transform_vector_frame(x, yaw, pitch, roll, aircraft_frame_input=False):
    """
    inputs: x: list or array size [3] of 3D vector, yaw, pitch, roll in degrees
    if aircraft_frame_input=False (default): transform x from inertial to aircraft frame
    if aircraft_frame_input=True: transform x from aircraft to inertial frame
    NOTE: DOES NOT MULTIPLY BY -1. PERFORM OUTSIDE THIS FUNCTION IF NECESSARY.
    """
    th_y = np.deg2rad(yaw)
    th_p = np.deg2rad(pitch)
    th_r = np.deg2rad(roll)

    #print(f"input: {x}\n")
    R = np.array([[1, 0, 0],
                  [0, np.cos(th_r), -np.sin(th_r)],
                  [0, np.sin(th_r),  np.cos(th_r)]])

    P = np.array([[np.cos(th_p), 0, np.sin(th_p)],
                  [0,            1, 0],
                  [-np.sin(th_p), 0, np.cos(th_p)]])

    Y = np.array([[np.cos(th_y), -np.sin(th_y), 0],
                  [np.sin(th_y),  np.cos(th_y), 0],
                  [0, 0, 1]])

    if aircraft_frame_input:
        # convert aircraft rame to inertial frame
        x_prime =  Y @ P @ R @ np.array(x)
    else:
        # convert intertial frame to aircraft frame
        x_prime = R.T @ P.T @ Y.T @ np.array(x)

    #print(f"output: {x_prime}\n")
    return(x_prime)



def read_perf_header(fname):
    """
    Read CHARM performance file header and extract configuration parameters
    Returns: (nrotor, npsi, nx, mrev)
    If there are multiple rotor, this code assumes they all have the same
    NPSI and NX
    """
    with open(fname) as f:
        lines = f.readlines()

    nrotor = int(lines[1].split()[0])
    npsi = int(lines[3].split()[1])
    nx = int(lines[3].split()[2])

    header_lines = 5

    # Calculate MREV from total data lines
    # Total lines = header_lines + (nrotor * mrev * npsi * nx)
    data_lines = len(lines) - header_lines
    lines_per_rev = nrotor * npsi * nx

    # CHARM includes the initial NREV plus MREV additional revolution
    # subtract 1 unless there is only 1 rev, meaning it's an NREV case.
    mrev = max(data_lines // lines_per_rev - 1 , 1)

    # Verify the calculation makes sense
    # Note: We use (mrev + 1) here because the file contains mrev + 1 blocks
    expected_data_lines = nrotor * (mrev + 1) * npsi * nx
    if abs(data_lines - expected_data_lines) > npsi * nx:
        print(f"WARNING: Data line count mismatch!")
        print(f"  Data lines: {data_lines}")
        print(f"  Expected for MREV={mrev} (+1 initial): {expected_data_lines}")
        print(f"  Difference: {data_lines - expected_data_lines}")

    return nrotor, npsi, nx, mrev

##########################################
# PERF FILE FUNCTIONS
##########################################

def read_perf(fname, nrotor, npsi, nx, mrev, var_ix):
    """
    Read CHARM performance data and extract specified variable

    Note: mrev represents the number of new revolutions (excluding the initial azimuth at 0°)
    The file contains mrev+1 data blocks total

    File format:
    - Top header (2 lines): NROTOR/RHO/SSPD
    - For each rotor block, 3-line header:
      Line 1: ROTOR NPSI NX RADIUS OMEGA ...
      Line 2: Numeric values
      Line 3: Column names (psi x=r/R dx ...)
    - Then data rows
    """
    with open(fname) as f:
        lines = f.readlines()

    rotor_delims = []
    for j in range(len(lines)):
        if "ROTOR" in lines[j]:
            rotor_delims.append(j)

    var = np.zeros((nrotor, mrev, npsi, nx))

    delim_ix = 1 # Start this at 1 because first 'ROTOR' instance is part of header
    for r in range(nrotor):
        for m in range(mrev):
            data = np.loadtxt(lines[rotor_delims[delim_ix]+3:rotor_delims[delim_ix]+3+npsi*nx])
            var[r, m, :, :] = data[:,var_ix].reshape((npsi,nx))
        delim_ix += 1

    psis = np.unique(data[:,0])
    xs = np.unique(data[:,1])

    var_mean = np.mean(var, axis=1)

    return var_mean, psis, xs
