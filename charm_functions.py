import numpy as np
import os
from matplotlib import pyplot as plt
# charm_functions to make polar contour plot of jobyperf variables
# ================================================================
# UTILITY FUNCTIONS
# ================================================================
def read_charm_geometry(bg_file):
    """Read station locations and chord lengths from BG file"""
    with open(bg_file) as f:
        lines = f.readlines()

    cutout = 0.
    radius = 0.
    widths = []
    adding = False
    for i, line in enumerate(lines):
        if 'CUTOUT' in line:
            cutout = float(lines[i+1].split()[0])
            radius += cutout
        elif 'SL' in line:
            adding = True
            continue
        # stop when the next input is reached
        if 'CHORD' in line:
            break

        if adding:
            values = line.strip().split()
            for value in values:
                widths.append(float(value))

            radius += np.sum(np.array(values).astype(float))

    # Add the widths to the root cutout to get the station locations (centers of strip)
    stations = []
    for i in range(len(widths)):
        # add the cutout and the midpoint of the current segment
        station = cutout + widths[i]/2
        # add all the previous segment widths
        for j in range(i):
            station += widths[j]
        stations.append(station)

    # Read chord values
    chords = []
    adding = False
    for i, line in enumerate(lines):
        if 'CHORD' in line:
            adding = True
            continue
        # stop when the next input is reached
        if 'ELOFSG' in line:
            break

        if adding:
            values = line.strip().split()
            for value in values:
                chords.append(float(value))

    # Report the chord at the center of each segment (average of ends)
    chords = (np.array(chords[:-1]) + np.array(chords[1:])) / 2

    return cutout, radius, np.array(stations), np.array(chords)

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

# ================================================================
# PART 1: CHARM PERFORMANCE DATA PROCESSING
# ================================================================

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

    Columns order in the perf file:
    psi x=r/R dx dCT/dx dCQI/dx dCQP/dx X-force Y-force -Z-force P-mom AOAG CL2D
    CD2D CM2D AOA2D MACH2D U-radial V-aft W-down W-induced Circulation
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
    

def plot_polar(psis, xs, polar_data, vmin=None, vmax=None, title=None, fname=None, nrotor=4):
    """Create polar plots of rotor data"""

    # Automatically determine global min and max if not specified
    if vmin is None:
        vmin = np.nanmin(polar_data[:nrotor])
    if vmax is None:
        vmax = np.nanmax(polar_data[:nrotor])

    # Convert psis to radians
    theta = np.radians(psis)
    r = xs

    # Prepare for cyclic closing (fixing the "Pac-Man" gap)
    theta_cyclic = np.append(theta, theta[0] + 2*np.pi)
    # Create Meshgrid (indexing='ij' matches shape (NPSI+1, NX))
    Theta, R = np.meshgrid(theta_cyclic, r, indexing='ij')

    # Determine subplot layout based on number of rotors
    if nrotor == 1:
        fig, axs = plt.subplots(1, 1, figsize=(8, 8), 
                                subplot_kw={'projection': 'polar'},
                                layout='constrained')
        axs_flat = [axs]
    elif nrotor == 2:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6), 
                                subplot_kw={'projection': 'polar'},
                                layout='constrained')
        axs_flat = axs.flatten()
    else:  # 3 or 4 rotors
        fig, axs = plt.subplots(2, 2, figsize=(12, 10), 
                                subplot_kw={'projection': 'polar'},
                                layout='constrained')
        axs_flat = axs.flatten(order='F')

    mesh_objects = []  # Store plots to define colorbar later
    for i in range(nrotor):
        ax = axs_flat[i]
        
        # Extract single rotor data
        Z = polar_data[i, :, :]
        
        # Make data cyclic (append first row to end) to match Theta_cyclic
        Z_cyclic = np.vstack((Z, Z[0, :]))
        
        c = ax.pcolormesh(Theta, R, Z_cyclic, cmap='jet', shading='auto', vmin=vmin, 
                          vmax=vmax, edgecolor='none')
        
        if i == 1 or i == 2:
            ax.set_theta_direction(-1)
        else:
            ax.set_theta_direction(1)

        mesh_objects.append(c)

        # Formatting individual plots
        ax.set_title(f"Rotor {i+1}", fontsize=40, pad=20)
        ax.grid(False)
        
        ax.tick_params(axis='x', labelsize=20, pad=10)
        ax.set_yticks([])
        ax.set_yticklabels([]) 
    
    # Hide unused subplots if nrotor < 4
    for i in range(nrotor, len(axs_flat)):
        axs_flat[i].set_visible(False)
        
    cb = fig.colorbar(mesh_objects[-1], ax=axs if nrotor > 1 else axs_flat[0], 
                      location='right', aspect=20, pad=0.05)
    
    cb.ax.yaxis.get_offset_text().set_fontsize(24)

    cb.formatter.set_scientific(True)
    cb.formatter.set_useMathText(True)
    cb.formatter.set_powerlimits((-2, 2))
    
    cb.update_ticks()
    
    cb.set_label(title, fontsize=40)
    cb.ax.tick_params(labelsize=24)

    if fname:
        plt.savefig(fname)
    else:
        plt.show()


def extract_charm_aoa_mach(perf_file_path, outputs_dir="photos_dir", npy_dir="polars_dir"):
    """Process CHARM performance data and save results - inputs for QuietFly"""
    print("\n" + "="*60)
    print("PART 1: Processing CHARM Performance Data")
    print("="*60)
    
    if not os.path.exists(perf_file_path):
        print(f"ERROR: File not found: {perf_file_path}")
        exit(1)
    
    print(f"Processing perf file: {perf_file_path}")
    
    # Auto-detect file parameters
    nrotor, npsi, nx, mrev = read_perf_header(perf_file_path)

    print(f"\nDetected from file:")
    print(f"  NROTOR = {nrotor}")
    print(f"  NPSI = {npsi}")
    print(f"  NX = {nx}")
    print(f"  MREV = {mrev}")
    
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(npy_dir, exist_ok=True)

    # Columns order in the perf file:
    # psi x=r/R dx dCT/dx dCQI/dx dCQP/dx X-force Y-force -Z-force P-mom AOAG CL2D
    # CD2D CM2D AOA2D MACH2D U-radial V-aft W-down W-induced Circulation

    # Read and plot AoA
    var_ix = 14
    polar_aoa, psis, xs = read_perf(perf_file_path, nrotor, npsi, nx, mrev, var_ix)
    plot_polar(psis, xs, polar_aoa, vmin=-10, vmax=10, title='Local Airfoil AoA (deg)', 
               fname=os.path.join(outputs_dir, 'polar_aoa.png'), nrotor=nrotor)

    # Read and plot Mach
    var_ix = 15
    polar_mach, psis, xs = read_perf(perf_file_path, nrotor, npsi, nx, mrev, var_ix)
    plot_polar(psis, xs, polar_mach, vmin=0, vmax=0.4, title='Local Airfoil Mach', 
               fname=os.path.join(outputs_dir, 'polar_mach.png'), nrotor=nrotor)

    # Save arrays
    np.save(os.path.join(npy_dir, "polar_aoa.npy"), polar_aoa)
    np.save(os.path.join(npy_dir, "polar_mach.npy"), polar_mach)
    np.savetxt(os.path.join(npy_dir, "radial_locs.txt"), xs)
    
    print("Saved aoa and mach arrays to npy_files/")
    print("Saved plots: polar_aoa.png and polar_mach.png")

    plt.close()
    plt.close()


    return polar_aoa, polar_mach, xs, nrotor, npsi
