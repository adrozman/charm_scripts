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
    Read CHARM performance data across all revolutions and extract specified variable(s).
    
    Parameters:
    - fname: path to *perf.dat file
    - nrotor: number of rotors
    - npsi: azimuthal resolution
    - nx: radial resolution
    - mrev: number of revolutions
    - var_ix: single column index (int) or list of indices
    
    Returns:
    - var: numpy array across all revolutions
           shape (nrotor, mrev, npsi, nx) if var_ix is scalar,
           or (nrotor, mrev, npsi, nx, nvars) if var_ix is a list
    - psis: array of unique azimuth angles (deg)
    - xs: array of unique radial stations (r/R)
    
    File format:
    - Top header (2 lines): NROTOR/RHO/SSPD
    - For each rotor block, 3-line header:
      Line 1: ROTOR NPSI NX RADIUS OMEGA ...
      Line 2: Numeric values
      Line 3: Column names (psi x=r/R dx ...)
    - Then data rows (MREV free-wake data is stacked contiguously)

    Columns order in the perf file:
    psi x=r/R dx dCT/dx dCQI/dx dCQP/dx X-force Y-force -Z-force P-mom AOAG CL2D
    CD2D CM2D AOA2D MACH2D U-radial V-aft W-down W-induced Circulation
    """

    # Normalize var_ix: handle both single integer and iterable
    is_scalar = np.ndim(var_ix) == 0
    indices = [var_ix] if is_scalar else list(var_ix)
    nvars = len(indices)

    with open(fname) as f:
        lines = f.readlines()

    # Locate each 'ROTOR' section header
    rotor_delims = [j for j, line in enumerate(lines) if "ROTOR" in line]

    var = np.zeros((nrotor, mrev, npsi, nx, nvars))

    if mrev == 1:
        # NREV only: each rotor block starts after its respective ROTOR header
        for r in range(nrotor):
            start = rotor_delims[r + 1] + 3
            end = start + npsi * nx
            data = np.loadtxt(lines[start:end])
            var[r, 0, :, :, :] = data[:, indices].reshape((npsi, nx, nvars))
    else:
        # MREV: time-stepping data starts after all initial rotor headers/data
        init_end = rotor_delims[nrotor] + 3 + (npsi * nx)
        total_mrev_lines = mrev * npsi * nrotor * nx
        data = np.loadtxt(lines[init_end : init_end + total_mrev_lines])
        data_reshaped = data[:, indices].reshape((mrev, npsi, nrotor, nx, nvars))
        var = np.transpose(data_reshaped, (2, 0, 1, 3, 4))

    psis = np.unique(data[:, 0])
    xs = np.unique(data[:, 1])

    if is_scalar:
        var = var[..., 0]

    return var, psis, xs
    

def plot_polar(psis, xs, polar_data, vmin=None, vmax=None, title=None, fname=None, nrotor=1, spins=["CCW"]):
    """
    Create polar contour plots of rotor data.
    
    Parameters:
    - psis: array of azimuth angles
    - xs: array of radial stations (r/R)
    - polar_data: 3D array of shape (nrotor, npsi, nx).
                  Note: if polar_data comes from read_perf(), average over revolutions
                  (e.g., np.mean(polar_data, axis=1) or np.average(polar_data[..., 0], axis=1))
                  or slice an individual revolution before passing to plot_polar.
    - vmin, vmax: colorbar limits
    - title: plot/colorbar title
    - fname: optional path to save figure (calls plt.show() if None)
    - nrotor: number of rotor subplots to display
    - spins: list of rotor rotation directions ('CW' or 'CCW')
    """

    if np.ndim(polar_data) != 3:
        raise ValueError(
            f"plot_polar expects a 3D array of shape (nrotor, npsi, nx), but received shape {np.shape(polar_data)}.\n"
            "Note: read_perf now outputs all revolutions (4D array).\n"
            "To plot, pass an average: np.mean(data, axis=1), or slice a single rev: data[:, rev, :, :].\n"
            "Please refer to /projectnb/turbomac/arozman/share/examples/ for up-to-date examples."
        )

    # Automatically determine global min and max if not specified
    if vmin is None:
        vmin = np.nanmin(polar_data[:nrotor])
    if vmax is None:
        vmax = np.nanmax(polar_data[:nrotor])

    # Ensure spins has an entry for each rotor
    spins = list(spins or []) + ["CCW"] * max(0, nrotor - len(spins or []))

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
                                constrained_layout=True)
        axs_flat = [axs]
    elif nrotor == 2:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6), 
                                subplot_kw={'projection': 'polar'},
                                constrained_layout=True)
        axs_flat = axs.flatten()
    else:  # 3 or 4 rotors
        fig, axs = plt.subplots(2, 2, figsize=(12, 10), 
                                subplot_kw={'projection': 'polar'},
                                constrained_layout=True)
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

        direction = str(spins[i]).strip().upper()
        if direction == "CCW":
            ax.set_theta_direction(1)
        elif direction == "CW":
            ax.set_theta_direction(-1)
        
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


def extract_charm_aoa_mach(perf_file_path, outputs_dir="photos_dir", npy_dir="polars_dir", spins=None):
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

    # Read and plot AoA (averaged across revolutions)
    var_ix = 14
    polar_aoa, psis, xs = read_perf(perf_file_path, nrotor, npsi, nx, mrev, var_ix)
    polar_aoa_mean = np.mean(polar_aoa, axis=1)
    plot_polar(psis, xs, polar_aoa_mean, vmin=-10, vmax=10, title='Local Airfoil AoA (deg)', 
               fname=os.path.join(outputs_dir, 'polar_aoa.png'), nrotor=nrotor, spins=spins)

    # Read and plot Mach (averaged across revolutions)
    var_ix = 15
    polar_mach, psis, xs = read_perf(perf_file_path, nrotor, npsi, nx, mrev, var_ix)
    polar_mach_mean = np.mean(polar_mach, axis=1)
    plot_polar(psis, xs, polar_mach_mean, vmin=0, vmax=0.4, title='Local Airfoil Mach', 
               fname=os.path.join(outputs_dir, 'polar_mach.png'), nrotor=nrotor, spins=spins)

    # Save arrays
    np.save(os.path.join(npy_dir, "polar_aoa.npy"), polar_aoa_mean)
    np.save(os.path.join(npy_dir, "polar_mach.npy"), polar_mach_mean)
    np.savetxt(os.path.join(npy_dir, "radial_locs.txt"), xs)
    
    print("Saved aoa and mach arrays to npy_files/")
    print("Saved plots: polar_aoa.png and polar_mach.png")

    plt.close()
    plt.close()

    return polar_aoa_mean, polar_mach_mean, xs, nrotor, npsi
