#!/usr/bin/env python3
"""
bwi_polar.py — Flexible Multi-Arc CHARM BWI Polar Plotter & Locus Extractor

Reads CHARM jobybwi.dat, extracts sectional blade pressure perturbations (|DelP-Blade|),
filters tip vortex trailers, identifies close encounters, and implements Method 3
(2D-Smoothed Radial Multi-Peak Tracking) to automatically recognize and separate
multiple concurrent BWI arcs across arbitrary inflow conditions and multi-rotor geometries.

Key Capabilities:
  1. Flexible & Scale-Free Multi-Arc Detection (Method 3):
     At each azimuth, detects radial peaks in the smoothed |DelP| field and links them
     into continuous BWI arc trajectories across azimuth (e.g. outer, middle, inner arcs).
  2. Multi-Rotor Support:
     Automatically detects single-rotor and multi-rotor runs (e.g. tandem rotors with Type=1, 2).
     Allows analyzing individual rotors or all rotors sequentially.
  3. Span Boundary Filtering (R_MAX = 0.95):
     Excludes the non-physical collocation boundary spike at r/R = 0.99 where blades graze
     the wake cylinder, preserving true acoustic BVI interactions.
  4. Analytical Multi-Arc Locus Export:
     Exports the complete physical state along each recognized arc
     (arc_id, psi, r_centroid, DelP_mean, MissDist, VtxCore, Gamma, Perp, Age, etc.)
     to a clean CSV file and structured dictionary for downstream analytical modeling.
  5. Aerodynamic Airload Comparison:
     Optionally overlays |dFz/dpsi| from jobyperf.dat for the matching rotor.
"""

import os
import sys
import io
import argparse
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Add shared folder to access charm_functions.py (for jobyperf.dat integration)
SHARE_DIR = Path("/projectnb/turbomac/arozman/share")
if SHARE_DIR.exists() and str(SHARE_DIR) not in sys.path:
    sys.path.append(str(SHARE_DIR))

try:
    from charm_functions import read_perf_header, read_perf
except ImportError:
    read_perf_header = None
    read_perf = None


# ==============================================================================
# USER CONFIGURATION PARAMETERS (Modify or override via CLI arguments)
# ==============================================================================

# Input / Output File Paths
BWI_FILE = "jobybwi.dat"              # CHARM BWI interaction file
PERF_FILE = "jobyperf.dat"            # CHARM performance airload file
OUTPUT_PLOT = "bwi_polar.png"         # Output polar plot image file
OUTPUT_CSV = "bwi_events.csv" # Output CSV file for the centroid trajectory

# ------------------------------------------------------------------------------
# 1. Flexible Rotor & Inflow Parameters
# ------------------------------------------------------------------------------
ROTOR_ID = 1                          # Rotor to analyze: 1, 2, ... or 'all' for multi-rotor cases
ROTOR_RPM = None                      # None: auto-detect from jobybwi.dat Omega column
REV_MODE = 'average'                  # 'average' (mean over converged revs), 'last_rev', 'all_raw'
N_REVS_AVG = 5                        # Number of final converged revolutions to analyze (e.g. 5)

# ------------------------------------------------------------------------------
# 2. Flexible Tip Vortex & Span Cutoff Controls
# ------------------------------------------------------------------------------
R_MAX = 0.95                          # Max blade span fraction to ignore artificial tip boundary spike
MAX_CORE_FT = 0.25                    # Maximum physical core radius [ft] to isolate tip vortices
MIN_NV = None                         # Optional trailer index filter (None: auto-filter via MAX_CORE_FT)
IGNORE_SAME_BLADE = False             # True: ignore blade self-interactions (NBB == NB)

# ------------------------------------------------------------------------------
# 3. Strike Encounters & Method 3 Multi-Arc Tracking Controls
# ------------------------------------------------------------------------------
SHOW_STRIKES = True                   # True: plot filtered strike dots AND multi-arc paths
MAX_STRIKE_MISS_FT = 0.05             # Miss distance threshold [ft] for strikes (0.05 ft = 0.60 in)
DOT_MIN_DELP_FRAC = 0.12              # Minimum fraction of peak |DelP| for plotting white strike dots

# Method 3 Radial Multi-Peak Tracking Parameters:
#   At each azimuth station psi_i, the tracker inspects the 1D radial profile |DelP(r)|
#   to locate local radial peaks (BWI ridges) and links them across adjacent azimuths:
#
#   - MIN_GLOBAL_CUTOFF_FRAC (default 0.10):
#       Minimum peak height as a fraction of the rotor-wide 99th-percentile |DelP|.
#       Acts as a global noise floor so quiet azimuths (e.g. wake shadow) produce no peaks.
#       * Increase (e.g. 0.15-0.25) to ignore weaker secondary/retreating arcs.
#       * Decrease (e.g. 0.05) to capture faint inner or downwind arcs.
#
#   - MIN_REL_SLICE_HEIGHT (default 0.18):
#       Minimum peak height as a fraction of the strongest peak at that SAME azimuth psi_i.
#       * Increase (e.g. 0.30-0.40) to suppress minor radial ripples next to a dominant arc.
#       * Decrease (e.g. 0.10-0.12) if a weaker inner arc is overshadowed by a strong outer arc.
#
#   - MIN_PEAK_PROM_FRAC (default 0.12):
#       Minimum vertical prominence (valley depth between adjacent radial peaks at the same
#       azimuth) as a fraction of the local slice maximum.
#       * Increase (e.g. 0.20) to merge closely blended double-ridges into a single arc.
#       * Decrease (e.g. 0.05) to split closely spaced parallel arcs separated by a shallow valley.
#
#   - MIN_PEAK_DIST_R (default 0.06):
#       Minimum radial separation dr/R between simultaneous peaks at the same azimuth.
#       * Increase (e.g. 0.08-0.10) to enforce wider radial spacing between separate arcs.
#       * Decrease (e.g. 0.03) if two physical BWI arcs lie very close in span.
#
#   - MAX_LINK_DR (default 0.055):
#       Maximum radial step |r(psi_{i+1}) - r(psi_i)| allowed when linking peaks from one
#       azimuth bin to the next.
#       * Increase (e.g. 0.08-0.10) if steeply sweeping oblique BVI arcs break into segments.
#       * Decrease (e.g. 0.04) to prevent jumping between adjacent parallel arcs.
#
#   - LOCAL_R_WINDOW (default 0.03):
#       Radial half-window (+/- LOCAL_R_WINDOW) around each detected ridge peak r_cen used to
#       compute the |DelP|-weighted centroid and average physical vortex properties (Age,
#       VtxCore, MissDist, Gamma). Kept tight (~1 radial station on each side) so adjacent
#       parallel arcs never pull each other's centroids.
#
#   - MIN_ARC_SPAN_DEG (default 25.0):
#       Minimum continuous azimuthal length [deg] required to retain a linked track as an arc.
#       * Increase (e.g. 45.0-60.0) to filter out short localized strike clusters.
#       * Decrease (e.g. 15.0) to retain brief, localized blade-vortex encounters.
MIN_GLOBAL_CUTOFF_FRAC = 0.10
MIN_REL_SLICE_HEIGHT = 0.18
MIN_PEAK_PROM_FRAC = 0.12
MIN_PEAK_DIST_R = 0.06
MAX_LINK_DR = 0.055
LOCAL_R_WINDOW = 0.03
MIN_ARC_SPAN_DEG = 25.0

# ------------------------------------------------------------------------------
# 4. Aerodynamic Airload Comparison Option
# ------------------------------------------------------------------------------
SHOW_PERF_DFZDPSI = False              # True: 2-panel comparison with |dFz/dpsi|; False: single polar

# Standard 26 columns in CHARM jobybwi.dat
BWI_COLUMNS = [
    'IBWI', 'Type', 'ITE', 'NBB', 'NAZ', 'NRB', 'N1', 'NV', 'ZONE', 'NB',
    'NAZ_src', 'NR_src', 'MissDist', 'GammaVortex', 'DelP_Blade', 'VtxCore',
    'Para', 'Perp', 'Age', 'Omega', 'r_R', 'PSI', 'u', 'v', 'w', 'Time'
]

# High-contrast color palette for distinct BWI arcs
ARC_PALETTE = ['#00ffff', '#39ff14', '#ffdd00', '#ff3366', '#b026ff', '#ff8800', '#00d4ff']


# ==============================================================================
# DATA EXTRACTION & MULTI-ARC IDENTIFICATION
# ==============================================================================

def extract_bwi_data(bwi_path=BWI_FILE, perf_path=PERF_FILE, rotor_id=ROTOR_ID,
                      r_max=R_MAX, max_core=MAX_CORE_FT, min_nv=MIN_NV,
                      rev_mode=REV_MODE, n_revs=N_REVS_AVG, rpm=ROTOR_RPM,
                      max_strike_miss=MAX_STRIKE_MISS_FT, dot_min_delp_frac=DOT_MIN_DELP_FRAC,
                      npsi_grid=None, nr_grid=60, csv_output=OUTPUT_CSV):
    """
    Reads jobybwi.dat, applies multi-rotor isolation, core filtering, and R_MAX cutoff,
    uses read_perf_header from charm_functions.py to align azimuthal bins with CHARM's
    native NPSI discretization, builds the revolution-averaged 2D polar grid, and applies
    Method 3 Multi-Peak Tracking to extract continuous separable BWI arcs.
    """
    bwi_path = Path(bwi_path)
    perf_path = Path(perf_path) if perf_path is not None else None
    if not bwi_path.exists():
        raise FileNotFoundError(f"CHARM BWI file not found: {bwi_path.resolve()}")

    # Auto-detect native azimuthal discretization NPSI using shared read_perf_header()
    if npsi_grid is None:
        if perf_path is not None and perf_path.exists() and read_perf_header is not None:
            try:
                _, npsi_perf, _, _ = read_perf_header(str(perf_path))
                if npsi_perf > 0:
                    npsi_grid = int(npsi_perf)
                    print(f"  Read native azimuthal resolution from {perf_path.name} via read_perf_header(): NPSI = {npsi_grid}")
            except Exception as e:
                print(f"  [Notice] Could not read NPSI from {perf_path.name} ({e}); falling back to BWI header.")

        if npsi_grid is None:
            # Fallback: parse 'NPSI = <int>' from first line of jobybwi.dat
            with open(bwi_path, 'r', errors='ignore') as f_hdr:
                first_line = f_hdr.readline()
            if 'NPSI' in first_line and '=' in first_line:
                try:
                    npsi_grid = int(first_line.split('=')[1].split()[0])
                    print(f"  Read native azimuthal resolution from {bwi_path.name} header: NPSI = {npsi_grid}")
                except Exception:
                    npsi_grid = 40
            else:
                npsi_grid = 40

    file_size_mb = bwi_path.stat().st_size / (1024 * 1024)
    print(f"Reading BWI file: {bwi_path.name} ({file_size_mb:.1f} MB)...")

    # Fast tail buffer for large files (>50 MB)
    if file_size_mb > 50.0 and rev_mode != 'all_raw':
        tail_lines = 1200000
        cmd = ['tail', '-n', str(tail_lines), str(bwi_path)]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, _ = proc.communicate()
        first_nl = out.find(b'\n')
        if first_nl != -1:
            out = out[first_nl + 1:]
        df = pd.read_csv(io.BytesIO(out), header=None, names=BWI_COLUMNS, sep=r'\s+', low_memory=False)
    else:
        df = pd.read_csv(bwi_path, skiprows=2, header=None, names=BWI_COLUMNS, sep=r'\s+', low_memory=False)

    if len(df) == 0:
        raise ValueError(f"No BWI interaction data found in {bwi_path}")

    # Robust numeric cleaning for all columns (drops intermediate periodic header lines)
    for col in BWI_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Time', 'NRB', 'DelP_Blade']).copy()

    int_cols = ['IBWI', 'Type', 'ITE', 'NBB', 'NAZ', 'NRB', 'N1', 'NV', 'ZONE', 'NB', 'NAZ_src', 'NR_src']
    for int_col in int_cols:
        if int_col in df.columns:
            df[int_col] = df[int_col].astype(int)

    # Multi-Rotor Detection & Selection based on Blade Rotor ID (NRB)
    available_rotors = sorted(df['NRB'].unique().tolist())
    if len(available_rotors) > 1:
        print(f"  Multi-Rotor case detected! Available Blade Rotors in BWI file: {available_rotors}")
        if rotor_id not in available_rotors:
            print(f"  [Notice] Specified Rotor {rotor_id} not in {available_rotors}. Defaulting to Rotor {available_rotors[0]}.")
            rotor_id = available_rotors[0]
        df_rotor = df[df['NRB'] == rotor_id].copy()
        print(f"  Selected Blade Rotor {rotor_id} ({len(df_rotor):,} events)")
    else:
        rotor_id = available_rotors[0]
        df_rotor = df.copy()

    # Auto-detect Rotor RPM from Omega if not specified
    if rpm is None:
        omega_val = df_rotor['Omega'].iloc[0]
        rpm = omega_val * 60.0 / (2.0 * np.pi)
        print(f"  Auto-detected Rotor RPM from Omega ({omega_val:.2f} rad/s): {rpm:.2f} RPM")
    else:
        print(f"  Using specified Rotor RPM: {rpm:.2f} RPM")

    t_rev = 60.0 / rpm if rpm > 0 else 1.0
    t_max = df_rotor['Time'].max()
    t_min = df_rotor['Time'].min()
    total_revs = max(1.0, (t_max - t_min) / t_rev)

    # Slice revolutions
    if rev_mode == 'last_rev':
        df_active = df_rotor[df_rotor['Time'] >= max(0.0, t_max - t_rev)].copy()
        n_revs_norm = 1.0
        print(f"  Mode: 'last_rev' -> Last revolution ({len(df_active):,} events)")
    elif rev_mode == 'average':
        if n_revs is not None and total_revs > n_revs:
            t_slice_min = t_max - float(n_revs) * t_rev
            df_active = df_rotor[df_rotor['Time'] >= t_slice_min].copy()
            n_revs_norm = float(n_revs)
            print(f"  Mode: 'average'  -> Averaged across last {n_revs_norm:.1f} converged revs ({len(df_active):,} events)")
        else:
            df_active = df_rotor.copy()
            n_revs_norm = total_revs
            print(f"  Mode: 'average'  -> Averaged across {total_revs:.1f} revs ({len(df_active):,} events)")
    else:  # 'all_raw'
        df_active = df_rotor.copy()
        n_revs_norm = 1.0
        print(f"  Mode: 'all_raw'  -> All raw events across {total_revs:.1f} revs ({len(df_active):,} events)")

    # Tip Trailer & Span Cutoff Filtering (R_MAX = 0.95 suppresses tip collocation boundary spike)
    mask_tip = (df_active['VtxCore'] < max_core) & (df_active['VtxCore'] > 0.0)
    if r_max is not None:
        mask_tip &= (df_active['r_R'] <= r_max)
    if min_nv is not None:
        mask_tip &= (df_active['NV'] >= min_nv)
    if IGNORE_SAME_BLADE:
        mask_tip &= (df_active['NBB'] != df_active['NB'])

    tip_df = df_active[mask_tip].copy()
    print(f"  Tip vortex events (Core < {max_core:.2f} ft, r/R <= {r_max:.2f}): {len(tip_df):,}")

    # Close strikes
    mask_strike = (tip_df['MissDist'] <= max_strike_miss)
    raw_strike_df = tip_df[mask_strike].copy()
    print(f"  Identified close strikes (Miss <= {max_strike_miss*12:.1f} in): {len(raw_strike_df):,}")

    # Build 2D Polar Grid centered on CHARM's native discrete azimuths [0, dpsi, ..., 360-dpsi]
    # and native discrete radial stations in tip_df['r_R'] to prevent radial bin aliasing
    r_data_min = tip_df['r_R'].min() if len(tip_df) > 0 else 0.20
    r_grid_min = max(0.20, np.floor(r_data_min * 10.0) / 10.0)
    r_grid_max = r_max if r_max is not None else 1.0

    dpsi = 360.0 / npsi_grid
    psi_edges = np.linspace(-0.5 * dpsi, 360.0 - 0.5 * dpsi, npsi_grid + 1)
    psi_centers = np.linspace(0.0, 360.0 - dpsi, npsi_grid)

    r_unique = np.sort(np.round(tip_df['r_R'].values, 4))
    r_unique = np.unique(r_unique)
    if len(r_unique) >= 5:
        r_centers = r_unique
        nr_grid = len(r_centers)
        dr_mid = 0.5 * (r_centers[:-1] + r_centers[1:])
        dr_first = r_centers[1] - r_centers[0]
        dr_last = r_centers[-1] - r_centers[-2]
        r_edges = np.concatenate([[r_centers[0] - 0.5 * dr_first], dr_mid, [r_centers[-1] + 0.5 * dr_last]])
        dr_bin = float(np.median(np.diff(r_centers)))
    else:
        r_edges = np.linspace(r_grid_min, r_grid_max, nr_grid + 1)
        r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
        dr_bin = (r_grid_max - r_grid_min) / nr_grid

    # Wrap PSI into [-0.5*dpsi, 360 - 0.5*dpsi) so each CHARM azimuth falls in the center of its bin
    psi_vals = ((tip_df['PSI'].values + 0.5 * dpsi) % 360.0) - 0.5 * dpsi
    r_vals = tip_df['r_R'].values
    weights = np.abs(tip_df['DelP_Blade'].values)

    grid_raw, _, _ = np.histogram2d(psi_vals, r_vals, bins=[psi_edges, r_edges], weights=weights)
    grid_raw = grid_raw / n_revs_norm

    pad = 6
    grid_pad = np.pad(grid_raw, ((pad, pad), (0, 0)), mode='wrap')
    grid_smooth = ndi.gaussian_filter(grid_pad, sigma=(1.0, 1.0))[pad:-pad, :]
    global_vmax = np.percentile(grid_smooth[grid_smooth > 0], 99.0) if np.any(grid_smooth > 0) else 1.0

    # ==========================================================================
    # METHOD 3: RADIAL MULTI-PEAK DETECTION & TRAJECTORY LINKING
    # ==========================================================================
    min_dist_bins = max(2, int(round(MIN_PEAK_DIST_R / dr_bin)))

    peaks_by_psi = []
    for i in range(npsi_grid):
        r_slice = grid_smooth[i, :]
        s_max = r_slice.max()
        if s_max < MIN_GLOBAL_CUTOFF_FRAC * global_vmax:
            peaks_by_psi.append([])
            continue

        # Pad outer boundary with 0.0 so peaks sitting on the outermost radial station are detected
        r_slice_padded = np.append(r_slice, 0.0)
        pks, _ = find_peaks(
            r_slice_padded,
            height=max(MIN_REL_SLICE_HEIGHT * s_max, MIN_GLOBAL_CUTOFF_FRAC * global_vmax),
            prominence=MIN_PEAK_PROM_FRAC * s_max,
            distance=min_dist_bins
        )
        peaks_by_psi.append([{'r': r_centers[p], 'psi': psi_centers[i], 'val': r_slice[p], 'psi_idx': i} for p in pks if p < len(r_centers)])

    # Nearest-neighbor linking across azimuth
    active_tracks = []
    finished_tracks = []

    for i in range(npsi_grid):
        current_peaks = peaks_by_psi[i]
        matched_tracks = set()
        matched_peaks = set()

        if active_tracks and current_peaks:
            for t_idx, track in enumerate(active_tracks):
                last_r = track['r'][-1]
                best_p_idx = None
                best_dr = float('inf')
                for p_idx, pk in enumerate(current_peaks):
                    if p_idx in matched_peaks:
                        continue
                    dr = abs(pk['r'] - last_r)
                    if dr < best_dr and dr <= MAX_LINK_DR:
                        best_dr = dr
                        best_p_idx = p_idx

                if best_p_idx is not None:
                    track['psi'].append(current_peaks[best_p_idx]['psi'])
                    track['r'].append(current_peaks[best_p_idx]['r'])
                    track['val'].append(current_peaks[best_p_idx]['val'])
                    matched_tracks.add(t_idx)
                    matched_peaks.add(best_p_idx)

        surviving = []
        for t_idx, track in enumerate(active_tracks):
            if t_idx in matched_tracks:
                surviving.append(track)
            else:
                finished_tracks.append(track)
        active_tracks = surviving

        for p_idx, pk in enumerate(current_peaks):
            if p_idx not in matched_peaks:
                active_tracks.append({
                    'psi': [pk['psi']],
                    'r': [pk['r']],
                    'val': [pk['val']]
                })

    finished_tracks.extend(active_tracks)

    # Filter by minimum span and rank by total impulse
    valid_tracks = []
    for track in finished_tracks:
        psi_arr = np.array(track['psi'])
        r_arr = np.array(track['r'])
        val_arr = np.array(track['val'])
        span_deg = psi_arr.max() - psi_arr.min()
        if span_deg >= MIN_ARC_SPAN_DEG and len(psi_arr) >= 3:
            valid_tracks.append({
                'psi_arr': psi_arr,
                'r_arr': r_arr,
                'val_arr': val_arr,
                'total_val': val_arr.sum(),
                'span_deg': span_deg,
                'r_min': r_arr.min(),
                'r_max': r_arr.max(),
                'mean_r': r_arr.mean()
            })

    valid_tracks.sort(key=lambda x: x['total_val'], reverse=True)
    print(f"  Method 3 identified {len(valid_tracks)} continuous BWI arc trajectories:")

    # Build Centroid Locus Table for all identified arcs
    arc_data_list = []
    all_locus_rows = []

    for arc_idx, tr in enumerate(valid_tracks):
        arc_id = arc_idx + 1
        arc_rows = []

        for p_deg, r_cen in zip(tr['psi_arr'], tr['r_arr']):
            # Count close strikes in the local vicinity of this ridge peak
            half_dpsi = 0.5 * dpsi + 1e-4
            dpsi_strikes = np.abs(((raw_strike_df['PSI'].values - p_deg + 180.0) % 360.0) - 180.0)
            sub_strikes = raw_strike_df[
                (dpsi_strikes <= half_dpsi) &
                (np.abs(raw_strike_df['r_R'].values - r_cen) <= LOCAL_R_WINDOW)
            ]
            n_str = len(sub_strikes)

            # Extract |DelP|-weighted centroid and vortex properties from tip_df within a tight
            # radial window (+/- LOCAL_R_WINDOW) around r_cen so adjacent arcs never pull each other
            dpsi_tip = np.abs(((tip_df['PSI'].values - p_deg + 180.0) % 360.0) - 180.0)
            sub = tip_df[
                (dpsi_tip <= half_dpsi) &
                (np.abs(tip_df['r_R'].values - r_cen) <= LOCAL_R_WINDOW)
            ]

            if len(sub) > 0:
                w = np.abs(sub['DelP_Blade'].values)
                w_sum = np.sum(w) if np.sum(w) > 0 else 1.0
                r_weighted = np.sum(sub['r_R'].values * w) / w_sum
                miss_bar = np.sum(sub['MissDist'].values * w) / w_sum
                core_bar = np.sum(sub['VtxCore'].values * w) / w_sum
                gamma_bar = np.sum(sub['GammaVortex'].values * w) / w_sum
                perp_bar = np.sum(sub['Perp'].values * w) / w_sum
                age_bar = np.sum(sub['Age'].values * w) / w_sum
                delp_mean = np.mean(w)
                delp_max = np.max(w)
                blade_prim = sub['NBB'].mode()[0] if len(sub) > 0 else 1
                vtx_blade = sub['NB'].mode()[0] if len(sub) > 0 else 1
            else:
                r_weighted = r_cen
                miss_bar = 0.0
                core_bar = 0.0
                gamma_bar = 0.0
                perp_bar = 0.0
                age_bar = 0.0
                delp_mean = tr['val_arr'][np.where(tr['psi_arr'] == p_deg)[0][0]]
                delp_max = delp_mean
                blade_prim = 1
                vtx_blade = 1

            row = {
                'arc_id': arc_id,
                'rotor_id': rotor_id,
                'psi_deg': p_deg,
                'r_centroid': r_weighted,
                'delp_blade_mean': delp_mean,
                'delp_blade_max': delp_max,
                'miss_dist_ft': miss_bar,
                'miss_dist_in': miss_bar * 12.0,
                'vortex_core_ft': core_bar,
                'vortex_gamma': gamma_bar,
                'sweep_perp': perp_bar,
                'vortex_age_rev': age_bar,
                'n_strikes': n_str,
                'primary_blade': blade_prim,
                'vortex_blade': vtx_blade
            }
            arc_rows.append(row)
            all_locus_rows.append(row)

        arc_df = pd.DataFrame(arc_rows)
        mean_age = arc_df['vortex_age_rev'].mean()
        print(f"    Arc {arc_id}: r/R={tr['r_min']:.2f}–{tr['r_max']:.2f} (ψ={tr['psi_arr'].min():.0f}–{tr['psi_arr'].max():.0f}°, Age={mean_age:.2f} rev, {len(arc_df)} stations)")
        arc_data_list.append({
            'arc_id': arc_id,
            'rotor_id': rotor_id,
            'r_span': (tr['r_min'], tr['r_max']),
            'psi_span': (tr['psi_arr'].min(), tr['psi_arr'].max()),
            'mean_age': mean_age,
            'df': arc_df
        })

    locus_df = pd.DataFrame(all_locus_rows)

    # Save to CSV
    if csv_output and len(locus_df) > 0:
        csv_file = Path(csv_output)
        locus_df.to_csv(csv_file, index=False)
        print(f"  Saved Multi-Arc Centroid Locus ({len(locus_df)} points across {len(valid_tracks)} arcs) to: {csv_file.resolve()}")

    # Package clean data structures
    dot_mask = (np.abs(raw_strike_df['DelP_Blade']) >= dot_min_delp_frac * np.abs(raw_strike_df['DelP_Blade']).max())
    vis_strike_df = raw_strike_df[dot_mask].copy()

    bwi_data = {
        'rotor_id': rotor_id,
        'available_rotors': available_rotors,
        'rev_mode': rev_mode,
        'rpm': rpm,
        'n_revs': n_revs_norm,
        'r_max': r_grid_max,

        # 2D Binned Grid
        'psi_grid': psi_centers,
        'r_grid': r_centers,
        'delp_grid': grid_smooth,
        'vmax': global_vmax,

        # Filtered strike scatter dots
        'vis_strikes': {
            'psi_deg': vis_strike_df['PSI'].to_numpy(dtype=float),
            'r_over_R': vis_strike_df['r_R'].to_numpy(dtype=float),
            'delp_psf': np.abs(vis_strike_df['DelP_Blade'].to_numpy(dtype=float))
        },

        # Separable Multi-Arc Data Structures
        'arcs': arc_data_list,
        'centroid_df': locus_df
    }

    return bwi_data


def extract_perf_airload(perf_path=PERF_FILE, rotor_id=ROTOR_ID):
    """
    Extracts sectional Fz and computes azimuthal loading rate |dFz/dpsi| from jobyperf.dat
    using the shared charm_functions.py module. Supports single and multi-rotor files.
    """
    perf_path = Path(perf_path)
    if not perf_path.exists():
        print(f"Notice: Performance file {perf_path} not found. Skipping airload panel.")
        return None

    if read_perf_header is None or read_perf is None:
        print("Notice: charm_functions.py not available. Skipping airload panel.")
        return None

    nrotor, npsi, nx, mrev = read_perf_header(str(perf_path))
    polar_data, psis, xs = read_perf(str(perf_path), nrotor, npsi, nx, mrev, [8])

    rotor_idx = max(0, min(nrotor - 1, (rotor_id - 1) if isinstance(rotor_id, int) else 0))
    Fz = np.average(polar_data[rotor_idx, ..., 0], axis=0)
    daz = 360.0 / npsi
    Fz_cyclic = np.concatenate((Fz, Fz[0:1, :]), axis=0)
    dFzdpsi = np.diff(Fz_cyclic, axis=0) / daz

    return {
        'rotor_id': rotor_idx + 1,
        'nrotor': nrotor,
        'psis': psis,
        'xs': xs,
        'Fz': Fz,
        'dFzdpsi': np.abs(dFzdpsi),
        'npsi': npsi,
        'nx': nx
    }


# ==============================================================================
# PLOTTING FUNCTIONS
# ==============================================================================

def plot_bwi_polar(bwi_data, perf_data=None, output_path=OUTPUT_PLOT, show_strikes=SHOW_STRIKES):
    """
    Renders the polar contour plot displaying:
      - 2D BWI pressure perturbation contour (|DelP-Blade|) up to R_MAX
      - Filtered strike encounter dots (semi-transparent scatter)
      - Continuous Multi-Arc Trajectory curves (Method 3) in high-contrast distinct colors
      - Optional side-by-side |dFz/dpsi| comparison panel from jobyperf.dat
    """
    psi_grid = bwi_data['psi_grid']
    r_grid = bwi_data['r_grid']
    delp_grid = bwi_data['delp_grid']
    vmax = bwi_data['vmax']
    arcs = bwi_data['arcs']
    vis_strikes = bwi_data['vis_strikes']
    rotor_id = bwi_data['rotor_id']
    n_revs = bwi_data['n_revs']
    rev_mode = bwi_data['rev_mode']
    r_max = bwi_data['r_max']

    theta_rad = np.radians(psi_grid)
    theta_cyclic = np.append(theta_rad, theta_rad[0] + 2.0 * np.pi)
    Theta, R_mesh = np.meshgrid(theta_cyclic, r_grid, indexing='ij')
    delp_cyclic = np.vstack((delp_grid, delp_grid[0, :]))

    has_perf = (perf_data is not None)
    if has_perf:
        fig, (ax_bwi, ax_perf) = plt.subplots(1, 2, figsize=(13.5, 7.5),
                                              subplot_kw={'projection': 'polar'},
                                              constrained_layout=True)
    else:
        fig, ax_bwi = plt.subplots(1, 1, figsize=(6.2, 7.5),
                                   subplot_kw={'projection': 'polar'},
                                   constrained_layout=True)
        ax_perf = None

    # Panel 1: CHARM Built-in |DelP-Blade| Polar Contour + Multi-Arc Overlay
    c_bwi = ax_bwi.pcolormesh(Theta, R_mesh, delp_cyclic, cmap='inferno',
                              shading='gouraud', vmin=0.0, vmax=vmax)

    ax_bwi.set_theta_direction(1)       # Counter-clockwise rotation
    ax_bwi.set_theta_zero_location('E') # 0 deg at East
    ax_bwi.set_ylim(0.0, 1.0)
    ax_bwi.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_bwi.set_rlabel_position(22.5)
    ax_bwi.tick_params(axis='x', labelsize=10)
    ax_bwi.tick_params(axis='y', labelsize=8, colors='lightgray')
    ax_bwi.grid(True, linestyle=':', alpha=0.5, color='gray')

    # Scatter filtered strike dots
    if show_strikes and len(vis_strikes['psi_deg']) > 0:
        dot_theta = np.radians(vis_strikes['psi_deg'])
        dot_r = vis_strikes['r_over_R']
        ax_bwi.scatter(dot_theta, dot_r, s=12, color='white', edgecolor='black',
                       linewidth=0.3, alpha=0.35, zorder=3,
                       label=f"Strikes (|ΔP|>{DOT_MIN_DELP_FRAC*100:.0f}%, N={len(dot_r):,})")

    # Overlay distinct Multi-Arc Trajectories (Method 3)
    for idx, arc in enumerate(arcs):
        col = ARC_PALETTE[idx % len(ARC_PALETTE)]
        arc_df = arc['df']
        arc_theta = np.radians(arc_df['psi_deg'].values)
        arc_r = arc_df['r_centroid'].values

        ax_bwi.plot(arc_theta, arc_r, color=col, linewidth=2.6, zorder=5,
                    label=f"Arc {arc['arc_id']}: r/R={arc['r_span'][0]:.2f}–{arc['r_span'][1]:.2f} (ψ={arc['psi_span'][0]:.0f}–{arc['psi_span'][1]:.0f}°, Age={arc['mean_age']:.2f} rev)")
        ax_bwi.scatter(arc_theta, arc_r, s=36, color=col, edgecolor='black',
                       linewidth=0.9, zorder=6)

    ax_bwi.legend(loc='lower left', bbox_to_anchor=(-0.05, -0.22), fontsize=8.5, frameon=True)

    mode_title_desc = {
        'average': f"Rev-Averaged ({n_revs:.1f} revs)",
        'last_rev': "Last Revolution Snapshot",
        'all_raw': f"All Revs Raw Sum ({n_revs:.1f} revs)"
    }.get(rev_mode, rev_mode)

    rotor_title = f" (Rotor {rotor_id})" if len(bwi_data['available_rotors']) > 1 else ""
    ax_bwi.set_title(f"CHARM BWI Severity: Multi-Arc |DelP-Blade|{rotor_title}\n"
                     f"[{mode_title_desc} | Core < {MAX_CORE_FT:.2f} ft | r/R ≤ {r_max:.2f}]",
                     fontsize=11, weight='bold', pad=15)

    cb_bwi = fig.colorbar(c_bwi, ax=ax_bwi, orientation='horizontal', fraction=0.046, pad=0.1, shrink=0.85)
    cb_bwi.set_label(r"Mean Delta Pressure $\langle |\Delta P_{\mathrm{blade}}| \rangle$ [$\mathrm{lbf/ft^2/rev}$]", fontsize=9)
    cb_bwi.ax.tick_params(labelsize=8)

    # Panel 2: Aerodynamic Loading Rate |dFz/dpsi| from jobyperf.dat
    if has_perf:
        perf_theta = np.radians(perf_data['psis'])
        perf_theta_cyclic = np.append(perf_theta, perf_theta[0] + 2.0 * np.pi)
        Perf_Theta, Perf_R = np.meshgrid(perf_theta_cyclic, perf_data['xs'], indexing='ij')
        perf_cyclic = np.vstack((perf_data['dFzdpsi'], perf_data['dFzdpsi'][0, :]))

        vmax_perf = np.percentile(perf_data['dFzdpsi'], 99.0)
        c_perf = ax_perf.pcolormesh(Perf_Theta, Perf_R, perf_cyclic, cmap='turbo',
                                    shading='gouraud', vmin=0.0, vmax=vmax_perf)

        ax_perf.set_theta_direction(1)
        ax_perf.set_theta_zero_location('E')
        ax_perf.set_ylim(0.0, 1.0)
        ax_perf.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax_perf.set_rlabel_position(22.5)
        ax_perf.tick_params(axis='x', labelsize=10)
        ax_perf.tick_params(axis='y', labelsize=8, colors='dimgray')
        ax_perf.grid(True, linestyle=':', alpha=0.6)

        ax_perf.set_title(f"Aero Loading Rate |dFz/dpsi| (Rotor {perf_data['rotor_id']})\n"
                          f"[from jobyperf.dat | {perf_data['npsi']} azimuths x {perf_data['nx']} radial]",
                          fontsize=11, weight='bold', pad=15)

        cb_perf = fig.colorbar(c_perf, ax=ax_perf, orientation='horizontal', fraction=0.046, pad=0.1, shrink=0.85)
        cb_perf.set_label(r"Loading Rate $|dF_z/d\psi|$ [$\mathrm{lbf/rad}$]", fontsize=9)
        cb_perf.ax.tick_params(labelsize=8)

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved Multi-Arc Polar Plot to: {Path(output_path).resolve()}")
    #plt.show()


# ==============================================================================
# MAIN WORKFLOW & CLI INTERFACE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="CHARM Multi-Arc BWI Polar Plotter & Locus Extractor")
    parser.add_argument("--case-dir", type=str, default=".", help="Directory containing jobybwi.dat & jobyperf.dat")
    parser.add_argument("--rotor-id", default=ROTOR_ID, help="Rotor ID to process (1, 2, ... or 'all')")
    parser.add_argument("--r-max", type=float, default=R_MAX, help="Max blade span fraction (default: 0.95)")
    parser.add_argument("--n-revs", type=int, default=N_REVS_AVG, help="Converged revolutions to average (default: 5)")
    parser.add_argument("--output-plot", type=str, default=OUTPUT_PLOT, help="Output plot filename")
    parser.add_argument("--output-csv", type=str, default=OUTPUT_CSV, help="Output CSV filename")
    args = parser.parse_args()

    case_path = Path(args.case_dir)
    bwi_path = case_path / BWI_FILE
    perf_path = case_path / PERF_FILE

    print("=" * 76)
    print("CHARM BWI Multi-Arc Extraction & Characterization")
    print("=" * 76)

    # Handle multi-rotor processing
    target_rotors = [args.rotor_id] if args.rotor_id != 'all' else [1, 2]

    for rid in target_rotors:
        plot_fname = args.output_plot if len(target_rotors) == 1 else f"bwi_polar_rotor{rid}.png"
        csv_fname = args.output_csv if len(target_rotors) == 1 else f"bwi_events_rotor{rid}.csv"

        bwi_data = extract_bwi_data(
            bwi_path=bwi_path,
            perf_path=perf_path,
            rotor_id=int(rid) if isinstance(rid, str) and rid.isdigit() else rid,
            r_max=args.r_max,
            n_revs=args.n_revs,
            csv_output=csv_fname
        )

        perf_data = extract_perf_airload(perf_path, rotor_id=bwi_data['rotor_id']) if SHOW_PERF_DFZDPSI else None

        plot_bwi_polar(bwi_data, perf_data=perf_data, output_path=plot_fname)

    print("\nExtraction & plotting workflow complete.")


if __name__ == '__main__':
    main()
