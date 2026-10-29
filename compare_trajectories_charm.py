#!/usr/bin/env python3
"""
compare_trajectories_charm.py — Compare Tip Vortex Trajectories: CHARM vs CFD vs Experiment

Hardcoded Inputs:
  - Experiment: /projectnb/turbomac/arozman/projects/ULI/experimental_data/SWT_PIV_Data_Oct-Nov2025/3806RPM_10mps_20F1flap_vortex_traj.mat
  - CFD:        /projectnb/turbomac/arozman/projects/ULI/piv_compare/velocity_and_vortex_data.h5

Usage:
  # Plot default cases:
  python3 compare_trajectories_charm.py

  # Plot 1 specific case:
  python3 compare_trajectories_charm.py CHARM_TILT-20

  # Plot specific multiple cases:
  python3 compare_trajectories_charm.py case1 case2 case3
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
import h5py
import matplotlib.pyplot as plt

# Hardcoded reference file paths
EXP_FILE = "/projectnb/turbomac/arozman/projects/ULI/experimental_data/SWT_PIV_Data_Oct-Nov2025/3806RPM_10mps_20F1flap_vortex_traj.mat"
CFD_FILE = "/projectnb/turbomac/arozman/projects/ULI/piv_compare/velocity_and_vortex_data.h5"

# Geometry and plot limits
D_IN = 24.0
D_MM = 609.6
YAW_DEG = 20.0
TIP_SLOPE = -np.tan(np.radians(YAW_DEG))
XLIM = (-0.32, 0.08)
YLIM = (-0.08, 0.38)
X_BINS = np.linspace(XLIM[0], 0.05, 45)

COLORS = ['#2ca02c', '#ff7f0e', '#7570b3', '#17becf', '#e377c2', '#8c564b', '#bcbd22']


def compute_binned_mean(x_list, z_list):
    """Bins trajectory points along X and returns clean continuous mean Z curve."""
    if not x_list:
        return None, None
    x_all = np.concatenate(x_list)
    z_all = np.concatenate(z_list)
    x_mid = 0.5 * (X_BINS[:-1] + X_BINS[1:])
    z_mean = np.full_like(x_mid, np.nan)
    for i in range(len(x_mid)):
        mask = (x_all >= X_BINS[i]) & (x_all < X_BINS[i + 1])
        if np.sum(mask) >= 3:
            z_mean[i] = np.mean(z_all[mask])
    valid = ~np.isnan(z_mean)
    if not np.any(valid):
        return None, None
    return x_mid[valid], z_mean[valid]


def load_experiment():
    """Loads experimental vortex tracks from hardcoded MAT file."""
    mat = sio.loadmat(EXP_FILE, squeeze_me=True, struct_as_record=False)
    cell = mat['RPM3806_10mps_20F1flap_vortex_traj']
    x_list, z_list = [], []
    for c in cell:
        if len(c.positions) >= 5:
            x_list.append(c.positions[:, 0] / D_MM)
            z_list.append(c.positions[:, 1] / D_MM)
    return compute_binned_mean(x_list, z_list)


def load_cfd():
    """Loads CFD vortex tracks from hardcoded HDF5 file."""
    x_list, z_list = [], []
    with h5py.File(CFD_FILE, 'r') as hf:
        for k in hf['vortices'].keys():
            x = hf['vortices'][k]['X_over_D'][:]
            z = hf['vortices'][k]['Z_over_D'][:]
            if len(x) >= 5:
                x_list.append(x)
                z_list.append(z)
    return compute_binned_mean(x_list, z_list)


def get_case_tilt(case_path):
    """Detects rotor tilt angle (in degrees) from companion Prop_000_000rw.inp or case name."""
    p = Path(case_path)
    case_dir = p if p.is_dir() else p.parent

    # 1. Inspect *rw.inp in case directory
    rw_files = list(case_dir.glob("*rw.inp"))
    if rw_files:
        try:
            with open(rw_files[0]) as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                if "tilt" in line.lower() and i + 1 < len(lines):
                    parts = lines[i + 1].split()
                    if len(parts) >= 6:
                        return abs(float(parts[5]))
        except Exception:
            pass

    # 2. Check filename/folder name for tilt pattern (e.g. tilt-10, TILT-10, tilt0)
    import re
    for s in [p.name, case_dir.name]:
        m = re.search(r'tilt[-_]?(\d+(?:\.\d+)?)', s, re.IGNORECASE)
        if m:
            return float(m.group(1))

    return YAW_DEG


def load_charm(case_path, auto_rotate=True):
    """Loads CHARM vortex crossings and aligns rotor disk tilt to reference frame if needed."""
    if os.path.isdir(case_path):
        cand = Path(case_path) / "vortex_crossings_Y8.4in.csv"
        if cand.exists():
            csv_path = str(cand)
        else:
            csv_candidates = sorted(Path(case_path).glob("vortex_crossings_*.csv"), key=lambda p: p.stat().st_size, reverse=True)
            if not csv_candidates:
                print(f"Warning: No vortex_crossings_*.csv found in {case_path}")
                return None, None
            csv_path = str(csv_candidates[0])
    else:
        csv_path = case_path

    if not os.path.exists(csv_path):
        print(f"Warning: File not found: {csv_path}")
        return None, None

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        print(f"Warning: CSV is empty: {csv_path}")
        return None, None

    # Invert X to downstream convention, normalize by D
    df['X_over_D'] = -df['x_in'] / D_IN
    df['Z_over_D'] = df['z_in'] / D_IN

    # Rotate non-reference tilt angles to align rotor disk with YAW_DEG
    tilt_deg = get_case_tilt(case_path)
    rot_applied = 0.0
    if auto_rotate and abs(tilt_deg - YAW_DEG) > 1e-3:
        delta_deg = YAW_DEG - tilt_deg
        phi = np.radians(-delta_deg)
        x_rot = df['X_over_D'] * np.cos(phi) - df['Z_over_D'] * np.sin(phi)
        z_rot = df['X_over_D'] * np.sin(phi) + df['Z_over_D'] * np.cos(phi)
        df['X_over_D'], df['Z_over_D'] = x_rot, z_rot
        rot_applied = np.degrees(phi)
        print(f"[{Path(case_path).name}] Aligned tilt {tilt_deg:.1f}° to {YAW_DEG:.1f}° disk (rotated {rot_applied:+.1f}°)")

    # Filter primary tip vortex
    df = df[(df['wake_age_deg'] <= 270.0) & (df['X_over_D'] >= -0.35) & (df['X_over_D'] <= 0.15) & (df['Z_over_D'] >= 0.0)]

    x_list, z_list = [], []
    for _, g in df.groupby(['revolution', 'blade']):
        g = g.sort_values('X_over_D')
        if len(g) >= 2:
            x_list.append(g['X_over_D'].values)
            z_list.append(g['Z_over_D'].values)

    if not x_list:
        if len(df) > 0:
            x_list.append(df['X_over_D'].values)
            z_list.append(df['Z_over_D'].values)
        else:
            print(f"Warning: No valid wake points in observation window for {case_path}")
            return None, None

    xm, zm = compute_binned_mean(x_list, z_list)
    return xm, zm, rot_applied


def main():
    parser = argparse.ArgumentParser(description="Compare CHARM, CFD, and Experiment trajectories.")
    parser.add_argument("cases", nargs="*", default=["CHARM_TILT0", "CHARM_TILT-10", "CHARM_TILT-20"],
                        help="CHARM case folders or CSV files (default: CHARM_TILT0 CHARM_TILT-10 CHARM_TILT-20)")
    parser.add_argument("--no-rotate", action="store_true", help="Disable rotor disk tilt alignment")
    parser.add_argument("-o", "--output", default="plots/compare_trajectories_charm.png", help="Output plot path")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(10, 7))

    # 1. Rotor tip path line and azimuth markers
    tip_x = np.linspace(XLIM[0], XLIM[1], 100)
    ax.plot(tip_x, TIP_SLOPE * tip_x, 'k-', linewidth=2.0, label='Rotor tip path')

    yaw_rad = np.radians(YAW_DEG)
    for psi_deg, r_lbl in [(120, "0.81R"), (105, "0.72R"), (90, "0.70R")]:
        psi_rad = np.radians(psi_deg)
        x_p = 0.5 * np.cos(psi_rad) * np.cos(yaw_rad)
        z_p = TIP_SLOPE * x_p
        ax.plot(x_p, z_p, 'r*', markersize=9)
        ax.annotate(rf"${psi_deg}^\circ$" + f"\n{r_lbl}", xy=(x_p, z_p), xytext=(x_p - 0.015, z_p - 0.05),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.1), fontsize=10, ha='center')

    # 2. Experiment (single clean line)
    x_exp, z_exp = load_experiment()
    ax.plot(x_exp, z_exp, 'r--', linewidth=2.8, label='SWT PIV Exp Mean')

    # 3. CFD (single clean line)
    x_cfd, z_cfd = load_cfd()
    ax.plot(x_cfd, z_cfd, color='#1f77b4', linestyle='-', linewidth=2.6, label='CFD Baseline Mean')

    # 4. CHARM cases
    plotted_any = False
    for idx, c in enumerate(args.cases):
        label = os.path.basename(c.rstrip("/")).replace(".csv", "")
        res = load_charm(c, auto_rotate=not args.no_rotate)
        if res is None:
            print(f"Skipping {label}: no valid trajectory data found.")
            continue
        x_ch, z_ch, rot_applied = res
        if x_ch is None or len(x_ch) == 0:
            print(f"Skipping {label}: no valid trajectory mean curve found.")
            continue
        color = COLORS[idx % len(COLORS)]
        rot_str = f", rot {rot_applied:+.0f}°" if rot_applied != 0.0 else ""
        ax.plot(x_ch, z_ch, color=color, linewidth=2.4, label=f"CHARM ({label}{rot_str})")
        plotted_any = True

    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.set_xlabel(r"Streamwise Coordinate $X / D$", fontsize=13)
    ax.set_ylabel(r"Vertical Coordinate $Z / D$", fontsize=13)
    ax.set_title("Tip Vortex Trajectory Comparison at Y = 8.4 in (0.70R)", fontsize=14, pad=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.92)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=200, bbox_inches='tight')
    print(f"Saved plot to: {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
