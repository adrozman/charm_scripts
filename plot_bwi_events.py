#!/usr/bin/env python3
"""
plot_bwi_events.py — CHARM BWI Multi-Arc Characteristics Visualizer

Visualizes extracted blade-vortex interaction (BWI) trajectories from `bwi_events.csv`
in a 3-panel stacked figure (`bwi_arc_characteristics.png`):
  (a) Blade-Vortex Miss Distance [in] vs. Azimuth
  (b) Tip Vortex Core Radius r_c [in] vs. Azimuth
  (c) Shedding Wake Age [revs] vs. Azimuth

Usage:
  python3 plot_bwi_events.py                        # Defaults to --csv bwi_events.csv
  python3 plot_bwi_events.py --csv <path.csv>       # Custom events CSV file
  python3 plot_bwi_events.py --include-pressure     # 4-panel layout adding |DelP_Blade|
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.patheffects as pe

# Arc color palette consistent with bwi_polar.py
ARC_PALETTE = ['#00ffff', '#39ff14', '#ffdd00', '#ff3366', '#b026ff', '#ff8800', '#00d4ff']

# Subtle outline path effect ensuring high contrast on white background for neon colors
LINE_EFFECTS = [pe.Stroke(linewidth=3.6, foreground='#444444', alpha=0.35), pe.Normal()]


# ==============================================================================
# DATA LOADING & PRE-PROCESSING
# ==============================================================================

def load_events_data(csv_path):
    """Loads and validates the BWI events CSV file."""
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"BWI events CSV file not found: {csv_file.resolve()}")

    df = pd.read_csv(csv_file)
    required_cols = ['arc_id', 'psi_deg', 'r_centroid', 'delp_blade_mean', 'miss_dist_in', 'vortex_age_rev']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {csv_file.name}")

    numeric_cols = [
        'arc_id', 'psi_deg', 'r_centroid', 'delp_blade_mean', 'delp_blade_max',
        'miss_dist_ft', 'miss_dist_in', 'vortex_core_ft', 'vortex_gamma',
        'sweep_perp', 'vortex_age_rev', 'x_1_rev', 'x_2_rev', 'n_strikes'
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    return df


# ==============================================================================
# MULTI-PANEL VORTEX CHARACTERISTICS (Miss Distance, Core Radius, Wake Age)
# ==============================================================================

def setup_azimuth_axis(ax, xlabel=None):
    """Configures azimuth axis with standard 0-360 range and 45 deg major ticks."""
    ax.set_xlim(0.0, 360.0)
    ax.xaxis.set_major_locator(MultipleLocator(45))
    ax.xaxis.set_minor_locator(MultipleLocator(15))
    ax.grid(True, which='major', linestyle=':', color='gray', alpha=0.55)
    ax.grid(True, which='minor', linestyle=':', color='lightgray', alpha=0.3)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10.5, weight='semibold')


def plot_arc_characteristics(df, output_path="bwi_arc_characteristics.png", rotor_id=None,
                             include_pressure=False):
    """
    Renders a multi-metric plot along the identified BWI arcs.
    By default, creates a 3-row stacked figure sharing the azimuth axis:
      (a) Blade-Vortex Miss Distance [in] vs. psi
      (b) Tip Vortex Core Radius r_c [in] vs. psi
      (c) Shedding Wake Age [revs] vs. psi (Cumulative Age & Secondary Elapsed Age x2)
    If include_pressure=True, generates a 4-panel (2x2) dashboard including surface |DelP|.
    Colors match bwi_polar.py palette index-for-index.
    """
    arc_ids = sorted(df['arc_id'].unique())
    rotor_str = f" (Rotor {rotor_id})" if rotor_id is not None else ""

    if not include_pressure:
        fig, (ax_m, ax_c, ax_a) = plt.subplots(3, 1, figsize=(9.0, 8.5), sharex=True, constrained_layout=True)

        max_miss_seen = 0.0
        for aid in arc_ids:
            idx = int(aid) - 1
            col = ARC_PALETTE[idx % len(ARC_PALETTE)]

            arc_df = df[df['arc_id'] == aid].sort_values('psi_deg')
            r_min, r_max = arc_df['r_centroid'].min(), arc_df['r_centroid'].max()
            valid = arc_df[arc_df['miss_dist_in'] > 0.0]
            sep_val = int(arc_df['blade_separation'].iloc[0]) if 'blade_separation' in arc_df.columns else None
            sep_str = f"Sep {sep_val}, " if sep_val is not None else ""

            if len(valid) > 0:
                max_miss_seen = max(max_miss_seen, valid['miss_dist_in'].max())
                core_in = valid['vortex_core_ft'] * 12.0

                ax_m.plot(valid['psi_deg'], valid['miss_dist_in'], color=col, linewidth=2.5,
                          path_effects=LINE_EFFECTS, label=f"Arc {aid} ({sep_str}$r/R={r_min:.2f}–{r_max:.2f}$)")
                ax_c.plot(valid['psi_deg'], core_in, color=col, linewidth=2.5,
                          path_effects=LINE_EFFECTS, label=f"Arc {aid}")

                # (c) Total Cumulative Wake Age and Secondary Elapsed Age x_2
                ax_a.plot(valid['psi_deg'], valid['vortex_age_rev'], color=col, linewidth=2.5,
                          path_effects=LINE_EFFECTS, label=f"Arc {aid} Age")

                if 'x_2_rev' in valid.columns and valid['x_2_rev'].notna().any():
                    v_x2 = valid[valid['x_2_rev'].notna() & (valid['x_2_rev'] > 0)]
                    if len(v_x2) > 0:
                        ax_a.plot(v_x2['psi_deg'], v_x2['x_2_rev'], color=col, linestyle='--', linewidth=2.0,
                                  label=f"Arc {aid} $x_2$ (Elapsed)")

        setup_azimuth_axis(ax_m)
        setup_azimuth_axis(ax_c)
        setup_azimuth_axis(ax_a, xlabel=r"Blade Azimuth $\psi$ [deg]")

        # (a) Miss Distance — scale cleanly to data starting at 0
        y_upper = max(0.5, max_miss_seen * 1.15)
        ax_m.set_ylim(0.0, y_upper)
        ax_m.set_ylabel("Miss Distance [in]", fontsize=10, weight='semibold')
        ax_m.set_title("(a) Blade-Vortex Miss Distance", fontsize=10.5, weight='bold')
        ax_m.legend(loc='upper right', fontsize=8.5, frameon=True)

        # (b) Vortex Core Radius
        ax_c.set_ylabel(r"Core Radius $r_c$ [in]", fontsize=10, weight='semibold')
        ax_c.set_title("(b) Tip Vortex Core Radius", fontsize=10.5, weight='bold')
        ax_c.legend(loc='upper right', fontsize=8.5, frameon=True)

        # (c) Shedding Wake Age
        ax_a.set_ylabel("Wake Age [revs]", fontsize=10, weight='semibold')
        ax_a.set_title("(c) Shedding Wake Age", fontsize=10.5, weight='bold')
        ax_a.legend(loc='upper right', fontsize=8.5, frameon=True)

        fig.suptitle(f"CHARM BWI Vortex Characteristics Along Arcs{rotor_str}",
                     fontsize=12, weight='bold')

    else:
        fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5), constrained_layout=True)
        ax_m, ax_p = axes[0, 0], axes[0, 1]
        ax_c, ax_a = axes[1, 0], axes[1, 1]

        max_miss_seen = 0.0
        for aid in arc_ids:
            idx = int(aid) - 1
            col = ARC_PALETTE[idx % len(ARC_PALETTE)]

            arc_df = df[df['arc_id'] == aid].sort_values('psi_deg')
            r_min, r_max = arc_df['r_centroid'].min(), arc_df['r_centroid'].max()
            valid = arc_df[arc_df['miss_dist_in'] > 0.0]
            sep_str = f"Sep {int(arc_df['blade_separation'].iloc[0])}, " if 'blade_separation' in arc_df.columns else ""

            if len(valid) > 0:
                max_miss_seen = max(max_miss_seen, valid['miss_dist_in'].max())
                core_in = valid['vortex_core_ft'] * 12.0

                ax_m.plot(valid['psi_deg'], valid['miss_dist_in'], color=col, linewidth=2.5,
                          path_effects=LINE_EFFECTS, label=f"Arc {aid} ({sep_str}$r/R={r_min:.2f}–{r_max:.2f}$)")
                ax_c.plot(valid['psi_deg'], core_in, color=col, linewidth=2.5,
                          path_effects=LINE_EFFECTS, label=f"Arc {aid}")
                ax_a.plot(valid['psi_deg'], valid['vortex_age_rev'], color=col, linewidth=2.5,
                          path_effects=LINE_EFFECTS, label=f"Arc {aid} Age")

                if 'x_2_rev' in valid.columns and valid['x_2_rev'].notna().any():
                    v_x2 = valid[valid['x_2_rev'].notna() & (valid['x_2_rev'] > 0)]
                    if len(v_x2) > 0:
                        ax_a.plot(v_x2['psi_deg'], v_x2['x_2_rev'], color=col, linestyle='--', linewidth=2.0,
                                  label=f"Arc {aid} $x_2$ (Elapsed)")

            ax_p.plot(arc_df['psi_deg'], arc_df['delp_blade_mean'], color=col, linewidth=2.5,
                      path_effects=LINE_EFFECTS, label=f"Arc {aid} Mean")
            if 'delp_blade_max' in arc_df.columns:
                ax_p.plot(arc_df['psi_deg'], arc_df['delp_blade_max'], color=col, linestyle=':',
                          linewidth=1.3, alpha=0.8, label=f"Arc {aid} Max")

        for ax in [ax_m, ax_p, ax_c, ax_a]:
            setup_azimuth_axis(ax, xlabel=r"Blade Azimuth $\psi$ [deg]")

        y_upper = max(0.5, max_miss_seen * 1.15)
        ax_m.set_ylim(0.0, y_upper)
        ax_m.set_ylabel("Miss Distance [in]", fontsize=10, weight='semibold')
        ax_m.set_title("(a) Blade-Vortex Miss Distance", fontsize=10.5, weight='bold')
        ax_m.legend(loc='upper right', fontsize=8.5, frameon=True)

        ax_p.set_ylabel(r"Delta Pressure $\langle |\Delta P| \rangle$ [$\mathrm{lbf/ft^2}$]", fontsize=10, weight='semibold')
        ax_p.set_title(r"(b) BWI Acoustic Source Severity ($\langle |\Delta P_{\mathrm{blade}}| \rangle$)", fontsize=10.5, weight='bold')
        ax_p.legend(loc='upper right', fontsize=8.5, frameon=True)

        ax_c.set_ylabel(r"Core Radius $r_c$ [in]", fontsize=10, weight='semibold')
        ax_c.set_title("(c) Tip Vortex Core Radius", fontsize=10.5, weight='bold')
        ax_c.legend(loc='upper left', fontsize=8.5, frameon=True)

        ax_a.set_ylabel("Wake Age [revs]", fontsize=10, weight='semibold')
        ax_a.set_title("(d) Shedding Wake Age", fontsize=10.5, weight='bold')
        ax_a.legend(loc='upper left', fontsize=8.5, frameon=True)

        fig.suptitle(f"CHARM BWI Physical Characteristics Dashboard{rotor_str}",
                     fontsize=12, weight='bold', y=1.01)

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"  Saved Arc Characteristics Plot to: {Path(output_path).resolve()}")
    return fig


# ==============================================================================
# DEDICATED X1 & X2 WAKE AGE DIAGNOSTIC DASHBOARD
# ==============================================================================

def plot_x2_wake_age_diagnostic(df, output_path="bwi_x2_diagnostic.png", rotor_id=None):
    """
    Renders a dedicated 4-panel diagnostic figure explaining:
      (a) Spatial Footprint: Where primary (x1 only) vs. secondary (has x2) interactions occur on rotor disk.
      (b) Cumulative Age Decomposition: Stacked visualization showing Total Age = x1 + x2 along azimuth.
      (c) Timing Comparison: Direct comparison of x1 vs. x2 relative to blade passage period (1/Nb = 0.20 rev).
      (d) Physical Consequence: Vortex core growth and radial drift during elapsed time x2.
    """
    arc_ids = sorted(df['arc_id'].unique())
    rotor_str = f" (Rotor {rotor_id})" if rotor_id is not None else ""

    fig = plt.figure(figsize=(15.0, 10.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    # --------------------------------------------------------------------------
    # Panel (a): Polar Footprint — Where x1 and x2 Occur
    # --------------------------------------------------------------------------
    ax_polar = fig.add_subplot(gs[0, 0], projection='polar')
    ax_polar.set_theta_direction(1)
    ax_polar.set_theta_zero_location('E')
    ax_polar.set_ylim(0.0, 1.0)
    ax_polar.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_polar.set_rlabel_position(22.5)
    ax_polar.tick_params(axis='x', labelsize=9.5)
    ax_polar.tick_params(axis='y', labelsize=8, colors='gray')
    ax_polar.grid(True, linestyle=':', alpha=0.55, color='gray')

    circle_theta = np.linspace(0, 2*np.pi, 200)
    ax_polar.plot(circle_theta, np.ones_like(circle_theta), color='gray', linestyle='--', linewidth=1.0, alpha=0.6)

    for aid in arc_ids:
        idx = int(aid) - 1
        col = ARC_PALETTE[idx % len(ARC_PALETTE)]
        arc_df = df[df['arc_id'] == aid].sort_values('psi_deg')
        sep_val = int(arc_df['blade_separation'].iloc[0]) if 'blade_separation' in arc_df.columns else None

        th = np.radians(arc_df['psi_deg'].values)
        r = arc_df['r_centroid'].values
        has_x2 = arc_df['x_2_rev'].notna().any()

        if sep_val == 1 or not has_x2:
            lbl = f"Arc {aid} (Sep 1): 1st Blade Hit [$x_1$ only, NO $x_2$]"
            marker = 'o'
        else:
            lbl = f"Arc {aid} (Sep {sep_val}): 2nd Blade Hit [Has BOTH $x_1$ & $x_2$]"
            marker = 's'

        ax_polar.plot(th, r, color=col, linewidth=2.5, label=lbl, zorder=5)
        ax_polar.scatter(th, r, s=40, color=col, edgecolor='black', linewidth=0.8, marker=marker, zorder=6)

    ax_polar.legend(loc='lower left', bbox_to_anchor=(-0.10, -0.22), fontsize=8.5, frameon=True)
    ax_polar.set_title("(a) Interaction Footprint on Rotor Disk\n[Cyan circles: 1st collision (x1 only) | Green squares: 2nd collision (has x2)]",
                       fontsize=10.5, weight='bold', pad=14)

    # --------------------------------------------------------------------------
    # Panel (b): Stacked Wake Age Decomposition (Total Age = x1 + x2)
    # --------------------------------------------------------------------------
    ax_stack = fig.add_subplot(gs[0, 1])
    setup_azimuth_axis(ax_stack, xlabel=r"Blade Azimuth $\psi$ [deg]")

    has_secondary = False
    for aid in arc_ids:
        idx = int(aid) - 1
        col = ARC_PALETTE[idx % len(ARC_PALETTE)]
        arc_df = df[df['arc_id'] == aid].sort_values('psi_deg')
        sep_val = int(arc_df['blade_separation'].iloc[0]) if 'blade_separation' in arc_df.columns else None

        if sep_val == 1:
            ax_stack.plot(arc_df['psi_deg'], arc_df['vortex_age_rev'], color=col, linewidth=2.5,
                          linestyle='--', label=f"Arc {aid} (Sep 1) Total Age = $x_1$ (No $x_2$)")
        elif sep_val == 2 or (arc_df['x_2_rev'].notna().any() and not has_secondary):
            has_secondary = True
            v = arc_df[arc_df['x_2_rev'].notna() & (arc_df['x_2_rev'] > 0)].copy()
            if len(v) > 0:
                psi_v = v['psi_deg'].values
                x1_v = v['x_1_rev'].values
                tot_v = v['vortex_age_rev'].values

                ax_stack.fill_between(psi_v, 0, x1_v, color='#00a8ff', alpha=0.45,
                                      label=r"Arc 2 Base Age $x_1$ (Vortex Shedding $\rightarrow$ 1st Blade)")
                ax_stack.fill_between(psi_v, x1_v, tot_v, color='#39ff14', alpha=0.45,
                                      label=r"Arc 2 Elapsed Age $x_2$ (1st Blade $\rightarrow$ 2nd Blade)")
                ax_stack.plot(psi_v, tot_v, color='#118800', linewidth=2.5,
                              label=r"Arc 2 Total Cumulative Age $\text{Age}_2 = x_1 + x_2$")

    ax_stack.axhline(0.20, color='blue', linestyle=':', linewidth=1.4, alpha=0.8,
                     label=r"1 Blade Pass Period ($1/N_b = 0.20$ rev = $72^\circ$)")
    ax_stack.axhline(0.40, color='purple', linestyle=':', linewidth=1.4, alpha=0.8,
                     label=r"2 Blade Pass Periods ($2/N_b = 0.40$ rev = $144^\circ$)")

    ax_stack.set_ylim(0.0, 0.55)
    ax_stack.set_ylabel("Wake Age [revs]", fontsize=10, weight='semibold')
    ax_stack.set_title(r"(b) Wake Age Decomposition: Total $\text{Age}_2 = x_1 + x_2$" + "\n" +
                       r"[Total Age of 2nd Hit = Time to 1st Hit ($x_1$) + Travel Time to 2nd Hit ($x_2$)]",
                       fontsize=10.5, weight='bold')
    ax_stack.legend(loc='upper right', fontsize=8.0, frameon=True)

    # --------------------------------------------------------------------------
    # Panel (c): Timing Comparison along Azimuth (x1 vs x2 on Secondary Arc)
    # --------------------------------------------------------------------------
    ax_time = fig.add_subplot(gs[1, 0])
    setup_azimuth_axis(ax_time, xlabel=r"Blade Azimuth $\psi$ [deg]")

    for aid in arc_ids:
        arc_df = df[df['arc_id'] == aid].sort_values('psi_deg')
        sep_val = int(arc_df['blade_separation'].iloc[0]) if 'blade_separation' in arc_df.columns else None
        if sep_val == 2 or (arc_df['x_2_rev'].notna().any() and sep_val != 1):
            v = arc_df[arc_df['x_2_rev'].notna() & (arc_df['x_2_rev'] > 0)]
            if len(v) > 0:
                ax_time.plot(v['psi_deg'], v['x_1_rev'], color='#0077cc', linewidth=2.4, marker='o',
                             markersize=5, label=f"Arc {aid} $x_1$ (Prior 1st Interaction Age)")
                ax_time.plot(v['psi_deg'], v['x_2_rev'], color='#22aa00', linewidth=2.4, marker='s',
                             markersize=5, label=f"Arc {aid} $x_2$ (Inter-Blade Elapsed Age)")
                ax_time.plot(v['psi_deg'], v['vortex_age_rev'], color='#7700cc', linewidth=2.0, linestyle='--',
                             marker='^', markersize=4, label=f"Arc {aid} Total $\\text{{Age}}_2 = x_1 + x_2$")

    ax_time.axhline(0.20, color='gray', linestyle=':', linewidth=1.5, alpha=0.9,
                    label=r"Nominal Blade Passage $\Delta \tau_b = 0.20$ rev")
    ax_time.set_ylim(0.10, 0.48)
    ax_time.set_ylabel("Wake Age [revs]", fontsize=10, weight='semibold')
    ax_time.set_title(r"(c) Inter-Blade Travel Duration: $x_1$ vs. $x_2$ Along Secondary Arc" + "\n" +
                      r"[$x_1 \approx 0.20$ rev & $x_2 \approx 0.18$ rev $\approx$ 1 blade passage each ($\approx 72^\circ$)]",
                      fontsize=10.5, weight='bold')
    ax_time.legend(loc='upper right', fontsize=8.0, frameon=True)

    # --------------------------------------------------------------------------
    # Panel (d): Physical Core Growth & Inboard Drift Between 1st and 2nd Hit
    # --------------------------------------------------------------------------
    ax_phys = fig.add_subplot(gs[1, 1])
    setup_azimuth_axis(ax_phys, xlabel=r"Blade Azimuth $\psi$ [deg]")

    for aid in arc_ids:
        idx = int(aid) - 1
        col = ARC_PALETTE[idx % len(ARC_PALETTE)]
        arc_df = df[df['arc_id'] == aid].sort_values('psi_deg')
        sep_val = int(arc_df['blade_separation'].iloc[0]) if 'blade_separation' in arc_df.columns else None
        v = arc_df[arc_df['vortex_core_ft'] > 0]
        if len(v) > 0:
            core_in = v['vortex_core_ft'] * 12.0
            sep_label = f"Sep {sep_val}" if sep_val is not None else f"Arc {aid}"
            ax_phys.plot(v['psi_deg'], core_in, color=col, linewidth=2.5,
                         label=f"Arc {aid} ({sep_label}): Core $r_c$ [in]")

    ax_phys.set_ylabel(r"Vortex Core Radius $r_c$ [in]", fontsize=10, weight='semibold')
    ax_phys.set_title("(d) Why $x_2$ Matters: Core Growth Between 1st & 2nd Blade Encounters\n" +
                      r"[Glegg Model: Core grows as $r_c \propto \sqrt{\nu (x_1 + x_2)}$, softening 2nd strike]",
                      fontsize=10.5, weight='bold')
    ax_phys.legend(loc='upper left', fontsize=8.5, frameon=True)

    fig.suptitle(f"BWI Physical Mechanism: $x_1$ and $x_2$ Wake Age Diagnostic Dashboard{rotor_str}",
                 fontsize=13, weight='bold', y=1.01)

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"  Saved x_2 Wake Age Diagnostic Plot to: {Path(output_path).resolve()}")
    return fig


# ==============================================================================
# SUMMARY TABLE REPORTER
# ==============================================================================

def print_summary_table(df, rotor_id=None):
    """Prints a clean ASCII summary table of BWI arc metrics to console, including x1 and x2."""
    rotor_title = f" (Rotor {rotor_id})" if rotor_id is not None else ""
    print("\n" + "=" * 98)
    print(f"BWI ARC CHARACTERISTICS & WAKE AGE SUMMARY TABLE{rotor_title}")
    print("=" * 98)
    header = (
        f"{'Arc':<5} | {'Sep':<4} | {'Azimuth Span':<14} | {'Span r/R':<12} | "
        f"{'Min Miss':<9} | {'Mean Age':<9} | {'x_1 (Base)':<10} | {'x_2 (Elapsed)':<13} | {'Strikes':<7}"
    )
    print(header)
    print("-" * len(header))

    for aid in sorted(df['arc_id'].unique()):
        sub = df[df['arc_id'] == aid]
        v_str = sub[sub['miss_dist_in'] > 0]
        sep_val = f"{int(sub['blade_separation'].iloc[0])}" if 'blade_separation' in sub.columns else "-"

        psi_str = f"{sub['psi_deg'].min():.0f}°–{sub['psi_deg'].max():.0f}°"
        r_str = f"{sub['r_centroid'].min():.2f}–{sub['r_centroid'].max():.2f}"
        tot_str = f"{int(sub['n_strikes'].sum()):,}" if 'n_strikes' in sub.columns else "0"

        if len(v_str) > 0:
            min_miss = f"{v_str['miss_dist_in'].min():.2f} in"
            mean_age = f"{v_str['vortex_age_rev'].mean():.2f} rev"
        else:
            min_miss = "N/A"
            mean_age = f"{sub['vortex_age_rev'].mean():.2f} rev"

        x1_val = f"{sub['x_1_rev'].dropna().mean():.2f} rev" if 'x_1_rev' in sub.columns and sub['x_1_rev'].notna().any() else ("N/A" if sep_val == "1" else "-")
        x2_val = f"{sub['x_2_rev'].dropna().mean():.2f} rev" if 'x_2_rev' in sub.columns and sub['x_2_rev'].notna().any() else ("Primary (None)" if sep_val == "1" else "N/A")

        print(f"Arc {aid:<2} | {sep_val:<4} | {psi_str:<14} | {r_str:<12} | {min_miss:<9} | {mean_age:<9} | {x1_val:<10} | {x2_val:<13} | {tot_str:<7}")
    print("=" * 98 + "\n")


# ==============================================================================
# MAIN CLI INTERFACE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="CHARM BWI Vortex Characteristics Plotter")
    parser.add_argument("--csv", type=str, default="bwi_events.csv", help="Path to BWI events CSV file (default: bwi_events.csv)")
    parser.add_argument("--output", type=str, default="bwi_arc_characteristics.png", help="Output plot filename (default: bwi_arc_characteristics.png)")
    parser.add_argument("--diag-output", type=str, default=None, help="Output filename for x2 diagnostic plot (default: bwi_x2_diagnostic.png)")
    parser.add_argument("--rotor-id", default=None, help="Optional Rotor ID label (1, 2, ...)")
    parser.add_argument("--include-pressure", action="store_true", help="Include 4th panel for unsteady surface Delta Pressure")
    args = parser.parse_args()

    csv_file = Path(args.csv)
    print(f"\nProcessing BWI Events File: {csv_file.resolve()}")
    df = load_events_data(csv_file)

    plot_arc_characteristics(df, output_path=args.output, rotor_id=args.rotor_id,
                             include_pressure=args.include_pressure)

    diag_fname = args.diag_output if args.diag_output else (Path(args.output).parent / f"{Path(args.output).stem.replace('bwi_arc_characteristics', 'bwi_x2_diagnostic')}.png")
    if diag_fname.name == args.output:
        diag_fname = Path(args.output).parent / "bwi_x2_diagnostic.png"
    plot_x2_wake_age_diagnostic(df, output_path=diag_fname, rotor_id=args.rotor_id)

    print_summary_table(df, rotor_id=args.rotor_id)
    print("BWI Characteristics Visualization complete.")
    #plt.show()


if __name__ == '__main__':
    main()
