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
        'sweep_perp', 'vortex_age_rev', 'n_strikes'
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


def plot_arc_characteristics(df, output_path="bwi_arc_characteristics.png", rotor_id=None, include_pressure=False):
    """
    Renders a multi-metric plot along the identified BWI arcs.
    By default, creates a 3-row stacked figure sharing the azimuth axis:
      (a) Blade-Vortex Miss Distance [in] vs. psi
      (b) Tip Vortex Core Radius r_c [in] vs. psi
      (c) Shedding Wake Age [revs] vs. psi
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

            if len(valid) > 0:
                max_miss_seen = max(max_miss_seen, valid['miss_dist_in'].max())
                core_in = valid['vortex_core_ft'] * 12.0

                ax_m.plot(valid['psi_deg'], valid['miss_dist_in'], color=col, linewidth=2.5,
                          path_effects=LINE_EFFECTS, label=f"Arc {aid} ($r/R={r_min:.2f}–{r_max:.2f}$)")
                ax_c.plot(valid['psi_deg'], core_in, color=col, linewidth=2.5,
                          path_effects=LINE_EFFECTS, label=f"Arc {aid}")
                ax_a.plot(valid['psi_deg'], valid['vortex_age_rev'], color=col, linewidth=2.5,
                          path_effects=LINE_EFFECTS, label=f"Arc {aid}")

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

            if len(valid) > 0:
                max_miss_seen = max(max_miss_seen, valid['miss_dist_in'].max())
                core_in = valid['vortex_core_ft'] * 12.0

                ax_m.plot(valid['psi_deg'], valid['miss_dist_in'], color=col, linewidth=2.5,
                          path_effects=LINE_EFFECTS, label=f"Arc {aid} ($r/R={r_min:.2f}–{r_max:.2f}$)")
                ax_c.plot(valid['psi_deg'], core_in, color=col, linewidth=2.5,
                          path_effects=LINE_EFFECTS, label=f"Arc {aid}")
                ax_a.plot(valid['psi_deg'], valid['vortex_age_rev'], color=col, linewidth=2.5,
                          path_effects=LINE_EFFECTS, label=f"Arc {aid}")

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

        ax_p.set_ylabel(r"Delta Pressure $\langle |\Delta P| \rangle$ [$\mathrm{lbf/ft^2/rev}$]", fontsize=10, weight='semibold')
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
# SUMMARY TABLE REPORTER
# ==============================================================================

def print_summary_table(df, rotor_id=None):
    """Prints a clean ASCII summary table of BWI arc metrics to console."""
    rotor_title = f" (Rotor {rotor_id})" if rotor_id is not None else ""
    print("\n" + "=" * 80)
    print(f"BWI ARC CHARACTERISTICS SUMMARY TABLE{rotor_title}")
    print("=" * 80)
    header = (
        f"{'Arc':<5} | {'Azimuth Span':<16} | {'Span r/R':<12} | "
        f"{'Min Miss':<10} | {'Peak |ΔP|':<12} | {'Vtx Age':<10} | {'Strikes':<8}"
    )
    print(header)
    print("-" * len(header))

    for aid in sorted(df['arc_id'].unique()):
        sub = df[df['arc_id'] == aid]
        v_str = sub[sub['miss_dist_in'] > 0]

        psi_str = f"{sub['psi_deg'].min():.0f}°–{sub['psi_deg'].max():.0f}°"
        r_str = f"{sub['r_centroid'].min():.2f}–{sub['r_centroid'].max():.2f}"
        peak_p = f"{sub['delp_blade_mean'].max():.1f} psf"
        tot_str = f"{int(sub['n_strikes'].sum()):,}" if 'n_strikes' in sub.columns else "0"

        if len(v_str) > 0:
            min_miss = f"{v_str['miss_dist_in'].min():.2f} in"
            mean_age = f"{v_str['vortex_age_rev'].mean():.2f} rev"
        else:
            min_miss = "N/A"
            mean_age = f"{sub['vortex_age_rev'].mean():.2f} rev"

        print(f"Arc {aid:<2} | {psi_str:<16} | {r_str:<12} | {min_miss:<10} | {peak_p:<12} | {mean_age:<10} | {tot_str:<8}")
    print("=" * 80 + "\n")


# ==============================================================================
# MAIN CLI INTERFACE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="CHARM BWI Vortex Characteristics Plotter")
    parser.add_argument("--csv", type=str, default="bwi_events.csv", help="Path to BWI events CSV file (default: bwi_events.csv)")
    parser.add_argument("--output", type=str, default="bwi_arc_characteristics.png", help="Output plot filename (default: bwi_arc_characteristics.png)")
    parser.add_argument("--rotor-id", default=None, help="Optional Rotor ID label (1, 2, ...)")
    parser.add_argument("--include-pressure", action="store_true", help="Include 4th panel for unsteady surface Delta Pressure")
    args = parser.parse_args()

    csv_file = Path(args.csv)
    print(f"\nProcessing BWI Events File: {csv_file.resolve()}")
    df = load_events_data(csv_file)

    plot_arc_characteristics(df, output_path=args.output, rotor_id=args.rotor_id, include_pressure=args.include_pressure)
    print_summary_table(df, rotor_id=args.rotor_id)
    print("BWI Characteristics Visualization complete.")
    #plt.show()


if __name__ == '__main__':
    main()
