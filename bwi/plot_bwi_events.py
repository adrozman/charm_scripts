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
    required_cols = ['arc_id', 'psi_deg', 'r_centroid', 'delp_blade_mean', 'miss_dist_ft', 'x_1_ft']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {csv_file.name}")

    numeric_cols = [
        'arc_id', 'psi_deg', 'r_centroid', 'delp_blade_mean', 'delp_blade_max',
        'miss_dist_ft', 'vortex_core_ft', 'vortex_gamma_bound',
        'sweep_perp', 'x_1_ft', 'x_2_ft',
        'r_prior', 'miss_dist_prior_ft', 'vortex_core_prior_ft', 'sweep_perp_prior',
        'n_strikes'
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
      (a) Blade-Vortex Miss Distance [ft] vs. psi
      (b) Tip Vortex Core Radius r_c [ft] vs. psi
      (c) Glegg Convection Distances x_1 and x_2 [ft] vs. psi
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
            valid = arc_df[arc_df['miss_dist_ft'] > 0.0]
            sep_val = int(arc_df['blade_separation'].iloc[0]) if 'blade_separation' in arc_df.columns else None
            sep_str = f"Sep {sep_val}, " if sep_val is not None else ""

            if len(valid) > 0:
                max_miss_seen = max(max_miss_seen, valid['miss_dist_ft'].max())
                core_ft = valid['vortex_core_ft']

                ax_m.plot(valid['psi_deg'], valid['miss_dist_ft'], color=col, linewidth=2.5,
                          path_effects=LINE_EFFECTS, label=f"Event {aid} ({sep_str}$r/R={r_min:.2f}–{r_max:.2f}$)")
                ax_c.plot(valid['psi_deg'], core_ft, color=col, linewidth=2.5,
                          path_effects=LINE_EFFECTS, label=f"Event {aid}")

                # (c) Glegg x_1 (Generating Blade -> Current) and x_2 (1st Chopping Blade -> Current)
                ax_a.plot(valid['psi_deg'], valid['x_1_ft'], color=col, linewidth=2.5,
                          path_effects=LINE_EFFECTS, label=f"Event {aid} $x_1$ (From Generating Blade)")

                if 'x_2_ft' in valid.columns and valid['x_2_ft'].notna().any():
                    v_x2 = valid[valid['x_2_ft'].notna() & (valid['x_2_ft'] > 0)]
                    if len(v_x2) > 0:
                        ax_a.plot(v_x2['psi_deg'], v_x2['x_2_ft'], color=col, linestyle='--', linewidth=2.0,
                                  label=f"Event {aid} $x_2$ (From 1st Chopping Blade)")

        setup_azimuth_axis(ax_m)
        setup_azimuth_axis(ax_c)
        setup_azimuth_axis(ax_a, xlabel=r"Blade Azimuth $\psi$ [deg]")

        # (a) Miss Distance — scale cleanly to data starting at 0
        y_upper = max(0.05, max_miss_seen * 1.15)
        ax_m.set_ylim(0.0, y_upper)
        ax_m.set_ylabel("Miss Distance [ft]", fontsize=10, weight='semibold')
        ax_m.set_title("(a) Blade-Vortex Miss Distance", fontsize=10.5, weight='bold')
        ax_m.legend(loc='upper right', fontsize=8.5, frameon=True)

        # (b) Vortex Core Radius
        ax_c.set_ylabel(r"Core Radius $r_c$ [ft]", fontsize=10, weight='semibold')
        ax_c.set_title("(b) Tip Vortex Core Radius", fontsize=10.5, weight='bold')
        ax_c.legend(loc='upper right', fontsize=8.5, frameon=True)

        # (c) Convection Distance
        ax_a.set_ylabel("Convection Distance [ft]", fontsize=10, weight='semibold')
        ax_a.set_title(r"(c) Glegg Wake Convection Distances $x_1$ and $x_2$ ($x = U_c \tau$)", fontsize=10.5, weight='bold')
        ax_a.legend(loc='upper right', fontsize=8.5, frameon=True)

        fig.suptitle(f"CHARM BWI Vortex Characteristics Along Events{rotor_str}",
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
            valid = arc_df[arc_df['miss_dist_ft'] > 0.0]
            sep_str = f"Sep {int(arc_df['blade_separation'].iloc[0])}, " if 'blade_separation' in arc_df.columns else ""

            if len(valid) > 0:
                max_miss_seen = max(max_miss_seen, valid['miss_dist_ft'].max())
                core_ft = valid['vortex_core_ft']

                ax_m.plot(valid['psi_deg'], valid['miss_dist_ft'], color=col, linewidth=2.5,
                          path_effects=LINE_EFFECTS, label=f"Event {aid} ({sep_str}$r/R={r_min:.2f}–{r_max:.2f}$)")
                ax_c.plot(valid['psi_deg'], core_ft, color=col, linewidth=2.5,
                          path_effects=LINE_EFFECTS, label=f"Event {aid}")
                ax_a.plot(valid['psi_deg'], valid['x_1_ft'], color=col, linewidth=2.5,
                          path_effects=LINE_EFFECTS, label=f"Event {aid} $x_1$ (From Generating Blade)")

                if 'x_2_ft' in valid.columns and valid['x_2_ft'].notna().any():
                    v_x2 = valid[valid['x_2_ft'].notna() & (valid['x_2_ft'] > 0)]
                    if len(v_x2) > 0:
                        ax_a.plot(v_x2['psi_deg'], v_x2['x_2_ft'], color=col, linestyle='--', linewidth=2.0,
                                  label=f"Event {aid} $x_2$ (From 1st Chopping Blade)")

            ax_p.plot(arc_df['psi_deg'], arc_df['delp_blade_mean'], color=col, linewidth=2.5,
                      path_effects=LINE_EFFECTS, label=f"Event {aid} Mean")
            if 'delp_blade_max' in arc_df.columns:
                ax_p.plot(arc_df['psi_deg'], arc_df['delp_blade_max'], color=col, linestyle=':',
                          linewidth=1.3, alpha=0.8, label=f"Event {aid} Max")

        for ax in [ax_m, ax_p, ax_c, ax_a]:
            setup_azimuth_axis(ax, xlabel=r"Blade Azimuth $\psi$ [deg]")

        y_upper = max(0.05, max_miss_seen * 1.15)
        ax_m.set_ylim(0.0, y_upper)
        ax_m.set_ylabel("Miss Distance [ft]", fontsize=10, weight='semibold')
        ax_m.set_title("(a) Blade-Vortex Miss Distance", fontsize=10.5, weight='bold')
        ax_m.legend(loc='upper right', fontsize=8.5, frameon=True)

        ax_p.set_ylabel(r"Delta Pressure $\langle |\Delta P| \rangle$ [$\mathrm{lbf/ft^2}$]", fontsize=10, weight='semibold')
        ax_p.set_title(r"(b) BWI Acoustic Source Severity ($\langle |\Delta P_{\mathrm{blade}}| \rangle$)", fontsize=10.5, weight='bold')
        ax_p.legend(loc='upper right', fontsize=8.5, frameon=True)

        ax_c.set_ylabel(r"Core Radius $r_c$ [ft]", fontsize=10, weight='semibold')
        ax_c.set_title("(c) Tip Vortex Core Radius", fontsize=10.5, weight='bold')
        ax_c.legend(loc='upper left', fontsize=8.5, frameon=True)

        ax_a.set_ylabel("Convection Distance [ft]", fontsize=10, weight='semibold')
        ax_a.set_title(r"(d) Glegg Wake Convection Distances $x_1$ and $x_2$ ($x = U_c \tau$)", fontsize=10.5, weight='bold')
        ax_a.legend(loc='upper left', fontsize=8.5, frameon=True)

        fig.suptitle(f"CHARM BWI Physical Characteristics Dashboard{rotor_str}",
                     fontsize=12, weight='bold', y=1.01)

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"  Saved Event Characteristics Plot to: {Path(output_path).resolve()}")
    return fig


# ==============================================================================
# DEDICATED X1 & X2 WAKE AGE DIAGNOSTIC DASHBOARD
# ==============================================================================

def plot_x2_wake_age_diagnostic(df, output_path="bwi_x2_diagnostic.png", rotor_id=None):
    """
    Renders a dedicated 4-panel diagnostic figure explaining Glegg's x1 and x2:
      (a) Spatial Footprint: Where primary (x1 only) vs. secondary (has x2) interactions occur on rotor disk.
      (b) Convection Distance Decomposition: Showing x1 (Generating -> Current) = (x1 - x2) + x2.
      (c) Convection Comparison: Direct comparison of x1 vs. x2 relative to blade passage convection distance.
      (d) Physical Consequence: Vortex core growth and radial drift during elapsed distance x2.
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
        has_x2 = arc_df['x_2_ft'].notna().any()

        if sep_val == 1 or not has_x2:
            lbl = f"Event {aid} (Sep 1): 1st Blade Hit [$x_1$ only, NO $x_2$]"
            marker = 'o'
        else:
            lbl = f"Event {aid} (Sep {sep_val}): Later Blade Hit [Has BOTH $x_1$ & $x_2$]"
            marker = 's'

        ax_polar.plot(th, r, color=col, linewidth=2.5, label=lbl, zorder=5)
        ax_polar.scatter(th, r, s=40, color=col, edgecolor='black', linewidth=0.8, marker=marker, zorder=6)

    ax_polar.legend(loc='lower left', bbox_to_anchor=(-0.10, -0.22), fontsize=8.5, frameon=True)
    ax_polar.set_title("(a) Interaction Footprint on Rotor Disk\n[Circles: 1st collision (x1 only) | Squares: 2nd+ collision (has x1 & x2)]",
                       fontsize=10.5, weight='bold', pad=14)

    # --------------------------------------------------------------------------
    # Panel (b): Stacked Convection Distance Decomposition (x1 = x_prior + x2)
    # --------------------------------------------------------------------------
    ax_stack = fig.add_subplot(gs[0, 1])
    setup_azimuth_axis(ax_stack, xlabel=r"Blade Azimuth $\psi$ [deg]")

    # Nominal 1 blade passage convection distance:
    # U_c = 32.81 ft/s, RPM = 3806, Nb = 5 -> d_b1 = 32.81 * (60/3806) / 5 = 0.1034 ft
    d_b1 = 32.81 * (60.0 / 3806.0) / 5.0
    d_b2 = 2.0 * d_b1

    has_secondary = False
    max_tot_dist = 0.35
    for aid in arc_ids:
        idx = int(aid) - 1
        col = ARC_PALETTE[idx % len(ARC_PALETTE)]
        arc_df = df[df['arc_id'] == aid].sort_values('psi_deg')
        sep_val = int(arc_df['blade_separation'].iloc[0]) if 'blade_separation' in arc_df.columns else None

        if sep_val == 1:
            ax_stack.plot(arc_df['psi_deg'], arc_df['x_1_ft'], color=col, linewidth=2.5,
                          linestyle='--', label=f"Event {aid} (Sep 1) $x_1$ (Shedding $\\rightarrow$ 1st Blade)")
        elif (sep_val == 2 or (arc_df['x_2_ft'].notna().any() and not has_secondary)):
            has_secondary = True
            v = arc_df[arc_df['x_2_ft'].notna() & (arc_df['x_2_ft'] > 0)].copy()
            if len(v) > 0:
                psi_v = v['psi_deg'].values
                x1_v = v['x_1_ft'].values
                x2_v = v['x_2_ft'].values
                x_prior_v = x1_v - x2_v
                max_tot_dist = max(max_tot_dist, x1_v.max() * 1.25)

                ax_stack.fill_between(psi_v, 0, x_prior_v, color='#00a8ff', alpha=0.45,
                                      label=f"Event {aid} Prior Dist $x_1 - x_2$ (Shedding $\\rightarrow$ 1st Blade)")
                ax_stack.fill_between(psi_v, x_prior_v, x1_v, color='#39ff14', alpha=0.45,
                                      label=f"Event {aid} Glegg $x_2$ (1st Blade $\\rightarrow$ 2nd Blade)")
                ax_stack.plot(psi_v, x1_v, color='#118800', linewidth=2.5,
                              label=rf"Event {aid} Glegg $x_1$ (Shedding $\rightarrow$ 2nd Blade)")

    ax_stack.axhline(d_b1, color='blue', linestyle=':', linewidth=1.4, alpha=0.8,
                     label=rf"1 Blade Pass Convection Dist ($U_c \Delta\tau_b \approx {d_b1:.3f}$ ft)")
    ax_stack.axhline(d_b2, color='purple', linestyle=':', linewidth=1.4, alpha=0.8,
                     label=rf"2 Blade Pass Convection Dist ($2 U_c \Delta\tau_b \approx {d_b2:.3f}$ ft)")

    ax_stack.set_ylim(0.0, max_tot_dist)
    ax_stack.set_ylabel("Convection Distance [ft]", fontsize=10, weight='semibold')
    ax_stack.set_title(r"(b) Glegg Convection Distance Definition: $x_1$ (Generating Blade) & $x_2$ (1st Chopping Blade)" + "\n" +
                       r"[$x_1$ = Total Dist from Generating Blade | $x_2$ = Elapsed Dist Since 1st Blade Chopped Vortex]",
                       fontsize=10.5, weight='bold')
    ax_stack.legend(loc='upper right', fontsize=8.0, frameon=True)

    # --------------------------------------------------------------------------
    # Panel (c): Distance Comparison along Azimuth (x1 vs x2 on Secondary Event)
    # --------------------------------------------------------------------------
    ax_time = fig.add_subplot(gs[1, 0])
    setup_azimuth_axis(ax_time, xlabel=r"Blade Azimuth $\psi$ [deg]")

    for aid in arc_ids:
        arc_df = df[df['arc_id'] == aid].sort_values('psi_deg')
        sep_val = int(arc_df['blade_separation'].iloc[0]) if 'blade_separation' in arc_df.columns else None
        if sep_val == 2 or (arc_df['x_2_ft'].notna().any() and sep_val != 1):
            v = arc_df[arc_df['x_2_ft'].notna() & (arc_df['x_2_ft'] > 0)]
            if len(v) > 0:
                x_prior_v = v['x_1_ft'] - v['x_2_ft']
                ax_time.plot(v['psi_deg'], v['x_1_ft'], color='#7700cc', linewidth=2.4, marker='^',
                             markersize=5, label=f"Event {aid} Glegg $x_1$ (Generating Blade $\\rightarrow$ Current)")
                ax_time.plot(v['psi_deg'], v['x_2_ft'], color='#22aa00', linewidth=2.4, marker='s',
                             markersize=5, label=f"Event {aid} Glegg $x_2$ (1st Blade $\\rightarrow$ Current)")
                ax_time.plot(v['psi_deg'], x_prior_v, color='#0077cc', linewidth=2.0, linestyle='--',
                             marker='o', markersize=4, label=f"Event {aid} $x_1 - x_2$ (Dist at 1st Ingestion)")

    ax_time.axhline(d_b1, color='gray', linestyle=':', linewidth=1.5, alpha=0.9,
                    label=rf"Nominal 1 Blade Pass Convection Dist $\approx {d_b1:.3f}$ ft")
    ax_time.set_ylim(0.0, max_tot_dist)
    ax_time.set_ylabel("Convection Distance [ft]", fontsize=10, weight='semibold')
    ax_time.set_title(r"(c) Glegg $x_1$ and $x_2$ Along Secondary Event (Sep 2)" + "\n" +
                      rf"[$x_1 \approx {d_b2:.3f}$ ft (2 blade passages) & $x_2 \approx {d_b1:.3f}$ ft (1 blade passage)]",
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
            core_ft = v['vortex_core_ft']
            sep_label = f"Sep {sep_val}" if sep_val is not None else f"Event {aid}"
            ax_phys.plot(v['psi_deg'], core_ft, color=col, linewidth=2.5,
                         label=f"Event {aid} ({sep_label}): Core $r_c$ [ft]")

    ax_phys.set_ylabel(r"Vortex Core Radius $r_c$ [ft]", fontsize=10, weight='semibold')
    ax_phys.set_title("(d) CHARM Vortex Core Radius Along Events\n" +
                      r"[In Glegg Eq. 11/12, turbulent radius $a_0$ combines $r_c(x_1)$ + wrapped wake growth from $x_1$ and $x_2$]",
                      fontsize=10.5, weight='bold')
    ax_phys.legend(loc='upper left', fontsize=8.5, frameon=True)

    fig.suptitle(f"BWI Physical Mechanism: Glegg $x_1$ and $x_2$ Convection Distance Diagnostic Dashboard{rotor_str}",
                 fontsize=13, weight='bold', y=1.01)

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"  Saved x_2 Convection Distance Diagnostic Plot to: {Path(output_path).resolve()}")
    return fig


# ==============================================================================
# SUMMARY TABLE REPORTER
# ==============================================================================

def print_summary_table(df, rotor_id=None):
    """Prints a clean ASCII summary table of BWI event metrics to console, including x1 and x2."""
    rotor_title = f" (Rotor {rotor_id})" if rotor_id is not None else ""
    print("\n" + "=" * 96)
    print(f"BWI EVENT CHARACTERISTICS & GLEGG CONVECTION DISTANCE SUMMARY TABLE{rotor_title}")
    print("=" * 96)
    header = (
        f"{'Event':<7} | {'Sep':<4} | {'Azimuth Span':<14} | {'Span r/R':<12} | "
        f"{'Min Miss [ft]':<13} | {'x_1 [ft]':<11} | {'x_2 [ft]':<14} | {'Strikes':<7}"
    )
    print(header)
    print("-" * len(header))

    for aid in sorted(df['arc_id'].unique()):
        sub = df[df['arc_id'] == aid]
        v_str = sub[sub['miss_dist_ft'] > 0]
        sep_val = f"{int(sub['blade_separation'].iloc[0])}" if 'blade_separation' in sub.columns else "-"

        psi_str = f"{sub['psi_deg'].min():.0f}°–{sub['psi_deg'].max():.0f}°"
        r_str = f"{sub['r_centroid'].min():.2f}–{sub['r_centroid'].max():.2f}"
        tot_str = f"{int(sub['n_strikes'].sum()):,}" if 'n_strikes' in sub.columns else "0"

        if len(v_str) > 0:
            min_miss = f"{v_str['miss_dist_ft'].min():.4f} ft"
        else:
            min_miss = "N/A"

        x1_val = f"{sub['x_1_ft'].dropna().mean():.4f} ft" if 'x_1_ft' in sub.columns and sub['x_1_ft'].notna().any() else "-"
        x2_val = f"{sub['x_2_ft'].dropna().mean():.4f} ft" if 'x_2_ft' in sub.columns and sub['x_2_ft'].notna().any() else ("Primary (None)" if sep_val == "1" else "N/A")

        print(f"Event {aid:<2} | {sep_val:<4} | {psi_str:<14} | {r_str:<12} | {min_miss:<13} | {x1_val:<11} | {x2_val:<14} | {tot_str:<7}")
    print("=" * 96 + "\n")


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
