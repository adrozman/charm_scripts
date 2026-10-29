#!/usr/bin/env python3
"""
extract_charm_wake.py — Standalone CHARM Wake Vortex Extractor

Extracts tip vortex crossing coordinates (X, Y, Z) at a cutting plane (+/- x, y, z).
Default: y = 8.4 inches.

Usage:
  python3 extract_charm_wake.py charm_runs/case1
  python3 extract_charm_wake.py charm_runs/case1 --plane -y=8.4
  python3 extract_charm_wake.py charm_runs/case1 --axis -y --val 8.4
  python3 extract_charm_wake.py charm_runs/case1 -y 8.4
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd


def parse_floats(line):
    """Parses whitespace-separated floats from a line."""
    parts = line.strip().split()
    return [float(p.replace('D', 'E').replace('d', 'e')) for p in parts] if parts else []


def parse_cli():
    """Simple, foolproof CLI parsing that never crashes on leading dashes."""
    input_path = None
    axis_char = 'y'
    sign = 1.0
    val_in = 8.4
    output_path = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        a = args[i]
        if a in ('--plane', '-p') and i + 1 < len(args):
            i += 1
            p_str = args[i]
            if '=' in p_str:
                ax_part, val_part = p_str.split('=', 1)
                sign = -1.0 if '-' in ax_part else 1.0
                axis_char = ax_part.strip('+-').lower()
                val_in = float(val_part)
            else:
                sign = -1.0 if '-' in p_str else 1.0
                axis_char = p_str.strip('+-').lower()
        elif a.startswith('--plane='):
            p_str = a.split('=', 1)[1]
            if '=' in p_str:
                ax_part, val_part = p_str.split('=', 1)
                sign = -1.0 if '-' in ax_part else 1.0
                axis_char = ax_part.strip('+-').lower()
                val_in = float(val_part)
            else:
                sign = -1.0 if '-' in p_str else 1.0
                axis_char = p_str.strip('+-').lower()
        elif a in ('--axis', '-a') and i + 1 < len(args):
            i += 1
            ax_part = args[i]
            sign = -1.0 if '-' in ax_part else 1.0
            axis_char = ax_part.strip('+-').lower()
        elif a in ('--val', '-v') and i + 1 < len(args):
            i += 1
            val_in = float(args[i])
        elif a in ('-o', '--output') and i + 1 < len(args):
            i += 1
            output_path = args[i]
        elif not a.startswith('-'):
            input_path = a
        elif a.lower() in ('-x', '-y', '-z') and i + 1 < len(args):
            axis_char = a.lstrip('-').lower()
            i += 1
            val_in = float(args[i])
        i += 1

    if not input_path:
        sys.exit("Usage: python3 extract_charm_wake.py <case_dir_or_file> [--plane -y=8.4]")

    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis_char]
    target_in = sign * val_in
    target_ft = target_in / 12.0
    return input_path, axis_char, axis_idx, target_in, target_ft, output_path


def extract_from_bvi(bvi_path, axis_idx, target_ft):
    """Extracts vortex crossings from CHARM jobybvi.dat centroid file."""
    print(f"Reading BVI centroid file: {bvi_path}")
    blocks = []
    current_az = 0.0
    current_pts = []
    reading = False

    with open(bvi_path, 'r', encoding='ascii', errors='replace') as f:
        for line in f:
            if "Azimuth angle" in line:
                current_az = float(line.split("=")[1].replace("degrees", "").strip())
                reading = False
            elif "Positive tip vortex location:" in line:
                reading = True
                current_pts = []
            elif "Negative tip vortex location:" in line:
                reading = False
                if len(current_pts) > 1:
                    blocks.append((current_az, np.array(current_pts)))
                    current_pts = []
            elif reading and line.strip() and not line.strip().startswith("WakeAge"):
                vals = parse_floats(line)
                if len(vals) >= 4:
                    current_pts.append(vals[:4])

    crossings = []
    rev = 1
    col = axis_idx + 1  # wake_age is col 0, x is 1, y is 2, z is 3

    for idx, (az, pts) in enumerate(blocks):
        if idx > 0 and az < blocks[idx - 1][0]:
            rev += 1
        v = pts[:, col]
        for i in range(len(pts) - 1):
            if (v[i] <= target_ft <= v[i + 1]) or (v[i + 1] <= target_ft <= v[i]):
                if v[i] != v[i + 1]:
                    t = max(0.0, min(1.0, (target_ft - v[i]) / (v[i + 1] - v[i])))
                    crossings.append({
                        'azimuth_deg': az,
                        'revolution': rev,
                        'blade': 1,
                        'wake_age_deg': pts[i, 0] + t * (pts[i + 1, 0] - pts[i, 0]),
                        'x_in': (pts[i, 1] + t * (pts[i + 1, 1] - pts[i, 1])) * 12.0,
                        'y_in': (pts[i, 2] + t * (pts[i + 1, 2] - pts[i, 2])) * 12.0,
                        'z_in': (pts[i, 3] + t * (pts[i + 1, 3] - pts[i, 3])) * 12.0,
                    })
    return crossings


def extract_from_plt(plt_path, axis_idx, target_ft):
    """Streams CHARM joby.plt and extracts tip trailer plane crossings."""
    print(f"Streaming PLT wake geometry file: {plt_path}")
    plt_dir = Path(plt_path).resolve().parent

    # Read NSPAN from companion input file
    nspan = 40
    bg_files = list(plt_dir.glob("*bg.inp"))
    if bg_files:
        with open(bg_files[0], 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if "NSPAN" in line:
                    idx = line.split().index("NSPAN")
                    nspan = abs(int(lines[i + 1].split()[idx]))
                    break

    blade_records = nspan * 4
    orient_records = 4
    inp_file = plt_dir / "joby.inp"
    if inp_file.exists():
        with open(inp_file) as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            parts = line.split()
            if "SFRAME" in parts:
                idx = parts.index("SFRAME")
                val = int(lines[i + 1].split()[idx])
                orient_records = 8 if val == 2 else 4
                break

    NULL_THRESHOLD = 1.0e9
    crossings = []

    def skip_floats(f, n):
        read = 0
        while read < n:
            read += len(parse_floats(f.readline()))

    with open(plt_path, 'r', encoding='ascii') as f:
        h1 = parse_floats(f.readline())
        nrotor, npsi = int(h1[0]), int(h1[1])

        r_lines = [f.readline() for _ in range(3 * nrotor)]
        h2 = parse_floats(r_lines[0])
        nblade = int(h2[0])
        nptfw = (int(h2[2]), int(h2[3]), int(h2[4]))
        nvort = (int(h2[5]), int(h2[6]), int(h2[7]))

        tip_idx_z1 = nvort[0] - 1
        tip_idx_z2 = nvort[1] - 1
        block_count = 0

        curr_marker = f.readline()
        while curr_marker:
            vals = parse_floats(curr_marker)
            if len(vals) != 1:
                break
            azimuth = vals[0]
            blade_idx = block_count % nblade
            rev_idx = block_count // (nrotor * nblade * npsi)

            # 1. Skip blade geometry
            for _ in range(blade_records):
                f.readline()

            # 2. Skip stalls & moments
            skip_floats(f, nspan)
            skip_floats(f, nspan)

            # 3. Skip orientation records
            for _ in range(orient_records):
                f.readline()

            # 4. Zone 1 trailers
            for tr in range(nvort[0]):
                pts = []
                for _ in range(nptfw[0] + 1):
                    p = parse_floats(f.readline())
                    pts.append((p[0], p[1], p[2]) if len(p) >= 3 and p[0] < NULL_THRESHOLD else None)

                if tr == tip_idx_z1:
                    for k in range(len(pts) - 1):
                        p1, p2 = pts[k], pts[k + 1]
                        if p1 and p2 and ((p1[axis_idx] <= target_ft <= p2[axis_idx]) or (p2[axis_idx] <= target_ft <= p1[axis_idx])):
                            if p1[axis_idx] != p2[axis_idx]:
                                t = max(0.0, min(1.0, (target_ft - p1[axis_idx]) / (p2[axis_idx] - p1[axis_idx])))
                                crossings.append({
                                    'azimuth_deg': azimuth,
                                    'revolution': rev_idx + 1,
                                    'blade': blade_idx + 1,
                                    'wake_age_deg': k * (360.0 / npsi),
                                    'x_in': (p1[0] + t * (p2[0] - p1[0])) * 12.0,
                                    'y_in': (p1[1] + t * (p2[1] - p1[1])) * 12.0,
                                    'z_in': (p1[2] + t * (p2[2] - p1[2])) * 12.0,
                                })

            # 5. Zone 2 trailers
            for tr in range(nvort[1]):
                pts = []
                for _ in range(nptfw[1] + 1):
                    p = parse_floats(f.readline())
                    pts.append((p[0], p[1], p[2]) if len(p) >= 3 and p[0] < NULL_THRESHOLD else None)

                if tr == tip_idx_z2:
                    for k in range(len(pts) - 1):
                        p1, p2 = pts[k], pts[k + 1]
                        if p1 and p2 and ((p1[axis_idx] <= target_ft <= p2[axis_idx]) or (p2[axis_idx] <= target_ft <= p1[axis_idx])):
                            if p1[axis_idx] != p2[axis_idx]:
                                t = max(0.0, min(1.0, (target_ft - p1[axis_idx]) / (p2[axis_idx] - p1[axis_idx])))
                                crossings.append({
                                    'azimuth_deg': azimuth,
                                    'revolution': rev_idx + 1,
                                    'blade': blade_idx + 1,
                                    'wake_age_deg': (nptfw[0] + k) * (360.0 / npsi),
                                    'x_in': (p1[0] + t * (p2[0] - p1[0])) * 12.0,
                                    'y_in': (p1[1] + t * (p2[1] - p1[1])) * 12.0,
                                    'z_in': (p1[2] + t * (p2[2] - p1[2])) * 12.0,
                                })

            # 6. Skip remainder of block until next azimuth marker or EOF
            next_line = f.readline()
            while next_line:
                parts = parse_floats(next_line)
                if len(parts) == 1:
                    break
                next_line = f.readline()
            curr_marker = next_line
            block_count += 1

    return crossings


def main():
    input_path, axis_char, axis_idx, target_in, target_ft, output_path = parse_cli()

    if os.path.isdir(input_path):
        plt = os.path.join(input_path, "joby.plt")
        bvi = os.path.join(input_path, "jobybvi.dat")
        target_file = plt if os.path.exists(plt) else bvi
        case_dir = input_path
    else:
        target_file = input_path
        case_dir = os.path.dirname(target_file) or "."

    sign_str = "-" if target_in < 0 else ""
    out_csv = output_path or os.path.join(case_dir, f"vortex_crossings_{sign_str}{axis_char.upper()}{abs(target_in):.1f}in.csv")

    print(f"Extracting crossings from {target_file} at {axis_char.upper()} = {target_in:+.2f} in...")
    if "bvi" in os.path.basename(target_file):
        crossings = extract_from_bvi(target_file, axis_idx, target_ft)
    else:
        crossings = extract_from_plt(target_file, axis_idx, target_ft)

    df = pd.DataFrame(crossings)
    df.to_csv(out_csv, index=False)
    print(f"Extracted {len(df):,} crossings -> {out_csv}")


if __name__ == "__main__":
    main()
