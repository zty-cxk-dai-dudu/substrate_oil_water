#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np


HEADER_RE = re.compile(r"i\s*=\s*(?P<step>-?\d+),\s*time\s*=\s*(?P<time>[-+0-9.Ee]+)")
C_CM_S = 2.99792458e10


def read_poscar(path: Path):
    lines = path.read_text().splitlines()
    scale = float(lines[1].split()[0])
    cell = np.array([[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)]) * scale
    species = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    coord_start = 8
    if lines[7].strip().lower().startswith("s"):
        coord_start = 9
    mode = lines[coord_start - 1].strip().lower()
    frac = []
    symbols = []
    for sym, count in zip(species, counts):
        for _ in range(count):
            symbols.append(sym)
            frac.append([float(x) for x in lines[coord_start + len(frac)].split()[:3]])
    if not mode.startswith("d"):
        raise ValueError("POSCAR coordinates must be Direct for this workflow.")
    return cell, species, counts, symbols, np.array(frac, dtype=np.float64)


def minimum_image_frac(delta: np.ndarray) -> np.ndarray:
    return delta - np.rint(delta)


def infer_waters(cell, species, counts, frac):
    starts = {}
    atom_id = 1
    for sym, count in zip(species, counts):
        starts[sym] = atom_id
        atom_id += count
    if "O" not in starts or "H" not in starts:
        raise ValueError("POSCAR must contain O and H.")
    o_ids = np.arange(starts["O"], starts["O"] + counts[species.index("O")])
    h_ids = np.arange(starts["H"], starts["H"] + counts[species.index("H")])
    waters = []
    for o_id in o_ids:
        d_frac = minimum_image_frac(frac[h_ids - 1] - frac[o_id - 1])
        d_cart = d_frac @ cell
        dist = np.linalg.norm(d_cart, axis=1)
        nearest = np.argsort(dist)[:2]
        if dist[nearest[1]] > 1.35:
            raise ValueError(f"O{o_id}: second-nearest H is {dist[nearest[1]]:.3f} A")
        z = float((frac[o_id - 1] @ cell)[2])
        waters.append(
            {
                "O_serial": int(o_id),
                "H1_serial": int(h_ids[nearest[0]]),
                "H2_serial": int(h_ids[nearest[1]]),
                "OH1_A": float(dist[nearest[0]]),
                "OH2_A": float(dist[nearest[1]]),
                "O_z_A": z,
            }
        )
    return waters


def build_layers(waters, width):
    z_min = min(w["O_z_A"] for w in waters)
    z_max = max(w["O_z_A"] for w in waters)
    layers = []
    lo = z_min
    index = 1
    while lo <= z_max + 1e-9:
        hi = lo + width
        if hi > z_max:
            selected = [w for w in waters if lo <= w["O_z_A"] <= hi + 1e-9]
        else:
            selected = [w for w in waters if lo <= w["O_z_A"] < hi]
        if selected:
            layers.append((index, lo, hi, selected))
            index += 1
        lo = hi
    return layers


def frame_count_xyz(path: Path, natoms: int) -> int:
    line_count = sum(1 for _ in path.open("r", errors="ignore"))
    block = natoms + 2
    if line_count % block != 0:
        raise ValueError(f"{path} line count {line_count} is not divisible by {block}.")
    return line_count // block


def read_velocity_window(path: Path, natoms: int, start_frame: int, selected_frames: int):
    velocities = np.empty((selected_frames, natoms, 3), dtype=np.float64)
    steps = []
    times = []
    with path.open("r", errors="ignore") as handle:
        for frame in range(start_frame + selected_frames):
            nat_line = handle.readline()
            if not nat_line:
                raise ValueError("Unexpected EOF in velocity XYZ.")
            n = int(nat_line.strip())
            if n != natoms:
                raise ValueError(f"Frame {frame}: expected {natoms} atoms, got {n}.")
            header = handle.readline()
            match = HEADER_RE.search(header)
            if frame >= start_frame:
                out_i = frame - start_frame
                steps.append(None if match is None else int(match.group("step")))
                times.append(None if match is None else float(match.group("time")))
                for atom_i in range(natoms):
                    parts = handle.readline().split()
                    velocities[out_i, atom_i] = [float(parts[1]), float(parts[2]), float(parts[3])]
            else:
                for _ in range(natoms):
                    handle.readline()
    return velocities, steps, times


def spectrum_for_atoms(velocities: np.ndarray, atom_serials: list[int], dt_fs: float):
    idx = np.array([serial - 1 for serial in atom_serials], dtype=np.int64)
    data = velocities[:, idx, :].reshape(velocities.shape[0], -1)
    data = data - data.mean(axis=0, keepdims=True)
    window = np.hanning(data.shape[0])[:, None]
    fft = np.fft.rfft(data * window, axis=0)
    power = np.mean(np.abs(fft) ** 2, axis=1)
    freq_hz = np.fft.rfftfreq(data.shape[0], d=dt_fs * 1.0e-15)
    wn = freq_hz / C_CM_S
    if power.max() > 0:
        power = power / power.max()
    return wn, power


def write_curve(path: Path, wn: np.ndarray, intensity: np.ndarray, max_cm: float):
    mask = wn <= max_cm + 1e-9
    with path.open("w") as handle:
        handle.write("# Frequency_cm-1 Normalized_VDOS\n")
        for x, y in zip(wn[mask], intensity[mask]):
            handle.write(f"{x:.8f} {y:.12g}\n")


def layer_name(index: int, lo: float, hi: float) -> str:
    return f"layer{index:02d}_z{lo:.3f}_{hi:.3f}".replace(".", "p")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poscar", default="POSCAR", type=Path)
    parser.add_argument("--velocity-xyz", default="merged-vel-1.xyz", type=Path)
    parser.add_argument("--last-frames", default=50000, type=int)
    parser.add_argument("--dt-fs", default=None, type=float)
    parser.add_argument("--layer-width", default=3.0, type=float)
    parser.add_argument("--max-cm", default=4000.0, type=float)
    parser.add_argument("--per-water-outdir", default="vdos_per_water_velocity_fft")
    parser.add_argument("--layer-outdir", default="vdos_water_layers_3A_velocity_fft")
    args = parser.parse_args()

    cell, species, counts, _symbols, frac = read_poscar(args.poscar)
    natoms = sum(counts)
    waters = infer_waters(cell, species, counts, frac)
    total_frames = frame_count_xyz(args.velocity_xyz, natoms)
    start_frame = max(0, total_frames - args.last_frames)
    selected_frames = total_frames - start_frame
    velocities, steps, times = read_velocity_window(args.velocity_xyz, natoms, start_frame, selected_frames)
    if args.dt_fs is None:
        valid_times = [t for t in times[:2] if t is not None]
        dt_fs = float(valid_times[1] - valid_times[0]) if len(valid_times) == 2 else 1.0
    else:
        dt_fs = args.dt_fs

    per_water = Path(args.per_water_outdir)
    per_water.mkdir(exist_ok=True)
    with (per_water / "water_atom_map.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["O_serial", "O_atom_id", "H1_atom_id", "H2_atom_id", "O_z_A", "OH1_A", "OH2_A"])
        writer.writeheader()
        for w in waters:
            writer.writerow({
                "O_serial": w["O_serial"],
                "O_atom_id": w["O_serial"],
                "H1_atom_id": w["H1_serial"],
                "H2_atom_id": w["H2_serial"],
                "O_z_A": f"{w['O_z_A']:.8f}",
                "OH1_A": f"{w['OH1_A']:.6f}",
                "OH2_A": f"{w['OH2_A']:.6f}",
            })
    with (per_water / "summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["O_serial", "status", "output_dat", "output_txt"])
        for i, w in enumerate(waters, start=1):
            wn, power = spectrum_for_atoms(velocities, [w["O_serial"], w["H1_serial"], w["H2_serial"]], dt_fs)
            dat = per_water / f"O{w['O_serial']}.dat"
            txt = per_water / f"O{w['O_serial']}.txt"
            write_curve(dat, wn, power, args.max_cm)
            write_curve(txt, wn, power, args.max_cm)
            writer.writerow([w["O_serial"], "ok", dat, txt])
            if i % 10 == 0 or i == len(waters):
                print(f"per-water VDOS {i}/{len(waters)}", flush=True)

    layer_dir = Path(args.layer_outdir)
    layer_dir.mkdir(exist_ok=True)
    layers = build_layers(waters, args.layer_width)
    with (layer_dir / "water_atom_map.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["O_serial", "O_atom_id", "H1_atom_id", "H2_atom_id", "O_z_A", "OH1_A", "OH2_A"])
        writer.writeheader()
        for w in waters:
            writer.writerow({
                "O_serial": w["O_serial"],
                "O_atom_id": w["O_serial"],
                "H1_atom_id": w["H1_serial"],
                "H2_atom_id": w["H2_serial"],
                "O_z_A": f"{w['O_z_A']:.8f}",
                "OH1_A": f"{w['OH1_A']:.6f}",
                "OH2_A": f"{w['OH2_A']:.6f}",
            })
    with (layer_dir / "layer_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["layer", "run_dir", "z_min_A", "z_max_A", "n_waters", "n_atoms", "O_serials", "status", "output_dat", "output_txt"])
        for index, lo, hi, selected in layers:
            name = layer_name(index, lo, hi)
            serials = []
            for w in selected:
                serials.extend([w["O_serial"], w["H1_serial"], w["H2_serial"]])
            wn, power = spectrum_for_atoms(velocities, serials, dt_fs)
            dat = layer_dir / f"{name}.dat"
            txt = layer_dir / f"{name}.txt"
            write_curve(dat, wn, power, args.max_cm)
            write_curve(txt, wn, power, args.max_cm)
            writer.writerow([index, name, f"{lo:.8f}", f"{hi:.8f}", len(selected), len(serials), " ".join(str(w["O_serial"]) for w in selected), "ok", dat, txt])
            print(f"{name}: waters={len(selected)} atoms={len(serials)} ok", flush=True)

    meta = {
        "method": "velocity_xyz_fft_power_spectrum",
        "note": "Used because native VASPKIT 728 segfaulted while reading this XDATCAR.",
        "velocity_xyz": str(args.velocity_xyz),
        "poscar": str(args.poscar),
        "total_frames": total_frames,
        "start_frame_index_0_based": start_frame,
        "selected_frames": selected_frames,
        "dt_fs": dt_fs,
        "max_wavenumber_cm-1": args.max_cm,
        "water_count": len(waters),
        "layer_width_A": args.layer_width,
        "normalization": "each curve divided by its own maximum",
    }
    (per_water / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    (layer_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
