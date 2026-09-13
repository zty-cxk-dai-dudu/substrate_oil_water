#!/usr/bin/env python3
"""Velocity-FFT VDOS for water molecules whose mean O z is in a fixed slab."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np

CM_PER_PS = 33.3564095198152  # (1/ps) to cm^-1


def water_topology(symbols: np.ndarray, positions: np.ndarray, cell: np.ndarray):
    """Assign two unique nearby H atoms to every oxygen by minimum-image distance."""
    oxygen = np.flatnonzero(symbols == "O")
    hydrogen = np.flatnonzero(symbols == "H")
    lengths = np.diag(cell)
    pairs = []
    for o in oxygen:
        delta = positions[hydrogen] - positions[o]
        delta -= lengths * np.rint(delta / lengths)
        for h, d in zip(hydrogen, np.linalg.norm(delta, axis=1)):
            if d <= 1.30:
                pairs.append((float(d), int(o), int(h)))
    pairs.sort()
    assigned_o = {int(o): [] for o in oxygen}
    used_h = set()
    for _, o, h in pairs:
        if len(assigned_o[o]) < 2 and h not in used_h:
            assigned_o[o].append(h)
            used_h.add(h)
    bad = {o: hs for o, hs in assigned_o.items() if len(hs) != 2}
    if bad:
        raise RuntimeError(f"water topology failed for O indices (zero-based): {bad}")
    return [(o, *assigned_o[o]) for o in oxygen]


def autocorrelation_fft(x: np.ndarray, nlag: int) -> np.ndarray:
    """Mean component-wise VACF over selected atoms using an unbiased FFT estimator."""
    n = x.shape[0]
    nfft = 1 << (2 * n - 1).bit_length()
    xf = np.fft.rfft(x, n=nfft, axis=0)
    corr = np.fft.irfft(xf * xf.conj(), n=nfft, axis=0)[:nlag]
    corr = corr.sum(axis=(1, 2)) / (np.arange(n, n - nlag, -1) * x.shape[1])
    return corr


def spectrum(v: np.ndarray, dt_fs: float, max_lag_ps: float):
    """Non-negative one-sided velocity power spectrum, averaged over time blocks."""
    nlag = min(v.shape[0], int(round(max_lag_ps * 1000.0 / dt_fs)))
    nblock = v.shape[0] // nlag
    if nblock < 1:
        raise RuntimeError("trajectory shorter than requested spectral window")
    x = v[: nblock * nlag].reshape(nblock, nlag, v.shape[1], 3)
    x = x - x.mean(axis=1, keepdims=True)
    window = np.blackman(nlag)
    nfft = 1 << (nlag * 4 - 1).bit_length()
    xf = np.fft.rfft(x * window[None, :, None, None], n=nfft, axis=1)
    spec = (np.abs(xf) ** 2).mean(axis=(0, 2, 3)) / np.sum(window ** 2)
    vacf = autocorrelation_fft(x.reshape(-1, v.shape[1], 3), nlag)
    vacf /= vacf[0]
    freq = np.fft.rfftfreq(nfft, d=dt_fs * 1e-3) * CM_PER_PS
    return freq, vacf, spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--z-min", type=float, default=22.0)
    ap.add_argument("--z-max", type=float, default=25.0)
    ap.add_argument("--max-lag-ps", type=float, default=10.0)
    args = ap.parse_args()
    segs = sorted((args.input / "segments").glob("*.h5"))
    if not segs:
        raise FileNotFoundError("No HDF5 segments found")
    args.output.mkdir(parents=True, exist_ok=True)

    with h5py.File(segs[0], "r") as f:
        symbols = np.asarray(f["symbols"]).astype(str)
        positions0 = np.asarray(f["positions_A"][0])
        cell = np.asarray(f["cell_A"])
        dt_fs = float(f.attrs["output_interval_fs"])
    waters = water_topology(symbols, positions0, cell)
    oidx = np.array([w[0] for w in waters], dtype=int)
    zsum = np.zeros(len(waters))
    nframes = 0
    for p in segs:
        with h5py.File(p, "r") as f:
            pos = f["positions_A"][:, oidx, 2]
            zsum += pos.sum(axis=0)
            nframes += pos.shape[0]
    mean_z = zsum / nframes
    selected = np.flatnonzero((mean_z >= args.z_min) & (mean_z < args.z_max))
    if len(selected) == 0:
        raise RuntimeError(f"No waters with mean O z in [{args.z_min}, {args.z_max}) A")
    # h5py fancy indexing requires monotonically increasing integer indices.
    all_atoms = np.sort(np.array([a for iw in selected for a in waters[iw]], dtype=int))
    h_atoms = np.array([a for iw in selected for a in waters[iw][1:]], dtype=int)
    o_atoms = oidx[selected]
    vel = np.empty((nframes, len(all_atoms), 3), dtype=np.float32)
    offset = 0
    for p in segs:
        with h5py.File(p, "r") as f:
            v = f["velocities_A_per_fs"][:, all_atoms, :]
            vel[offset:offset + len(v)] = v
            offset += len(v)
    local_o = np.array([np.where(all_atoms == x)[0][0] for x in o_atoms])
    local_h = np.array([np.where(all_atoms == x)[0][0] for x in h_atoms])
    freq, vacf_all, spec_all = spectrum(vel, dt_fs, args.max_lag_ps)
    _, vacf_h, spec_h = spectrum(vel[:, local_h], dt_fs, args.max_lag_ps)
    _, vacf_o, spec_o = spectrum(vel[:, local_o], dt_fs, args.max_lag_ps)
    mask = (freq >= 1200.0) & (freq <= 4000.0)
    norm = lambda x: x / np.max(x[mask])
    with (args.output / "selected_water_mean_z.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["water_index", "O_serial_1based", "mean_O_z_A", "H1_serial_1based", "H2_serial_1based", "selected_22_25A"])
        for i, (water, z) in enumerate(zip(waters, mean_z)):
            w.writerow([i + 1, water[0] + 1, f"{z:.8f}", water[1] + 1, water[2] + 1, int(i in selected)])
    with (args.output / "vdos_z22_25A_source_data.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["wavenumber_cm-1", "VDOS_all_normalized", "VDOS_H_normalized", "VDOS_O_normalized"])
        for row in zip(freq[mask], norm(spec_all)[mask], norm(spec_h)[mask], norm(spec_o)[mask]): w.writerow([f"{x:.10g}" for x in row])
    np.savez_compressed(args.output / "vdos_z22_25A_full.npz", wavenumber_cm1=freq, vacf_all=vacf_all, vacf_H=vacf_h, vacf_O=vacf_o, vdos_all=norm(spec_all), vdos_H=norm(spec_h), vdos_O=norm(spec_o))
    meta = {"input_segments": len(segs), "frames": nframes, "sampling_interval_fs": dt_fs, "z_selection_A": [args.z_min, args.z_max], "selection_basis": "mean O Cartesian z over all 500 ps", "water_count_total": len(waters), "water_count_selected": int(len(selected)), "atom_count_selected": int(len(all_atoms)), "spectral_block_ps": args.max_lag_ps, "spectral_blocks_averaged": nframes // int(round(args.max_lag_ps * 1000.0 / dt_fs)), "method": "one-sided velocity power spectrum averaged over atoms, Cartesian components, and non-overlapping Blackman-tapered blocks", "spectrum_range_export_cm-1": [1200, 4000]}
    (args.output / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
