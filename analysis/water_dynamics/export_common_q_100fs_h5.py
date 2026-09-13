#!/usr/bin/env python3
"""Export a common 100 fs-sampled S(q,w) for fair cross-trajectory comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

import analyze_six_advanced_collective as core


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--candidate-screen", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--q-orders", required=True)
    parser.add_argument("--source-frames", default=500000, type=int)
    parser.add_argument("--layer-width", default=2.0, type=float)
    parser.add_argument("--n-layers", default=12, type=int)
    args = parser.parse_args()
    paths = sorted(args.input.glob("*segment_*.h5"))
    if not paths:
        raise RuntimeError("No source HDF5 segments")
    with h5py.File(paths[0], "r") as handle:
        symbols = np.asarray(handle["symbols"]).astype(str)
        cell = np.asarray(handle["cell_A"], dtype=float)
        position0 = np.asarray(handle["positions_A"][0], dtype=float)
    waters = core.water_topology(symbols, position0, cell)
    oxygen = waters[:, 0]
    mean_z_table = core.load_candidate_mean_z(args.candidate_screen)
    mean_z = np.asarray([mean_z_table[int(index) + 1] for index in oxygen])
    fixed_layer = np.floor(mean_z / args.layer_width).astype(int)
    margin = max(1, int(round(args.n_layers / 6.0)))
    members = np.flatnonzero(
        (fixed_layer >= margin) & (fixed_layer < args.n_layers - margin)
    )
    samples = []
    for part in core.source_parts(paths, args.source_frames, 100, ("positions_A",)):
        samples.append(np.asarray(part["positions_A"][:, oxygen[members], 0]))
    oxygen_x = np.concatenate(samples)
    if len(oxygen_x) != 5000:
        raise RuntimeError(f"Expected 5000 samples, found {len(oxygen_x)}")
    q_orders = np.asarray([int(value) for value in args.q_orders.split(",")])
    q_actual = 2.0 * np.pi * q_orders / cell[0, 0]
    samples_per_block = 100
    frequency_full = np.fft.fftfreq(samples_per_block, d=0.1) * core.CM_PER_PS
    keep = (frequency_full >= 0.0) & (frequency_full <= 150.0)
    frequency = frequency_full[keep]
    window = np.blackman(samples_per_block)
    window_norm = float(np.sum(window ** 2))
    spectrum = np.zeros((len(q_orders), len(frequency)), dtype=float)
    for block in range(50):
        x = oxygen_x[block * samples_per_block:(block + 1) * samples_per_block]
        rho = np.asarray(
            [np.sum(np.exp(1.0j * q * x), axis=1) for q in q_actual]
        ).T
        rho -= np.mean(rho, axis=0, keepdims=True)
        transformed = np.fft.fft(rho * window[:, None], axis=0)[keep]
        spectrum += (np.abs(transformed) ** 2 / (len(members) * window_norm)).T
    spectrum /= 50
    rows = []
    peaks = []
    for iq, (order, q) in enumerate(zip(q_orders, q_actual)):
        rows.extend(
            [iq + 1, int(order), q, f, value]
            for f, value in zip(frequency, spectrum[iq])
        )
        band = (frequency >= 5.0) & (frequency <= 150.0)
        peak = np.flatnonzero(band)[np.argmax(spectrum[iq, band])]
        peaks.append([iq + 1, int(order), q, frequency[peak], spectrum[iq, peak]])
    core.write_csv(
        args.output / "common_q_dynamic_structure_factor_100fs.csv",
        ["common_q_index", "reciprocal_order_x", "actual_q_A-1",
         "frequency_cm-1", "S_q_omega"], rows,
    )
    core.write_csv(
        args.output / "common_q_peak_summary_100fs.csv",
        ["common_q_index", "reciprocal_order_x", "actual_q_A-1",
         "peak_frequency_5_150cm-1", "peak_S_q_omega"], peaks,
    )
    metadata = {
        "status": "complete",
        "position_sampling_fs": 100.0,
        "frequency_limit_cm-1": 150.0,
        "blocks": 50,
        "block_ps": 10.0,
        "central_water_count": len(members),
        "q_orders": q_orders.tolist(),
        "q_actual_A-1": q_actual.tolist(),
    }
    (args.output / "common_q_100fs_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
