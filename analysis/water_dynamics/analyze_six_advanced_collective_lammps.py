#!/usr/bin/env python3
"""Run the six collective-water analyses on split LAMMPS position/velocity dumps."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import time
from pathlib import Path

import numpy as np

import analyze_six_advanced_collective as core


def read_frame(handle, wanted_ids: set[int] | None = None):
    while True:
        marker = handle.readline()
        if not marker:
            return None
        if marker.strip() == "ITEM: TIMESTEP":
            break
    step = int(handle.readline())
    if handle.readline().strip() != "ITEM: NUMBER OF ATOMS":
        raise RuntimeError("Invalid atom-count header")
    natoms = int(handle.readline())
    if not handle.readline().startswith("ITEM: BOX BOUNDS"):
        raise RuntimeError("Invalid box header")
    bounds = np.asarray(
        [[float(value) for value in handle.readline().split()[:2]] for _ in range(3)]
    )
    header = handle.readline().split()
    if header[:2] != ["ITEM:", "ATOMS"]:
        raise RuntimeError("Invalid atom header")
    columns = header[2:]
    rows = []
    for _ in range(natoms):
        fields = handle.readline().split()
        if not fields:
            raise EOFError("Incomplete final frame")
        atom_id = int(fields[0])
        if wanted_ids is None or atom_id in wanted_ids:
            rows.append(fields)
    return step, natoms, bounds, columns, rows


def build_topology(position_dump: Path, restraints: Path):
    restrained_h = {
        int(row["H_id"])
        for row in json.loads(restraints.read_text())["pairs"]
    }
    with position_dump.open() as handle:
        step, natoms, bounds, columns, rows = read_frame(handle)
    if step != 0:
        raise RuntimeError(f"Initial position step is {step}, expected 0")
    col = {name: index for index, name in enumerate(columns)}
    ids = np.asarray([int(row[col["id"]]) for row in rows])
    types = np.asarray([int(row[col["type"]]) for row in rows])
    xyz = np.asarray(
        [[float(row[col[axis]]) for axis in ("x", "y", "z")] for row in rows]
    )
    order = np.argsort(ids)
    ids, types, xyz = ids[order], types[order], xyz[order]
    lengths = bounds[:, 1] - bounds[:, 0]
    substrate_o = ((ids >= 1) & (ids <= 48)) | ((ids >= 411) & (ids <= 458))
    water_o = ids[(types == 1) & ~substrate_o]
    water_h = ids[(types == 3) & ~np.isin(ids, list(restrained_h))]
    id_to_index = {int(atom_id): index for index, atom_id in enumerate(ids)}
    candidates = []
    for oxygen_id in water_o:
        delta = xyz[[id_to_index[int(h)] for h in water_h]] \
            - xyz[id_to_index[int(oxygen_id)]]
        delta -= lengths * np.rint(delta / lengths)
        for hydrogen_id, distance in zip(water_h, np.linalg.norm(delta, axis=1)):
            if distance <= 1.30:
                candidates.append((float(distance), int(oxygen_id), int(hydrogen_id)))
    candidates.sort()
    assigned = {int(oxygen_id): [] for oxygen_id in water_o}
    used_h = set()
    for _distance, oxygen_id, hydrogen_id in candidates:
        if len(assigned[oxygen_id]) < 2 and hydrogen_id not in used_h:
            assigned[oxygen_id].append(hydrogen_id)
            used_h.add(hydrogen_id)
    waters = np.asarray(
        [(int(o), *assigned[int(o)]) for o in water_o if len(assigned[int(o)]) == 2],
        dtype=np.int64,
    )
    if len(waters) != len(water_o) or len(used_h) != 2 * len(water_o):
        raise RuntimeError(
            f"Water topology incomplete: {len(waters)}/{len(water_o)} waters"
        )
    return natoms, np.diag(lengths), waters


def prepare_position_cache(args):
    if args.position_cache.exists() and args.mapping.exists():
        with np.load(args.position_cache) as cache:
            return {name: np.asarray(cache[name]) for name in cache.files}

    natoms, cell, waters = build_topology(args.position_dump, args.restraints)
    wanted_ids = set(map(int, waters.ravel()))
    atom_column = {int(atom_id): index for index, atom_id in enumerate(waters.ravel())}
    frames = []
    steps = []
    with args.position_dump.open() as handle:
        while True:
            try:
                frame = read_frame(handle, wanted_ids)
            except EOFError:
                break
            if frame is None:
                break
            step, frame_atoms, _bounds, columns, rows = frame
            if frame_atoms != natoms or len(rows) != len(wanted_ids):
                raise RuntimeError(f"Invalid position frame at step {step}")
            col = {name: index for index, name in enumerate(columns)}
            flat = np.empty((len(waters) * 3, 3), dtype=np.float32)
            for row in rows:
                flat[atom_column[int(row[col["id"]])]] = [
                    float(row[col[axis]]) for axis in ("x", "y", "z")
                ]
            frames.append(flat.reshape(len(waters), 3, 3))
            steps.append(step)
    position = np.asarray(frames, dtype=np.float32)
    steps = np.asarray(steps, dtype=np.int64)
    if len(position) != 5001 or steps[0] != 0 or steps[-1] != 1_000_000:
        raise RuntimeError(
            f"Position dump incomplete: frames={len(position)}, "
            f"first={steps[0]}, last={steps[-1]}"
        )
    mean_z = np.mean(position[:, :, 0, 2], axis=0)
    z_origin = math.floor(float(np.min(mean_z)) / args.layer_width) * args.layer_width
    n_layers = int(math.ceil((float(np.max(mean_z)) - z_origin) / args.layer_width))
    fixed_layer = np.floor((mean_z - z_origin) / args.layer_width).astype(np.int64)
    counts = np.bincount(fixed_layer, minlength=n_layers)
    if np.any(counts == 0):
        raise RuntimeError(f"Empty native 2 A layer: counts={counts.tolist()}")

    mass_total = core.MASS_O + 2 * core.MASS_H
    with args.mapping.open("w") as stream:
        stream.write(f"{natoms} {n_layers} 500000\n")
        for water_index, (oxygen, hydrogen1, hydrogen2) in enumerate(waters):
            layer = int(fixed_layer[water_index])
            for atom_id, mass in (
                (oxygen, core.MASS_O),
                (hydrogen1, core.MASS_H),
                (hydrogen2, core.MASS_H),
            ):
                stream.write(
                    f"{int(atom_id)} {layer} "
                    f"{mass / mass_total / counts[layer]:.17g} "
                    f"{mass / mass_total / len(waters):.17g}\n"
                )

    np.savez_compressed(
        args.position_cache,
        position_A=position,
        step=steps,
        cell_A=cell,
        waters_id=waters,
        mean_oxygen_z_A=mean_z,
        z_origin_A=np.asarray(z_origin),
        fixed_layer=fixed_layer,
        layer_counts=counts,
        natoms=np.asarray(natoms),
    )
    return {
        "position_A": position,
        "step": steps,
        "cell_A": cell,
        "waters_id": waters,
        "mean_oxygen_z_A": mean_z,
        "z_origin_A": np.asarray(z_origin),
        "fixed_layer": fixed_layer,
        "layer_counts": counts,
        "natoms": np.asarray(natoms),
    }


def hbond_graph(position, cell, fixed_layer, n_layers, args):
    nwater = len(fixed_layer)
    upper_i, upper_j = np.triu_indices(nwater, 1)
    valid = (fixed_layer[upper_i] >= 0) & (fixed_layer[upper_j] >= 0)
    left = np.minimum(fixed_layer[upper_i], fixed_layer[upper_j])
    right = np.maximum(fixed_layer[upper_i], fixed_layer[upper_j])
    pair_code = left * n_layers + right
    total = np.zeros((n_layers, n_layers), dtype=float)
    samples = 0
    # Match the other two systems: exactly 5000 samples over [0, 500 ps).
    for start in range(0, 5000, 10):
        chunk = position[start:start + 10]
        directed = core.directed_hbond(
            chunk[:, :, 0], chunk[:, :, 1:], cell,
            args.hbond_ho, args.hbond_oo, args.hbond_angle,
        )
        adjacency = directed | np.swapaxes(directed, 1, 2)
        for frame in adjacency:
            edge = frame[upper_i, upper_j] & valid
            total += np.bincount(
                pair_code[edge], minlength=n_layers ** 2
            ).reshape(n_layers, n_layers)
            samples += 1
        if start == 0 or (start // 10 + 1) % 50 == 0:
            print(json.dumps({"graph_samples": samples, "graph_total": 5000}), flush=True)
    upper = total / samples
    mean_edges = upper + upper.T
    np.fill_diagonal(mean_edges, np.diag(upper))
    counts = np.bincount(fixed_layer, minlength=n_layers)
    weight = np.zeros_like(mean_edges)
    for i in range(n_layers):
        for j in range(i + 1, n_layers):
            weight[i, j] = weight[j, i] = mean_edges[i, j] / math.sqrt(
                max(int(counts[i]) * int(counts[j]), 1)
            )
    degree = np.sum(weight, axis=1)
    invsqrt = np.zeros_like(degree)
    invsqrt[degree > 0] = 1.0 / np.sqrt(degree[degree > 0])
    laplacian = np.eye(n_layers) - invsqrt[:, None] * weight * invsqrt[None, :]
    laplacian[degree == 0] = 0.0
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    return mean_edges, weight, counts, eigenvalues, eigenvectors, samples


def sampled_layer_positions(position, cell, fixed_layer, n_layers):
    groups = [np.flatnonzero(fixed_layer == layer) for layer in range(n_layers)]
    oxygen = np.asarray(position[:5000, :, 0], dtype=float)
    unwrapped = np.empty_like(oxygen)
    unwrapped[0] = oxygen[0]
    for frame in range(1, len(oxygen)):
        unwrapped[frame] = unwrapped[frame - 1] + core.minimum_image(
            oxygen[frame] - oxygen[frame - 1], cell
        )
    layer_position = np.empty((len(unwrapped), n_layers, 3), dtype=float)
    for layer, members in enumerate(groups):
        layer_position[:, layer] = np.mean(unwrapped[:, members], axis=1)
    return layer_position, np.mean(unwrapped, axis=1)


def velocity_spectra(args, reduced_velocity, graph_eigenvectors, n_layers):
    record_values = (n_layers + 1) * 3
    expected_bytes = args.source_frames * record_values * np.dtype(np.float32).itemsize
    if reduced_velocity.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"Reduced velocity size {reduced_velocity.stat().st_size}/{expected_bytes}"
        )
    data = np.memmap(
        reduced_velocity, dtype=np.float32, mode="r",
        shape=(args.source_frames, n_layers + 1, 3),
    )
    layer_signal = data[:, :n_layers]
    global_signal = data[:, n_layers]
    block_frames = int(round(args.block_ps * 1000.0 / args.source_dt_fs))
    nblocks = args.source_frames // block_frames
    window = np.blackman(block_frames)
    window_norm = float(np.sum(window ** 2))
    full_frequency = np.fft.rfftfreq(
        block_frames, d=args.source_dt_fs / 1000.0
    ) * core.CM_PER_PS
    export_mask = full_frequency <= args.max_frequency_cm
    frequency = full_frequency[export_mask]
    band_masks = {
        band: (frequency >= band[0]) & (frequency < band[1]) for band in core.BANDS
    }
    variants = ("raw", "global_COM_corrected")
    cross_accum = {
        variant: np.zeros((len(frequency), n_layers, n_layers), dtype=np.complex128)
        for variant in variants
    }
    band_block = {variant: [] for variant in variants}
    graph_spectrum = {
        variant: np.zeros((n_layers, len(frequency)), dtype=float)
        for variant in variants
    }
    for block in range(nblocks):
        start = block * block_frames
        # LAMMPS metal units store velocity in A/ps; the HDF5 comparison
        # trajectories use A/fs.  Apply the explicit 1e-3 conversion before
        # any dimensional spectral power is accumulated.
        raw = (np.asarray(layer_signal[start:start + block_frames], dtype=float)
               * args.velocity_scale_Afs_per_Aps)
        global_velocity = (
            np.asarray(global_signal[start:start + block_frames], dtype=float)
            * args.velocity_scale_Afs_per_Aps
        )
        signals = {
            "raw": raw,
            "global_COM_corrected": raw - global_velocity[:, None, :],
        }
        for variant, signal in signals.items():
            centered = signal - np.mean(signal, axis=0, keepdims=True)
            transformed = np.fft.rfft(
                centered * window[:, None, None], axis=0
            )[export_mask]
            spectrum = core.cross_spectrum(transformed, window_norm)
            cross_accum[variant] += spectrum
            band_block[variant].append({
                band: np.sum(spectrum[mask], axis=0)
                for band, mask in band_masks.items()
            })
            projected = np.einsum(
                "tla,lm->tma", signal, graph_eigenvectors, optimize=True
            )
            projected -= np.mean(projected, axis=0, keepdims=True)
            mode_fft = np.fft.rfft(
                projected * window[:, None, None], axis=0
            )[export_mask]
            graph_spectrum[variant] += (
                np.sum(np.abs(mode_fft) ** 2, axis=2).T / window_norm
            )
        print(json.dumps({"native_blocks_done": block + 1,
                          "native_blocks_total": nblocks}), flush=True)
    for variant in variants:
        cross_accum[variant] /= nblocks
        graph_spectrum[variant] /= nblocks
    return {
        "frequency": frequency,
        "band_masks": band_masks,
        "cross_spectrum": cross_accum,
        "band_block": band_block,
        "graph_spectrum": graph_spectrum,
        "layer_signal": layer_signal,
        "global_signal": global_signal,
        "nblocks": nblocks,
        "block_frames": block_frames,
    }


def export_sparse_q(args, position, fixed_layer, cell):
    margin = max(1, int(round(args.n_layers / 6.0)))
    members = np.flatnonzero(
        (fixed_layer >= margin) & (fixed_layer < args.n_layers - margin)
    )
    q_orders = np.asarray([int(value) for value in args.q_orders.split(",")])
    q_actual = 2.0 * np.pi * q_orders / cell[0, 0]
    samples_per_block = int(round(args.block_ps / 0.1))
    frequency_full = np.fft.fftfreq(samples_per_block, d=0.1) * core.CM_PER_PS
    keep = (frequency_full >= 0.0) & (frequency_full <= 150.0)
    frequency = frequency_full[keep]
    spectrum = np.zeros((len(q_orders), len(frequency)), dtype=float)
    window = np.blackman(samples_per_block)
    window_norm = float(np.sum(window ** 2))
    oxygen_x = position[:5000, members, 0, 0]
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
            [iq + 1, int(order), q, freq, value]
            for freq, value in zip(frequency, spectrum[iq])
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
    return q_orders, q_actual, len(members)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--position-dump", required=True, type=Path)
    parser.add_argument("--velocity-dump", required=True, type=Path)
    parser.add_argument("--restraints", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--layer-width", default=2.0, type=float)
    parser.add_argument("--source-frames", default=500000, type=int)
    parser.add_argument("--source-dt-fs", default=1.0, type=float)
    parser.add_argument("--velocity-scale-Afs-per-Aps", default=1.0e-3, type=float)
    parser.add_argument("--block-ps", default=10.0, type=float)
    parser.add_argument("--position-stride", default=100, type=int)
    parser.add_argument("--max-frequency-cm", default=300.0, type=float)
    parser.add_argument("--q-orders", default="2,4,6,8")
    parser.add_argument("--pca-modes", default=6, type=int)
    parser.add_argument("--stft-window-ps", default=10.0, type=float)
    parser.add_argument("--stft-step-ps", default=1.0, type=float)
    parser.add_argument("--hbond-ho", default=2.45, type=float)
    parser.add_argument("--hbond-oo", default=3.50, type=float)
    parser.add_argument("--hbond-angle", default=150.0, type=float)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.position_cache = args.output / "sio2_water_position_100fs_cache.npz"
    args.mapping = args.output / "water_atom_to_native_2A_layer_mapping.txt"
    args.reduced_velocity = args.output / "water_layer_COM_velocity_1fs.bin"
    started = time.time()
    cache = prepare_position_cache(args)
    args.n_layers = len(cache["layer_counts"])
    print(json.dumps({
        "position_prepass": "complete",
        "water_count": len(cache["waters_id"]),
        "position_frames": len(cache["position_A"]),
        "native_layers": args.n_layers,
        "layer_counts": cache["layer_counts"].tolist(),
        "z_origin_A": float(cache["z_origin_A"]),
    }, indent=2), flush=True)
    if args.prepare_only:
        return
    if not args.reduced_velocity.exists():
        raise RuntimeError(
            f"Run reduce_lammps_water_layer_velocity first; missing {args.reduced_velocity}"
        )

    graph = hbond_graph(
        cache["position_A"], cache["cell_A"], cache["fixed_layer"],
        args.n_layers, args,
    )
    mean_edges, graph_weight, graph_counts, graph_values, graph_vectors, samples = graph
    layer_position, global_position = sampled_layer_positions(
        cache["position_A"], cache["cell_A"], cache["fixed_layer"], args.n_layers
    )
    native = velocity_spectra(args, args.reduced_velocity, graph_vectors, args.n_layers)
    native["sampled_layer_position"] = layer_position
    native["sampled_global_position"] = global_position

    core.export_complex_coherence(args, native)
    core.export_pca(args, native)
    core.export_correlation_length(args, native)
    core.export_stft(args, native)
    q_orders, q_actual, q_members = export_sparse_q(
        args, cache["position_A"], cache["fixed_layer"], cache["cell_A"]
    )
    core.export_graph_modes(
        args, mean_edges, graph_weight, graph_counts,
        graph_values, graph_vectors, native,
    )

    metadata = {
        "status": "complete",
        "system": "sio2_fixed_water",
        "source_frames": args.source_frames,
        "source_sampling_interval_fs": args.source_dt_fs,
        "source_velocity_unit": "A/ps (LAMMPS metal units)",
        "spectral_velocity_unit": "A/fs after explicit 1e-3 conversion",
        "position_sampling_interval_fs": 100.0,
        "water_count": len(cache["waters_id"]),
        "native_layer_width_A": args.layer_width,
        "native_layer_count": args.n_layers,
        "native_layer_z_origin_A": float(cache["z_origin_A"]),
        "fixed_layer_water_counts": graph_counts.tolist(),
        "spectral_blocks": native["nblocks"],
        "spectral_block_ps": args.block_ps,
        "frequency_resolution_cm-1": float(native["frequency"][1] - native["frequency"][0]),
        "bands_cm-1": core.BANDS,
        "velocity_variants": {
            "raw": "molecular COM layer velocities with collective water translation retained",
            "global_COM_corrected": "instantaneous all-water mean COM velocity subtracted",
        },
        "common_q_actual_A-1": q_actual.tolist(),
        "common_q_reciprocal_orders": q_orders.tolist(),
        "common_q_position_sampling_fs": 100.0,
        "common_q_frequency_limit_cm-1": 150.0,
        "common_q_central_water_count": q_members,
        "hbond_graph_samples": samples,
        "method_limits": [
            "Native SiO2 film thickness gives 16 nonempty 2 A layers; depth-normalized plotting is required against the 12-layer CaF2/oil films.",
            "The four common q values use the same reciprocal orders as CaF2 and differ by less than 1 percent in actual q.",
            "Common-q S(q,omega) uses the available 100 fs position sampling and is therefore limited to 150 cm-1; all three systems must be downsampled identically for direct comparison.",
            "H-bond graph modes use a time-averaged layer graph and are not Hessian phonon normal modes.",
        ],
        "elapsed_s": time.time() - started,
    }
    (args.output / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    (args.output / "COMPLETE").write_text("complete\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
