#!/usr/bin/env python3
"""Export a final oil/water(-on-CaF2) frame and audit ice-like water dynamics.

The analysis is deliberately multi-observable.  It combines the last-50-ps
hydrogen-bond network, tetrahedral order, lateral mobility, frequency-domain
intermolecular coherence, and whole-trajectory substrate/temperature checks.
An ice assignment is never made from a single OH-stretch peak.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


CM_PER_PS = 33.3564095198152
AMU_KG = 1.66053906660e-27
KB_J_K = 1.380649e-23
ANGSTROM_PER_FS_TO_M_PER_S = 1.0e5
MASSES_AMU = {"H": 1.00794, "C": 12.0107, "O": 15.9994,
              "F": 18.998403163, "Ca": 40.078}


def minimum_image(delta: np.ndarray, cell: np.ndarray) -> np.ndarray:
    frac = delta @ np.linalg.inv(cell)
    frac -= np.rint(frac)
    return frac @ cell


def water_topology(symbols: np.ndarray, positions: np.ndarray,
                   cell: np.ndarray, cutoff: float = 1.30) -> np.ndarray:
    oxygen = np.flatnonzero(symbols == "O")
    hydrogen = np.flatnonzero(symbols == "H")
    candidates: list[tuple[float, int, int]] = []
    for o in oxygen:
        delta = minimum_image(positions[hydrogen] - positions[o], cell)
        distance = np.linalg.norm(delta, axis=1)
        candidates.extend(
            (float(d), int(o), int(h))
            for h, d in zip(hydrogen, distance) if d <= cutoff
        )
    candidates.sort()
    assigned = {int(o): [] for o in oxygen}
    used_h: set[int] = set()
    for _distance, o, h in candidates:
        if len(assigned[o]) < 2 and h not in used_h:
            assigned[o].append(h)
            used_h.add(h)
    waters = [(int(o), *assigned[int(o)]) for o in oxygen
              if len(assigned[int(o)]) == 2]
    if len(waters) != len(oxygen):
        missing = [int(o) + 1 for o in oxygen if len(assigned[int(o)]) != 2]
        raise RuntimeError(f"Only {len(waters)}/{len(oxygen)} waters assigned; "
                           f"missing O serials: {missing[:20]}")
    return np.asarray(waters, dtype=np.int64)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_poscar(path: Path, symbols: np.ndarray, positions: np.ndarray,
                  cell: np.ndarray, step: int, time_ps: float,
                  system_label: str) -> dict[str, object]:
    order = [element for element in ("H", "C", "O", "F", "Ca")
             if np.any(symbols == element)]
    atom_order = np.concatenate([np.flatnonzero(symbols == e) for e in order])
    counts = [int(np.sum(symbols == e)) for e in order]
    frac = positions[atom_order] @ np.linalg.inv(cell)
    frac %= 1.0
    with path.open("w") as stream:
        stream.write(
            f"{system_label} final complete frame step={step} "
            f"time={time_ps:.6f} ps\n1.0\n"
        )
        for vector in cell:
            stream.write("  " + "  ".join(f"{x:20.12f}" for x in vector) + "\n")
        stream.write("  " + "  ".join(order) + "\n")
        stream.write("  " + "  ".join(str(x) for x in counts) + "\n")
        stream.write("Direct\n")
        for xyz in frac:
            stream.write("  " + "  ".join(f"{x:20.12f}" for x in xyz) + "\n")
    return {
        "species_order": order,
        "counts": counts,
        "source_to_poscar_index_1based": (np.argsort(atom_order) + 1).tolist(),
        "poscar_to_source_index_1based": (atom_order + 1).tolist(),
    }


def tetrahedral_q(o_xyz: np.ndarray, cell: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return q and O-neighbour coordination within 3.5 A for one frame."""
    delta = minimum_image(o_xyz[None, :, :] - o_xyz[:, None, :], cell)
    distance = np.linalg.norm(delta, axis=2)
    np.fill_diagonal(distance, np.inf)
    nearest = np.argpartition(distance, 4, axis=1)[:, :4]
    vectors = np.take_along_axis(delta, nearest[:, :, None], axis=1)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=2, keepdims=True), 1.0e-12)
    terms = np.zeros(len(o_xyz), dtype=np.float64)
    for j in range(3):
        for k in range(j + 1, 4):
            cosine = np.sum(vectors[:, j] * vectors[:, k], axis=1)
            terms += (cosine + 1.0 / 3.0) ** 2
    q = 1.0 - 3.0 * terms / 8.0
    coordination = np.sum(distance <= 3.50, axis=1)
    return q, coordination


def largest_component_fraction(adjacency: np.ndarray) -> float:
    n = len(adjacency)
    seen = np.zeros(n, dtype=bool)
    largest = 0
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for nxt in np.flatnonzero(adjacency[node] & ~seen):
                seen[nxt] = True
                stack.append(int(nxt))
        largest = max(largest, size)
    return largest / max(n, 1)


def hbond_state(o_xyz: np.ndarray, h_xyz: np.ndarray, cell: np.ndarray,
                ho_cutoff: float, oo_cutoff: float,
                angle_cutoff: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized H-bond state for a chunk: (frame, donor, H, acceptor)."""
    h_to_acceptor = minimum_image(
        o_xyz[:, None, None, :, :] - h_xyz[:, :, :, None, :], cell
    )
    ho_distance = np.linalg.norm(h_to_acceptor, axis=-1)
    donor_to_acceptor = minimum_image(
        o_xyz[:, None, :, :] - o_xyz[:, :, None, :], cell
    )
    oo_distance = np.linalg.norm(donor_to_acceptor, axis=-1)
    h_to_donor = minimum_image(
        o_xyz[:, :, None, :] - h_xyz, cell
    )[:, :, :, None, :]
    oh_distance = np.linalg.norm(h_to_donor, axis=-1)
    cosine = np.sum(h_to_donor * h_to_acceptor, axis=-1) / np.maximum(
        oh_distance * ho_distance, 1.0e-12
    )
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    not_self = ~np.eye(o_xyz.shape[1], dtype=bool)[None, :, None, :]
    state = (
        (ho_distance <= ho_cutoff)
        & (oo_distance[:, :, None, :] <= oo_cutoff)
        & (angle >= angle_cutoff)
        & not_self
    )
    return state, ho_distance, angle


def temperature_K(velocities: np.ndarray, symbols: np.ndarray) -> np.ndarray:
    mass_amu = np.asarray([MASSES_AMU[str(s)] for s in symbols])
    total_mass = mass_amu.sum()
    com_velocity = np.sum(
        velocities * mass_amu[None, :, None], axis=1
    ) / total_mass
    relative = velocities - com_velocity[:, None, :]
    kinetic = 0.5 * np.sum(
        mass_amu[None, :, None] * AMU_KG
        * (relative * ANGSTROM_PER_FS_TO_M_PER_S) ** 2,
        axis=(1, 2),
    )
    dof = 3 * len(symbols) - 3
    return 2.0 * kinetic / (dof * KB_J_K)


def bond_distances(positions: np.ndarray, pairs: np.ndarray,
                   cell: np.ndarray) -> np.ndarray:
    if not len(pairs):
        return np.empty((len(positions), 0), dtype=float)
    delta = minimum_image(
        positions[:, pairs[:, 1]] - positions[:, pairs[:, 0]], cell
    )
    return np.linalg.norm(delta, axis=2)


def spectral_metrics(signal: np.ndarray, subset: np.ndarray,
                     window: np.ndarray, nfft: int) -> tuple[np.ndarray, ...]:
    selected = np.asarray(signal[:, subset], dtype=np.float64)
    selected -= selected.mean(axis=0, keepdims=True)
    transformed = np.fft.rfft(selected * window[:, None, None], n=nfft, axis=0)
    power_by_water = np.sum(np.abs(transformed) ** 2, axis=2)
    norm = float(np.sum(window ** 2))
    self_mean = np.sum(power_by_water, axis=1) / (len(subset) * norm)
    coherent = np.sum(np.abs(np.sum(transformed, axis=1)) ** 2, axis=1)
    coherent /= len(subset) * norm
    cross = coherent - self_mean
    participation = np.sum(power_by_water, axis=1) ** 2 / np.maximum(
        len(subset) * np.sum(power_by_water ** 2, axis=1), 1.0e-30
    )
    return self_mean, coherent, cross, participation


def water_signals(positions: np.ndarray, velocities: np.ndarray,
                  waters: np.ndarray, cell: np.ndarray) -> dict[str, np.ndarray]:
    o, h1, h2 = waters.T
    p_o, p_h1, p_h2 = positions[:, o], positions[:, h1], positions[:, h2]
    v_o, v_h1, v_h2 = velocities[:, o], velocities[:, h1], velocities[:, h2]
    mass_o, mass_h = MASSES_AMU["O"], MASSES_AMU["H"]
    v_com = (mass_o * v_o + mass_h * (v_h1 + v_h2)) / (mass_o + 2 * mass_h)

    r1 = minimum_image(p_h1 - p_o, cell)
    r2 = minimum_image(p_h2 - p_o, cell)
    d1 = np.maximum(np.linalg.norm(r1, axis=2, keepdims=True), 1.0e-12)
    d2 = np.maximum(np.linalg.norm(r2, axis=2, keepdims=True), 1.0e-12)
    u1, u2 = r1 / d1, r2 / d2
    dv1, dv2 = v_h1 - v_o, v_h2 - v_o
    s1 = np.sum(dv1 * u1, axis=2)
    s2 = np.sum(dv2 * u2, axis=2)
    stretch = np.stack((s1, s2), axis=2)

    du1 = (dv1 - u1 * np.sum(dv1 * u1, axis=2, keepdims=True)) / d1
    du2 = (dv2 - u2 * np.sum(dv2 * u2, axis=2, keepdims=True)) / d2
    raw = u1 + u2
    raw_norm = np.maximum(np.linalg.norm(raw, axis=2, keepdims=True), 1.0e-12)
    bisector = raw / raw_norm
    draw = du1 + du2
    bisector_dot = (
        draw - bisector * np.sum(draw * bisector, axis=2, keepdims=True)
    ) / raw_norm
    return {"translation": v_com, "libration": bisector_dot, "stretch": stretch}


def write_csv(path: Path, header: list[str], rows) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gds", required=True, type=float)
    parser.add_argument("--layer-width", default=4.0, type=float)
    parser.add_argument("--hbond-last-ps", default=50.0, type=float)
    parser.add_argument("--hbond-ho", default=2.45, type=float)
    parser.add_argument("--hbond-oo", default=3.50, type=float)
    parser.add_argument("--hbond-angle", default=150.0, type=float)
    parser.add_argument("--stable-occupancy", default=0.10, type=float)
    parser.add_argument("--structure-stride", default=100, type=int,
                        help="Whole-trajectory stability stride in source frames")
    parser.add_argument("--order-stride", default=10, type=int,
                        help="Last-window q/network stride in source frames")
    parser.add_argument("--spectral-block-ps", default=10.0, type=float)
    parser.add_argument("--segment-glob", default="*segment_*.h5")
    parser.add_argument("--expected-frames", default=0, type=int)
    parser.add_argument("--system-label", default="CaF2/oil/water ax2")
    parser.add_argument("--interface-description", default=(
        "Water is above the oil layer; in the supported system it is not "
        "directly on bare CaF2."
    ))
    args = parser.parse_args()
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)

    segments = sorted(args.input.glob(args.segment_glob))
    if not segments:
        raise RuntimeError(
            f"No complete trajectory files match {args.segment_glob!r} in {args.input}"
        )
    with h5py.File(segments[0], "r") as handle:
        if not bool(handle.attrs.get("complete", False)):
            raise RuntimeError(f"Incomplete segment: {segments[0]}")
        symbols = np.asarray(handle["symbols"]).astype(str)
        atomic_numbers = np.asarray(handle["atomic_numbers"], dtype=int)
        cell = np.asarray(handle["cell_A"], dtype=float)
        positions0 = np.asarray(handle["positions_A"][0], dtype=float)
        dt_fs = float(handle.attrs["output_interval_fs"])
        first_step = int(handle["step"][0])
        first_time = float(handle["time_ps"][0])
    if not np.allclose(cell, np.diag(np.diag(cell)), atol=1.0e-7):
        raise RuntimeError("This audit currently requires an orthorhombic cell")
    waters = water_topology(symbols, positions0, cell)
    nwater = len(waters)
    oxygen, h1, h2 = waters.T
    h_by_water = np.stack((h1, h2), axis=1)
    substrate = np.flatnonzero((symbols == "Ca") | (symbols == "F"))
    calcium = np.flatnonzero(symbols == "Ca")
    fluorine = np.flatnonzero(symbols == "F")
    carbon = np.flatnonzero(symbols == "C")
    has_caf2 = bool(len(calcium) and len(fluorine))

    # Initial solid/oil topology for the whole-trajectory structural audit.
    ca_f_delta = minimum_image(
        positions0[fluorine][None, :, :] - positions0[calcium][:, None, :], cell
    )
    initial_ca_f = np.linalg.norm(ca_f_delta, axis=2) <= 3.00
    cc_delta = minimum_image(
        positions0[carbon][None, :, :] - positions0[carbon][:, None, :], cell
    )
    cc_dist = np.linalg.norm(cc_delta, axis=2)
    cc_pairs_local = np.argwhere(np.triu((cc_dist <= 1.90) & (cc_dist > 0), 1))
    cc_pairs = np.column_stack((carbon[cc_pairs_local[:, 0]],
                                carbon[cc_pairs_local[:, 1]])).astype(int)
    all_h = np.flatnonzero(symbols == "H")
    ch_delta = minimum_image(
        positions0[all_h][None, :, :] - positions0[carbon][:, None, :], cell
    )
    ch_dist = np.linalg.norm(ch_delta, axis=2)
    ch_pairs_local = np.argwhere(ch_dist <= 1.25)
    ch_pairs = np.column_stack((carbon[ch_pairs_local[:, 0]],
                                all_h[ch_pairs_local[:, 1]])).astype(int)

    # First pass: mean water z, final frame, and sparse whole-trajectory stability.
    z_sum = np.zeros(nwater, dtype=np.float64)
    total_frames = 0
    stability_rows: list[list[float]] = []
    cc_min, cc_max = np.inf, -np.inf
    ch_min, ch_max = np.inf, -np.inf
    previous_step = 0
    final_positions = None
    final_step = None
    final_time = None
    for si, path in enumerate(segments):
        with h5py.File(path, "r") as handle:
            if not bool(handle.attrs.get("complete", False)):
                raise RuntimeError(f"Incomplete segment: {path}")
            steps = np.asarray(handle["step"], dtype=np.int64)
            expected = np.arange(previous_step + 2, previous_step + 2 + 2 * len(steps), 2)
            if not np.array_equal(steps, expected):
                raise RuntimeError(f"Step discontinuity in {path}")
            previous_step = int(steps[-1])
            z = np.asarray(handle["positions_A"][:, oxygen, 2], dtype=np.float64)
            z_sum += np.sum(z % cell[2, 2], axis=0)
            total_frames += len(z)
            pos = np.asarray(handle["positions_A"][::args.structure_stride], dtype=float)
            vel = np.asarray(handle["velocities_A_per_fs"][::args.structure_stride], dtype=float)
            sample_steps = np.asarray(handle["step"][::args.structure_stride], dtype=int)
            sample_times = np.asarray(handle["time_ps"][::args.structure_stride], dtype=float)
            temp = temperature_K(vel, symbols)

            # Remove a periodic rigid translation before measuring a CaF2
            # distortion.  A simple mean of individually minimum-imaged
            # displacements fails when the whole slab crosses half a box:
            # equivalent atoms then choose different image branches.  The
            # circular mean resolves one common translation modulo the cell.
            cc = bond_distances(pos, cc_pairs, cell)
            ch = bond_distances(pos, ch_pairs, cell)
            if has_caf2:
                sub_mass = np.asarray([MASSES_AMU[str(symbols[i])] for i in substrate])
                frac_disp = ((pos[:, substrate] - positions0[substrate])
                             @ np.linalg.inv(cell))
                phase = np.exp(2.0j * np.pi * frac_disp)
                mean_phase = np.sum(
                    phase * sub_mass[None, :, None], axis=1
                ) / sub_mass.sum()
                drift_frac = np.angle(mean_phase) / (2.0 * np.pi)
                residual_frac = frac_disp - drift_frac[:, None, :]
                residual_frac -= np.rint(residual_frac)
                residual = residual_frac @ cell
                drift = drift_frac @ cell
                structure_rmsd = np.sqrt(
                    np.mean(np.sum(residual ** 2, axis=2), axis=1)
                )
                ca_mask = symbols[substrate] == "Ca"
                metric_1 = np.sqrt(
                    np.mean(np.sum(residual[:, ca_mask] ** 2, axis=2), axis=1)
                )
                metric_2 = np.sqrt(
                    np.mean(np.sum(residual[:, ~ca_mask] ** 2, axis=2), axis=1)
                )
                current_caf = np.linalg.norm(
                    minimum_image(
                        pos[:, fluorine][..., None, :, :]
                        - pos[:, calcium][:, :, None, :], cell
                    ), axis=-1
                ) <= 3.00
                retained = np.sum(
                    current_caf & initial_ca_f[None, :, :], axis=(1, 2)
                )
                retention = retained / max(int(np.sum(initial_ca_f)), 1)
            else:
                initial_cc_lengths = bond_distances(
                    positions0[None, :, :], cc_pairs, cell
                )[0]
                initial_ch_lengths = bond_distances(
                    positions0[None, :, :], ch_pairs, cell
                )[0]
                cc_rmsd = np.sqrt(np.mean(
                    (cc - initial_cc_lengths[None, :]) ** 2, axis=1
                ))
                ch_rmsd = np.sqrt(np.mean(
                    (ch - initial_ch_lengths[None, :]) ** 2, axis=1
                ))
                structure_rmsd = np.sqrt(
                    (np.sum((cc - initial_cc_lengths[None, :]) ** 2, axis=1)
                     + np.sum((ch - initial_ch_lengths[None, :]) ** 2, axis=1))
                    / max(len(cc_pairs) + len(ch_pairs), 1)
                )
                metric_1, metric_2 = cc_rmsd, ch_rmsd
                retention = np.mean(
                    np.concatenate((cc <= 2.00, ch <= 1.50), axis=1), axis=1
                )
                drift = np.zeros((len(pos), 3), dtype=float)
            if cc.size:
                cc_min, cc_max = min(cc_min, float(cc.min())), max(cc_max, float(cc.max()))
            if ch.size:
                ch_min, ch_max = min(ch_min, float(ch.min())), max(ch_max, float(ch.max()))
            water_z = pos[:, oxygen, 2] % cell[2, 2]
            for j in range(len(pos)):
                stability_rows.append([
                    int(sample_steps[j]), float(sample_times[j]), float(temp[j]),
                    float(structure_rmsd[j]), float(metric_1[j]), float(metric_2[j]),
                    float(retention[j]), float(drift[j, 0]), float(drift[j, 1]),
                    float(drift[j, 2]), float(np.mean(water_z[j])),
                    int(np.sum(water_z[j] < args.gds)),
                    int(np.sum(water_z[j] >= args.gds + 24.0)),
                    float(np.max(cc[j])) if cc.size else math.nan,
                    float(np.max(ch[j])) if ch.size else math.nan,
                ])
            if si == len(segments) - 1:
                final_positions = np.asarray(handle["positions_A"][-1], dtype=float)
                final_step = int(handle["step"][-1])
                final_time = float(handle["time_ps"][-1])
        if si == 0 or (si + 1) % 20 == 0:
            print(json.dumps({"first_pass_segments_done": si + 1,
                              "segments_total": len(segments)}), flush=True)

    assert final_positions is not None and final_step is not None and final_time is not None
    if args.expected_frames and total_frames != args.expected_frames:
        raise RuntimeError(
            f"Expected {args.expected_frames} frames, found {total_frames}"
        )
    mean_z = z_sum / total_frames
    relative_z = mean_z - args.gds
    layer_id = np.floor(relative_z / args.layer_width).astype(int)
    layer_ids = sorted(int(x) for x in np.unique(layer_id) if x >= 0)
    layer_members = {lid: np.flatnonzero(layer_id == lid) for lid in layer_ids}
    first_layer = layer_members[min(layer_ids)]
    interior = np.flatnonzero((relative_z >= 8.0) & (relative_z < 16.0))
    if not len(first_layer) or not len(interior):
        raise RuntimeError("First-layer or 8-16 A interior control is empty")

    poscar_filename = f"POSCAR_final_{int(round(final_time))}ps"
    poscar_info = export_poscar(
        args.output / poscar_filename, symbols, final_positions, cell,
        final_step, final_time, args.system_label,
    )
    write_csv(
        args.output / "water_topology_and_layers.csv",
        ["water_index_1based", "source_O_serial_1based", "source_H1_serial_1based",
         "source_H2_serial_1based", "mean_O_z_A", "z_minus_GDS_A", "fixed_4A_layer"],
        [[i + 1, int(o) + 1, int(a) + 1, int(b) + 1, mean_z[i], relative_z[i],
          (int(layer_id[i]) + 1 if layer_id[i] >= 0 else 0)]
         for i, (o, a, b) in enumerate(waters)],
    )
    write_csv(
        args.output / "whole_trajectory_stability_stride100fs.csv",
        ["step", "time_ps", "temperature_K",
         ("CaF2_RMSD_A" if has_caf2 else "oil_covalent_bond_RMSD_A"),
         ("Ca_RMSD_A" if has_caf2 else "oil_CC_bond_RMSD_A"),
         ("F_RMSD_A" if has_caf2 else "oil_CH_bond_RMSD_A"),
         ("initial_CaF_coordination_retained_fraction" if has_caf2 else
          "initial_oil_covalent_bonds_retained_fraction"),
         ("CaF2_drift_x_A" if has_caf2 else "not_applicable_x"),
         ("CaF2_drift_y_A" if has_caf2 else "not_applicable_y"),
         ("CaF2_drift_z_A" if has_caf2 else "not_applicable_z"),
         "mean_water_O_z_A",
         "waters_below_GDS", "waters_at_or_above_GDS_plus24A",
         "max_initial_CC_bond_A", "max_initial_CH_bond_A"],
        stability_rows,
    )

    # Last 50 ps positions for full-resolution H-bonds and mobility.
    last_frames = int(round(args.hbond_last_ps * 1000.0 / dt_fs))
    o_parts, h_parts, last_time_parts = [], [], []
    collected = 0
    selected_last_segments = []
    for path in reversed(segments):
        selected_last_segments.append(path)
        with h5py.File(path, "r") as handle:
            collected += len(handle["step"])
        if collected >= last_frames:
            break
    for path in reversed(selected_last_segments):
        with h5py.File(path, "r") as handle:
            pos = np.asarray(handle["positions_A"], dtype=np.float32)
            o_parts.append(pos[:, oxygen])
            h_parts.append(pos[:, h_by_water])
            last_time_parts.append(np.asarray(handle["time_ps"], dtype=float))
    o_last = np.concatenate(o_parts, axis=0)[-last_frames:]
    h_last = np.concatenate(h_parts, axis=0)[-last_frames:]
    times_last = np.concatenate(last_time_parts)[-last_frames:]

    shape = (nwater, 2, nwater)
    occupancy_count = np.zeros(shape, dtype=np.int32)
    run = np.zeros(shape, dtype=np.int32)
    episode_count = np.zeros(shape, dtype=np.int32)
    episode_frames = np.zeros(shape, dtype=np.int64)
    episode_max = np.zeros(shape, dtype=np.int32)
    donor_sum = np.zeros(nwater, dtype=np.int64)
    acceptor_sum = np.zeros(nwater, dtype=np.int64)
    q_samples: list[np.ndarray] = []
    coordination_samples: list[np.ndarray] = []
    network_rows: list[list[float]] = []
    global_frame = 0
    chunk = 25
    for start in range(0, last_frames, chunk):
        stop = min(start + chunk, last_frames)
        state, _ho_distance, _angle = hbond_state(
            o_last[start:stop], h_last[start:stop], cell,
            args.hbond_ho, args.hbond_oo, args.hbond_angle,
        )
        occupancy_count += np.sum(state, axis=0, dtype=np.int32)
        donor_sum += np.sum(state, axis=(0, 2, 3), dtype=np.int64)
        acceptor_sum += np.sum(state, axis=(0, 1, 2), dtype=np.int64)
        for local in range(len(state)):
            current = state[local]
            ended = (run > 0) & ~current
            episode_count[ended] += 1
            episode_frames[ended] += run[ended]
            episode_max[ended] = np.maximum(episode_max[ended], run[ended])
            run[current] += 1
            run[~current] = 0
            if global_frame % args.order_stride == 0:
                q, coordination = tetrahedral_q(o_last[global_frame], cell)
                q_samples.append(q.astype(np.float32))
                coordination_samples.append(coordination.astype(np.int16))
                directed = np.any(current, axis=1)
                adjacency = directed | directed.T
                degree = np.sum(adjacency, axis=1)
                network_rows.append([
                    float(times_last[global_frame]), int(np.sum(current)),
                    float(np.mean(degree)), float(np.mean(q)),
                    float(np.mean(q >= 0.80)), largest_component_fraction(adjacency),
                ])
            global_frame += 1
        if stop % 5000 == 0 or stop == last_frames:
            print(json.dumps({"hbond_frames_done": stop,
                              "hbond_frames_total": last_frames}), flush=True)
    active = run > 0
    episode_count[active] += 1
    episode_frames[active] += run[active]
    episode_max[active] = np.maximum(episode_max[active], run[active])
    occupancy = occupancy_count / last_frames
    stable = occupancy >= args.stable_occupancy

    pair_rows = []
    observed = np.argwhere(occupancy_count > 0)
    for donor, hi, acceptor in observed:
        pair_rows.append([
            int(oxygen[donor]) + 1, int(h_by_water[donor, hi]) + 1,
            int(oxygen[acceptor]) + 1,
            int(layer_id[donor]) + 1 if layer_id[donor] >= 0 else 0,
            int(layer_id[acceptor]) + 1 if layer_id[acceptor] >= 0 else 0,
            int(occupancy_count[donor, hi, acceptor]),
            float(occupancy[donor, hi, acceptor]),
            bool(stable[donor, hi, acceptor]),
            int(episode_count[donor, hi, acceptor]),
            float(episode_frames[donor, hi, acceptor]
                  / max(episode_count[donor, hi, acceptor], 1) * dt_fs / 1000.0),
            float(episode_max[donor, hi, acceptor] * dt_fs / 1000.0),
        ])
    pair_rows.sort(key=lambda row: row[6], reverse=True)
    write_csv(
        args.output / "hbond_pair_stability_last50ps.csv",
        ["donor_O_serial", "donor_H_serial", "acceptor_O_serial",
         "donor_fixed_4A_layer", "acceptor_fixed_4A_layer", "bonded_frames",
         "occupancy_fraction", "stable_occupancy_ge_0p10", "episodes",
         "mean_continuous_lifetime_ps", "max_continuous_lifetime_ps"],
        pair_rows,
    )
    write_csv(
        args.output / "hbond_network_time_stride10fs.csv",
        ["time_ps", "directed_H_bonds", "mean_undirected_degree",
         "mean_tetrahedral_q", "fraction_q_ge_0p80", "largest_component_fraction"],
        network_rows,
    )

    q_values = np.asarray(q_samples, dtype=float)
    coord_values = np.asarray(coordination_samples, dtype=float)
    layer_hbond_rows = []
    for lid in layer_ids:
        members = layer_members[lid]
        donor_mean = float(np.sum(donor_sum[members]) / (last_frames * len(members)))
        accept_mean = float(np.sum(acceptor_sum[members]) / (last_frames * len(members)))
        involvement = np.zeros(shape, dtype=bool)
        involvement[members, :, :] = True
        involvement[:, :, members] = True
        observations = np.sum(occupancy_count[involvement])
        stable_observations = np.sum(occupancy_count[involvement & stable])
        episodes = np.sum(episode_count[involvement])
        episode_total_frames = np.sum(episode_frames[involvement])
        layer_hbond_rows.append([
            lid + 1, lid * args.layer_width, (lid + 1) * args.layer_width,
            len(members), donor_mean, accept_mean, donor_mean + accept_mean,
            float(np.mean(q_values[:, members])),
            float(np.mean(q_values[:, members] >= 0.80)),
            float(np.mean(q_values[:, members] >= 0.90)),
            float(np.mean(coord_values[:, members])),
            int(np.sum(stable[involvement])),
            float(stable_observations / max(observations, 1)),
            float(episode_total_frames / max(episodes, 1) * dt_fs / 1000.0),
            float(np.max(episode_max[involvement]) * dt_fs / 1000.0),
        ])
    write_csv(
        args.output / "hbond_tetrahedral_order_by_4A_layer_last50ps.csv",
        ["layer", "z_minus_GDS_min_A", "z_minus_GDS_max_A", "waters",
         "mean_donated_HB_per_water", "mean_accepted_HB_per_water",
         "mean_HB_degree_per_water", "mean_tetrahedral_q", "fraction_q_ge_0p80",
         "fraction_q_ge_0p90", "mean_O_neighbors_within_3p50A",
         "stable_directed_pairs_involving_layer", "fraction_HB_observations_from_stable_pairs",
         "mean_continuous_HB_lifetime_ps", "max_continuous_HB_lifetime_ps"],
        layer_hbond_rows,
    )
    q_hist_rows = []
    q_edges = np.linspace(-0.2, 1.0, 121)
    for label, members in (("L1_0_4A", first_layer), ("interior_8_16A", interior)):
        density, _ = np.histogram(q_values[:, members].ravel(), bins=q_edges, density=True)
        for lo, hi, value in zip(q_edges[:-1], q_edges[1:], density):
            q_hist_rows.append([label, 0.5 * (lo + hi), value])
    write_csv(args.output / "tetrahedral_q_distribution_last50ps.csv",
              ["region", "q_bin_center", "probability_density"], q_hist_rows)

    # Last-50-ps molecular mobility using unwrapped oxygen positions.
    delta = minimum_image(np.diff(o_last.astype(np.float64), axis=0), cell)
    unwrapped_lab = np.empty_like(o_last, dtype=np.float64)
    unwrapped_lab[0] = o_last[0]
    unwrapped_lab[1:] = o_last[0] + np.cumsum(delta, axis=0)
    # The finite slab undergoes a coherent lateral translation.  Remove the
    # instantaneous all-water oxygen COM before calling residual motion
    # molecular mobility; retain the lab-frame MSD in the source table.
    water_o_com = np.mean(unwrapped_lab, axis=1, keepdims=True)
    unwrapped = unwrapped_lab - water_o_com
    lag_ps_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0]
    mobility_rows = []
    mobility_by_layer: dict[int, dict[float, tuple[float, float]]] = {}
    for lid in layer_ids:
        members = layer_members[lid]
        mobility_by_layer[lid] = {}
        for lag_ps in lag_ps_values:
            lag = int(round(lag_ps * 1000.0 / dt_fs))
            displacement = unwrapped[lag:, members] - unwrapped[:-lag, members]
            displacement_lab = (
                unwrapped_lab[lag:, members] - unwrapped_lab[:-lag, members]
            )
            msd_xy = float(np.mean(np.sum(displacement[:, :, :2] ** 2, axis=2)))
            msd_3d = float(np.mean(np.sum(displacement ** 2, axis=2)))
            msd_xy_lab = float(np.mean(
                np.sum(displacement_lab[:, :, :2] ** 2, axis=2)
            ))
            msd_3d_lab = float(np.mean(np.sum(displacement_lab ** 2, axis=2)))
            mobility_by_layer[lid][lag_ps] = (msd_xy, msd_3d)
            mobility_rows.append([lid + 1, lid * args.layer_width,
                                  (lid + 1) * args.layer_width, len(members),
                                  lag_ps, msd_xy, msd_3d,
                                  msd_xy_lab, msd_3d_lab])
    write_csv(args.output / "water_O_mobility_by_4A_layer_last50ps.csv",
              ["layer", "z_minus_GDS_min_A", "z_minus_GDS_max_A", "waters",
               "lag_ps", "COM_corrected_MSD_xy_A2", "COM_corrected_MSD_3D_A2",
               "lab_frame_MSD_xy_A2", "lab_frame_MSD_3D_A2"], mobility_rows)
    diffusion_rows = []
    fit_lags = np.asarray([5.0, 10.0, 20.0, 30.0])
    for lid in layer_ids:
        y = np.asarray([mobility_by_layer[lid][float(t)][0] for t in fit_lags])
        slope, intercept = np.polyfit(fit_lags, y, 1)
        fitted = slope * fit_lags + intercept
        residual_sum = float(np.sum((y - fitted) ** 2))
        total_sum = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = 1.0 - residual_sum / max(total_sum, 1.0e-30)
        diffusion_rows.append([
            lid + 1, len(layer_members[lid]), slope, intercept, r_squared,
            max(float(slope / 4.0), 0.0), max(float(slope / 4.0), 0.0) * 1.0e-4,
        ])
    write_csv(args.output / "water_lateral_diffusion_by_4A_layer_last50ps.csv",
              ["layer", "waters", "COM_corrected_MSD_xy_slope_A2_per_ps",
               "fit_intercept_A2", "fit_R_squared", "effective_D_xy_A2_per_ps",
               "effective_D_xy_cm2_per_s"], diffusion_rows)

    # Full-trajectory spectra: self, q=0 coherent total, signed cross, and PR.
    frames_per_block = int(round(args.spectral_block_ps * 1000.0 / dt_fs))
    if total_frames % frames_per_block:
        raise RuntimeError(
            f"{total_frames} frames are not divisible into {frames_per_block}-frame blocks"
        )
    blocks = total_frames // frames_per_block
    nfft = 1 << (4 * frames_per_block - 1).bit_length()
    window = np.blackman(frames_per_block)
    frequency = np.fft.rfftfreq(nfft, d=dt_fs * 1.0e-3) * CM_PER_PS
    regions = {"L1_0_4A": first_layer, "interior_8_16A": interior}
    modes = ("translation", "libration", "stretch")
    metric_names = ("self", "coherent", "cross", "participation")
    spectral_sums = {
        (region, mode, metric): np.zeros_like(frequency)
        for region in regions for mode in modes for metric in metric_names
    }
    pos_parts: list[np.ndarray] = []
    vel_parts: list[np.ndarray] = []
    pending = 0
    block = 0
    for path in segments:
        with h5py.File(path, "r") as handle:
            file_pos = np.asarray(handle["positions_A"], dtype=np.float32)
            file_vel = np.asarray(handle["velocities_A_per_fs"], dtype=np.float32)
        offset = 0
        while offset < len(file_pos):
            take = min(frames_per_block - pending, len(file_pos) - offset)
            pos_parts.append(file_pos[offset:offset + take])
            vel_parts.append(file_vel[offset:offset + take])
            offset += take
            pending += take
            if pending != frames_per_block:
                continue
            positions = np.concatenate(pos_parts, axis=0)
            velocities = np.concatenate(vel_parts, axis=0)
            signals = water_signals(positions, velocities, waters, cell)
            del positions, velocities, pos_parts, vel_parts
            for mode, signal in signals.items():
                for region, members in regions.items():
                    values = spectral_metrics(signal, members, window, nfft)
                    for metric, value in zip(metric_names, values):
                        spectral_sums[(region, mode, metric)] += value
                del signal
            del signals
            block += 1
            if block == 1 or block % 10 == 0:
                print(json.dumps({"spectral_blocks_done": block,
                                  "spectral_blocks_total": blocks}), flush=True)
            pos_parts, vel_parts, pending = [], [], 0
        del file_pos, file_vel
    if pending or block != blocks:
        raise RuntimeError(
            f"Spectral streaming ended with pending={pending}, blocks={block}/{blocks}"
        )
    for key in spectral_sums:
        spectral_sums[key] /= blocks
    export = frequency <= 4000.0
    spacing = float(np.median(np.diff(frequency)))
    smooth_sigma_bins = 15.0 / spacing
    smoothed = {
        key: (gaussian_filter1d(value, smooth_sigma_bins, mode="nearest")
              if key[2] != "participation" else
              gaussian_filter1d(value, smooth_sigma_bins, mode="nearest"))
        for key, value in spectral_sums.items()
    }
    spectral_header = ["frequency_cm-1"]
    spectral_columns = [frequency[export]]
    for region in regions:
        for mode in modes:
            for metric in metric_names:
                spectral_header.append(f"{region}_{mode}_{metric}")
                spectral_columns.append(smoothed[(region, mode, metric)][export])
    np.savetxt(
        args.output / "collective_vibration_spectra_full_trajectory.csv",
        np.column_stack(spectral_columns), delimiter=",",
        header=",".join(spectral_header), comments="",
    )

    mode_bands = {"translation": (20.0, 300.0),
                  "libration": (300.0, 1000.0),
                  "stretch": (2800.0, 3900.0)}
    collective_rows = []
    collective_summary: dict[str, dict[str, dict[str, float]]] = {}
    for region in regions:
        collective_summary[region] = {}
        for mode, (lo, hi) in mode_bands.items():
            mask = (frequency >= lo) & (frequency <= hi)
            self_s = smoothed[(region, mode, "self")]
            coherent_s = smoothed[(region, mode, "coherent")]
            cross_s = smoothed[(region, mode, "cross")]
            pr_s = smoothed[(region, mode, "participation")]
            idx = np.flatnonzero(mask)[np.argmax(self_s[mask])]
            integral_self = float(np.trapezoid(self_s[mask], frequency[mask]))
            integral_coh = float(np.trapezoid(coherent_s[mask], frequency[mask]))
            integral_cross = float(np.trapezoid(cross_s[mask], frequency[mask]))
            pr_weighted = float(np.sum(pr_s[mask] * self_s[mask])
                                / max(np.sum(self_s[mask]), 1.0e-30))
            row = [region, mode, lo, hi, float(frequency[idx]), integral_self,
                   integral_coh, integral_cross,
                   integral_cross / max(integral_self, 1.0e-30), pr_weighted,
                   float(pr_s[idx])]
            collective_rows.append(row)
            collective_summary[region][mode] = {
                "self_peak_cm-1": row[4],
                "integrated_cross_to_self_ratio": row[8],
                "self_weighted_participation_ratio": row[9],
                "participation_ratio_at_self_peak": row[10],
            }
    write_csv(
        args.output / "collective_vibration_band_summary.csv",
        ["region", "mode", "band_min_cm-1", "band_max_cm-1", "self_peak_cm-1",
         "integrated_self", "integrated_coherent", "integrated_signed_cross",
         "integrated_cross_to_self_ratio", "self_weighted_participation_ratio",
         "participation_ratio_at_self_peak"], collective_rows,
    )

    # Aggregate whole-system stability and evidence-based classification.
    stability = np.asarray(stability_rows, dtype=float)
    layer_hbond = {int(row[0]): row for row in layer_hbond_rows}
    diffusion_by_layer = {int(row[0]): row for row in diffusion_rows}
    layer_ice_rows = []
    for layer_number, hb_row in sorted(layer_hbond.items()):
        diffusion_row = diffusion_by_layer[layer_number]
        q_mean = float(hb_row[7])
        q80_fraction = float(hb_row[8])
        stable_fraction = float(hb_row[12])
        effective_dxy = float(diffusion_row[5])
        structure_flag = bool(q_mean >= 0.90 and q80_fraction >= 0.80)
        immobility_flag = bool(effective_dxy <= 0.01)
        persistent_flag = bool(stable_fraction >= 0.80)
        if structure_flag and immobility_flag and persistent_flag:
            classification = "ice-I-like local solid"
        elif q_mean >= 0.80 and immobility_flag and persistent_flag:
            classification = "ordered caged ice-like water; below ice-I q threshold"
        elif immobility_flag and persistent_flag:
            classification = "caged but tetrahedrally disordered interfacial water"
        else:
            classification = "more mobile/disordered boundary water"
        layer_ice_rows.append([
            layer_number, float(hb_row[1]), float(hb_row[2]), int(hb_row[3]),
            q_mean, q80_fraction, stable_fraction, float(hb_row[13]),
            effective_dxy, float(diffusion_row[4]), structure_flag,
            immobility_flag, persistent_flag, classification,
        ])
    write_csv(
        args.output / "ice_like_diagnostic_by_4A_layer.csv",
        ["layer", "z_minus_GDS_min_A", "z_minus_GDS_max_A", "waters",
         "mean_tetrahedral_q", "fraction_q_ge_0p80",
         "fraction_HB_observations_from_stable_pairs",
         "mean_continuous_HB_lifetime_ps", "COM_corrected_effective_D_xy_A2_per_ps",
         "mobility_fit_R_squared", "ice_I_like_local_structure",
         "cage_relative_immobility", "persistent_pair_identity", "classification"],
        layer_ice_rows,
    )
    l1_row = layer_hbond[min(layer_ids) + 1]
    l1_diff = diffusion_by_layer[min(layer_ids) + 1]
    interior_q = float(np.mean(q_values[:, interior]))
    l1_q = float(l1_row[7])
    l1_q80 = float(l1_row[8])
    l1_stable_observation_fraction = float(l1_row[12])
    l1_hb_lifetime = float(l1_row[13])
    l1_dxy = float(l1_diff[5])
    l1_trans_cross = collective_summary["L1_0_4A"]["translation"][
        "integrated_cross_to_self_ratio"
    ]
    l1_lib_cross = collective_summary["L1_0_4A"]["libration"][
        "integrated_cross_to_self_ratio"
    ]
    ice_structure = bool(l1_q >= 0.90 and l1_q80 >= 0.80)
    ice_immobility = bool(l1_dxy <= 0.01)
    ice_persistent_network = bool(l1_stable_observation_fraction >= 0.80)
    spectral_correlation_detected = bool(
        abs(l1_trans_cross) >= 0.10 or abs(l1_lib_cross) >= 0.10
    )
    ice_supported = bool(
        ice_structure and ice_immobility and ice_persistent_network
        and spectral_correlation_detected
    )
    any_ice_like_layer = any(bool(row[10] and row[11] and row[12])
                             for row in layer_ice_rows)
    ordered_caged_layers = [int(row[0]) for row in layer_ice_rows
                            if row[4] >= 0.80 and row[11] and row[12]]
    last100 = stability[:, 1] >= stability[:, 1].max() - 100.0
    structure_rmsd_last100_slope = float(np.polyfit(
        stability[last100, 1], stability[last100, 3], 1
    )[0])
    structure_name = "CaF2" if has_caf2 else "oil covalent network"
    structure_rmsd_limit = 1.50 if has_caf2 else 0.30
    retention_limit = 0.90 if has_caf2 else 1.00
    overall_stable = bool(
        abs(float(np.mean(stability[:, 2])) - 300.0) <= 5.0
        and float(np.max(stability[:, 3])) <= structure_rmsd_limit
        and float(np.min(stability[:, 6])) >= retention_limit
        and cc_max <= 2.00 and ch_max <= 1.50
    )
    result = {
        "status": "complete",
        "system": args.system_label,
        "interface_description": args.interface_description,
        "trajectory": {
            "trajectory_files": len(segments), "frames": total_frames,
            "first_step": first_step, "last_step": final_step,
            "time_ps": [first_time, final_time], "sampling_interval_fs": dt_fs,
            "atoms": len(symbols), "waters": nwater,
        },
        "final_poscar": {
            "file": poscar_filename, "source_segment": str(segments[-1]),
            "step": final_step, "time_ps": final_time,
            "sha256": sha256(args.output / poscar_filename),
            **poscar_info,
        },
        "water_regions": {
            "GDS_A": args.gds, "layer_width_A": args.layer_width,
            "L1_0_4A_waters": int(len(first_layer)),
            "interior_8_16A_waters": int(len(interior)),
        },
        "hbond_definition": {
            "H_to_acceptor_O_max_A": args.hbond_ho,
            "donor_O_to_acceptor_O_max_A": args.hbond_oo,
            "O_H_O_min_degree": args.hbond_angle,
            "window_ps": args.hbond_last_ps,
            "stable_pair_occupancy_threshold": args.stable_occupancy,
        },
        "first_layer_ice_diagnostics": {
            "mean_tetrahedral_q": l1_q,
            "interior_8_16A_mean_tetrahedral_q": interior_q,
            "fraction_q_ge_0p80": l1_q80,
            "mean_HB_degree_per_water": float(l1_row[6]),
            "fraction_HB_observations_from_pairs_with_occupancy_ge_0p10":
                l1_stable_observation_fraction,
            "mean_continuous_HB_lifetime_ps": l1_hb_lifetime,
            "COM_corrected_effective_D_xy_A2_per_ps": l1_dxy,
            "translation_20_300cm-1_cross_to_self": l1_trans_cross,
            "libration_300_1000cm-1_cross_to_self": l1_lib_cross,
            "collective_spectral_correlation_detected": spectral_correlation_detected,
        },
        "layer_resolved_ice_diagnostics": [
            {
                "layer": int(row[0]),
                "z_minus_GDS_A": [float(row[1]), float(row[2])],
                "waters": int(row[3]),
                "mean_tetrahedral_q": float(row[4]),
                "fraction_q_ge_0p80": float(row[5]),
                "stable_pair_observation_fraction": float(row[6]),
                "mean_continuous_HB_lifetime_ps": float(row[7]),
                "COM_corrected_effective_D_xy_A2_per_ps": float(row[8]),
                "mobility_fit_R_squared": float(row[9]),
                "ice_I_like_local_structure": bool(row[10]),
                "cage_relative_immobility": bool(row[11]),
                "persistent_pair_identity": bool(row[12]),
                "classification": str(row[13]),
            }
            for row in layer_ice_rows
        ],
        "whole_trajectory_stability": {
            "temperature_K_mean_std_min_max": [
                float(np.mean(stability[:, 2])), float(np.std(stability[:, 2])),
                float(np.min(stability[:, 2])), float(np.max(stability[:, 2]))],
            "structure_metric": structure_name,
            "structure_bond_or_position_RMSD_A_mean_p95_max": [
                float(np.mean(stability[:, 3])), float(np.quantile(stability[:, 3], 0.95)),
                float(np.max(stability[:, 3]))],
            "initial_structure_connectivity_retained_fraction_min":
                float(np.min(stability[:, 6])),
            "oil_initial_CC_bonds": int(len(cc_pairs)),
            "oil_initial_CH_bonds": int(len(ch_pairs)),
            "oil_CC_distance_A_min_max": [float(cc_min), float(cc_max)],
            "oil_CH_distance_A_min_max": [float(ch_min), float(ch_max)],
            "structure_RMSD_last100ps_slope_A_per_ps":
                structure_rmsd_last100_slope,
            "structure_RMSD_acceptance_limit_A": structure_rmsd_limit,
            "connectivity_retention_acceptance_limit": retention_limit,
            "finite_and_step_continuous": True,
            "overall_stable": overall_stable,
            "verdict": (
                f"The full {args.system_label} trajectory is structurally and thermally stable."
                if overall_stable else
                "At least one whole-system stability criterion failed; inspect source tables."
            ),
            "water_geometry_audit_reference": "trajectory_geometry_audit.json",
        },
        "decision": {
            "low_frequency_collective_translation_detected": spectral_correlation_detected,
            "ice_I_like_local_structure_threshold_met": ice_structure,
            "ice_like_immobility_threshold_met": ice_immobility,
            "persistent_fixed_hbond_network_threshold_met": ice_persistent_network,
            "first_layer_ice_like_collective_dynamics_supported": ice_supported,
            "any_4A_layer_meets_ice_I_like_local_solid_diagnostics": any_ice_like_layer,
            "ordered_caged_4A_layers": ordered_caged_layers,
            "confirmed_crystalline_ice_normal_mode": False,
            "verdict": (
                "Low-frequency collective translation is present and at least one internal "
                "4 A layer has ice-I-like local solid diagnostics. This supports ice-like "
                "collective dynamics, but does not by itself prove a crystalline-ice normal mode."
                if any_ice_like_layer and spectral_correlation_detected else
                "Low-frequency cross-molecular correlation is present, but the combined "
                "structure, cage-relative mobility, and persistent-network diagnostics do not "
                "support an ice-like collective solid assignment."
                if spectral_correlation_detected else
                "The low-frequency signed cross/self ratios are below the 0.10 collective "
                "threshold, and no layer satisfies the combined ice-like solid diagnostics."
            ),
        },
        "method_limits": [
            "The collective diagnostic is a trajectory cross spectrum and participation ratio, not a Hessian normal-mode eigenvector calculation.",
            "The 8-16 A region is an internal control from the same finite slab, not an independently simulated bulk-water or ice reference.",
            "Tetrahedral q alone detects local order and is insufficient to label ice.",
        ],
        "elapsed_s": time.time() - started,
    }
    (args.output / "ice_hbond_stability_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )

    # Compact review figure with source tables written above.
    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8,
                         "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(3, 2, figsize=(10.2, 11.0), constrained_layout=True)
    colors = {"L1_0_4A": "#0072B2", "interior_8_16A": "#D55E00"}
    for label, members in (("L1_0_4A", first_layer), ("interior_8_16A", interior)):
        axes[0, 0].hist(q_values[:, members].ravel(), bins=q_edges, density=True,
                        histtype="step", lw=1.6, label=label.replace("_", " "),
                        color=colors[label])
    axes[0, 0].axvline(0.90, color="0.35", ls="--", lw=1.0)
    axes[0, 0].set(xlabel="Tetrahedral order q", ylabel="Probability density",
                   title="a  Local tetrahedral order (last 50 ps)")
    axes[0, 0].legend(frameon=False)

    x = np.arange(len(layer_hbond_rows)) + 1
    axes[0, 1].plot(x, [r[7] for r in layer_hbond_rows], "o-", color="#009E73",
                    label="mean q")
    axes[0, 1].plot(x, [r[12] for r in layer_hbond_rows], "s-", color="#CC79A7",
                    label="HB observations from stable pairs")
    axes[0, 1].set(xlabel="Fixed 4 A layer", ylabel="Fraction / order",
                   ylim=(0, 1.02), title="b  Order and H-bond persistence")
    axes[0, 1].legend(frameon=False)

    for lid in layer_ids:
        if lid not in (min(layer_ids), 2, 3, max(layer_ids)):
            continue
        lags = np.asarray(lag_ps_values)
        msd = [mobility_by_layer[lid][float(t)][0] for t in lags]
        axes[1, 0].plot(lags, msd, marker="o", ms=3, label=f"L{lid + 1}")
    axes[1, 0].set(xlabel="Lag time (ps)", ylabel=r"COM-corrected lateral O MSD ($\AA^2$)",
                   title="c  Cage-relative water mobility (last 50 ps)")
    axes[1, 0].legend(frameon=False, ncol=2)

    for region in regions:
        y = smoothed[(region, "translation", "self")]
        mask = (frequency >= 0) & (frequency <= 400)
        y = y / max(float(np.max(y[mask])), 1.0e-30)
        axes[1, 1].plot(frequency[mask], y[mask], lw=1.3,
                        color=colors[region], label=region.replace("_", " "))
    axes[1, 1].set(xlabel=r"Wavenumber (cm$^{-1}$)", ylabel="Normalized self VDOS",
                   title="d  Intermolecular translation")
    axes[1, 1].legend(frameon=False)

    for region in regions:
        self_y = smoothed[(region, "libration", "self")]
        cross_y = smoothed[(region, "libration", "cross")]
        mask = (frequency >= 250) & (frequency <= 1100)
        scale = max(float(np.max(self_y[mask])), 1.0e-30)
        axes[2, 0].plot(frequency[mask], self_y[mask] / scale, lw=1.3,
                        color=colors[region], label=region.replace("_", " "))
        axes[2, 0].plot(frequency[mask], cross_y[mask] / scale, lw=0.9,
                        color=colors[region], ls="--", alpha=0.8)
    axes[2, 0].axhline(0, color="0.5", lw=0.7)
    axes[2, 0].set(xlabel=r"Wavenumber (cm$^{-1}$)",
                   ylabel="Self (solid) / signed cross (dashed)",
                   title="e  Libration and cross-molecular correlation")
    axes[2, 0].legend(frameon=False)

    ax = axes[2, 1]
    ax.plot(stability[:, 1], stability[:, 2], color="#E69F00", lw=0.8,
            label="Temperature")
    ax.set(xlabel="Time (ps)", ylabel="Temperature (K)",
           title="f  Whole-trajectory stability")
    ax2 = ax.twinx()
    structure_curve_label = (r"CaF$_2$ RMSD" if has_caf2 else
                             "Oil covalent-bond RMSD")
    ax2.plot(stability[:, 1], stability[:, 3], color="#56B4E9", lw=0.8,
             label=structure_curve_label)
    ax2.set_ylabel(structure_curve_label + r" ($\AA$)")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], frameon=False,
              loc="upper left")

    figure_stem = args.output / "oil_water_ice_hbond_stability"
    fig.savefig(figure_stem.with_suffix(".png"), dpi=300)
    fig.savefig(figure_stem.with_suffix(".pdf"))
    fig.savefig(figure_stem.with_suffix(".svg"))
    plt.close(fig)

    readme = f"""# {args.system_label} final frame and ice-mode audit

## Final frame

- `{poscar_filename}` is the last complete trajectory frame: step {final_step}, {final_time:.3f} ps.
- POSCAR species are regrouped as {' '.join(poscar_info['species_order'])}; the source-index mapping is retained in `final_frame_metadata.json`.

## Scope and definitions

- {args.interface_description}
- Hydrogen bonds use H...O <= {args.hbond_ho:.2f} A, O...O <= {args.hbond_oo:.2f} A, and O-H...O >= {args.hbond_angle:.0f} degrees over the final {args.hbond_last_ps:.0f} ps.
- A persistent directed pair has occupancy >= {args.stable_occupancy:.0%}.
- The tetrahedral order parameter uses the four nearest water oxygens. q=1 is an ideal tetrahedron; q alone is not an ice label.
- The spectral collective diagnostic compares the self spectrum with the signed intermolecular q=0 cross term and reports a spectral participation ratio. It is not a Hessian normal-mode calculation.

## First 4 A layer result

- Mean tetrahedral q: {l1_q:.4f} (8-16 A internal control: {interior_q:.4f}).
- Fraction q >= 0.80: {l1_q80:.4f}.
- Mean H-bond degree per water: {float(l1_row[6]):.4f}.
- Fraction of H-bond observations from >=10% occupancy pairs: {l1_stable_observation_fraction:.4f}.
- Mean continuous H-bond lifetime: {l1_hb_lifetime:.4f} ps.
- All-water-COM-corrected effective lateral Dxy: {l1_dxy:.6f} A^2/ps.
- Translation 20-300 cm-1 signed cross/self: {l1_trans_cross:.4f}.
- Libration 300-1000 cm-1 signed cross/self: {l1_lib_cross:.4f}.

## Layer-resolved decision

`ice_like_diagnostic_by_4A_layer.csv` distinguishes four cases: ice-I-like local solid,
ordered/caged ice-like water below the strict ice-I q threshold, caged but
tetrahedrally disordered interface water, and more mobile boundary water.

## Decision

{result['decision']['verdict']}

The decision requires structure, all-water-COM-corrected mobility, persistent H-bond
pair identity, and spectral correlation together. A broad or shifted OH-stretch
feature by itself is not evidence of ice. The current trajectory analysis supports
ice-like collective dynamics where these diagnostics coincide, but it is not a
Hessian/phonon calculation and therefore does not prove a crystalline-ice normal mode.

## Whole-system stability

{result['whole_trajectory_stability']['verdict']}
"""
    (args.output / "README.md").write_text(readme)
    (args.output / "final_frame_metadata.json").write_text(
        json.dumps(result["final_poscar"], indent=2) + "\n"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
