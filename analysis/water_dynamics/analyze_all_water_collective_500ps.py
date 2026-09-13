#!/usr/bin/env python3
"""Unbiased all-water structure, collective dynamics, network and dipole audit.

The workflow compares systems with one fixed analysis contract:
  * 0-24 A above the water GDS, mutually exclusive 2 A layers;
  * structure/H-bond network at 10 fs over 500 ps;
  * COM-corrected mobility and displacement covariance at 100 fs;
  * translation self/cross spectra and low-q S(q,w) from native 1 fs data;
  * layer dipole correlations at the native 100 fs label cadence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np


CM_PER_PS = 33.3564095198152
DEBYE_PER_E_ANGSTROM = 4.803204712570263
MASS_O = 15.9994
MASS_H = 1.00794
WATER_DENSITY_FACTOR_G_CM3_PER_A3 = (MASS_O + 2 * MASS_H) / 0.602214076


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
        distance = np.linalg.norm(
            minimum_image(positions[hydrogen] - positions[o], cell), axis=1
        )
        candidates.extend((float(d), int(o), int(h))
                          for h, d in zip(hydrogen, distance) if d <= cutoff)
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
        raise RuntimeError(f"Only {len(waters)}/{len(oxygen)} waters assigned")
    return np.asarray(waters, dtype=np.int64)


def write_csv(path: Path, header: list[str], rows) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def trajectory_parts(paths: list[Path], total_frames: int, stride: int,
                     datasets: tuple[str, ...]):
    global_start = 0
    remaining = total_frames
    for path in paths:
        with h5py.File(path, "r") as handle:
            if not bool(handle.attrs.get("complete", False)):
                raise RuntimeError(f"Incomplete source file: {path}")
            available = min(len(handle["step"]), remaining)
            if available <= 0:
                break
            offset = (-global_start) % stride
            indices = np.arange(offset, available, stride, dtype=np.int64)
            yield {name: np.asarray(handle[name][indices]) for name in datasets}
            global_start += available
            remaining -= available
    if remaining:
        raise RuntimeError(
            f"Trajectory has {total_frames - remaining} usable frames; "
            f"{total_frames} requested"
        )


def load_candidate_rows(path: Path) -> dict[int, dict[str, float | int | str]]:
    rows: dict[int, dict[str, float | int | str]] = {}
    with path.open(newline="") as stream:
        for raw in csv.DictReader(stream):
            serial = int(raw["O_serial_1based"])
            row: dict[str, float | int | str] = dict(raw)
            for key in (
                "mean_O_z_A", "std_O_z_A", "mean_z_minus_GDS_A",
                "mean_HB_degree", "fraction_frames_degree_eq4",
                "fraction_frames_degree_ge4", "mean_donated_HB",
                "mean_accepted_HB", "top4_partner_identity_fraction",
                "fourth_partner_occupancy", "top4_mean_partner_occupancy",
                "four_partner_stability_score",
            ):
                row[key] = float(raw[key])
            row["unique_HB_partners"] = int(raw["unique_HB_partners"])
            rows[serial] = row
    return rows


def tetrahedral_q_chunk(o_xyz: np.ndarray, cell: np.ndarray) -> np.ndarray:
    delta = minimum_image(
        o_xyz[:, None, :, :] - o_xyz[:, :, None, :], cell
    )
    distance = np.linalg.norm(delta, axis=-1)
    diagonal = np.arange(o_xyz.shape[1])
    distance[:, diagonal, diagonal] = np.inf
    nearest = np.argpartition(distance, 4, axis=2)[:, :, :4]
    vectors = np.take_along_axis(delta, nearest[:, :, :, None], axis=2)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=3, keepdims=True), 1.0e-12)
    terms = np.zeros(o_xyz.shape[:2], dtype=np.float64)
    for j in range(3):
        for k in range(j + 1, 4):
            cosine = np.sum(vectors[:, :, j] * vectors[:, :, k], axis=2)
            terms += (cosine + 1.0 / 3.0) ** 2
    return 1.0 - 3.0 * terms / 8.0


def directed_hbond(o_xyz: np.ndarray, h_xyz: np.ndarray, cell: np.ndarray,
                   ho_cutoff: float, oo_cutoff: float,
                   angle_cutoff: float) -> np.ndarray:
    h_to_acceptor = minimum_image(
        o_xyz[:, None, None, :, :] - h_xyz[:, :, :, None, :], cell
    )
    ho_distance = np.linalg.norm(h_to_acceptor, axis=-1)
    donor_to_acceptor = minimum_image(
        o_xyz[:, None, :, :] - o_xyz[:, :, None, :], cell
    )
    oo_distance = np.linalg.norm(donor_to_acceptor, axis=-1)
    h_to_donor = minimum_image(o_xyz[:, :, None, :] - h_xyz, cell)
    h_to_donor = h_to_donor[:, :, :, None, :]
    cosine = np.sum(h_to_donor * h_to_acceptor, axis=-1) / np.maximum(
        np.linalg.norm(h_to_donor, axis=-1)
        * np.linalg.norm(h_to_acceptor, axis=-1), 1.0e-12
    )
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    not_self = ~np.eye(o_xyz.shape[1], dtype=bool)[None, :, None, :]
    state = (
        (ho_distance <= ho_cutoff)
        & (oo_distance[:, :, None, :] <= oo_cutoff)
        & (angle >= angle_cutoff)
        & not_self
    )
    return np.any(state, axis=2)


def largest_component_fraction(adjacency: np.ndarray,
                               members: np.ndarray | None = None) -> float:
    if members is None:
        members = np.arange(len(adjacency))
    members = np.asarray(members, dtype=int)
    if not len(members):
        return math.nan
    sub = adjacency[np.ix_(members, members)]
    seen = np.zeros(len(members), dtype=bool)
    largest = 0
    for start in range(len(members)):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            nxt = np.flatnonzero(sub[node] & ~seen)
            seen[nxt] = True
            stack.extend(int(value) for value in nxt)
        largest = max(largest, size)
    return largest / len(members)


def cycle_basis_lengths(adjacency: np.ndarray) -> list[int]:
    """Return lengths in a Paton fundamental cycle basis."""
    nodes = set(range(len(adjacency)))
    cycles: list[int] = []
    while nodes:
        root = nodes.pop()
        stack = [root]
        pred = {root: root}
        used: dict[int, set[int]] = {root: set()}
        while stack:
            current = stack.pop()
            current_used = used[current]
            for nbr_raw in np.flatnonzero(adjacency[current]):
                nbr = int(nbr_raw)
                if nbr not in used:
                    pred[nbr] = current
                    stack.append(nbr)
                    used[nbr] = {current}
                elif nbr == current:
                    cycles.append(1)
                elif nbr not in current_used:
                    nbr_used = used[nbr]
                    length = 2
                    parent = pred[current]
                    while parent not in nbr_used:
                        length += 1
                        parent = pred[parent]
                    cycles.append(length + 1)
                    used[nbr].add(current)
        nodes.difference_update(pred)
    return cycles


def autocorrelation(signal: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(signal, dtype=float)
    x -= np.mean(x)
    n = len(x)
    nfft = 1 << (2 * n - 1).bit_length()
    transformed = np.fft.rfft(x, n=nfft)
    result = np.fft.irfft(transformed * np.conjugate(transformed), n=nfft)[:n]
    result /= np.arange(n, 0, -1)
    if result[0] <= 1.0e-30:
        return np.full(max_lag + 1, np.nan)
    return result[:max_lag + 1] / result[0]


def spectral_metrics(signal: np.ndarray, members: np.ndarray,
                     window: np.ndarray) -> tuple[np.ndarray, ...]:
    selected = np.asarray(signal[:, members], dtype=np.float64)
    selected -= selected.mean(axis=0, keepdims=True)
    transformed = np.fft.rfft(selected * window[:, None, None], axis=0)
    power = np.sum(np.abs(transformed) ** 2, axis=2)
    norm = float(np.sum(window ** 2))
    self_s = np.sum(power, axis=1) / (len(members) * norm)
    coherent = np.sum(np.abs(np.sum(transformed, axis=1)) ** 2, axis=1)
    coherent /= len(members) * norm
    cross = coherent - self_s
    participation = np.sum(power, axis=1) ** 2 / np.maximum(
        len(members) * np.sum(power ** 2, axis=1), 1.0e-30
    )
    return self_s, coherent, cross, participation


def structure_network_pass(args, paths, symbols, cell, waters, fixed_layer,
                           candidate_rows, source_dt_fs):
    output = args.output
    oxygen, h1, h2 = waters.T
    h_by_water = np.stack((h1, h2), axis=1)
    nwater = len(waters)
    samples = len(range(0, args.source_frames, args.structure_stride))
    sample_dt_ps = source_dt_fs * args.structure_stride / 1000.0
    nblocks = int(round(args.source_frames * source_dt_fs / 1000.0
                        / args.block_ps))
    samples_per_block = int(round(args.block_ps / sample_dt_ps))
    if samples != nblocks * samples_per_block:
        raise RuntimeError("Structure samples do not divide into requested blocks")

    o_comcorr_xy = np.empty((samples, nwater, 2), dtype=np.float32)
    dynamic_layer = np.empty((samples, nwater), dtype=np.int8)
    upper_i, upper_j = np.triu_indices(nwater, 1)
    npair = len(upper_i)
    edge_occupancy = np.zeros(npair, dtype=np.int64)
    edge_run = np.zeros(npair, dtype=np.int64)
    edge_episode_count = np.zeros(npair, dtype=np.int64)
    edge_episode_frames = np.zeros(npair, dtype=np.int64)
    edge_episode_max = np.zeros(npair, dtype=np.int64)
    adjacency_100fs = np.empty((samples // 10, npair), dtype=bool)

    shape = (nblocks, args.n_layers)
    profile_count = np.zeros(shape, dtype=np.int64)
    profile_q = np.zeros(shape, dtype=np.float64)
    profile_q2 = np.zeros(shape, dtype=np.float64)
    profile_degree = np.zeros(shape, dtype=np.float64)
    profile_degree2 = np.zeros(shape, dtype=np.float64)
    profile_degree4 = np.zeros(shape, dtype=np.float64)
    profile_p1 = np.zeros(shape, dtype=np.float64)
    profile_p2 = np.zeros(shape, dtype=np.float64)
    network_rows = []
    network_layer_rows = []
    ring_sum = np.zeros((args.n_layers, 3), dtype=np.float64)
    ring_samples = np.zeros(args.n_layers, dtype=np.int64)

    previous_all_o = None
    unwrapped_all_o = None
    previous_edge_100fs = None
    index = 0
    for part_number, part in enumerate(trajectory_parts(
        paths, args.source_frames, args.structure_stride,
        ("positions_A", "step", "time_ps")), start=1
    ):
        positions = np.asarray(part["positions_A"], dtype=np.float32)
        steps = np.asarray(part["step"], dtype=int)
        times = np.asarray(part["time_ps"], dtype=float)
        for start in range(0, len(positions), 20):
            stop = min(start + 20, len(positions))
            chunk = positions[start:stop]
            o_xyz = chunk[:, oxygen]
            h_xyz = chunk[:, h_by_water]
            directed = directed_hbond(
                o_xyz, h_xyz, cell, args.hbond_ho,
                args.hbond_oo, args.hbond_angle,
            )
            adjacency = directed | np.swapaxes(directed, 1, 2)
            degree = np.sum(adjacency, axis=2)
            q = tetrahedral_q_chunk(o_xyz, cell)
            r1 = minimum_image(chunk[:, h1] - o_xyz, cell)
            r2 = minimum_image(chunk[:, h2] - o_xyz, cell)
            bisector = r1 + r2
            bisector /= np.maximum(np.linalg.norm(bisector, axis=2, keepdims=True),
                                   1.0e-12)
            p1 = bisector[:, :, 2]
            p2 = 0.5 * (3.0 * p1 ** 2 - 1.0)
            relz = (o_xyz[:, :, 2] % cell[2, 2]) - args.gds
            layer = np.floor(relz / args.layer_width).astype(np.int16)

            for local in range(len(chunk)):
                global_index = index + local
                wrapped_all_o = np.asarray(o_xyz[local], dtype=float)
                if previous_all_o is None:
                    unwrapped_all_o = wrapped_all_o.copy()
                else:
                    unwrapped_all_o = unwrapped_all_o + minimum_image(
                        wrapped_all_o - previous_all_o, cell
                    )
                previous_all_o = wrapped_all_o
                com = np.mean(unwrapped_all_o, axis=0)
                o_comcorr_xy[global_index] = unwrapped_all_o[:, :2] - com[:2]
                dynamic_layer[global_index] = np.clip(layer[local], -1, 127)

                block = global_index // samples_per_block
                valid = (layer[local] >= 0) & (layer[local] < args.n_layers)
                bins = layer[local, valid]
                count = np.bincount(bins, minlength=args.n_layers)
                profile_count[block] += count
                for accumulator, values in (
                    (profile_q, q[local]), (profile_q2, q[local] ** 2),
                    (profile_degree, degree[local]),
                    (profile_degree2, degree[local] ** 2),
                    (profile_degree4, degree[local] == 4),
                    (profile_p1, p1[local]), (profile_p2, p2[local]),
                ):
                    accumulator[block] += np.bincount(
                        bins, weights=np.asarray(values)[valid],
                        minlength=args.n_layers,
                    )

                edge = adjacency[local, upper_i, upper_j]
                edge_occupancy += edge
                ended = (edge_run > 0) & ~edge
                edge_episode_count[ended] += 1
                edge_episode_frames[ended] += edge_run[ended]
                edge_episode_max[ended] = np.maximum(
                    edge_episode_max[ended], edge_run[ended]
                )
                edge_run[edge] += 1
                edge_run[~edge] = 0

                if global_index % 10 == 0:
                    sample100 = global_index // 10
                    adjacency_100fs[sample100] = edge
                    full_lcc = largest_component_fraction(adjacency[local])
                    edge_total = int(np.sum(edge))
                    different_layer = fixed_layer[upper_i] != fixed_layer[upper_j]
                    valid_pair = ((fixed_layer[upper_i] >= 0)
                                  & (fixed_layer[upper_j] >= 0))
                    cross_edges = int(np.sum(edge & different_layer & valid_pair))
                    if previous_edge_100fs is None:
                        rewiring = math.nan
                        retained = math.nan
                    else:
                        union = edge | previous_edge_100fs
                        rewiring = float(np.sum(edge ^ previous_edge_100fs)
                                         / max(np.sum(union), 1))
                        retained = float(np.sum(edge & previous_edge_100fs)
                                         / max(np.sum(previous_edge_100fs), 1))
                    network_rows.append([
                        int(steps[start + local]), float(times[start + local]),
                        edge_total, float(np.mean(degree[local])), full_lcc,
                        cross_edges / max(edge_total, 1), rewiring, retained,
                    ])
                    for layer_index in range(args.n_layers):
                        members = np.flatnonzero(fixed_layer == layer_index)
                        if previous_edge_100fs is None or len(members) < 2:
                            layer_rewiring = math.nan
                        else:
                            within = np.isin(upper_i, members) & np.isin(upper_j, members)
                            union = edge[within] | previous_edge_100fs[within]
                            layer_rewiring = float(
                                np.sum(edge[within] ^ previous_edge_100fs[within])
                                / max(np.sum(union), 1)
                            )
                        network_layer_rows.append([
                            float(times[start + local]), layer_index + 1,
                            layer_index * args.layer_width,
                            (layer_index + 1) * args.layer_width,
                            len(members),
                            largest_component_fraction(adjacency[local], members),
                            float(np.mean(degree[local, members])) if len(members)
                            else math.nan, layer_rewiring,
                        ])
                    previous_edge_100fs = edge.copy()

                if global_index % 100 == 0:
                    for layer_index in range(args.n_layers):
                        members = np.flatnonzero(fixed_layer == layer_index)
                        if len(members) < 3:
                            continue
                        sub = adjacency[local][np.ix_(members, members)]
                        lengths = cycle_basis_lengths(sub)
                        ring_sum[layer_index, 0] += sum(value == 5 for value in lengths)
                        ring_sum[layer_index, 1] += sum(value == 6 for value in lengths)
                        ring_sum[layer_index, 2] += len(lengths)
                        ring_samples[layer_index] += 1
            index += len(chunk)
        if part_number == 1 or part_number % 10 == 0:
            print(json.dumps({"structure_parts_done": part_number,
                              "structure_samples": index}), flush=True)
    if index != samples:
        raise RuntimeError(f"Expected {samples} structure samples, obtained {index}")

    active = edge_run > 0
    edge_episode_count[active] += 1
    edge_episode_frames[active] += edge_run[active]
    edge_episode_max[active] = np.maximum(edge_episode_max[active], edge_run[active])

    profile_block_rows = []
    metrics_by_block: dict[str, np.ndarray] = {}
    slice_volume_A3 = float(np.linalg.norm(np.cross(cell[0], cell[1]))
                            * args.layer_width)
    mean_count_by_block = profile_count / samples_per_block
    number_density_by_block = mean_count_by_block / slice_volume_A3
    mass_density_by_block = (number_density_by_block
                             * WATER_DENSITY_FACTOR_G_CM3_PER_A3)
    safe_count = np.maximum(profile_count, 1)
    for name, raw_values in (
        ("mean_tetrahedral_q", profile_q / safe_count),
        ("mean_HB_degree", profile_degree / safe_count),
        ("fraction_degree_eq4", profile_degree4 / safe_count),
        ("mean_P1_cos_theta", profile_p1 / safe_count),
        ("mean_P2", profile_p2 / safe_count),
    ):
        values = np.asarray(raw_values, dtype=float)
        values[profile_count == 0] = np.nan
        metrics_by_block[name] = values
    for block in range(nblocks):
        for layer_index in range(args.n_layers):
            profile_block_rows.append([
                block + 1, block * args.block_ps,
                (block + 1) * args.block_ps, layer_index + 1,
                layer_index * args.layer_width,
                (layer_index + 1) * args.layer_width,
                int(profile_count[block, layer_index]),
                float(mean_count_by_block[block, layer_index]),
                float(number_density_by_block[block, layer_index]),
                float(mass_density_by_block[block, layer_index]),
                *[metrics_by_block[name][block, layer_index]
                  for name in metrics_by_block],
            ])
    write_csv(
        output / "z_profile_2A_block_statistics.csv",
        ["block", "time_min_ps", "time_max_ps", "layer",
         "z_minus_GDS_min_A", "z_minus_GDS_max_A", "water_frame_observations",
         "mean_instantaneous_water_count", "water_number_density_A-3",
         "water_mass_density_g_cm-3",
         *metrics_by_block], profile_block_rows,
    )
    profile_summary_rows = []
    for layer_index in range(args.n_layers):
        row = [layer_index + 1, layer_index * args.layer_width,
               (layer_index + 1) * args.layer_width,
               float(np.mean(mean_count_by_block[:, layer_index])),
               float(np.mean(number_density_by_block[:, layer_index])),
               float(np.mean(mass_density_by_block[:, layer_index]))]
        for name in metrics_by_block:
            values = metrics_by_block[name][:, layer_index]
            finite = np.isfinite(values)
            row.extend([
                float(np.nanmean(values)) if np.any(finite) else math.nan,
                float(np.nanstd(values, ddof=1) / np.sqrt(np.sum(finite)))
                if np.sum(finite) > 1 else math.nan,
            ])
        fixed_members = np.flatnonzero(fixed_layer == layer_index)
        candidate_values = [candidate_rows[int(oxygen[i]) + 1]
                            for i in fixed_members]
        row.extend([
            len(fixed_members),
            float(np.mean([float(value["four_partner_stability_score"])
                           for value in candidate_values])) if candidate_values else math.nan,
            float(np.mean([float(value["top4_partner_identity_fraction"])
                           for value in candidate_values])) if candidate_values else math.nan,
        ])
        profile_summary_rows.append(row)
    profile_header = ["layer", "z_minus_GDS_min_A", "z_minus_GDS_max_A",
                      "mean_instantaneous_water_count",
                      "water_number_density_A-3", "water_mass_density_g_cm-3"]
    for name in metrics_by_block:
        profile_header.extend([name, name + "_SEM_50x10ps"])
    profile_header.extend(["fixed_mean_z_water_count",
                           "mean_four_partner_stability_score",
                           "mean_top4_partner_identity_fraction"])
    write_csv(output / "z_profile_2A_summary.csv", profile_header,
              profile_summary_rows)

    write_csv(
        output / "hbond_network_time_series_100fs.csv",
        ["step", "time_ps", "edges", "mean_degree",
         "largest_component_fraction", "cross_fixed_layer_edge_fraction",
         "edge_rewiring_Jaccard_since_previous_100fs",
         "previous_edges_retained_fraction"],
        network_rows,
    )
    write_csv(
        output / "hbond_network_layer_time_series_100fs.csv",
        ["time_ps", "layer", "z_minus_GDS_min_A", "z_minus_GDS_max_A",
         "fixed_waters", "largest_component_fraction", "mean_total_HB_degree",
         "edge_rewiring_Jaccard_since_previous_100fs"],
        network_layer_rows,
    )
    ring_rows = []
    for layer_index in range(args.n_layers):
        ring_rows.append([
            layer_index + 1, layer_index * args.layer_width,
            (layer_index + 1) * args.layer_width, int(ring_samples[layer_index]),
            *(ring_sum[layer_index] / max(ring_samples[layer_index], 1)),
        ])
    write_csv(
        output / "hbond_cycle_basis_by_layer_1ps.csv",
        ["layer", "z_minus_GDS_min_A", "z_minus_GDS_max_A", "samples_1ps",
         "mean_5_member_cycles", "mean_6_member_cycles", "mean_all_basis_cycles"],
        ring_rows,
    )

    pair_mid_z = 0.5 * (
        np.asarray([float(candidate_rows[int(oxygen[i]) + 1]["mean_z_minus_GDS_A"])
                    for i in upper_i])
        + np.asarray([float(candidate_rows[int(oxygen[j]) + 1]["mean_z_minus_GDS_A"])
                      for j in upper_j])
    )
    pair_layer = np.floor(pair_mid_z / args.layer_width).astype(int)
    lifetime_rows = []
    for layer_index in range(args.n_layers):
        members = pair_layer == layer_index
        episodes = np.sum(edge_episode_count[members])
        occupied_frames = np.sum(edge_episode_frames[members])
        active_pairs = np.sum(edge_occupancy[members] > 0)
        lifetime_rows.append([
            layer_index + 1, layer_index * args.layer_width,
            (layer_index + 1) * args.layer_width, int(np.sum(members)),
            int(active_pairs), float(np.sum(edge_occupancy[members])
                                     / max(np.sum(members) * samples, 1)),
            int(episodes), float(occupied_frames / max(episodes, 1) * sample_dt_ps),
            float(np.max(edge_episode_max[members]) * sample_dt_ps)
            if np.any(members) else math.nan,
        ])
    write_csv(
        output / "hbond_edge_continuous_lifetime_by_layer.csv",
        ["layer", "pair_mid_z_min_A", "pair_mid_z_max_A", "possible_pairs",
         "observed_pairs", "mean_pair_occupancy", "continuous_episodes",
         "mean_continuous_lifetime_ps", "max_continuous_lifetime_ps"], lifetime_rows,
    )

    intermittent_rows = []
    lag100 = np.unique(np.rint(np.geomspace(1, 500, 45)).astype(int))
    lag100 = lag100[lag100 < len(adjacency_100fs)]
    for layer_index in range(args.n_layers):
        members = pair_layer == layer_index
        if not np.any(members):
            continue
        state = adjacency_100fs[:, members]
        occupancy = float(np.mean(state))
        for lag in lag100:
            joint = float(np.mean(state[:-lag] & state[lag:]))
            conditional = joint / max(occupancy, 1.0e-30)
            normalized = ((joint - occupancy ** 2)
                          / max(occupancy - occupancy ** 2, 1.0e-30))
            intermittent_rows.append([
                layer_index + 1, layer_index * args.layer_width,
                (layer_index + 1) * args.layer_width, lag * 0.1,
                occupancy, conditional, normalized,
            ])
    write_csv(
        output / "hbond_intermittent_correlation_by_layer_100fs.csv",
        ["layer", "pair_mid_z_min_A", "pair_mid_z_max_A", "lag_ps",
         "pair_occupancy", "conditional_survival", "normalized_correlation"],
        intermittent_rows,
    )

    coords100 = np.asarray(o_comcorr_xy[::10], dtype=np.float64)
    mobility_rows = []
    lag_steps = np.unique(np.rint(np.geomspace(1, 1000, 55)).astype(int))
    lag_steps = lag_steps[lag_steps < len(coords100)]
    for layer_index in range(args.n_layers):
        members = np.flatnonzero(fixed_layer == layer_index)
        if not len(members):
            continue
        for lag in lag_steps:
            delta = coords100[lag:, members] - coords100[:-lag, members]
            by_water = np.mean(np.sum(delta ** 2, axis=2), axis=0)
            mobility_rows.append([
                layer_index + 1, layer_index * args.layer_width,
                (layer_index + 1) * args.layer_width, len(members), lag * 0.1,
                float(np.mean(by_water)),
                float(np.std(by_water, ddof=1) / np.sqrt(len(members)))
                if len(members) > 1 else math.nan,
            ])
    write_csv(
        output / "COM_corrected_MSD_xy_by_layer_100fs.csv",
        ["layer", "z_minus_GDS_min_A", "z_minus_GDS_max_A", "waters",
         "lag_ps", "mean_MSD_xy_A2", "water_to_water_SEM_A2"], mobility_rows,
    )

    episode_count = np.zeros(args.n_layers, dtype=int)
    episode_frames = np.zeros(args.n_layers, dtype=int)
    episode_max = np.zeros(args.n_layers, dtype=int)
    transition_count = np.zeros(args.n_layers, dtype=int)
    for water in range(nwater):
        sequence = dynamic_layer[:, water]
        start = 0
        for end in range(1, len(sequence) + 1):
            if end == len(sequence) or sequence[end] != sequence[start]:
                layer_index = int(sequence[start])
                if 0 <= layer_index < args.n_layers:
                    length = end - start
                    episode_count[layer_index] += 1
                    episode_frames[layer_index] += length
                    episode_max[layer_index] = max(episode_max[layer_index], length)
                    if end < len(sequence):
                        transition_count[layer_index] += 1
                start = end
    residence_rows = []
    for layer_index in range(args.n_layers):
        residence_rows.append([
            layer_index + 1, layer_index * args.layer_width,
            (layer_index + 1) * args.layer_width, int(episode_count[layer_index]),
            float(episode_frames[layer_index] / max(episode_count[layer_index], 1)
                  * sample_dt_ps),
            float(episode_max[layer_index] * sample_dt_ps),
            int(transition_count[layer_index]),
        ])
    write_csv(
        output / "layer_residence_time_10fs.csv",
        ["layer", "z_minus_GDS_min_A", "z_minus_GDS_max_A", "episodes",
         "mean_residence_ps", "max_residence_ps", "outgoing_transitions"],
        residence_rows,
    )

    displacement = coords100[10:] - coords100[:-10]
    covariance = np.einsum("tix,tjx->ij", displacement, displacement)
    covariance /= len(displacement)
    rms = np.sqrt(np.mean(np.sum(displacement ** 2, axis=2), axis=0))
    correlation = covariance / np.maximum(rms[:, None] * rms[None, :], 1.0e-30)
    serials = oxygen + 1
    matrix_header = ["O_serial_1based", "fixed_layer"] + [f"O{value}" for value in serials]
    matrix_rows = [[int(serials[i]), int(fixed_layer[i]) + 1,
                    *correlation[i].tolist()] for i in range(nwater)]
    write_csv(output / "displacement_correlation_matrix_lag1ps.csv",
              matrix_header, matrix_rows)
    layer_corr_rows = []
    for left in range(args.n_layers):
        li = np.flatnonzero(fixed_layer == left)
        for right in range(args.n_layers):
            rj = np.flatnonzero(fixed_layer == right)
            if not len(li) or not len(rj):
                value = math.nan
            else:
                block = correlation[np.ix_(li, rj)]
                if left == right:
                    value = float(np.mean(block[~np.eye(len(li), dtype=bool)])) \
                        if len(li) > 1 else math.nan
                else:
                    value = float(np.mean(block))
            layer_corr_rows.append([left + 1, right + 1, value])
    write_csv(
        output / "layer_displacement_correlation_matrix_lag1ps.csv",
        ["layer_i", "layer_j", "mean_normalized_vector_correlation"],
        layer_corr_rows,
    )
    flattened = displacement.reshape(len(displacement), -1)
    flattened -= flattened.mean(axis=0, keepdims=True)
    eig = np.linalg.eigvalsh(flattened.T @ flattened / len(flattened))
    eig = np.maximum(eig, 0.0)
    pr = float(np.sum(eig) ** 2 / max(np.sum(eig ** 2), 1.0e-30))
    collective_summary = {
        "lag_ps": 1.0,
        "degrees_of_freedom_xy": 2 * nwater,
        "eigenvalue_participation_ratio": pr,
        "normalized_participation_ratio": pr / (2 * nwater),
        "largest_eigenvalue_fraction": float(eig[-1] / max(np.sum(eig), 1.0e-30)),
        "mean_offdiagonal_correlation": float(
            np.mean(correlation[~np.eye(nwater, dtype=bool)])
        ),
        "COM_constraint_expected_uniform_baseline": -1.0 / (nwater - 1),
    }
    (output / "displacement_collective_summary.json").write_text(
        json.dumps(collective_summary, indent=2) + "\n"
    )
    return {
        "structure_samples": samples,
        "structure_sampling_fs": source_dt_fs * args.structure_stride,
        "nblocks": nblocks,
        "displacement_collective": collective_summary,
    }


def spectrum_pass(args, paths, cell, waters, fixed_layer, source_dt_fs):
    output = args.output
    oxygen, h1, h2 = waters.T
    block_frames = int(round(args.spectral_block_ps * 1000.0 / source_dt_fs))
    expected_blocks = args.source_frames // block_frames
    window = np.blackman(block_frames).astype(np.float64)
    frequency = np.fft.rfftfreq(block_frames, d=source_dt_fs / 1000.0) * CM_PER_PS
    groups = {layer: np.flatnonzero(fixed_layer == layer)
              for layer in range(args.n_layers)}
    metrics = ("self", "coherent", "cross", "participation")
    spectral_sum = {(layer, metric): np.zeros(len(frequency), dtype=np.float64)
                    for layer in groups for metric in metrics}
    sq_sum = {(layer, order): np.zeros(len(frequency), dtype=np.float64)
              for layer in groups for order in (1, 2)}
    block_band_rows = []
    buffers_pos: list[np.ndarray] = []
    buffers_vel: list[np.ndarray] = []
    buffered = 0
    block_index = 0

    def process_block(positions: np.ndarray, velocities: np.ndarray) -> None:
        nonlocal block_index
        vcom = (MASS_O * velocities[:, oxygen]
                + MASS_H * (velocities[:, h1] + velocities[:, h2])) \
            / (MASS_O + 2 * MASS_H)
        for layer, members in groups.items():
            if len(members) < 2:
                continue
            values = spectral_metrics(vcom, members, window)
            for metric, spectrum in zip(metrics, values):
                spectral_sum[(layer, metric)] += spectrum
            q1 = 2.0 * np.pi / cell[0, 0]
            for order in (1, 2):
                phase = np.exp(1.0j * order * q1
                               * positions[:, oxygen[members], 0])
                rho = np.sum(phase, axis=1)
                rho -= np.mean(rho)
                transformed = np.fft.fft(rho * window)[:len(frequency)]
                spectrum = np.abs(transformed) ** 2 / (
                    len(members) * np.sum(window ** 2)
                )
                sq_sum[(layer, order)] += spectrum
            band = (frequency >= 20.0) & (frequency <= 300.0)
            self_s, _coherent, cross_s, participation = values
            self_int = float(np.trapezoid(self_s[band], frequency[band]))
            cross_int = float(np.trapezoid(cross_s[band], frequency[band]))
            pr_weighted = float(np.sum(participation[band] * self_s[band])
                                / max(np.sum(self_s[band]), 1.0e-30))
            block_band_rows.append([
                block_index + 1, layer + 1, layer * args.layer_width,
                (layer + 1) * args.layer_width, len(members),
                cross_int / max(self_int, 1.0e-30), pr_weighted,
            ])
        block_index += 1
        print(json.dumps({"spectral_blocks_done": block_index,
                          "spectral_blocks_total": expected_blocks}), flush=True)

    remaining = args.source_frames
    for path in paths:
        if remaining <= 0:
            break
        with h5py.File(path, "r") as handle:
            available = min(len(handle["step"]), remaining)
            start = 0
            while start < available:
                take = min(block_frames - buffered, available - start)
                buffers_pos.append(np.asarray(
                    handle["positions_A"][start:start + take], dtype=np.float32
                ))
                buffers_vel.append(np.asarray(
                    handle["velocities_A_per_fs"][start:start + take], dtype=np.float32
                ))
                buffered += take
                start += take
                if buffered == block_frames:
                    process_block(np.concatenate(buffers_pos, axis=0),
                                  np.concatenate(buffers_vel, axis=0))
                    buffers_pos.clear(); buffers_vel.clear(); buffered = 0
            remaining -= available
    if remaining or buffered or block_index != expected_blocks:
        raise RuntimeError(
            f"Spectrum stream incomplete: remaining={remaining}, buffered={buffered}, "
            f"blocks={block_index}/{expected_blocks}"
        )

    export = frequency <= args.max_spectral_cm
    rows = []
    for i in np.flatnonzero(export):
        for layer in range(args.n_layers):
            members = groups[layer]
            if len(members) < 2:
                continue
            rows.append([
                float(frequency[i]), layer + 1,
                layer * args.layer_width, (layer + 1) * args.layer_width,
                len(members),
                *[float(spectral_sum[(layer, metric)][i] / expected_blocks)
                  for metric in metrics],
                float(sq_sum[(layer, 1)][i] / expected_blocks),
                float(sq_sum[(layer, 2)][i] / expected_blocks),
            ])
    write_csv(
        output / "collective_translation_and_lowq_spectra.csv",
        ["frequency_cm-1", "layer", "z_minus_GDS_min_A", "z_minus_GDS_max_A",
         "waters", "translation_self", "translation_coherent",
         "translation_signed_cross", "translation_participation_ratio",
         "S_q1", "S_q2"], rows,
    )
    write_csv(
        output / "collective_translation_block_band_metrics.csv",
        ["block", "layer", "z_minus_GDS_min_A", "z_minus_GDS_max_A", "waters",
         "translation_20_300cm-1_cross_to_self",
         "translation_20_300cm-1_self_weighted_participation"],
        block_band_rows,
    )
    summary_rows = []
    for layer in range(args.n_layers):
        members = groups[layer]
        if len(members) < 2:
            continue
        selected = [row for row in block_band_rows if row[1] == layer + 1]
        cross_values = np.asarray([row[5] for row in selected], dtype=float)
        pr_values = np.asarray([row[6] for row in selected], dtype=float)
        q_band = (frequency >= 5.0) & (frequency <= 300.0)
        sq = sq_sum[(layer, 1)] / expected_blocks
        peak = float(frequency[np.flatnonzero(q_band)[np.argmax(sq[q_band])]])
        summary_rows.append([
            layer + 1, layer * args.layer_width,
            (layer + 1) * args.layer_width, len(members),
            float(np.mean(cross_values)),
            float(np.std(cross_values, ddof=1) / np.sqrt(expected_blocks)),
            float(np.mean(pr_values)),
            float(np.std(pr_values, ddof=1) / np.sqrt(expected_blocks)), peak,
        ])
    write_csv(
        output / "collective_translation_layer_summary.csv",
        ["layer", "z_minus_GDS_min_A", "z_minus_GDS_max_A", "waters",
         "mean_translation_cross_to_self", "cross_to_self_SEM_50x10ps",
         "mean_translation_participation", "participation_SEM_50x10ps",
         "S_q1_peak_5_300cm-1"], summary_rows,
    )
    return {"spectral_blocks": expected_blocks,
            "spectral_block_ps": args.spectral_block_ps,
            "frequency_resolution_cm-1": float(frequency[1] - frequency[0])}


def dipole_pass(args, dipole_paths, oxygen, fixed_layer):
    mu_parts = []
    time_parts = []
    topology_ref = None
    sampling_fs = None
    model = None
    model_sha = None
    for path in dipole_paths:
        with h5py.File(path, "r") as handle:
            if not bool(handle.attrs.get("complete", False)):
                raise RuntimeError(f"Incomplete dipole file: {path}")
            topology = np.asarray(handle["water_topology_indices"], dtype=int)
            if topology_ref is None:
                topology_ref = topology
                sampling_fs = float(handle.attrs["sampling_interval_fs"])
                model = str(handle.attrs["model"])
                model_sha = str(handle.attrs["model_sha256"])
            elif not np.array_equal(topology, topology_ref):
                raise RuntimeError(f"Dipole topology changed in {path}")
            mu_parts.append(np.asarray(handle["water_dipoles_eA"], dtype=np.float64)
                            * DEBYE_PER_E_ANGSTROM)
            time_parts.append(np.asarray(handle["time_ps"], dtype=float))
    mu = np.concatenate(mu_parts, axis=0)
    times = np.concatenate(time_parts)
    if len(mu) != 5000:
        raise RuntimeError(f"Expected 5000 dipole frames, obtained {len(mu)}")
    if not np.array_equal(topology_ref[:, 0], oxygen):
        raise RuntimeError("Dipole and trajectory water ordering differ")
    dt_ps = sampling_fs / 1000.0
    max_lag = int(round(50.0 / dt_ps))
    summary_rows = []
    acf_rows = []
    layer_mz = np.full((len(mu), args.n_layers), np.nan, dtype=float)
    block_rows = []
    for layer in range(args.n_layers):
        members = np.flatnonzero(fixed_layer == layer)
        if not len(members):
            continue
        selected = mu[:, members]
        magnitude = np.linalg.norm(selected, axis=2)
        cos = selected[:, :, 2] / np.maximum(magnitude, 1.0e-30)
        mz = np.sum(selected[:, :, 2], axis=1)
        layer_mz[:, layer] = mz
        centered = selected[:, :, 2] - np.mean(selected[:, :, 2], axis=0)
        self_var = float(np.sum(np.mean(centered ** 2, axis=0)))
        coherent_var = float(np.var(mz))
        cross_ratio = (coherent_var - self_var) / max(self_var, 1.0e-30)
        summary_rows.append([
            layer + 1, layer * args.layer_width,
            (layer + 1) * args.layer_width, len(members),
            float(np.mean(magnitude)), float(np.std(magnitude)),
            float(np.mean(cos)), float(np.mean(0.5 * (3 * cos ** 2 - 1))),
            float(np.mean(mz)), float(np.std(mz)), self_var,
            coherent_var, cross_ratio,
        ])
        acf = autocorrelation(mz, max_lag)
        for lag, value in enumerate(acf):
            acf_rows.append([layer + 1, lag * dt_ps, float(value)])
        block_frames = int(round(args.block_ps / dt_ps))
        for block in range(len(mu) // block_frames):
            part = selected[block * block_frames:(block + 1) * block_frames, :, 2]
            centered_part = part - np.mean(part, axis=0)
            self_part = float(np.sum(np.mean(centered_part ** 2, axis=0)))
            coherent_part = float(np.var(np.sum(part, axis=1)))
            block_rows.append([
                block + 1, layer + 1,
                (coherent_part - self_part) / max(self_part, 1.0e-30),
                float(np.mean(np.sum(part, axis=1))),
            ])
    write_csv(
        args.output / "dipole_layer_summary_100fs.csv",
        ["layer", "z_minus_GDS_min_A", "z_minus_GDS_max_A", "waters",
         "mean_latent_molecular_dipole_D", "std_latent_molecular_dipole_D",
         "mean_P1_mu_z", "mean_P2_mu_z", "mean_layer_total_Mz_D",
         "std_layer_total_Mz_D", "sum_single_water_Mz_variance_D2",
         "layer_total_Mz_variance_D2", "Mz_cross_to_self_variance_ratio"],
        summary_rows,
    )
    write_csv(args.output / "dipole_layer_Mz_autocorrelation_100fs.csv",
              ["layer", "lag_ps", "normalized_Mz_autocorrelation"], acf_rows)
    write_csv(args.output / "dipole_layer_block_statistics_10ps.csv",
              ["block", "layer", "Mz_cross_to_self_variance_ratio",
               "mean_layer_total_Mz_D"], block_rows)
    valid_layers = [layer for layer in range(args.n_layers)
                    if np.all(np.isfinite(layer_mz[:, layer]))]
    corr = np.corrcoef(layer_mz[:, valid_layers], rowvar=False)
    corr_rows = []
    for i, left in enumerate(valid_layers):
        for j, right in enumerate(valid_layers):
            corr_rows.append([left + 1, right + 1, float(corr[i, j])])
    write_csv(args.output / "dipole_layer_cross_correlation_matrix.csv",
              ["layer_i", "layer_j", "Pearson_correlation_total_Mz"], corr_rows)
    return {
        "dipole_frames": len(mu), "dipole_sampling_fs": sampling_fs,
        "model": model, "model_sha256": model_sha,
        "interpretation_warning": (
            "Per-water dipoles are O+H+H sums of latent atomic contributions from "
            "a model trained on total-cell dipoles; absolute molecular partitions "
            "are not uniquely supervised."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--segment-glob", default="*segment_*.h5")
    parser.add_argument("--dipole-dir", required=True, type=Path)
    parser.add_argument("--candidate-screen", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--system", required=True)
    parser.add_argument("--gds", required=True, type=float)
    parser.add_argument("--source-frames", default=500000, type=int)
    parser.add_argument("--structure-stride", default=10, type=int)
    parser.add_argument("--layer-width", default=2.0, type=float)
    parser.add_argument("--n-layers", default=12, type=int)
    parser.add_argument("--block-ps", default=10.0, type=float)
    parser.add_argument("--spectral-block-ps", default=10.0, type=float)
    parser.add_argument("--max-spectral-cm", default=1200.0, type=float)
    parser.add_argument("--hbond-ho", default=2.45, type=float)
    parser.add_argument("--hbond-oo", default=3.50, type=float)
    parser.add_argument("--hbond-angle", default=150.0, type=float)
    args = parser.parse_args()
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.input.glob(args.segment_glob))
    if not paths:
        raise RuntimeError(f"No files match {args.segment_glob!r} in {args.input}")
    with h5py.File(paths[0], "r") as handle:
        symbols = np.asarray(handle["symbols"]).astype(str)
        cell = np.asarray(handle["cell_A"], dtype=float)
        positions0 = np.asarray(handle["positions_A"][0], dtype=float)
        source_dt_fs = float(handle.attrs["output_interval_fs"])
    waters = water_topology(symbols, positions0, cell)
    oxygen = waters[:, 0]
    candidate_rows = load_candidate_rows(args.candidate_screen)
    mean_relative_z = np.asarray([
        float(candidate_rows[int(o) + 1]["mean_z_minus_GDS_A"]) for o in oxygen
    ])
    fixed_layer = np.floor(mean_relative_z / args.layer_width).astype(int)
    fixed_layer[(fixed_layer < 0) | (fixed_layer >= args.n_layers)] = -1
    dipole_paths = sorted(args.dipole_dir.glob("*.dipole.h5"))
    if not dipole_paths:
        raise RuntimeError(f"No dipole HDF5 files in {args.dipole_dir}")

    structure = structure_network_pass(
        args, paths, symbols, cell, waters, fixed_layer,
        candidate_rows, source_dt_fs,
    )
    spectra = spectrum_pass(args, paths, cell, waters, fixed_layer, source_dt_fs)
    dipole = dipole_pass(args, dipole_paths, oxygen, fixed_layer)
    metadata = {
        "status": "complete", "system": args.system,
        "trajectory_source": str(args.input), "trajectory_files": len(paths),
        "source_frames": args.source_frames,
        "source_sampling_interval_fs": source_dt_fs,
        "water_count": len(waters), "GDS_A": args.gds,
        "common_z_minus_GDS_interval_A": [0.0,
                                          args.layer_width * args.n_layers],
        "layer_width_A": args.layer_width,
        "layer_assignment_for_dynamics": "fixed by each water's 500 ps mean O z-GDS",
        "layer_assignment_for_z_profile": "instantaneous O z-GDS",
        "hbond_definition": {
            "H_to_acceptor_O_max_A": args.hbond_ho,
            "donor_O_to_acceptor_O_max_A": args.hbond_oo,
            "O_H_O_min_degree": args.hbond_angle,
        },
        "structure_network": structure, "spectra": spectra,
        "dipole": dipole,
        "method_limits": [
            "COM correction removes rigid water-layer translation; it does not prove a mode is collective.",
            "Signed cross spectra and displacement covariance are trajectory correlations, not Hessian eigenmodes.",
            "Fundamental cycle-basis counts are graph descriptors, not unique primitive-ring statistics.",
            "Selected MACE per-water dipole partitions are latent and non-unique; layer correlations are diagnostic.",
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
