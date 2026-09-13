#!/usr/bin/env python3
"""SiO2 extension of the best-contrast and four-relative-z-water workflows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


CM_PER_PS = 33.3564095198152
MASS_O = 15.9994
MASS_H = 1.00794
WATER_MASS_G_CM3_PER_A3 = (MASS_O + 2 * MASS_H) / 0.602214076
TARGET_CAF2_RELATIVE_DEPTH_A = np.asarray([
    5.035273105273362, 10.54223173118126,
    14.580528025279158, 19.93898509784233,
])
CAF2_OIL_COMMON_THICKNESS_A = 23.194765803637615


def minimum_image(delta: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Apply the triclinic minimum-image convention to Cartesian vectors."""
    frac = delta @ np.linalg.inv(cell)
    frac -= np.rint(frac)
    return frac @ cell


def directed_hbond(o_xyz: np.ndarray, h_xyz: np.ndarray, cell: np.ndarray,
                   ho_cutoff: float, oo_cutoff: float,
                   angle_cutoff: float) -> np.ndarray:
    """Return donor-water by acceptor-water H-bond states."""
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
        * np.linalg.norm(h_to_acceptor, axis=-1), 1.0e-30
    )
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    not_self = ~np.eye(o_xyz.shape[1], dtype=bool)[None, :, None, :]
    state = ((ho_distance <= ho_cutoff)
             & (oo_distance[:, :, None, :] <= oo_cutoff)
             & (angle >= angle_cutoff) & not_self)
    return np.any(state, axis=2)


def write_csv(path: Path, header, rows):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def file_sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tetrahedral_q_chunk(o_xyz, cell):
    delta = minimum_image(o_xyz[:, None, :, :] - o_xyz[:, :, None, :], cell)
    distance = np.linalg.norm(delta, axis=-1)
    diagonal = np.arange(o_xyz.shape[1])
    distance[:, diagonal, diagonal] = np.inf
    nearest = np.argpartition(distance, 4, axis=2)[:, :, :4]
    vectors = np.take_along_axis(delta, nearest[:, :, :, None], axis=2)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=3, keepdims=True), 1e-12)
    terms = np.zeros(o_xyz.shape[:2], dtype=float)
    for j in range(3):
        for k in range(j + 1, 4):
            cosine = np.sum(vectors[:, :, j] * vectors[:, :, k], axis=2)
            terms += (cosine + 1.0 / 3.0) ** 2
    return 1.0 - 3.0 * terms / 8.0


def candidate_table(waters_id, mean_z, z_std, z0, thickness, partner_counts,
                    degree, donated, accepted):
    samples, nwater = degree.shape
    rows = []
    details = []
    for water in range(nwater):
        order = np.argsort(partner_counts[water])[::-1]
        top4 = order[:4]
        occupancy = partner_counts[water, top4] / samples
        total = int(np.sum(partner_counts[water]))
        identity = float(np.sum(partner_counts[water, top4]) / max(total, 1))
        fourth = float(np.min(occupancy))
        degree4 = float(np.mean(degree[:, water] == 4))
        score = fourth * identity * degree4
        relative_A = float(mean_z[water] - z0)
        row = {
            "water_index_1based": water + 1,
            "O_serial_1based": int(waters_id[water, 0]),
            "H1_serial_1based": int(waters_id[water, 1]),
            "H2_serial_1based": int(waters_id[water, 2]),
            "mean_O_z_A": float(mean_z[water]),
            "std_O_z_A": float(z_std[water]),
            "mean_z_relative_water_film_A": relative_A,
            "mean_relative_depth_fraction": relative_A / thickness,
            "mean_HB_degree": float(np.mean(degree[:, water])),
            "fraction_frames_degree_eq4": degree4,
            "fraction_frames_degree_ge4": float(np.mean(degree[:, water] >= 4)),
            "mean_donated_HB": float(np.mean(donated[:, water])),
            "mean_accepted_HB": float(np.mean(accepted[:, water])),
            "unique_HB_partners": int(np.sum(partner_counts[water] > 0)),
            "top4_partner_identity_fraction": identity,
            "fourth_partner_occupancy": fourth,
            "top4_mean_partner_occupancy": float(np.mean(occupancy)),
            "four_partner_stability_score": score,
            "top4_partner_O_serials_1based": ";".join(
                str(int(waters_id[index, 0])) for index in top4
            ),
            "top4_partner_occupancies": ";".join(f"{value:.8f}" for value in occupancy),
        }
        rows.append(row)
        details.append((water, row, top4))
    return rows, details


def choose_four(details, thickness):
    target_fraction = TARGET_CAF2_RELATIVE_DEPTH_A / CAF2_OIL_COMMON_THICKNESS_A
    selected = []
    used = set()
    for layer, target in enumerate(target_fraction, start=1):
        quarter_min = (layer - 1) / 4.0
        quarter_max = layer / 4.0
        candidates = [item for item in details if item[0] not in used
                      and quarter_min <= item[1]["mean_relative_depth_fraction"] < quarter_max]
        close = [item for item in candidates if abs(
            item[1]["mean_relative_depth_fraction"] - target
        ) <= 0.05]
        pool = close if close else candidates
        if not pool:
            raise RuntimeError(f"No SiO2 candidate in relative quarter {layer}")
        chosen = max(pool, key=lambda item: (
            item[1]["four_partner_stability_score"],
            -abs(item[1]["mean_relative_depth_fraction"] - target),
        ))
        used.add(chosen[0])
        selected.append((layer, float(target), *chosen))
    return selected


def unwrapped_oxygen(o_xyz, cell):
    unwrapped = np.empty_like(o_xyz, dtype=float)
    unwrapped[0] = o_xyz[0]
    for frame in range(1, len(o_xyz)):
        unwrapped[frame] = unwrapped[frame - 1] + minimum_image(
            o_xyz[frame] - o_xyz[frame - 1], cell
        )
    return unwrapped


def partner_summary(adjacency, selected, waters_id, dt_ps):
    state = adjacency[:, selected]
    rows = []
    for partner in np.argsort(np.sum(state, axis=0))[::-1]:
        connected = state[:, partner]
        count = int(np.sum(connected))
        if count == 0:
            continue
        changes = np.diff(np.r_[False, connected, False].astype(np.int8))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        lengths = ends - starts
        rows.append([
            int(waters_id[partner, 0]), count / len(state), len(lengths),
            float(np.mean(lengths) * dt_ps), float(np.max(lengths) * dt_ps),
        ])
    return rows


def structure_mode(args):
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.position_cache) as cache:
        position = np.asarray(cache["position_A"][:5000], dtype=np.float32)
        steps = np.asarray(cache["step"][:5000], dtype=np.int64)
        cell = np.asarray(cache["cell_A"], dtype=float)
        waters_id = np.asarray(cache["waters_id"], dtype=np.int64)
        natoms = int(cache["natoms"])
    nframes, nwater = position.shape[:2]
    o_xyz = position[:, :, 0]
    h_xyz = position[:, :, 1:]
    mean_z = np.mean(o_xyz[:, :, 2], axis=0)
    z_std = np.std(o_xyz[:, :, 2], axis=0)
    z0 = args.z0
    thickness = args.n_layers * args.layer_width
    fixed_layer = np.floor((mean_z - z0) / args.layer_width).astype(int)
    if np.any((fixed_layer < 0) | (fixed_layer >= args.n_layers)):
        raise RuntimeError("Mean-z water outside the configured film interval")

    adjacency_store = np.empty((nframes, nwater, nwater), dtype=bool)
    degree_store = np.empty((nframes, nwater), dtype=np.uint8)
    donated_store = np.empty((nframes, nwater), dtype=np.uint8)
    accepted_store = np.empty((nframes, nwater), dtype=np.uint8)
    q_store = np.empty((nframes, nwater), dtype=np.float32)
    bisector_store = np.empty((nframes, nwater, 3), dtype=np.float32)
    nblocks = 50
    samples_per_block = nframes // nblocks
    shape = (nblocks, args.n_layers)
    count = np.zeros(shape, dtype=np.int64)
    sums = {name: np.zeros(shape, dtype=float) for name in
            ("q", "degree", "degree4", "p1", "p2")}

    for start in range(0, nframes, 10):
        stop = min(start + 10, nframes)
        chunk = position[start:stop]
        oxygen = chunk[:, :, 0]
        hydrogens = chunk[:, :, 1:]
        directed = directed_hbond(
            oxygen, hydrogens, cell, args.hbond_ho, args.hbond_oo,
            args.hbond_angle,
        )
        adjacency = directed | np.swapaxes(directed, 1, 2)
        degree = np.sum(adjacency, axis=2)
        donated = np.sum(directed, axis=2)
        accepted = np.sum(directed, axis=1)
        q = tetrahedral_q_chunk(oxygen, cell)
        r1 = minimum_image(hydrogens[:, :, 0] - oxygen, cell)
        r2 = minimum_image(hydrogens[:, :, 1] - oxygen, cell)
        bisector = r1 + r2
        bisector /= np.maximum(np.linalg.norm(bisector, axis=2, keepdims=True), 1e-12)
        p1 = bisector[:, :, 2]
        p2 = 0.5 * (3.0 * p1 ** 2 - 1.0)
        adjacency_store[start:stop] = adjacency
        degree_store[start:stop] = degree
        donated_store[start:stop] = donated
        accepted_store[start:stop] = accepted
        q_store[start:stop] = q
        bisector_store[start:stop] = bisector
        layer = np.floor((oxygen[:, :, 2] - z0) / args.layer_width).astype(int)
        for local in range(stop - start):
            global_frame = start + local
            block = global_frame // samples_per_block
            valid = (layer[local] >= 0) & (layer[local] < args.n_layers)
            bins = layer[local, valid]
            count[block] += np.bincount(bins, minlength=args.n_layers)
            for name, values in (
                ("q", q[local]), ("degree", degree[local]),
                ("degree4", degree[local] == 4), ("p1", p1[local]),
                ("p2", p2[local]),
            ):
                sums[name][block] += np.bincount(
                    bins, weights=np.asarray(values)[valid], minlength=args.n_layers
                )
        if start == 0 or stop % 500 == 0:
            print(json.dumps({"structure_frames_done": stop,
                              "structure_frames_total": nframes}), flush=True)

    partner_counts = np.sum(adjacency_store, axis=0, dtype=np.int64)
    candidate_rows, details = candidate_table(
        waters_id, mean_z, z_std, z0, thickness, partner_counts,
        degree_store, donated_store, accepted_store,
    )
    candidate_path = args.output / "all_water_hbond_candidate_screen_500ps_stride100fs.csv"
    write_csv(candidate_path, list(candidate_rows[0]),
              [[row[key] for key in row] for row in candidate_rows])
    selected = choose_four(details, thickness)

    safe = np.maximum(count, 1)
    metric = {name: values / safe for name, values in sums.items()}
    mean_count = count / samples_per_block
    volume = cell[0, 0] * cell[1, 1] * args.layer_width
    density = mean_count / volume
    block_rows = []
    for block in range(nblocks):
        for layer in range(args.n_layers):
            block_rows.append([
                block + 1, block * 10.0, (block + 1) * 10.0,
                layer + 1, layer * args.layer_width,
                (layer + 1) * args.layer_width, int(count[block, layer]),
                mean_count[block, layer], density[block, layer],
                density[block, layer] * WATER_MASS_G_CM3_PER_A3,
                metric["q"][block, layer], metric["degree"][block, layer],
                metric["degree4"][block, layer], metric["p1"][block, layer],
                metric["p2"][block, layer],
            ])
    write_csv(
        args.output / "z_profile_2A_block_statistics.csv",
        ["block", "time_min_ps", "time_max_ps", "layer",
         "z_minus_GDS_min_A", "z_minus_GDS_max_A", "water_frame_observations",
         "mean_instantaneous_water_count", "water_number_density_A-3",
         "water_mass_density_g_cm-3", "mean_tetrahedral_q", "mean_HB_degree",
         "fraction_degree_eq4", "mean_P1_cos_theta", "mean_P2"], block_rows,
    )
    summary_rows = []
    for layer in range(args.n_layers):
        members = np.flatnonzero(fixed_layer == layer)
        candidate_values = [candidate_rows[index] for index in members]
        values = []
        for name in ("q", "degree", "degree4", "p1", "p2"):
            x = metric[name][:, layer]
            values.extend([float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(nblocks))])
        summary_rows.append([
            layer + 1, layer * args.layer_width, (layer + 1) * args.layer_width,
            float(np.mean(mean_count[:, layer])), float(np.mean(density[:, layer])),
            float(np.mean(density[:, layer]) * WATER_MASS_G_CM3_PER_A3),
            *values, len(members),
            float(np.mean([row["four_partner_stability_score"] for row in candidate_values])),
            float(np.mean([row["top4_partner_identity_fraction"] for row in candidate_values])),
        ])
    write_csv(
        args.output / "z_profile_2A_summary.csv",
        ["layer", "z_minus_GDS_min_A", "z_minus_GDS_max_A",
         "mean_instantaneous_water_count", "water_number_density_A-3",
         "water_mass_density_g_cm-3", "mean_tetrahedral_q",
         "mean_tetrahedral_q_SEM_50x10ps", "mean_HB_degree",
         "mean_HB_degree_SEM_50x10ps", "fraction_degree_eq4",
         "fraction_degree_eq4_SEM_50x10ps", "mean_P1_cos_theta",
         "mean_P1_cos_theta_SEM_50x10ps", "mean_P2",
         "mean_P2_SEM_50x10ps", "fixed_mean_z_water_count",
         "mean_four_partner_stability_score",
         "mean_top4_partner_identity_fraction"], summary_rows,
    )

    unwrapped = unwrapped_oxygen(o_xyz, cell)
    water_com = np.mean(unwrapped, axis=1)
    time_ps = steps * 0.0005
    manifest_rows = []
    combined_rows = []
    combined_header = None
    for layer, target_fraction, selected_index, row, top4 in selected:
        serial = int(row["O_serial_1based"])
        directory = args.output / "four_relative_z_waters" / f"layer{layer}_O{serial}"
        directory.mkdir(parents=True, exist_ok=True)
        selected_unwrapped = unwrapped[:, selected_index]
        rel_com = selected_unwrapped - water_com
        first = selected_unwrapped[0]
        first_rel = rel_com[0]
        adjacency_selected = adjacency_store[:, selected_index]
        top4_set = set(map(int, top4))
        rows = []
        for frame in range(nframes):
            connected = np.flatnonzero(adjacency_selected[frame])
            top4_count = sum(int(index) in top4_set for index in connected)
            vector = bisector_store[frame, selected_index]
            angle = float(np.degrees(np.arccos(np.clip(vector[2], -1, 1))))
            rows.append([
                int(steps[frame]), float(time_ps[frame]), float(time_ps[frame] - time_ps[0]),
                *map(float, o_xyz[frame, selected_index]),
                float(o_xyz[frame, selected_index, 2] - z0),
                float((o_xyz[frame, selected_index, 2] - z0) / thickness),
                *map(float, selected_unwrapped[frame]),
                float(selected_unwrapped[frame, 0] - first[0]),
                float(selected_unwrapped[frame, 1] - first[1]),
                *map(float, water_com[frame]),
                float(water_com[frame, 0] - water_com[0, 0]),
                float(water_com[frame, 1] - water_com[0, 1]),
                float(rel_com[frame, 0] - first_rel[0]),
                float(rel_com[frame, 1] - first_rel[1]),
                *map(float, vector), angle,
                int(donated_store[frame, selected_index]),
                int(accepted_store[frame, selected_index]),
                int(degree_store[frame, selected_index]),
                int(degree_store[frame, selected_index] == 4),
                top4_count, int(degree_store[frame, selected_index]) - top4_count,
                ";".join(str(int(waters_id[index, 0])) for index in connected),
            ])
        header = [
            "step", "source_time_ps", "elapsed_time_ps", "O_x_wrapped_A",
            "O_y_wrapped_A", "O_z_wrapped_A", "O_z_relative_water_film_A",
            "O_relative_depth_fraction", "O_x_unwrapped_A", "O_y_unwrapped_A",
            "O_z_unwrapped_A", "O_dx_unwrapped_from_start_A",
            "O_dy_unwrapped_from_start_A", "water_O_COM_x_unwrapped_A",
            "water_O_COM_y_unwrapped_A", "water_O_COM_z_unwrapped_A",
            "water_O_COM_dx_from_start_A", "water_O_COM_dy_from_start_A",
            "O_dx_relative_water_COM_from_start_A",
            "O_dy_relative_water_COM_from_start_A", "structural_dipole_unit_x",
            "structural_dipole_unit_y", "structural_dipole_unit_z",
            "structural_dipole_angle_to_plus_z_degree", "donated_HB", "accepted_HB",
            "unique_HB_degree", "degree_equals_4", "connected_top4_partner_count",
            "connected_other_partner_count", "connected_partner_O_serials_1based",
        ]
        time_series_path = directory / "selected_water_timeseries_500ps_stride100fs.csv"
        write_csv(time_series_path, header, rows)
        write_csv(
            directory / "selected_water_partner_hbond_summary_500ps_stride100fs.csv",
            ["partner_O_serial_1based", "occupancy_fraction", "continuous_episodes",
             "mean_continuous_lifetime_ps", "max_continuous_lifetime_ps"],
            partner_summary(adjacency_store, selected_index, waters_id, 0.1),
        )
        metadata = {
            "status": "complete", "system": "SiO2-fixed/water",
            "layer_index_relative_water_film": layer,
            "target_relative_depth_fraction_from_CaF2_selection": target_fraction,
            "selected_water": row,
            "position_sampling_interval_fs": 100.0,
            "water_film_reference_z0_A": z0,
            "water_film_thickness_A": thickness,
            "water_film_reference_note": "native nonempty 2 A interval; not an independently fitted GDS",
            "model_dipole": {"status": "unavailable", "reason": "No validated SiO2 per-water dipole model"},
            "time_series_sha256": file_sha256(time_series_path),
        }
        (directory / "selection_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        mismatch = abs(row["mean_relative_depth_fraction"] - target_fraction)
        manifest_rows.append([
            layer, serial, target_fraction, row["mean_relative_depth_fraction"],
            mismatch, row["mean_z_relative_water_film_A"],
            row["four_partner_stability_score"], row["fraction_frames_degree_eq4"],
            row["fourth_partner_occupancy"], len(rows), str(directory),
        ])
        if combined_header is None:
            combined_header = ["system", "layer_index_relative_water_film",
                               "selected_O_serial_1based", "mean_relative_depth_fraction", *header]
        combined_rows.extend([
            ["SiO2-fixed/water", layer, serial, row["mean_relative_depth_fraction"], *values]
            for values in rows
        ])
    write_csv(
        args.output / "four_relative_z_waters" / "selected_four_water_manifest.csv",
        ["layer_index_relative_water_film", "O_serial_1based",
         "target_relative_depth_fraction", "selected_mean_relative_depth_fraction",
         "absolute_relative_depth_mismatch", "mean_z_relative_water_film_A",
         "four_partner_stability_score", "fraction_frames_degree_eq4",
         "fourth_partner_occupancy", "samples_100fs", "output_directory"],
        manifest_rows,
    )
    write_csv(
        args.output / "four_relative_z_waters" / "combined_four_water_timeseries_500ps_100fs.csv",
        combined_header, combined_rows,
    )
    mapping = args.output / "water_atom_to_water_COM_mapping_Afs.txt"
    total_mass = MASS_O + 2 * MASS_H
    with mapping.open("w") as stream:
        stream.write(f"{natoms} {nwater} 500000\n")
        for water, ids in enumerate(waters_id):
            for atom_id, mass in zip(ids, (MASS_O, MASS_H, MASS_H)):
                stream.write(f"{int(atom_id)} {water} {mass / total_mass * 1e-3:.17g}\n")
    metadata = {
        "status": "structure_complete", "system": "SiO2-fixed/water",
        "position_frames": nframes, "position_sampling_fs": 100.0,
        "water_count": nwater, "water_film_reference_z0_A": z0,
        "water_film_thickness_A": thickness, "native_2A_layers": args.n_layers,
        "hbond_definition": {"H_to_O_max_A": args.hbond_ho,
                             "O_to_O_max_A": args.hbond_oo,
                             "O_H_O_min_degree": args.hbond_angle},
        "selected_O_serials_1based": [int(item[3]["O_serial_1based"]) for item in selected],
        "model_dipole": "not evaluated; no validated SiO2 molecular dipole model",
    }
    (args.output / "structure_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output / "STRUCTURE_COMPLETE").write_text("complete\n")
    print(json.dumps(metadata, indent=2), flush=True)


def spectral_metrics(signal, members, window):
    selected = np.asarray(signal[:, members], dtype=float)
    selected -= np.mean(selected, axis=0, keepdims=True)
    transformed = np.fft.rfft(selected * window[:, None, None], axis=0)
    power = np.sum(np.abs(transformed) ** 2, axis=2)
    norm = float(np.sum(window ** 2))
    self_s = np.sum(power, axis=1) / (len(members) * norm)
    coherent = np.sum(np.abs(np.sum(transformed, axis=1)) ** 2, axis=1)
    coherent /= len(members) * norm
    cross = coherent - self_s
    participation = np.sum(power, axis=1) ** 2 / np.maximum(
        len(members) * np.sum(power ** 2, axis=1), 1e-30
    )
    return self_s, coherent, cross, participation


def spectrum_mode(args):
    with np.load(args.position_cache) as cache:
        mean_z = np.mean(np.asarray(cache["position_A"][:5000, :, 0, 2]), axis=0)
    fixed_layer = np.floor((mean_z - args.z0) / args.layer_width).astype(int)
    nwater = len(mean_z)
    expected_bytes = 500000 * nwater * 3 * 4
    if args.water_com_velocity.stat().st_size != expected_bytes:
        raise RuntimeError(f"COM velocity binary size mismatch: {args.water_com_velocity.stat().st_size}/{expected_bytes}")
    velocity = np.memmap(args.water_com_velocity, dtype=np.float32, mode="r",
                         shape=(500000, nwater, 3))
    block_frames = 10000
    window = np.blackman(block_frames)
    frequency = np.fft.rfftfreq(block_frames, d=0.001) * CM_PER_PS
    band = (frequency >= 20.0) & (frequency <= 300.0)
    groups = [np.flatnonzero(fixed_layer == layer) for layer in range(args.n_layers)]
    accum = {(layer, metric): np.zeros(len(frequency), dtype=float)
             for layer in range(args.n_layers)
             for metric in ("self", "coherent", "cross", "participation")}
    block_rows = []
    for block in range(50):
        signal = velocity[block * block_frames:(block + 1) * block_frames]
        for layer, members in enumerate(groups):
            if len(members) < 2:
                continue
            values = spectral_metrics(signal, members, window)
            for metric, value in zip(("self", "coherent", "cross", "participation"), values):
                accum[(layer, metric)] += value
            self_s, _coherent, cross, participation = values
            self_int = float(np.trapezoid(self_s[band], frequency[band]))
            cross_int = float(np.trapezoid(cross[band], frequency[band]))
            weighted_pr = float(np.sum(participation[band] * self_s[band])
                                / max(np.sum(self_s[band]), 1e-30))
            block_rows.append([
                block + 1, layer + 1, layer * args.layer_width,
                (layer + 1) * args.layer_width, len(members),
                cross_int / max(self_int, 1e-30), weighted_pr,
            ])
        print(json.dumps({"spectral_blocks_done": block + 1,
                          "spectral_blocks_total": 50}), flush=True)
    write_csv(
        args.output / "collective_translation_block_band_metrics.csv",
        ["block", "layer", "z_minus_GDS_min_A", "z_minus_GDS_max_A", "waters",
         "translation_20_300cm-1_cross_to_self",
         "translation_20_300cm-1_self_weighted_participation"], block_rows,
    )
    summary_rows = []
    for layer, members in enumerate(groups):
        selected = [row for row in block_rows if row[1] == layer + 1]
        cross = np.asarray([row[5] for row in selected])
        pr = np.asarray([row[6] for row in selected])
        summary_rows.append([
            layer + 1, layer * args.layer_width, (layer + 1) * args.layer_width,
            len(members), float(np.mean(cross)), float(np.std(cross, ddof=1) / np.sqrt(50)),
            float(np.mean(pr)), float(np.std(pr, ddof=1) / np.sqrt(50)), math.nan,
        ])
    write_csv(
        args.output / "collective_translation_layer_summary.csv",
        ["layer", "z_minus_GDS_min_A", "z_minus_GDS_max_A", "waters",
         "mean_translation_cross_to_self", "cross_to_self_SEM_50x10ps",
         "mean_translation_participation", "participation_SEM_50x10ps",
         "S_q1_peak_5_300cm-1"], summary_rows,
    )
    spectrum_rows = []
    export = frequency <= 300.0
    for index in np.flatnonzero(export):
        for layer, members in enumerate(groups):
            if len(members) < 2:
                continue
            spectrum_rows.append([
                float(frequency[index]), layer + 1, len(members),
                *[float(accum[(layer, metric)][index] / 50)
                  for metric in ("self", "coherent", "cross", "participation")],
            ])
    write_csv(
        args.output / "collective_translation_spectra.csv",
        ["frequency_cm-1", "layer", "waters", "translation_self",
         "translation_coherent", "translation_signed_cross",
         "translation_participation_ratio"], spectrum_rows,
    )
    metadata = {
        "status": "complete", "source_frames": 500000,
        "source_sampling_fs": 1.0, "spectral_blocks": 50,
        "spectral_block_ps": 10.0, "velocity_unit": "A/fs",
        "frequency_resolution_cm-1": float(frequency[1] - frequency[0]),
    }
    (args.output / "spectrum_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output / "COMPLETE").write_text("complete\n")
    print(json.dumps(metadata, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("structure", "spectrum"), required=True)
    parser.add_argument("--position-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--water-com-velocity", type=Path)
    parser.add_argument("--z0", default=18.0, type=float)
    parser.add_argument("--layer-width", default=2.0, type=float)
    parser.add_argument("--n-layers", default=16, type=int)
    parser.add_argument("--hbond-ho", default=2.45, type=float)
    parser.add_argument("--hbond-oo", default=3.50, type=float)
    parser.add_argument("--hbond-angle", default=150.0, type=float)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "structure":
        structure_mode(args)
    else:
        if args.water_com_velocity is None:
            raise RuntimeError("--water-com-velocity is required in spectrum mode")
        spectrum_mode(args)


if __name__ == "__main__":
    main()
