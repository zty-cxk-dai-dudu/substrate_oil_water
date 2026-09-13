#!/usr/bin/env python3
"""Select one water by 500-ps H-bond stability and export 10-fs observables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np


DEBYE_PER_E_ANGSTROM = 4.803204712570263


def minimum_image(delta: np.ndarray, cell: np.ndarray) -> np.ndarray:
    frac = delta @ np.linalg.inv(cell)
    frac -= np.rint(frac)
    return frac @ cell


def water_topology(symbols: np.ndarray, positions: np.ndarray,
                   cell: np.ndarray, cutoff: float = 1.30) -> np.ndarray:
    oxygen = np.flatnonzero(symbols == "O")
    hydrogen = np.flatnonzero(symbols == "H")
    candidates: list[tuple[float, int, int]] = []
    for oxygen_index in oxygen:
        distance = np.linalg.norm(
            minimum_image(positions[hydrogen] - positions[oxygen_index], cell), axis=1
        )
        candidates.extend(
            (float(value), int(oxygen_index), int(hydrogen_index))
            for hydrogen_index, value in zip(hydrogen, distance) if value <= cutoff
        )
    candidates.sort()
    assigned = {int(oxygen_index): [] for oxygen_index in oxygen}
    used_hydrogen: set[int] = set()
    for _distance, oxygen_index, hydrogen_index in candidates:
        if len(assigned[oxygen_index]) < 2 and hydrogen_index not in used_hydrogen:
            assigned[oxygen_index].append(hydrogen_index)
            used_hydrogen.add(hydrogen_index)
    waters = [(int(oxygen_index), *assigned[int(oxygen_index)])
              for oxygen_index in oxygen if len(assigned[int(oxygen_index)]) == 2]
    if len(waters) != len(oxygen):
        raise RuntimeError(f"Only {len(waters)}/{len(oxygen)} waters were assigned")
    return np.asarray(waters, dtype=np.int64)


def directed_hbond_state(o_xyz: np.ndarray, h_xyz: np.ndarray, cell: np.ndarray,
                         ho_cutoff: float, oo_cutoff: float,
                         angle_cutoff: float) -> np.ndarray:
    """Return (frame, donor water, acceptor water) directed H-bond state."""
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
    return np.any(state, axis=2)


def selected_water_hbond_state(
    o_xyz: np.ndarray, h_xyz: np.ndarray, selected: int, cell: np.ndarray,
    ho_cutoff: float, oo_cutoff: float, angle_cutoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return selected-water donation, acceptance and undirected partners."""
    selected_o = o_xyz[:, selected]
    selected_h = h_xyz[:, selected]

    # Selected water donates through either of its two H atoms to every acceptor O.
    h_to_acceptor = minimum_image(
        o_xyz[:, None, :, :] - selected_h[:, :, None, :], cell
    )
    ho_distance = np.linalg.norm(h_to_acceptor, axis=-1)
    donor_to_acceptor = minimum_image(o_xyz - selected_o[:, None, :], cell)
    oo_distance = np.linalg.norm(donor_to_acceptor, axis=-1)
    h_to_donor = minimum_image(
        selected_o[:, None, :] - selected_h, cell
    )[:, :, None, :]
    cosine = np.sum(h_to_donor * h_to_acceptor, axis=-1) / np.maximum(
        np.linalg.norm(h_to_donor, axis=-1)
        * np.linalg.norm(h_to_acceptor, axis=-1), 1.0e-12
    )
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    donated_to = np.any(
        (ho_distance <= ho_cutoff)
        & (oo_distance[:, None, :] <= oo_cutoff)
        & (angle >= angle_cutoff), axis=1,
    )

    # Every other water donates through either H atom to the selected acceptor O.
    h_to_selected = minimum_image(
        selected_o[:, None, None, :] - h_xyz, cell
    )
    hs_distance = np.linalg.norm(h_to_selected, axis=-1)
    donor_to_selected = minimum_image(
        selected_o[:, None, :] - o_xyz, cell
    )
    ds_distance = np.linalg.norm(donor_to_selected, axis=-1)
    h_to_donor_all = minimum_image(o_xyz[:, :, None, :] - h_xyz, cell)
    cosine = np.sum(h_to_donor_all * h_to_selected, axis=-1) / np.maximum(
        np.linalg.norm(h_to_donor_all, axis=-1)
        * np.linalg.norm(h_to_selected, axis=-1), 1.0e-12
    )
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    accepted_from = np.any(
        (hs_distance <= ho_cutoff)
        & (ds_distance[:, :, None] <= oo_cutoff)
        & (angle >= angle_cutoff), axis=2,
    )
    donated_to[:, selected] = False
    accepted_from[:, selected] = False
    return donated_to, accepted_from, donated_to | accepted_from


def sampled_segment_parts(paths: list[Path], total_source_frames: int,
                          stride: int, datasets: tuple[str, ...]):
    global_start = 0
    remaining = total_source_frames
    for path in paths:
        with h5py.File(path, "r") as handle:
            if not bool(handle.attrs.get("complete", False)):
                raise RuntimeError(f"Incomplete source file: {path}")
            available = min(len(handle["step"]), remaining)
            if available <= 0:
                break
            offset = (-global_start) % stride
            local_indices = np.arange(offset, available, stride, dtype=np.int64)
            part = {name: np.asarray(handle[name][local_indices]) for name in datasets}
            yield path, global_start, local_indices, part
            global_start += available
            remaining -= available
    if remaining:
        raise RuntimeError(
            f"Trajectory has {total_source_frames - remaining} usable frames; "
            f"{total_source_frames} requested"
        )


def write_csv(path: Path, header: list[str], rows) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model_dipole(dipole_dir: Path, selected_oxygen: int,
                       output: Path) -> dict[str, object]:
    paths = sorted(dipole_dir.glob("*.dipole.h5"))
    if not paths:
        raise RuntimeError(f"No .dipole.h5 files in {dipole_dir}")
    rows = []
    topology_reference = None
    sampling_interval_fs = None
    for path in paths:
        with h5py.File(path, "r") as handle:
            if not bool(handle.attrs.get("complete", False)):
                raise RuntimeError(f"Incomplete dipole file: {path}")
            topology = np.asarray(handle["water_topology_indices"], dtype=int)
            if topology_reference is None:
                topology_reference = topology
                matches = np.flatnonzero(topology[:, 0] == selected_oxygen)
                if len(matches) != 1:
                    raise RuntimeError(
                        f"Selected O index {selected_oxygen} occurs {len(matches)} times "
                        "in dipole topology"
                    )
                selected_water = int(matches[0])
                sampling_interval_fs = float(handle.attrs["sampling_interval_fs"])
                definition = str(handle.attrs["water_dipole_definition"])
                model = str(handle.attrs["model"])
                model_sha256 = str(handle.attrs["model_sha256"])
            elif not np.array_equal(topology, topology_reference):
                raise RuntimeError(f"Dipole topology changed in {path}")
            dipole = np.asarray(handle["water_dipoles_eA"][:, selected_water], dtype=float)
            magnitude_ea = np.linalg.norm(dipole, axis=1)
            angle = np.degrees(np.arccos(np.clip(
                dipole[:, 2] / np.maximum(magnitude_ea, 1.0e-15), -1.0, 1.0
            )))
            times = np.asarray(handle["time_ps"], dtype=float)
            steps = np.asarray(handle["step"], dtype=int)
            for i in range(len(times)):
                rows.append([
                    int(steps[i]), float(times[i]),
                    float(dipole[i, 0]), float(dipole[i, 1]), float(dipole[i, 2]),
                    float(magnitude_ea[i]),
                    float(magnitude_ea[i] * DEBYE_PER_E_ANGSTROM), float(angle[i]),
                ])
    write_csv(
        output / "selected_water_model_dipole_100fs.csv",
        ["step", "source_time_ps", "latent_mu_x_eA", "latent_mu_y_eA",
         "latent_mu_z_eA", "latent_mu_magnitude_eA", "latent_mu_magnitude_D",
         "latent_mu_angle_to_plus_z_degree"], rows,
    )
    return {
        "status": "available",
        "files": len(paths),
        "frames": len(rows),
        "sampling_interval_fs": sampling_interval_fs,
        "water_dipole_definition": definition,
        "model": model,
        "model_sha256": model_sha256,
        "interpretation_warning": (
            "The per-water dipole is an O+H+H sum of latent atomic contributions "
            "from a model supervised on total-cell dipoles; it is not a uniquely "
            "supervised molecular dipole."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--glob", default="*segment_*.h5")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--system", required=True)
    parser.add_argument("--gds", required=True, type=float)
    parser.add_argument("--source-frames", default=500000, type=int)
    parser.add_argument("--stride", default=10, type=int)
    parser.add_argument("--bulk-min", default=8.0, type=float)
    parser.add_argument("--bulk-max", default=16.0, type=float)
    parser.add_argument("--selection", choices=("stable", "unstable"), required=True)
    parser.add_argument("--target-relative-z", type=float)
    parser.add_argument("--target-z-tolerance", default=1.0, type=float)
    parser.add_argument("--hbond-ho", default=2.45, type=float)
    parser.add_argument("--hbond-oo", default=3.50, type=float)
    parser.add_argument("--hbond-angle", default=150.0, type=float)
    parser.add_argument("--dipole-dir", type=Path)
    args = parser.parse_args()
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.input.glob(args.glob))
    if not paths:
        raise RuntimeError(f"No source files match {args.glob!r} in {args.input}")
    with h5py.File(paths[0], "r") as handle:
        symbols = np.asarray(handle["symbols"]).astype(str)
        cell = np.asarray(handle["cell_A"], dtype=float)
        positions0 = np.asarray(handle["positions_A"][0], dtype=float)
        source_dt_fs = float(handle.attrs["output_interval_fs"])
    waters = water_topology(symbols, positions0, cell)
    oxygen, h1, h2 = waters.T
    h_by_water = np.stack((h1, h2), axis=1)
    nwater = len(waters)
    expected_samples = len(range(0, args.source_frames, args.stride))

    partner_counts = np.zeros((nwater, nwater), dtype=np.int64)
    degree_sum = np.zeros(nwater, dtype=np.int64)
    donated_sum = np.zeros(nwater, dtype=np.int64)
    accepted_sum = np.zeros(nwater, dtype=np.int64)
    degree4_count = np.zeros(nwater, dtype=np.int64)
    degree_ge4_count = np.zeros(nwater, dtype=np.int64)
    z_sum = np.zeros(nwater, dtype=np.float64)
    z_sq_sum = np.zeros(nwater, dtype=np.float64)
    samples = 0
    datasets = ("positions_A", "step", "time_ps")
    for file_number, (_path, _global_start, _indices, part) in enumerate(
        sampled_segment_parts(paths, args.source_frames, args.stride, datasets), start=1
    ):
        positions = np.asarray(part["positions_A"], dtype=np.float32)
        for start in range(0, len(positions), 25):
            stop = min(start + 25, len(positions))
            chunk = positions[start:stop]
            o_xyz = chunk[:, oxygen]
            h_xyz = chunk[:, h_by_water]
            directed = directed_hbond_state(
                o_xyz, h_xyz, cell, args.hbond_ho, args.hbond_oo,
                args.hbond_angle,
            )
            adjacency = directed | np.swapaxes(directed, 1, 2)
            degree = np.sum(adjacency, axis=2)
            partner_counts += np.sum(adjacency, axis=0, dtype=np.int64)
            degree_sum += np.sum(degree, axis=0, dtype=np.int64)
            donated_sum += np.sum(directed, axis=(0, 2), dtype=np.int64)
            accepted_sum += np.sum(directed, axis=(0, 1), dtype=np.int64)
            degree4_count += np.sum(degree == 4, axis=0, dtype=np.int64)
            degree_ge4_count += np.sum(degree >= 4, axis=0, dtype=np.int64)
            z = np.asarray(o_xyz[:, :, 2] % cell[2, 2], dtype=float)
            z_sum += np.sum(z, axis=0)
            z_sq_sum += np.sum(z ** 2, axis=0)
            samples += len(chunk)
        if file_number == 1 or file_number % 10 == 0:
            print(json.dumps({"screen_files_done": file_number,
                              "screen_samples": samples}), flush=True)
    if samples != expected_samples:
        raise RuntimeError(f"Expected {expected_samples} samples, obtained {samples}")

    mean_z = z_sum / samples
    z_std = np.sqrt(np.maximum(z_sq_sum / samples - mean_z ** 2, 0.0))
    relative_z = mean_z - args.gds
    candidate_rows = []
    metrics = []
    for water_index in range(nwater):
        partner_order = np.argsort(partner_counts[water_index])[::-1]
        top4 = partner_order[:4]
        top4_occupancy = partner_counts[water_index, top4] / samples
        total_partner_observations = int(np.sum(partner_counts[water_index]))
        identity_fraction = float(
            np.sum(partner_counts[water_index, top4])
            / max(total_partner_observations, 1)
        )
        fourth_occupancy = float(np.min(top4_occupancy))
        degree4_fraction = float(degree4_count[water_index] / samples)
        score = fourth_occupancy * identity_fraction * degree4_fraction
        row = {
            "water_index_1based": water_index + 1,
            "O_serial_1based": int(oxygen[water_index]) + 1,
            "H1_serial_1based": int(h1[water_index]) + 1,
            "H2_serial_1based": int(h2[water_index]) + 1,
            "mean_O_z_A": float(mean_z[water_index]),
            "std_O_z_A": float(z_std[water_index]),
            "mean_z_minus_GDS_A": float(relative_z[water_index]),
            "mean_HB_degree": float(degree_sum[water_index] / samples),
            "fraction_frames_degree_eq4": degree4_fraction,
            "fraction_frames_degree_ge4": float(degree_ge4_count[water_index] / samples),
            "mean_donated_HB": float(donated_sum[water_index] / samples),
            "mean_accepted_HB": float(accepted_sum[water_index] / samples),
            "unique_HB_partners": int(np.sum(partner_counts[water_index] > 0)),
            "top4_partner_identity_fraction": identity_fraction,
            "fourth_partner_occupancy": fourth_occupancy,
            "top4_mean_partner_occupancy": float(np.mean(top4_occupancy)),
            "four_partner_stability_score": score,
            "top4_partner_O_serials_1based": ";".join(
                str(int(oxygen[index]) + 1) for index in top4
            ),
            "top4_partner_occupancies": ";".join(
                f"{float(value):.8f}" for value in top4_occupancy
            ),
        }
        candidate_rows.append(row)
        metrics.append((water_index, row, top4))
    write_csv(
        args.output / "all_water_hbond_candidate_screen_500ps_stride10fs.csv",
        list(candidate_rows[0]), [[row[key] for key in row] for row in candidate_rows],
    )

    bulk = [item for item in metrics
            if args.bulk_min <= item[1]["mean_z_minus_GDS_A"] < args.bulk_max]
    if not bulk:
        raise RuntimeError("No water falls inside the requested bulk interval")
    if args.selection == "stable":
        selected_water, selected_row, selected_top4 = max(
            bulk, key=lambda item: (
                item[1]["four_partner_stability_score"],
                item[1]["fourth_partner_occupancy"],
                item[1]["fraction_frames_degree_eq4"],
            )
        )
        selection_note = (
            "Maximum fourth-partner occupancy x top-four identity fraction x "
            "degree-equals-four fraction within the requested bulk interval."
        )
    else:
        if args.target_relative_z is None:
            raise RuntimeError("--target-relative-z is required for unstable selection")
        same_depth = [item for item in bulk if abs(
            item[1]["mean_z_minus_GDS_A"] - args.target_relative_z
        ) <= args.target_z_tolerance]
        if not same_depth:
            same_depth = sorted(
                bulk, key=lambda item: abs(
                    item[1]["mean_z_minus_GDS_A"] - args.target_relative_z
                )
            )[:10]
        selected_water, selected_row, selected_top4 = min(
            same_depth, key=lambda item: (
                item[1]["four_partner_stability_score"],
                abs(item[1]["mean_z_minus_GDS_A"] - args.target_relative_z),
            )
        )
        selection_note = (
            "Minimum four-partner stability score among bulk waters within the "
            "requested matched relative-z tolerance; if empty, among the ten closest."
        )

    # Second pass: exact selected-water time series at the same 10 fs cadence.
    top4_set = set(int(value) for value in selected_top4)
    series_rows = []
    partner_episode_run = np.zeros(nwater, dtype=np.int64)
    partner_episode_count = np.zeros(nwater, dtype=np.int64)
    partner_episode_frames = np.zeros(nwater, dtype=np.int64)
    partner_episode_max = np.zeros(nwater, dtype=np.int64)
    previous_wrapped_o = None
    unwrapped_o = None
    first_unwrapped_o = None
    for file_number, (_path, _global_start, _indices, part) in enumerate(
        sampled_segment_parts(paths, args.source_frames, args.stride, datasets), start=1
    ):
        positions = np.asarray(part["positions_A"], dtype=np.float32)
        steps = np.asarray(part["step"], dtype=int)
        times = np.asarray(part["time_ps"], dtype=float)
        for start in range(0, len(positions), 25):
            stop = min(start + 25, len(positions))
            chunk = positions[start:stop]
            o_xyz = chunk[:, oxygen]
            h_xyz = chunk[:, h_by_water]
            donated_to, accepted_from, selected_adjacency = selected_water_hbond_state(
                o_xyz, h_xyz, selected_water, cell, args.hbond_ho,
                args.hbond_oo, args.hbond_angle,
            )
            for local in range(len(chunk)):
                frame_index = start + local
                wrapped_o = np.asarray(o_xyz[local, selected_water], dtype=float)
                if previous_wrapped_o is None:
                    unwrapped_o = wrapped_o.copy()
                    first_unwrapped_o = unwrapped_o.copy()
                else:
                    unwrapped_o = unwrapped_o + minimum_image(
                        wrapped_o - previous_wrapped_o, cell
                    )
                previous_wrapped_o = wrapped_o
                r1 = minimum_image(
                    chunk[local, h1[selected_water]] - wrapped_o, cell
                )
                r2 = minimum_image(
                    chunk[local, h2[selected_water]] - wrapped_o, cell
                )
                structural_dipole = r1 + r2
                structural_norm = float(np.linalg.norm(structural_dipole))
                structural_unit = structural_dipole / max(structural_norm, 1.0e-15)
                structural_angle = float(np.degrees(np.arccos(np.clip(
                    structural_unit[2], -1.0, 1.0
                ))))
                connected = selected_adjacency[local]
                ended = (partner_episode_run > 0) & ~connected
                partner_episode_count[ended] += 1
                partner_episode_frames[ended] += partner_episode_run[ended]
                partner_episode_max[ended] = np.maximum(
                    partner_episode_max[ended], partner_episode_run[ended]
                )
                partner_episode_run[connected] += 1
                partner_episode_run[~connected] = 0
                connected_indices = np.flatnonzero(connected)
                series_rows.append([
                    int(steps[frame_index]), float(times[frame_index]),
                    float(times[frame_index] - series_rows[0][1]) if series_rows else 0.0,
                    float(wrapped_o[0]), float(wrapped_o[1]), float(wrapped_o[2]),
                    float(wrapped_o[2] - args.gds),
                    float(unwrapped_o[0]), float(unwrapped_o[1]), float(unwrapped_o[2]),
                    float(unwrapped_o[0] - first_unwrapped_o[0]),
                    float(unwrapped_o[1] - first_unwrapped_o[1]),
                    float(structural_unit[0]), float(structural_unit[1]),
                    float(structural_unit[2]), structural_angle,
                    int(np.sum(donated_to[local])),
                    int(np.sum(accepted_from[local])),
                    int(np.sum(connected)), int(np.sum(connected) == 4),
                    int(np.sum(connected[list(top4_set)])),
                    int(np.sum(connected)) - int(np.sum(connected[list(top4_set)])),
                    ";".join(str(int(oxygen[index]) + 1)
                              for index in connected_indices),
                ])
        if file_number == 1 or file_number % 10 == 0:
            print(json.dumps({"export_files_done": file_number,
                              "export_samples": len(series_rows)}), flush=True)
    active = partner_episode_run > 0
    partner_episode_count[active] += 1
    partner_episode_frames[active] += partner_episode_run[active]
    partner_episode_max[active] = np.maximum(
        partner_episode_max[active], partner_episode_run[active]
    )
    time_series_header = [
        "step", "source_time_ps", "elapsed_time_ps", "O_x_wrapped_A",
        "O_y_wrapped_A", "O_z_wrapped_A", "O_z_minus_GDS_A",
        "O_x_unwrapped_A", "O_y_unwrapped_A", "O_z_unwrapped_A",
        "O_dx_unwrapped_from_start_A", "O_dy_unwrapped_from_start_A",
        "structural_dipole_unit_x", "structural_dipole_unit_y",
        "structural_dipole_unit_z", "structural_dipole_angle_to_plus_z_degree",
        "donated_HB", "accepted_HB", "unique_HB_degree", "degree_equals_4",
        "connected_top4_partner_count", "connected_other_partner_count",
        "connected_partner_O_serials_1based",
    ]
    time_series_path = args.output / "selected_water_timeseries_500ps_stride10fs.csv"
    write_csv(time_series_path, time_series_header, series_rows)

    partner_rows = []
    for partner in np.argsort(partner_counts[selected_water])[::-1]:
        if not partner_counts[selected_water, partner]:
            continue
        partner_rows.append([
            int(oxygen[partner]) + 1,
            float(partner_counts[selected_water, partner] / samples),
            bool(int(partner) in top4_set),
            int(partner_episode_count[partner]),
            float(partner_episode_frames[partner]
                  / max(partner_episode_count[partner], 1)
                  * source_dt_fs * args.stride / 1000.0),
            float(partner_episode_max[partner]
                  * source_dt_fs * args.stride / 1000.0),
        ])
    write_csv(
        args.output / "selected_water_partner_hbond_summary_500ps_stride10fs.csv",
        ["partner_O_serial_1based", "occupancy_fraction", "is_top4_partner",
         "continuous_episodes", "mean_continuous_lifetime_ps",
         "max_continuous_lifetime_ps"], partner_rows,
    )

    if args.dipole_dir:
        dipole_metadata = load_model_dipole(
            args.dipole_dir, int(oxygen[selected_water]), args.output
        )
    else:
        dipole_metadata = {
            "status": "not_supplied",
            "note": "Only the 10-fs structural O-to-HH-midpoint orientation is exported.",
        }
    metadata = {
        "status": "complete",
        "system": args.system,
        "source": str(args.input),
        "source_glob": args.glob,
        "source_files": len(paths),
        "source_frames_used": args.source_frames,
        "source_sampling_interval_fs": source_dt_fs,
        "analysis_stride_frames": args.stride,
        "analysis_sampling_interval_fs": source_dt_fs * args.stride,
        "analysis_samples": samples,
        "time_window_ps": [float(series_rows[0][1]), float(series_rows[-1][1])],
        "GDS_A": args.gds,
        "bulk_relative_z_interval_A": [args.bulk_min, args.bulk_max],
        "hbond_definition": {
            "H_to_acceptor_O_max_A": args.hbond_ho,
            "donor_O_to_acceptor_O_max_A": args.hbond_oo,
            "O_H_O_min_degree": args.hbond_angle,
            "undirected_partner_definition": (
                "connected if either water donates a geometrically valid H-bond to the other"
            ),
        },
        "selection": args.selection,
        "selection_note": selection_note,
        "target_relative_z_A": args.target_relative_z,
        "selected_water": selected_row,
        "structural_dipole_definition": (
            "unit vector from O toward the midpoint direction of its two minimum-image O-H vectors"
        ),
        "model_dipole": dipole_metadata,
        "time_series_sha256": file_sha256(time_series_path),
        "elapsed_s": time.time() - started,
    }
    (args.output / "selection_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
