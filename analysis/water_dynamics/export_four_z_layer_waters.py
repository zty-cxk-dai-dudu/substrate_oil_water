#!/usr/bin/env python3
"""Export four preselected matched-depth waters from one 500 ps trajectory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np

from select_and_export_matched_water import (
    load_model_dipole,
    minimum_image,
    sampled_segment_parts,
    selected_water_hbond_state,
    water_topology,
    write_csv,
)


TIME_SERIES_HEADER = [
    "step", "source_time_ps", "elapsed_time_ps", "O_x_wrapped_A",
    "O_y_wrapped_A", "O_z_wrapped_A", "O_z_minus_GDS_A",
    "O_x_unwrapped_A", "O_y_unwrapped_A", "O_z_unwrapped_A",
    "O_dx_unwrapped_from_start_A", "O_dy_unwrapped_from_start_A",
    "water_O_COM_x_unwrapped_A", "water_O_COM_y_unwrapped_A",
    "water_O_COM_z_unwrapped_A", "water_O_COM_dx_from_start_A",
    "water_O_COM_dy_from_start_A",
    "O_dx_relative_water_COM_from_start_A",
    "O_dy_relative_water_COM_from_start_A",
    "structural_dipole_unit_x", "structural_dipole_unit_y",
    "structural_dipole_unit_z", "structural_dipole_angle_to_plus_z_degree",
    "donated_HB", "accepted_HB", "unique_HB_degree", "degree_equals_4",
    "connected_top4_partner_count", "connected_other_partner_count",
    "connected_partner_O_serials_1based",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_candidate_rows(path: Path) -> dict[int, dict[str, object]]:
    numeric_float = {
        "mean_O_z_A", "std_O_z_A", "mean_z_minus_GDS_A", "mean_HB_degree",
        "fraction_frames_degree_eq4", "fraction_frames_degree_ge4",
        "mean_donated_HB", "mean_accepted_HB",
        "top4_partner_identity_fraction", "fourth_partner_occupancy",
        "top4_mean_partner_occupancy", "four_partner_stability_score",
    }
    numeric_int = {
        "water_index_1based", "O_serial_1based", "H1_serial_1based",
        "H2_serial_1based", "unique_HB_partners",
    }
    rows: dict[int, dict[str, object]] = {}
    with path.open(newline="") as stream:
        for raw in csv.DictReader(stream):
            row: dict[str, object] = dict(raw)
            for key in numeric_float:
                row[key] = float(raw[key])
            for key in numeric_int:
                row[key] = int(raw[key])
            rows[int(row["O_serial_1based"])] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--glob", default="*segment_*.h5")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--system", required=True)
    parser.add_argument("--gds", required=True, type=float)
    parser.add_argument("--candidate-screen", required=True, type=Path)
    parser.add_argument("--selected-o-serials", required=True)
    parser.add_argument("--source-frames", default=500000, type=int)
    parser.add_argument("--stride", default=10, type=int)
    parser.add_argument("--hbond-ho", default=2.45, type=float)
    parser.add_argument("--hbond-oo", default=3.50, type=float)
    parser.add_argument("--hbond-angle", default=150.0, type=float)
    parser.add_argument("--dipole-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.time()

    selected_serials = [int(value) for value in args.selected_o_serials.split(",")]
    if len(selected_serials) != 4 or len(set(selected_serials)) != 4:
        raise RuntimeError("Exactly four unique oxygen serials are required")
    candidate_rows = load_candidate_rows(args.candidate_screen)
    missing = [serial for serial in selected_serials if serial not in candidate_rows]
    if missing:
        raise RuntimeError(f"Selected O serials absent from candidate screen: {missing}")

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
    oxygen_to_water = {int(atom) + 1: i for i, atom in enumerate(oxygen)}

    states = []
    for layer, serial in enumerate(selected_serials, start=1):
        if serial not in oxygen_to_water:
            raise RuntimeError(f"O serial {serial} is not a water oxygen")
        selected = oxygen_to_water[serial]
        candidate = candidate_rows[serial]
        top4_serials = [
            int(value) for value in str(candidate["top4_partner_O_serials_1based"]).split(";")
        ]
        top4 = [oxygen_to_water[value] for value in top4_serials]
        directory = args.output / f"layer{layer}_O{serial}"
        directory.mkdir(parents=True, exist_ok=True)
        time_series_path = directory / "selected_water_timeseries_500ps_stride10fs.csv"
        stream = time_series_path.open("w", newline="")
        writer = csv.writer(stream)
        writer.writerow(TIME_SERIES_HEADER)
        states.append({
            "layer": layer,
            "serial": serial,
            "selected": selected,
            "candidate": candidate,
            "top4": top4,
            "top4_set": set(top4),
            "directory": directory,
            "time_series_path": time_series_path,
            "stream": stream,
            "writer": writer,
            "previous_wrapped_o": None,
            "unwrapped_o": None,
            "first_unwrapped_o": None,
            "first_relative_to_water_com": None,
            "first_time_ps": None,
            "samples": 0,
            "partner_counts": np.zeros(len(waters), dtype=np.int64),
            "episode_run": np.zeros(len(waters), dtype=np.int64),
            "episode_count": np.zeros(len(waters), dtype=np.int64),
            "episode_frames": np.zeros(len(waters), dtype=np.int64),
            "episode_max": np.zeros(len(waters), dtype=np.int64),
        })

    datasets = ("positions_A", "step", "time_ps")
    previous_all_o = None
    unwrapped_all_o = None
    first_water_com = None
    try:
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
                water_com_unwrapped = []
                for local in range(len(chunk)):
                    wrapped_all_o = np.asarray(o_xyz[local], dtype=float)
                    if previous_all_o is None:
                        unwrapped_all_o = wrapped_all_o.copy()
                    else:
                        unwrapped_all_o = unwrapped_all_o + minimum_image(
                            wrapped_all_o - previous_all_o, cell
                        )
                    previous_all_o = wrapped_all_o
                    current_com = np.mean(unwrapped_all_o, axis=0)
                    if first_water_com is None:
                        first_water_com = current_com.copy()
                    water_com_unwrapped.append(current_com)
                water_com_unwrapped = np.asarray(water_com_unwrapped, dtype=float)
                for state in states:
                    selected = int(state["selected"])
                    donated_to, accepted_from, adjacency = selected_water_hbond_state(
                        o_xyz, h_xyz, selected, cell, args.hbond_ho,
                        args.hbond_oo, args.hbond_angle,
                    )
                    state["partner_counts"] += np.sum(adjacency, axis=0, dtype=np.int64)
                    for local in range(len(chunk)):
                        wrapped_o = np.asarray(o_xyz[local, selected], dtype=float)
                        if state["previous_wrapped_o"] is None:
                            state["unwrapped_o"] = wrapped_o.copy()
                            state["first_unwrapped_o"] = wrapped_o.copy()
                            state["first_time_ps"] = float(times[start + local])
                        else:
                            state["unwrapped_o"] = state["unwrapped_o"] + minimum_image(
                                wrapped_o - state["previous_wrapped_o"], cell
                            )
                        state["previous_wrapped_o"] = wrapped_o

                        connected = adjacency[local]
                        ended = (state["episode_run"] > 0) & ~connected
                        state["episode_count"][ended] += 1
                        state["episode_frames"][ended] += state["episode_run"][ended]
                        state["episode_max"][ended] = np.maximum(
                            state["episode_max"][ended], state["episode_run"][ended]
                        )
                        state["episode_run"][connected] += 1
                        state["episode_run"][~connected] = 0

                        r1 = minimum_image(chunk[local, h1[selected]] - wrapped_o, cell)
                        r2 = minimum_image(chunk[local, h2[selected]] - wrapped_o, cell)
                        structural = r1 + r2
                        structural /= max(float(np.linalg.norm(structural)), 1.0e-15)
                        angle = float(np.degrees(np.arccos(np.clip(structural[2], -1.0, 1.0))))
                        connected_indices = np.flatnonzero(connected)
                        top4_count = sum(bool(connected[index]) for index in state["top4_set"])
                        unwrapped = state["unwrapped_o"]
                        first = state["first_unwrapped_o"]
                        water_com = water_com_unwrapped[local]
                        relative_to_water_com = unwrapped - water_com
                        if state["first_relative_to_water_com"] is None:
                            state["first_relative_to_water_com"] = relative_to_water_com.copy()
                        first_relative = state["first_relative_to_water_com"]
                        frame = start + local
                        state["writer"].writerow([
                            int(steps[frame]), float(times[frame]),
                            float(times[frame] - state["first_time_ps"]),
                            float(wrapped_o[0]), float(wrapped_o[1]), float(wrapped_o[2]),
                            float(wrapped_o[2] - args.gds),
                            float(unwrapped[0]), float(unwrapped[1]), float(unwrapped[2]),
                            float(unwrapped[0] - first[0]), float(unwrapped[1] - first[1]),
                            float(water_com[0]), float(water_com[1]), float(water_com[2]),
                            float(water_com[0] - first_water_com[0]),
                            float(water_com[1] - first_water_com[1]),
                            float(relative_to_water_com[0] - first_relative[0]),
                            float(relative_to_water_com[1] - first_relative[1]),
                            float(structural[0]), float(structural[1]), float(structural[2]), angle,
                            int(np.sum(donated_to[local])), int(np.sum(accepted_from[local])),
                            int(np.sum(connected)), int(np.sum(connected) == 4), top4_count,
                            int(np.sum(connected)) - top4_count,
                            ";".join(str(int(oxygen[index]) + 1)
                                      for index in connected_indices),
                        ])
                        state["samples"] += 1
            if file_number == 1 or file_number % 10 == 0:
                print(json.dumps({"files_done": file_number,
                                  "samples_per_water": states[0]["samples"]}), flush=True)
    finally:
        for state in states:
            state["stream"].close()

    expected_samples = len(range(0, args.source_frames, args.stride))
    manifest_rows = []
    for state in states:
        if state["samples"] != expected_samples:
            raise RuntimeError(
                f"O{state['serial']} has {state['samples']} samples, expected {expected_samples}"
            )
        active = state["episode_run"] > 0
        state["episode_count"][active] += 1
        state["episode_frames"][active] += state["episode_run"][active]
        state["episode_max"][active] = np.maximum(
            state["episode_max"][active], state["episode_run"][active]
        )
        partner_rows = []
        for partner in np.argsort(state["partner_counts"])[::-1]:
            count = int(state["partner_counts"][partner])
            if not count:
                continue
            episodes = int(state["episode_count"][partner])
            partner_rows.append([
                int(oxygen[partner]) + 1, count / expected_samples,
                bool(int(partner) in state["top4_set"]), episodes,
                float(state["episode_frames"][partner] / max(episodes, 1)
                      * source_dt_fs * args.stride / 1000.0),
                float(state["episode_max"][partner]
                      * source_dt_fs * args.stride / 1000.0),
            ])
        write_csv(
            state["directory"] / "selected_water_partner_hbond_summary_500ps_stride10fs.csv",
            ["partner_O_serial_1based", "occupancy_fraction", "is_top4_partner",
             "continuous_episodes", "mean_continuous_lifetime_ps",
             "max_continuous_lifetime_ps"], partner_rows,
        )
        dipole = load_model_dipole(
            args.dipole_dir, int(oxygen[state["selected"]]), state["directory"]
        )
        metadata = {
            "status": "complete",
            "system": args.system,
            "layer_index_from_GDS": state["layer"],
            "source": str(args.input),
            "source_files": len(paths),
            "source_frames_used": args.source_frames,
            "source_sampling_interval_fs": source_dt_fs,
            "analysis_sampling_interval_fs": source_dt_fs * args.stride,
            "analysis_samples": expected_samples,
            "time_window_ps": [state["first_time_ps"],
                               state["first_time_ps"] + (expected_samples - 1)
                               * source_dt_fs * args.stride / 1000.0],
            "GDS_A": args.gds,
            "selected_water": state["candidate"],
            "hbond_definition": {
                "H_to_acceptor_O_max_A": args.hbond_ho,
                "donor_O_to_acceptor_O_max_A": args.hbond_oo,
                "O_H_O_min_degree": args.hbond_angle,
            },
            "xy_reference_frames": {
                "lab_frame": "minimum-image unwrapped selected-water oxygen",
                "water_COM_corrected": (
                    "selected-water unwrapped oxygen minus the instantaneous COM "
                    "of all unwrapped water oxygens, referenced to the first frame"
                ),
            },
            "model_dipole": dipole,
            "time_series_sha256": sha256(state["time_series_path"]),
        }
        (state["directory"] / "selection_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        manifest_rows.append([
            state["layer"], state["serial"],
            state["candidate"]["mean_z_minus_GDS_A"],
            state["candidate"]["four_partner_stability_score"],
            state["candidate"]["fraction_frames_degree_eq4"],
            state["candidate"]["fourth_partner_occupancy"],
            str(state["directory"]), expected_samples, dipole["frames"],
        ])

    write_csv(
        args.output / "selected_four_water_manifest.csv",
        ["layer_index_from_GDS", "O_serial_1based", "mean_z_minus_GDS_A",
         "four_partner_stability_score", "fraction_frames_degree_eq4",
         "fourth_partner_occupancy", "output_directory", "samples_10fs",
         "model_dipole_samples_100fs"], manifest_rows,
    )
    summary = {
        "status": "complete",
        "system": args.system,
        "selected_O_serials_1based": selected_serials,
        "analysis_samples_per_water": expected_samples,
        "elapsed_s": time.time() - started,
    }
    (args.output / "COMPLETE.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
