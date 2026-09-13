#!/usr/bin/env python3
"""Export one unstable SiO2 bulk water at the CaF2-matched relative depth."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def minimum_image(delta, cell):
    frac = delta @ np.linalg.inv(cell)
    frac -= np.rint(frac)
    return frac @ cell


def selected_water_hbond_state(o_xyz, h_xyz, selected, cell,
                               ho_cutoff=2.45, oo_cutoff=3.50,
                               angle_cutoff=150.0):
    selected_o = o_xyz[:, selected]
    selected_h = h_xyz[:, selected]
    h_to_acceptor = minimum_image(
        o_xyz[:, None, :, :] - selected_h[:, :, None, :], cell)
    ho_distance = np.linalg.norm(h_to_acceptor, axis=-1)
    donor_to_acceptor = minimum_image(o_xyz - selected_o[:, None, :], cell)
    oo_distance = np.linalg.norm(donor_to_acceptor, axis=-1)
    h_to_donor = minimum_image(selected_o[:, None, :] - selected_h, cell)[:, :, None, :]
    cosine = np.sum(h_to_donor * h_to_acceptor, axis=-1) / np.maximum(
        np.linalg.norm(h_to_donor, axis=-1)
        * np.linalg.norm(h_to_acceptor, axis=-1), 1.0e-12)
    angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    donated = np.any((ho_distance <= ho_cutoff)
                     & (oo_distance[:, None, :] <= oo_cutoff)
                     & (angle >= angle_cutoff), axis=1)
    h_to_selected = minimum_image(selected_o[:, None, None, :] - h_xyz, cell)
    hs_distance = np.linalg.norm(h_to_selected, axis=-1)
    donor_to_selected = minimum_image(selected_o[:, None, :] - o_xyz, cell)
    ds_distance = np.linalg.norm(donor_to_selected, axis=-1)
    h_to_donor_all = minimum_image(o_xyz[:, :, None, :] - h_xyz, cell)
    cosine = np.sum(h_to_donor_all * h_to_selected, axis=-1) / np.maximum(
        np.linalg.norm(h_to_donor_all, axis=-1)
        * np.linalg.norm(h_to_selected, axis=-1), 1.0e-12)
    angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    accepted = np.any((hs_distance <= ho_cutoff)
                      & (ds_distance[:, :, None] <= oo_cutoff)
                      & (angle >= angle_cutoff), axis=2)
    donated[:, selected] = False
    accepted[:, selected] = False
    return donated, accepted, donated | accepted


def unwrap(values, cell):
    output = np.empty_like(values, dtype=float); output[0] = values[0]
    for frame in range(1, len(values)):
        output[frame] = output[frame - 1] + minimum_image(
            values[frame] - values[frame - 1], cell)
    return output


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, fields, rows):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(fields); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-O", default=531, type=int)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    with args.candidates.open(newline="") as stream:
        candidates = list(csv.DictReader(stream))
    matches = [row for row in candidates if int(row["O_serial_1based"]) == args.target_O]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one O{args.target_O} candidate")
    selected_row = matches[0]
    with np.load(args.cache) as cache:
        position = np.asarray(cache["position_A"][:5000], dtype=np.float32)
        steps = np.asarray(cache["step"][:5000], dtype=np.int64)
        cell = np.asarray(cache["cell_A"], dtype=float)
        waters_id = np.asarray(cache["waters_id"], dtype=np.int64)
    found = np.flatnonzero(waters_id[:, 0] == args.target_O)
    if len(found) != 1:
        raise RuntimeError(f"O{args.target_O} not uniquely mapped")
    selected = int(found[0]); nframes, nwater = position.shape[:2]
    adjacency = np.empty((nframes, nwater), dtype=bool)
    donated_count = np.empty(nframes, dtype=np.uint8)
    accepted_count = np.empty(nframes, dtype=np.uint8)
    for start in range(0, nframes, 250):
        stop = min(start + 250, nframes)
        donated, accepted, connected = selected_water_hbond_state(
            position[start:stop, :, 0], position[start:stop, :, 1:], selected, cell)
        adjacency[start:stop] = connected
        donated_count[start:stop] = np.sum(donated, axis=1)
        accepted_count[start:stop] = np.sum(accepted, axis=1)
        print(json.dumps({"frames_done": stop, "frames_total": nframes}), flush=True)
    counts = np.sum(adjacency, axis=0)
    top4 = np.argsort(counts)[::-1][:4]
    top4_set = set(map(int, top4))
    selected_o = position[:, selected, 0]
    selected_h = position[:, selected, 1:]
    unwrapped_o = unwrap(selected_o, cell)
    vector = minimum_image(selected_h - selected_o[:, None, :], cell).sum(axis=1)
    vector /= np.maximum(np.linalg.norm(vector, axis=1, keepdims=True), 1e-12)
    angle = np.degrees(np.arccos(np.clip(vector[:, 2], -1, 1)))
    time_ps = steps * 0.0005
    first = unwrapped_o[0]
    rows = []
    for frame in range(nframes):
        connected = np.flatnonzero(adjacency[frame])
        top4_count = sum(int(index) in top4_set for index in connected)
        degree = len(connected)
        rows.append([
            int(steps[frame]), float(time_ps[frame]), float(time_ps[frame] - time_ps[0]),
            *map(float, selected_o[frame]), float(selected_o[frame, 2] - 18.0),
            float((selected_o[frame, 2] - 18.0) / 32.0),
            *map(float, unwrapped_o[frame]),
            float(unwrapped_o[frame, 0] - first[0]),
            float(unwrapped_o[frame, 1] - first[1]),
            *map(float, vector[frame]), float(angle[frame]),
            int(donated_count[frame]), int(accepted_count[frame]), degree,
            int(degree == 4), top4_count, degree - top4_count,
            ";".join(str(int(waters_id[index, 0])) for index in connected),
        ])
    fields = [
        "step", "source_time_ps", "elapsed_time_ps", "O_x_wrapped_A",
        "O_y_wrapped_A", "O_z_wrapped_A", "O_z_relative_water_film_A",
        "O_relative_depth_fraction", "O_x_unwrapped_A", "O_y_unwrapped_A",
        "O_z_unwrapped_A", "O_dx_unwrapped_from_start_A",
        "O_dy_unwrapped_from_start_A", "structural_dipole_unit_x",
        "structural_dipole_unit_y", "structural_dipole_unit_z",
        "structural_dipole_angle_to_plus_z_degree", "donated_HB", "accepted_HB",
        "unique_HB_degree", "degree_equals_4", "connected_top4_partner_count",
        "connected_other_partner_count", "connected_partner_O_serials_1based",
    ]
    series_path = args.output / "selected_water_timeseries_500ps_stride100fs.csv"
    write_csv(series_path, fields, rows)
    partner_rows = []
    for partner in np.argsort(counts)[::-1]:
        if not counts[partner]:
            continue
        state = adjacency[:, partner]
        changes = np.diff(np.r_[False, state, False].astype(np.int8))
        starts = np.flatnonzero(changes == 1); stops = np.flatnonzero(changes == -1)
        lengths = stops - starts
        partner_rows.append([
            int(waters_id[partner, 0]), counts[partner] / nframes, len(lengths),
            float(np.mean(lengths) * 0.1), float(np.max(lengths) * 0.1),
        ])
    write_csv(args.output / "selected_water_partner_hbond_summary_500ps_stride100fs.csv",
              ["partner_O_serial_1based", "occupancy_fraction", "continuous_episodes",
               "mean_continuous_lifetime_ps", "max_continuous_lifetime_ps"], partner_rows)
    selected = {}
    for key, value in selected_row.items():
        try:
            selected[key] = float(value) if any(char in value for char in ".eE") else int(value)
        except (ValueError, TypeError):
            selected[key] = value
    metadata = {
        "status": "complete", "system": "SiO2-fixed/water",
        "selection": "unstable", "selection_scope": "mapped bulk target +/- 1 A",
        "target_relative_depth_fraction_from_CaF2": 0.4545090828003932,
        "mapped_target_depth_A": 14.54429064961258,
        "selection_tolerance_A": 1.0, "eligible_water_count": 15,
        "selection_rule": "minimum four_partner_stability_score",
        "selected_water": selected, "position_sampling_interval_fs": 100.0,
        "water_film_reference_z0_A": 18.0, "water_film_thickness_A": 32.0,
        "hbond_definition": {"H_to_O_max_A": 2.45, "O_to_O_max_A": 3.5,
                             "O_H_O_min_degree": 150.0},
        "model_dipole": {"status": "unavailable",
                         "reason": "No validated SiO2 per-water dipole model"},
        "time_series_sha256": sha256(series_path),
    }
    (args.output / "selection_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output / "COMPLETE").write_text("complete\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
