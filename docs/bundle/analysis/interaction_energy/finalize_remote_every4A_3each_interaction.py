#!/usr/bin/env python3
"""Validate and summarize one every-4-A interaction-energy dataset."""

from pathlib import Path
import argparse
import csv
import re
import statistics


ROOT = Path(__file__).resolve().parent
EV_TO_KJMOL = 96.4853321233


def final_energy(path):
    pattern = re.compile(r"free\s+energy\s+TOTEN\s+=\s+([-+0-9.Ee]+)")
    value = None
    with path.open(errors="replace") as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                value = float(match.group(1))
    if value is None:
        raise RuntimeError(f"Missing TOTEN in {path}")
    return value


def last_scf_iteration(path):
    pattern = re.compile(r"(?:DAV|RMM):\s+(\d+)")
    value = None
    with path.open(errors="replace") as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                value = int(match.group(1))
    if value is None or value >= 300:
        raise RuntimeError(f"Invalid final SCF iteration in {path}: {value}")
    return value


def complete(path):
    marker = "General timing and accounting informations"
    with path.open(errors="replace") as handle:
        return any(marker in line for line in handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT, help="Prepared every-4-A calculation directory")
    args = parser.parse_args()
    root = args.input.resolve()
    with (root / "selected_waters.tsv").open() as handle:
        selected = list(csv.DictReader(handle, delimiter="\t"))
    if not selected:
        raise RuntimeError("No selected waters")
    ab = root / "ab_reference"
    if not complete(ab / "OUTCAR"):
        raise RuntimeError("Incomplete AB reference")
    e_ab = final_energy(ab / "OUTCAR")
    iter_ab = last_scf_iteration(ab / "vasp.log")
    header = [
        "system_label", "plane_index", "target_z_unwrapped_A", "target_z_mod_cell_A",
        "water_rank", "O_serial", "H1_serial", "H2_serial", "O_z_A", "abs_from_target_A",
        "reused", "E_ab_eV", "E_a_eV", "E_b_eV", "E_interaction_eV",
        "E_interaction_kJ_mol", "ab_scf_iter", "a_scf_iter", "b_scf_iter", "status",
    ]
    output_rows, by_plane = [], {}
    for row in selected:
        water = root / row["directory"]
        a, b = water / "a_without_water", water / "b_single_water"
        if not complete(a / "OUTCAR") or not complete(b / "OUTCAR"):
            raise RuntimeError(f"Incomplete water result: {water}")
        e_a, e_b = final_energy(a / "OUTCAR"), final_energy(b / "OUTCAR")
        e_int = e_ab - e_a - e_b
        values = [
            row["system_label"], row["plane_index"], row["target_z_unwrapped_A"],
            row["target_z_mod_cell_A"], row["water_rank"], row["O_serial"], row["H1_serial"],
            row["H2_serial"], row["O_z_A"], row["abs_from_target_A"], row["reused"],
            f"{e_ab:.10f}", f"{e_a:.10f}", f"{e_b:.10f}", f"{e_int:.10f}",
            f"{e_int * EV_TO_KJMOL:.6f}", str(iter_ab),
            str(last_scf_iteration(a / "vasp.log")), str(last_scf_iteration(b / "vasp.log")), "ok",
        ]
        output_rows.append(values)
        by_plane.setdefault(row["plane_index"], []).append((row, e_int))
    with (root / "interaction_energy_summary.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(output_rows)
    plane_header = [
        "system_label", "plane_index", "target_z_unwrapped_A", "target_z_mod_cell_A",
        "water_count", "mean_interaction_eV", "sd_interaction_eV", "mean_interaction_kJ_mol",
    ]
    plane_rows = []
    for plane_index in sorted(by_plane, key=int):
        items = by_plane[plane_index]
        if len(items) != 3:
            raise RuntimeError(f"Plane {plane_index} has {len(items)} waters")
        energies = [value for _, value in items]
        row = items[0][0]
        mean = statistics.mean(energies)
        sd = statistics.stdev(energies)
        plane_rows.append([
            row["system_label"], plane_index, row["target_z_unwrapped_A"], row["target_z_mod_cell_A"],
            str(len(items)), f"{mean:.10f}", f"{sd:.10f}", f"{mean * EV_TO_KJMOL:.6f}",
        ])
    with (root / "plane_interaction_summary.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(plane_header)
        writer.writerows(plane_rows)
    print(
        f"Verified {selected[0]['system_label']}: {len(by_plane)} planes, "
        f"{len(selected)} waters, {1 + 2 * len(selected)} energy calculations"
    )


if __name__ == "__main__":
    main()
