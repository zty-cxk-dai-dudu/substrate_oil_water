#!/usr/bin/env python3
"""Validate local three-water interaction-energy calculations."""

from pathlib import Path
import argparse
import csv
import re


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
    parser.add_argument("--input", type=Path, default=ROOT, help="Prepared three-water calculation directory")
    args = parser.parse_args()
    root = args.input.resolve()
    with (root / "selected_waters.tsv").open() as handle:
        selected = list(csv.DictReader(handle, delimiter="\t"))
    if len(selected) != 3:
        raise RuntimeError(f"Expected 3 selected waters, found {len(selected)}")
    ab = root / "ab_full"
    if not complete(ab / "OUTCAR"):
        raise RuntimeError("Incomplete AB calculation")
    e_ab = final_energy(ab / "OUTCAR")
    iter_ab = last_scf_iteration(ab / "vasp.log")
    rows = [
        "rank\tO_serial\tH1_serial\tH2_serial\tO_z_A\tabs_from_middle_A\t"
        "E_ab_eV\tE_a_eV\tE_b_eV\tE_interaction_eV\tE_interaction_kJ_mol\t"
        "ab_scf_iter\ta_scf_iter\tb_scf_iter\tstatus"
    ]
    for row in selected:
        water = root / row["directory"]
        a, b = water / "a_without_water", water / "b_single_water"
        if not complete(a / "OUTCAR") or not complete(b / "OUTCAR"):
            raise RuntimeError(f"Incomplete result: {water}")
        e_a, e_b = final_energy(a / "OUTCAR"), final_energy(b / "OUTCAR")
        e_int = e_ab - e_a - e_b
        rows.append("\t".join((
            row["rank"], row["O_serial"], row["H1_serial"], row["H2_serial"],
            row["O_z_A"], row["abs_from_middle_A"], f"{e_ab:.10f}", f"{e_a:.10f}",
            f"{e_b:.10f}", f"{e_int:.10f}", f"{e_int * EV_TO_KJMOL:.6f}",
            str(iter_ab), str(last_scf_iteration(a / "vasp.log")),
            str(last_scf_iteration(b / "vasp.log")), "ok",
        )))
    (root / "interaction_energy_summary.tsv").write_text("\n".join(rows) + "\n")
    print("Verified 7 calculations and 3 interaction energies")


if __name__ == "__main__":
    main()
