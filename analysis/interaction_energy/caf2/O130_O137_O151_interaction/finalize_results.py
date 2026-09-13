#!/usr/bin/env python3
"""Validate completed VASP runs and write the interaction-energy table."""

from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EV_TO_KJMOL = 96.4853321233


def validate(calc: Path):
    outcar = (calc / "OUTCAR").read_text(errors="replace")
    if "General timing and accounting informations" not in outcar:
        raise RuntimeError(f"Incomplete calculation: {calc}")
    energies = [float(x) for x in re.findall(r"energy\s+without entropy\s*=\s*([-+0-9.Ee]+)", outcar)]
    if not energies:
        raise RuntimeError(f"Missing final energy: {calc}")
    log = (calc / "vasp.log").read_text(errors="replace")
    iterations = [int(x) for x in re.findall(r"(?:DAV|RMM|SDA|CGA|DMP):\s+(\d+)", log)]
    if not iterations or iterations[-1] >= 300:
        raise RuntimeError(f"Invalid SCF completion: {calc}; last iteration={iterations[-1] if iterations else 'none'}")
    return energies[-1], iterations[-1]


def main():
    rows = []
    with (ROOT / "selected_waters.tsv").open() as handle:
        selected = list(csv.DictReader(handle, delimiter="\t"))
    e_ab, i_ab = validate(ROOT / "ab_full")
    for row in selected:
        oxygen, h1, h2 = row["O_serial"], row["H1_serial"], row["H2_serial"]
        e_a, i_a = validate(ROOT / f"a_without_O{oxygen}")
        e_b, i_b = validate(ROOT / f"b_O{oxygen}_H{h1}_H{h2}")
        e_int = e_ab - e_a - e_b
        rows.append({
            "O_serial": oxygen,
            "H1_serial": h1,
            "H2_serial": h2,
            "O-H1_A": row["O-H1_A"],
            "O-H2_A": row["O-H2_A"],
            "E_ab_eV": f"{e_ab:.10f}",
            "E_a_eV": f"{e_a:.10f}",
            "E_b_eV": f"{e_b:.10f}",
            "E_interaction_eV": f"{e_int:.10f}",
            "E_interaction_kJ_mol": f"{e_int * EV_TO_KJMOL:.6f}",
            "ab_scf_iter": str(i_ab),
            "a_scf_iter": str(i_a),
            "b_scf_iter": str(i_b),
            "status": "ok",
        })
    fields = list(rows[0])
    with (ROOT / "interaction_energy_summary.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print((ROOT / "interaction_energy_summary.tsv").read_text(), end="")


if __name__ == "__main__":
    main()
