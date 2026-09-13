#!/usr/bin/env python3
"""Prepare frozen-geometry D2O--system interaction-energy calculations."""

from __future__ import annotations

import csv
import hashlib
import re
import shutil
import argparse
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "O130_O137_O151_interaction"
SOURCE = ROOT / "CONTCAR"
TARGET_O = [130, 137, 151]
EV_TO_KJMOL = 96.4853321233


def parse_poscar(path: Path):
    lines = path.read_text().splitlines()
    scale = float(lines[1].split()[0])
    cell = np.array([[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)]) * scale
    species = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    n_atoms = sum(counts)
    line_no = 7
    selective = lines[line_no].strip().lower().startswith("s")
    if selective:
        line_no += 1
    mode = lines[line_no].strip().lower()
    line_no += 1
    direct = []
    for j in range(n_atoms):
        fields = lines[line_no + j].split()
        direct.append([float(x) for x in fields[:3]])
    direct = np.asarray(direct, dtype=float)
    if mode.startswith("c"):
        direct = direct @ np.linalg.inv(cell)
    labels = []
    for symbol, count in zip(species, counts):
        labels.extend([symbol] * count)
    return scale, cell, species, labels, direct


def atom_numbers(labels, species):
    by_element = {symbol: [] for symbol in species}
    for serial, symbol in enumerate(labels, start=1):
        by_element[symbol].append(serial)
    return by_element


def write_poscar(path: Path, title: str, cell, species, labels, direct):
    counts = [labels.count(symbol) for symbol in species]
    with path.open("w") as handle:
        handle.write(f"{title}\n1.0\n")
        for vector in cell:
            handle.write("  " + " ".join(f"{x:.16f}" for x in vector) + "\n")
        handle.write("  " + "  ".join(species) + "\n")
        handle.write("  " + "  ".join(str(x) for x in counts) + "\n")
        handle.write("Direct\n")
        for xyz in direct:
            handle.write("  " + " ".join(f"{x:.16f}" for x in xyz) + "\n")


def potcar_blocks(path: Path):
    text = path.read_text()
    starts = [m.start() for m in re.finditer(r"(?m)^\s*PAW_PBE\s+", text)]
    blocks = {}
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        block = text[start:end]
        match = re.search(r"TITEL\s*=\s*PAW_PBE\s+(\S+)", block)
        if not match:
            raise RuntimeError(f"Could not identify POTCAR block starting at {start}")
        symbol = match.group(1)
        if symbol.endswith("_h") or symbol.endswith("_sv"):
            symbol = symbol.split("_")[0]
        blocks[symbol] = block
    return blocks


def write_inputs(calc: Path, title: str, cell, species, labels, direct, blocks):
    calc.mkdir(parents=True, exist_ok=True)
    write_poscar(calc / "POSCAR", title, cell, species, labels, direct)
    (calc / "INCAR").write_text(
        f"SYSTEM = {title}\n"
        "ISTART = 0\n"
        "ICHARG = 2\n"
        "GGA = PE\n"
        "IVDW = 12\n"
        "PREC = Accurate\n"
        "ENCUT = 450\n"
        "EDIFF = 1E-6\n"
        "NELMIN = 6\n"
        "NELM = 300\n"
        "ALGO = Normal\n"
        "ISMEAR = 0\n"
        "SIGMA = 0.06\n"
        "ISPIN = 1\n"
        "ISYM = 0\n"
        "LREAL = Auto\n"
        "IBRION = -1\n"
        "NSW = 0\n"
        "LCHARG = .FALSE.\n"
        "LWAVE = .FALSE.\n"
        "NCORE = 6\n"
    )
    (calc / "KPOINTS").write_text("#KPOINTS\n0\nGamma\n  1  1  1\n0.0  0.0  0.0\n")
    (calc / "POTCAR").write_text("".join(blocks[symbol] for symbol in species))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("CONTCAR"))
    parser.add_argument("--output", type=Path, default=Path("O130_O137_O151_interaction"))
    parser.add_argument("--potcar", type=Path, default=None)
    parser.add_argument("--source-label", default=None)
    parser.add_argument("--targets", type=int, nargs="+", default=[130, 137, 151])
    args = parser.parse_args()
    global ROOT, OUT, SOURCE, TARGET_O
    SOURCE = args.source.resolve()
    ROOT = SOURCE.parent
    OUT = args.output.resolve()
    TARGET_O = args.targets
    source_label = args.source_label or str(SOURCE)
    if OUT.exists():
        existing = {path.name for path in OUT.iterdir()}
        allowed = {"finalize_results.py", "run_all.sh"}
        unexpected = existing - allowed
        if unexpected:
            raise SystemExit(f"Refusing to overwrite existing directory {OUT}; unexpected files: {sorted(unexpected)}")
    scale, cell, source_species, labels, direct = parse_poscar(SOURCE)
    numbers = atom_numbers(labels, source_species)
    h_serials = numbers["H"]
    o_serials = numbers["O"]
    cart = direct @ cell
    selected = []
    for o_serial in TARGET_O:
        if labels[o_serial - 1] != "O":
            raise RuntimeError(f"Atom {o_serial} is {labels[o_serial - 1]}, not O")
        candidates = []
        for h_serial in h_serials:
            delta = cart[h_serial - 1] - cart[o_serial - 1]
            fractional_delta = delta @ np.linalg.inv(cell)
            fractional_delta -= np.rint(fractional_delta)
            distance = float(np.linalg.norm(fractional_delta @ cell))
            if distance <= 1.25:
                candidates.append((distance, h_serial))
        candidates.sort()
        if len(candidates) != 2:
            raise RuntimeError(f"O{o_serial} has {len(candidates)} H neighbours within 1.25 A: {candidates}")
        selected.append((o_serial, candidates[0][1], candidates[1][1], candidates[0][0], candidates[1][0]))

    OUT.mkdir(exist_ok=True)
    shutil.copy2(SOURCE, OUT / "source_CONTCAR")
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    with (OUT / "selected_waters.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["O_serial", "H1_serial", "H2_serial", "O-H1_A", "O-H2_A", "source", "source_sha256"])
        for o_serial, h1, h2, d1, d2 in selected:
            writer.writerow([o_serial, h1, h2, f"{d1:.6f}", f"{d2:.6f}", source_label, source_hash])

    blocks = potcar_blocks(args.potcar.resolve() if args.potcar else ROOT / "POTCAR")
    write_inputs(OUT / "ab_full", "ab_CONTCAR", cell, source_species, labels, direct, blocks)
    for o_serial, h1, h2, *_ in selected:
        removed = {o_serial, h1, h2}
        a_labels = [symbol for serial, symbol in enumerate(labels, start=1) if serial not in removed]
        a_direct = np.asarray([xyz for serial, xyz in enumerate(direct, start=1) if serial not in removed])
        write_inputs(OUT / f"a_without_O{o_serial}", f"a_without_O{o_serial}", cell, source_species, a_labels, a_direct, blocks)

        b_serials = [h1, h2, o_serial]
        b_labels = [labels[serial - 1] for serial in b_serials]
        b_direct = np.asarray([direct[serial - 1] for serial in b_serials])
        write_inputs(OUT / f"b_O{o_serial}_H{h1}_H{h2}", f"b_O{o_serial}_H{h1}_H{h2}", cell, ["H", "O"], b_labels, b_direct, blocks)

    (OUT / "README.md").write_text(
        "# D2O-water/system interaction energies\n\n"
        f"Source: `{source_label}`\n"
        f"Source SHA-256: `{source_hash}`\n"
        f"Selected O serials: {', '.join(str(x) for x in TARGET_O)}. D atoms are represented by the existing H species/POTCAR (POMASS=2.000).\n\n"
        "For each selected water, static PBE-D3 calculations use `E_int = E_AB - E_A - E_B`, with the same cell and frozen geometry in all three calculations.\n"
    )
    print(f"Prepared {len(selected)} waters and {1 + 2 * len(selected)} static calculations in {OUT}")
    for row in selected:
        print(f"O{row[0]}: H{row[1]} H{row[2]} (O-H = {row[3]:.4f}, {row[4]:.4f} A)")


if __name__ == "__main__":
    main()
