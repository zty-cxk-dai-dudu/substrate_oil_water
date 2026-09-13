#!/usr/bin/env python3
"""Prepare full/A/B CP2K single points with identical grids and ghost bases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np


OH_CUTOFF_A = 1.25


def read_poscar(path: Path):
    lines = path.read_text().splitlines()
    scale = float(lines[1].split()[0])
    cell = np.array([[float(x) for x in lines[i].split()] for i in range(2, 5)]) * scale
    elements = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    cursor = 7
    if lines[cursor].strip().lower().startswith("s"):
        cursor += 1
    mode = lines[cursor].strip().lower()
    cursor += 1
    coordinates = np.array(
        [[float(x) for x in lines[cursor + i].split()[:3]] for i in range(sum(counts))]
    )
    if mode.startswith("c"):
        cart = coordinates * scale
        frac = cart @ np.linalg.inv(cell)
    else:
        frac = coordinates
        cart = frac @ cell
    return cell, elements, counts, frac, cart


def distances(frac_a, frac_b, cell):
    delta = frac_a[:, None, :] - frac_b[None, :, :]
    delta -= np.rint(delta)
    return np.linalg.norm(delta @ cell, axis=2)


def kind_block(kind: str, element: str, ghost: bool = False) -> str:
    lines = [
        f"    &KIND {kind}",
        f"      ELEMENT {element}",
        "      BASIS_SET DZVP-MOLOPT-SR-GTH",
    ]
    if ghost:
        lines.append("      GHOST TRUE")
    else:
        lines.append("      POTENTIAL GTH-PBE")
    lines.append("    &END KIND")
    return "\n".join(lines)


def make_input(project: str, labels, cart, cell, print_density: bool = True,
               max_scf: int = 200, ignore_failure: bool = False,
               data_dir: Path | str = ".") -> str:
    used = []
    for label in labels:
        if label not in used:
            used.append(label)
    kind_specs = {
        "O": ("O", False), "Si": ("Si", False), "H": ("H", False), "C": ("C", False),
        "OG": ("O", True), "SiG": ("Si", True), "HG": ("H", True), "CG": ("C", True),
    }
    kinds = "\n".join(kind_block(label, *kind_specs[label]) for label in used)
    coords = "\n".join(
        f"      {label:<3s} {xyz[0]:.15f} {xyz[1]:.15f} {xyz[2]:.15f}"
        for label, xyz in zip(labels, cart)
    )
    density = ""
    if print_density:
        density = """
      &E_DENSITY_CUBE ON
        FILENAME =electron_density.cube
        STRIDE 1 1 1
        ADD_LAST NO
      &END E_DENSITY_CUBE"""
    ignore = "\n      IGNORE_CONVERGENCE_FAILURE TRUE" if ignore_failure else ""
    return f"""&GLOBAL
  PROJECT {project}
  RUN_TYPE ENERGY
  PRINT_LEVEL MEDIUM
&END GLOBAL

&FORCE_EVAL
  METHOD Quickstep
  &DFT
    BASIS_SET_FILE_NAME "{data_dir}/BASIS_MOLOPT"
    POTENTIAL_FILE_NAME "{data_dir}/GTH_POTENTIALS"
    CHARGE 0
    MULTIPLICITY 1
    &QS
      METHOD GPW
      EPS_DEFAULT 1.0E-12
    &END QS
    &MGRID
      NGRIDS 4
      CUTOFF 280
      REL_CUTOFF 30
    &END MGRID
    &SCF
      SCF_GUESS ATOMIC
      EPS_SCF 1.0E-6
      EPS_DIIS 1.0E-5
      MAX_SCF {max_scf}{ignore}
      &OT
        PRECONDITIONER FULL_KINETIC
        MINIMIZER DIIS
        MAX_SCF_DIIS 100
        LINESEARCH 3PNT
        STEPSIZE 0.05
      &END OT
    &END SCF
    &XC
      &XC_FUNCTIONAL PBE
      &END XC_FUNCTIONAL
    &END XC
    &POISSON
      PERIODIC XYZ
      POISSON_SOLVER PERIODIC
    &END POISSON
    &PRINT{density}
    &END PRINT
  &END DFT
  &SUBSYS
    &CELL
      A {cell[0, 0]:.15f} {cell[0, 1]:.15f} {cell[0, 2]:.15f}
      B {cell[1, 0]:.15f} {cell[1, 1]:.15f} {cell[1, 2]:.15f}
      C {cell[2, 0]:.15f} {cell[2, 1]:.15f} {cell[2, 2]:.15f}
      PERIODIC XYZ
    &END CELL
    &COORD
{coords}
    &END COORD
{kinds}
  &END SUBSYS
&END FORCE_EVAL
"""


def as_preflight(text: str, project: str) -> str:
    """Turn a production input into one real SCF iteration without cube output."""
    text = re.sub(r"(?m)^  PROJECT \S+$", f"  PROJECT {project}", text, count=1)
    text = text.replace(
        "      MAX_SCF 200\n",
        "      MAX_SCF 1\n      IGNORE_CONVERGENCE_FAILURE TRUE\n",
        1,
    )
    text = re.sub(
        r"(?ms)^      &E_DENSITY_CUBE ON\n.*?^      &END E_DENSITY_CUBE\n",
        "",
        text,
        count=1,
    )
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="820-atom SiO2/oil/water POSCAR (O296 Si48 H452 C24)")
    parser.add_argument("--out", type=Path, required=True, help="Directory for prepared full/fragment inputs")
    parser.add_argument("--cp2k-data", type=Path, default=os.environ.get("CP2K_DATA_DIR"), help="Directory containing BASIS_MOLOPT and GTH_POTENTIALS (or set CP2K_DATA_DIR)")
    args = parser.parse_args()
    if args.cp2k_data is None:
        parser.error("provide --cp2k-data or set CP2K_DATA_DIR")
    source = args.source.resolve()
    root = args.out.resolve()
    data_dir = Path(args.cp2k_data).resolve()
    for name in ("BASIS_MOLOPT", "GTH_POTENTIALS"):
        if not (data_dir / name).is_file():
            parser.error(f"missing CP2K data file: {data_dir / name}")
    if any((root / name).exists() for name in ("ab_full", "a_sio2oil", "b_h2o")):
        parser.error("output calculation directories already exist; choose a new --out directory")
    cell, elements, counts, frac, cart = read_poscar(source)
    if elements != ["O", "Si", "H", "C"] or counts != [296, 48, 452, 24]:
        raise RuntimeError(f"Unexpected POSCAR composition: {elements} {counts}")
    offsets = np.cumsum([0] + counts)
    slices = {e: np.arange(offsets[i], offsets[i + 1]) for i, e in enumerate(elements)}
    oh = distances(frac[slices["O"]], frac[slices["H"]], cell)
    hits = oh <= OH_CUTOFF_A
    o_coordination = hits.sum(axis=1)
    if dict(zip(*np.unique(o_coordination, return_counts=True))) != {0: 96, 2: 200}:
        raise RuntimeError("O-H topology is not exactly SiO2 O96 plus H2O O200")
    if np.any(hits.sum(axis=0) > 1):
        raise RuntimeError("A H is ambiguously assigned to more than one O")
    water_o_local = np.where(o_coordination == 2)[0]
    water_h_local = np.where(hits[water_o_local].any(axis=0))[0]
    if len(water_h_local) != 400:
        raise RuntimeError("Expected 400 water H")
    water = np.zeros(sum(counts), dtype=bool)
    water[slices["O"][water_o_local]] = True
    water[slices["H"][water_h_local]] = True

    atomic_elements = []
    for element, indices in slices.items():
        atomic_elements.extend([element] * len(indices))
    full_labels = atomic_elements
    a_labels = [f"{element}G" if water[i] else element for i, element in enumerate(atomic_elements)]
    b_labels = [element if water[i] else f"{element}G" for i, element in enumerate(atomic_elements)]

    (root / "ab_full").mkdir(parents=True, exist_ok=False)
    (root / "a_sio2oil").mkdir(parents=True, exist_ok=False)
    (root / "b_h2o").mkdir(parents=True, exist_ok=False)
    shutil.copy2(source, root / "POSCAR.source")
    ab_input = make_input("ab_full", full_labels, cart, cell, data_dir=data_dir)
    a_input = make_input("a_sio2oil", a_labels, cart, cell, data_dir=data_dir)
    b_input = make_input("b_h2o", b_labels, cart, cell, data_dir=data_dir)
    (root / "ab_full" / "input.inp").write_text(ab_input)
    (root / "a_sio2oil" / "input.inp").write_text(a_input)
    (root / "b_h2o" / "input.inp").write_text(b_input)
    (root / "preflight.inp").write_text(
        make_input("preflight", full_labels, cart, cell, print_density=False,
                   max_scf=1, ignore_failure=True, data_dir=data_dir)
    )
    (root / "preflight_a_sio2oil.inp").write_text(as_preflight(a_input, "preflight_a"))
    (root / "preflight_b_h2o.inp").write_text(as_preflight(b_input, "preflight_b"))

    with (root / "atom_partition.tsv").open("w") as handle:
        handle.write("global_serial\telement\tcomponent\tfull_kind\ta_kind\tb_kind\tx_A\ty_A\tz_A\n")
        for i, (element, xyz) in enumerate(zip(atomic_elements, cart)):
            component = "h2o" if water[i] else "sio2oil"
            handle.write(
                f"{i + 1}\t{element}\t{component}\t{full_labels[i]}\t{a_labels[i]}\t"
                f"{b_labels[i]}\t{xyz[0]:.15f}\t{xyz[1]:.15f}\t{xyz[2]:.15f}\n"
            )
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    summary = {
        "source_poscar": str(source),
        "source_sha256": source_hash,
        "cp2k_data_directory": str(data_dir),
        "cp2k_data_sha256": {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest() for name in ("BASIS_MOLOPT", "GTH_POTENTIALS")},
        "method": "CP2K 2026.1 Quickstep GPW PBE",
        "basis": "DZVP-MOLOPT-SR-GTH",
        "potential": "GTH-PBE",
        "cutoff_Ry": 280,
        "rel_cutoff_Ry": 30,
        "cube_stride": [1, 1, 1],
        "difference": "rho(ab_full)-rho(a_sio2oil)-rho(b_h2o)",
        "ghost_basis": True,
        "full_atoms": 820,
        "a_sio2oil_real_atoms": int((~water).sum()),
        "b_h2o_real_atoms": int(water.sum()),
        "h2o_formula": {"O": 200, "H": 400},
        "sio2oil_formula": {"O": 96, "Si": 48, "H": 52, "C": 24},
        "water_oh_min_A": float(oh[hits].min()),
        "water_oh_max_A": float(oh[hits].max()),
    }
    (root / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    (root / "source_sha256.txt").write_text(f"{source_hash}  POSCAR.source\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
