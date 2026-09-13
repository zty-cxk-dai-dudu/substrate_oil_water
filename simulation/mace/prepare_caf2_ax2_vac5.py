#!/usr/bin/env python3
"""Build the CaF2/oil/water a*2, top-vacuum +5 A initial cell."""

import argparse
from pathlib import Path

from ase.io import read, write


Z_OF_TYPE = {1: 1, 2: 6, 3: 8, 4: 9, 5: 20}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Atomic-style LAMMPS data file (H, C, O, F, Ca types 1-5)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Directory for the initial extxyz and POSCAR")
    args = parser.parse_args()
    atoms = read(args.source, format="lammps-data", atom_style="atomic", Z_of_type=Z_OF_TYPE)
    atoms = atoms.repeat((2, 1, 1))
    cell = atoms.cell.array.copy()
    old_c = float((cell[2] @ cell[2]) ** 0.5)
    if old_c <= 0:
        parser.error("source cell must have a positive c-vector length")
    cell[2] *= (old_c + 5.0) / old_c
    atoms.set_cell(cell, scale_atoms=False)
    atoms.wrap()

    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output / "initial_from_original_conf_ax2_vac5A.extxyz", atoms, format="extxyz")
    write(args.output / "POSCAR_initial_from_original_conf_ax2_vac5A", atoms,
          format="vasp", direct=True, sort=True)

    print(f"source={args.source.resolve()}")
    print(f"atoms={len(atoms)} formula={atoms.get_chemical_formula()}")
    print("cell_A=")
    print(atoms.cell.array)
    print(f"z_range_A={atoms.positions[:, 2].min():.8f}..{atoms.positions[:, 2].max():.8f}")
    print(f"top_vacuum_added_A=5.0 old_c_A={old_c:.8f} new_c_A={atoms.cell.lengths()[2]:.8f}")


if __name__ == "__main__":
    main()
