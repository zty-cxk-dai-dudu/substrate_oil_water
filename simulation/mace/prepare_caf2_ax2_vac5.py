#!/usr/bin/env python3
"""Build the requested CaF2/oil/water a*2, top-vacuum +5 A initial cell."""

from pathlib import Path

from ase.io import read, write


SOURCE = Path("/mnt/d/jacs/caf2_mace_weight0p25_20260805/conf.lmp")
OUT = Path("/mnt/d/jacs/caf2_mace_ax2_vac5_cueq_500ps_20260808")
Z_OF_TYPE = {1: 1, 2: 6, 3: 8, 4: 9, 5: 20}


atoms = read(SOURCE, format="lammps-data", atom_style="atomic", Z_of_type=Z_OF_TYPE)
atoms = atoms.repeat((2, 1, 1))
cell = atoms.cell.array.copy()
old_c = float((cell[2] @ cell[2]) ** 0.5)
cell[2] *= (old_c + 5.0) / old_c
atoms.set_cell(cell, scale_atoms=False)
atoms.wrap()

OUT.mkdir(parents=True, exist_ok=True)
write(OUT / "initial_from_original_conf_ax2_vac5A.extxyz", atoms, format="extxyz")
write(OUT / "POSCAR_initial_from_original_conf_ax2_vac5A", atoms, format="vasp", direct=True, sort=True)

print(f"source={SOURCE}")
print(f"atoms={len(atoms)} formula={atoms.get_chemical_formula()}")
print("cell_A=")
print(atoms.cell.array)
print(f"z_range_A={atoms.positions[:, 2].min():.8f}..{atoms.positions[:, 2].max():.8f}")
print(f"top_vacuum_added_A=5.0 old_c_A={old_c:.8f} new_c_A={atoms.cell.lengths()[2]:.8f}")
