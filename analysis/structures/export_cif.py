#!/usr/bin/env python3
"""Convert a periodic atomic structure to CIF using ASE."""

import argparse
from pathlib import Path

import numpy as np
from ase.io import read, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--format', default=None, help='ASE input format, e.g. vasp or lammps-data')
    parser.add_argument('--index', type=int, default=-1, help='Frame index, starting from zero')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('The output file already exists. Choose a new filename.')
    kwargs = {'atom_style': 'atomic', 'sort_by_id': True} if args.format == 'lammps-data' else {}
    atoms = read(args.input, format=args.format, index=args.index, **kwargs)
    if not np.isfinite(atoms.positions).all() or atoms.cell.volume <= 0:
        parser.error('A finite periodic cell and finite atomic positions are required.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    labels = [[f'{symbol}{i + 1}' for i, symbol in enumerate(atoms.get_chemical_symbols())]]
    write(args.output, atoms, format='cif', wrap=True, labels=labels)
    print(f'{args.output}: {len(atoms)} atoms, {atoms.get_chemical_formula()}')


if __name__ == '__main__':
    main()
