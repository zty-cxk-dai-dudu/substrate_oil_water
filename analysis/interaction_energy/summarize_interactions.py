#!/usr/bin/env python3
"""Calculate interaction energies from a prepared VASP calculation directory."""

import argparse
import importlib.util
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True,
                        help='Directory containing selected_waters.tsv and the component calculations')
    args = parser.parse_args()
    source = Path(__file__).resolve().parent / 'caf2/O130_O137_O151_interaction/finalize_results.py'
    spec = importlib.util.spec_from_file_location('interaction_energy', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = args.input.resolve()
    module.main()


if __name__ == '__main__':
    main()
