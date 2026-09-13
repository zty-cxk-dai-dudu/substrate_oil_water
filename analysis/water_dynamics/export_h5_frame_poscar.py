#!/usr/bin/env python3
"""Export one exact HDF5 trajectory frame as a grouped-species POSCAR."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


PREFERRED_ORDER = ("H", "C", "O", "F", "Ca")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--frame", type=int, default=-1)
    parser.add_argument(
        "--selection",
        default="selected frame from the production trajectory",
    )
    args = parser.parse_args()

    with h5py.File(args.input, "r") as handle:
        if not bool(handle.attrs.get("complete", False)):
            raise RuntimeError(f"Incomplete trajectory segment: {args.input}")
        symbols = np.asarray(handle["symbols"]).astype(str)
        positions = np.asarray(handle["positions_A"][args.frame], dtype=float)
        cell = np.asarray(handle["cell_A"], dtype=float)
        step = int(handle["step"][args.frame])
        time_ps = float(handle["time_ps"][args.frame])

    order = [element for element in PREFERRED_ORDER if np.any(symbols == element)]
    unexpected = sorted(set(symbols) - set(order))
    order.extend(unexpected)
    atom_order = np.concatenate([np.flatnonzero(symbols == e) for e in order])
    counts = [int(np.sum(symbols == e)) for e in order]
    fractional = positions[atom_order] @ np.linalg.inv(cell)
    fractional %= 1.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        stream.write(
            f"{args.label} representative final complete frame "
            f"step={step} time={time_ps:.6f} ps\n1.0\n"
        )
        for vector in cell:
            stream.write("  " + "  ".join(f"{x:20.12f}" for x in vector) + "\n")
        stream.write("  " + "  ".join(order) + "\n")
        stream.write("  " + "  ".join(str(value) for value in counts) + "\n")
        stream.write("Direct\n")
        for xyz in fractional:
            stream.write("  " + "  ".join(f"{x:20.12f}" for x in xyz) + "\n")

    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    metadata = {
        "file": args.output.name,
        "source_segment": str(args.input),
        "source_frame_index": args.frame,
        "selection": args.selection,
        "step": step,
        "time_ps": time_ps,
        "atoms": int(len(symbols)),
        "species_order": order,
        "counts": counts,
        "sha256": digest,
        "source_to_poscar_index_1based": (np.argsort(atom_order) + 1).tolist(),
        "poscar_to_source_index_1based": (atom_order + 1).tolist(),
    }
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({key: metadata[key] for key in (
        "file", "step", "time_ps", "atoms", "species_order", "counts", "sha256"
    )}, indent=2))


if __name__ == "__main__":
    main()
