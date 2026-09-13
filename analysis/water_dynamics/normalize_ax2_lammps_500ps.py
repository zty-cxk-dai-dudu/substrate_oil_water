#!/usr/bin/env python3
"""Merge the 0--115 ps ASE HDF5 prefix and 115--500 ps LAMMPS dump.

The normalized trajectory contains exactly 100 complete 5 ps HDF5 segments,
with wrapped Cartesian positions in angstrom and velocities in angstrom/fs.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np


NUMBER_BY_SYMBOL = {"H": 1, "C": 6, "O": 8, "F": 9, "Ca": 20}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def complete_h5(path: Path, frames: int = 5000) -> bool:
    try:
        with h5py.File(path, "r") as handle:
            return (
                bool(handle.attrs.get("complete", False))
                and int(handle["positions_A"].shape[0]) == frames
                and int(handle["velocities_A_per_fs"].shape[0]) == frames
            )
    except OSError:
        return False


def write_segment(
    path: Path,
    index: int,
    steps: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    cell: np.ndarray,
    symbols: np.ndarray,
    numbers: np.ndarray,
    atom_types: np.ndarray,
) -> None:
    partial = path.with_suffix(".partial.h5")
    if partial.exists():
        partial.unlink()
    with h5py.File(partial, "w") as handle:
        handle.create_dataset("step", data=steps)
        handle.create_dataset("time_ps", data=steps.astype(np.float64) * 0.0005)
        handle.create_dataset("cell_A", data=cell)
        handle.create_dataset("symbols", data=np.asarray(symbols, dtype="S2"))
        handle.create_dataset("atomic_numbers", data=numbers.astype(np.int16))
        handle.create_dataset("type", data=atom_types.astype(np.int8))
        handle.create_dataset(
            "positions_A", data=positions.astype(np.float32), chunks=(64, len(symbols), 3),
            compression="gzip", compression_opts=1, shuffle=True,
        )
        handle.create_dataset(
            "velocities_A_per_fs", data=velocities.astype(np.float32),
            chunks=(64, len(symbols), 3), compression="gzip", compression_opts=1,
            shuffle=True,
        )
        handle.attrs.update({
            "complete": True,
            "completed_frames": len(steps),
            "dt_fs": 0.5,
            "output_interval_steps": 2,
            "output_interval_fs": 1.0,
            "first_absolute_step": int(steps[0]),
            "segment_index": index,
            "positions_unit": "angstrom",
            "velocities_unit": "angstrom/fs",
            "time_unit": "ps",
            "source_engine": "LAMMPS MLIAP MACE",
            "source_velocity_unit": "angstrom/ps",
            "velocity_conversion_to_stored": 0.001,
        })
    os.replace(partial, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)

    prefix = sorted(
        path for path in args.prefix.glob("segment_*.h5")
        if ".partial" not in path.name and ".abandoned" not in path.name
    )
    if len(prefix) != 23:
        raise RuntimeError(f"expected 23 complete ASE prefix segments, found {len(prefix)}")
    for index, source in enumerate(prefix):
        if not complete_h5(source):
            raise RuntimeError(f"invalid prefix segment: {source}")
        target = args.output / source.name
        if target.is_symlink() and target.resolve() == source.resolve():
            continue
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(source)

    with h5py.File(prefix[0], "r") as first:
        symbols = np.asarray(first["symbols"]).astype(str)
        numbers = np.asarray(first["atomic_numbers"], dtype=np.int16)
        atom_types = np.asarray(first["type"], dtype=np.int8)
        cell = np.asarray(first["cell_A"], dtype=np.float64)
    with h5py.File(prefix[-1], "r") as last:
        boundary_positions = np.asarray(last["positions_A"][-1], dtype=np.float64)
        boundary_velocities = np.asarray(last["velocities_A_per_fs"][-1], dtype=np.float64)
        boundary_step = int(last["step"][-1])
    if boundary_step != 230000:
        raise RuntimeError(f"unexpected ASE boundary step: {boundary_step}")

    natoms = len(symbols)
    expected_ids = np.arange(1, natoms + 1)
    expected_step = boundary_step
    dump_frames = 0
    written_frames = 0
    segment_index = 23
    steps_buffer: list[int] = []
    positions_buffer: list[np.ndarray] = []
    velocities_buffer: list[np.ndarray] = []
    boundary_position_rms_A = None
    boundary_velocity_rms_A_per_fs = None
    box_lower = None

    with gzip.open(args.dump, "rt") as stream:
        while True:
            line = stream.readline()
            if not line:
                break
            if line.strip() != "ITEM: TIMESTEP":
                raise RuntimeError(f"malformed dump before frame {dump_frames}: {line!r}")
            step = int(stream.readline())
            if stream.readline().strip() != "ITEM: NUMBER OF ATOMS":
                raise RuntimeError("missing atom-count header")
            count = int(stream.readline())
            if count != natoms:
                raise RuntimeError(f"atom count changed at step {step}: {count}")
            if not stream.readline().startswith("ITEM: BOX BOUNDS"):
                raise RuntimeError("missing box header")
            bounds = np.loadtxt(stream, max_rows=3, usecols=(0, 1), dtype=np.float64)
            lower = bounds[:, 0]
            this_cell = np.diag(bounds[:, 1] - bounds[:, 0])
            if not np.allclose(this_cell, cell, atol=1.0e-7):
                raise RuntimeError(f"cell changed at step {step}")
            atom_header = stream.readline().strip()
            if atom_header != "ITEM: ATOMS id type element xu yu zu vx vy vz":
                raise RuntimeError(f"unexpected atom columns: {atom_header}")
            atom_data = np.loadtxt(
                stream, max_rows=natoms, usecols=(0, 1, 3, 4, 5, 6, 7, 8),
                dtype=np.float64,
            )
            order = np.argsort(atom_data[:, 0].astype(np.int64))
            atom_data = atom_data[order]
            ids = atom_data[:, 0].astype(np.int64)
            types = atom_data[:, 1].astype(np.int8)
            if not np.array_equal(ids, expected_ids) or not np.array_equal(types, atom_types):
                raise RuntimeError(f"atom identity/order mismatch at step {step}")
            if step != expected_step:
                raise RuntimeError(f"step discontinuity: expected {expected_step}, got {step}")
            expected_step += 2
            positions = (atom_data[:, 2:5] - lower) % np.diag(cell)
            velocities = atom_data[:, 5:8] * 0.001
            dump_frames += 1

            if dump_frames == 1:
                delta = positions - boundary_positions
                delta -= np.diag(cell) * np.rint(delta / np.diag(cell))
                boundary_position_rms_A = float(np.sqrt(np.mean(delta ** 2)))
                boundary_velocity_rms_A_per_fs = float(
                    np.sqrt(np.mean((velocities - boundary_velocities) ** 2))
                )
                box_lower = lower.tolist()
                if boundary_position_rms_A > 2.0e-4 or boundary_velocity_rms_A_per_fs > 2.0e-5:
                    raise RuntimeError(
                        "ASE/LAMMPS boundary mismatch: "
                        f"position RMS={boundary_position_rms_A:g} A, "
                        f"velocity RMS={boundary_velocity_rms_A_per_fs:g} A/fs"
                    )
                continue

            steps_buffer.append(step)
            positions_buffer.append(positions.astype(np.float32))
            velocities_buffer.append(velocities.astype(np.float32))
            if len(steps_buffer) == 5000:
                target = args.output / (
                    f"segment_{segment_index:04d}_step{steps_buffer[0]:07d}-"
                    f"{steps_buffer[-1]:07d}.h5"
                )
                if not complete_h5(target):
                    write_segment(
                        target, segment_index, np.asarray(steps_buffer, dtype=np.int64),
                        np.asarray(positions_buffer), np.asarray(velocities_buffer), cell,
                        symbols, numbers, atom_types,
                    )
                written_frames += len(steps_buffer)
                segment_index += 1
                steps_buffer.clear()
                positions_buffer.clear()
                velocities_buffer.clear()
                print(json.dumps({
                    "segments_complete": segment_index,
                    "frames_complete": 23 * 5000 + written_frames,
                    "last_step": step,
                    "elapsed_s": round(time.time() - started, 1),
                }), flush=True)

    if steps_buffer:
        raise RuntimeError(f"trailing incomplete normalized segment: {len(steps_buffer)} frames")
    if dump_frames != 385001 or written_frames != 385000 or segment_index != 100:
        raise RuntimeError(
            f"unexpected totals: dump={dump_frames}, written={written_frames}, "
            f"segments={segment_index}"
        )
    all_segments = sorted(args.output.glob("segment_*.h5"))
    if len(all_segments) != 100 or any(not complete_h5(path) for path in all_segments):
        raise RuntimeError("normalized 100-segment audit failed")

    metadata = {
        "status": "complete",
        "analysis_trajectory": "continuous a*2 CaF2/oil/water 0-500 ps trajectory",
        "prefix_source": str(args.prefix),
        "prefix_segments": 23,
        "prefix_frames": 115000,
        "lammps_dump": str(args.dump),
        "lammps_dump_sha256": sha256(args.dump),
        "lammps_dump_frames_including_boundary_duplicate": dump_frames,
        "lammps_frames_used_after_boundary": written_frames,
        "boundary_duplicate_step": boundary_step,
        "boundary_position_RMS_A": boundary_position_rms_A,
        "boundary_velocity_RMS_A_per_fs": boundary_velocity_rms_A_per_fs,
        "dump_box_lower_A": box_lower,
        "velocity_conversion": "LAMMPS metal vx/vy/vz divided by 1000 from A/ps to A/fs",
        "normalized_segments": 100,
        "normalized_frames": 500000,
        "time_ps": [0.001, 500.0],
        "sampling_interval_fs": 1.0,
        "atoms": natoms,
        "composition": {
            element: int(np.sum(symbols == element)) for element in np.unique(symbols)
        },
        "cell_A": cell.tolist(),
        "elapsed_s": time.time() - started,
    }
    (args.output / "normalization_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
