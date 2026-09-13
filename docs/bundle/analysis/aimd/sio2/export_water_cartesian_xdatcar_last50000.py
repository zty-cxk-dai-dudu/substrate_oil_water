#!/usr/bin/env python3
"""Export Cartesian coordinates for each water molecule in the last XDATCAR window."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


CONFIG_RE = re.compile(r"configuration\s*=\s*(?P<step>-?\d+)", re.I)


def parse_poscar_cell(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    scale = float(lines[1].split()[0])
    cell = np.array([[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)])
    return cell * scale


def parse_step(line: str) -> int | None:
    match = CONFIG_RE.search(line)
    return None if match is None else int(match.group("step"))


def iter_xdatcar_frames(
    path: Path,
    cell: np.ndarray,
    natoms: int,
    start_frame: int,
    max_frames: int,
):
    with path.open("r", errors="ignore") as handle:
        for _ in range(7):
            handle.readline()
        frame_no = 0
        yielded = 0
        while yielded < max_frames:
            config_line = handle.readline()
            if not config_line:
                break
            if not config_line.strip():
                break
            frac = np.empty((natoms, 3), dtype=np.float64)
            for idx in range(natoms):
                line = handle.readline()
                if not line:
                    raise ValueError("Unexpected EOF inside XDATCAR frame.")
                parts = line.split()
                if len(parts) < 3:
                    raise ValueError(f"Bad XDATCAR coordinate line: {line!r}")
                frac[idx] = (float(parts[0]), float(parts[1]), float(parts[2]))
            if frame_no >= start_frame:
                yield frame_no, yielded, parse_step(config_line), None, frac @ cell
                yielded += 1
            frame_no += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xdatcar", default="XDATCAR", type=Path)
    parser.add_argument("--poscar", default="POSCAR", type=Path)
    parser.add_argument(
        "--analysis-metadata", default="hbonds_last50000/metadata.json", type=Path
    )
    parser.add_argument("--output-dir", default="hbonds_last50000_by_O_serial", type=Path)
    parser.add_argument("--coord-filename", default="water_xyz_cartesian.txt")
    args = parser.parse_args()

    metadata = json.loads(args.analysis_metadata.read_text())
    start_frame = int(metadata["start_frame_index_0_based"])
    selected_frames = int(metadata["selected_frames"])
    natoms = int(metadata["natoms"])
    topology = metadata["water_topology"]
    cell = parse_poscar_cell(args.poscar)

    handles: dict[int, object] = {}
    try:
        for water in topology:
            o_serial = int(water["O_serial"])
            folder = args.output_dir / str(o_serial)
            folder.mkdir(parents=True, exist_ok=True)
            handle = (folder / args.coord_filename).open("w")
            handle.write(
                "# frame_0_based step time_fs "
                "O_x O_y O_z H1_x H1_y H1_z H2_x H2_y H2_z\n"
            )
            handles[o_serial] = handle

        serial_triplets = [
            (
                int(water["O_serial"]),
                int(water["H1_serial"]),
                int(water["H2_serial"]),
            )
            for water in topology
        ]

        written_frames = 0
        for _abs_frame, rel_frame, step, time_fs, coords in iter_xdatcar_frames(
            args.xdatcar, cell, natoms, start_frame, selected_frames
        ):
            step_s = "" if step is None else str(step)
            time_s = step_s if time_fs is None else f"{time_fs:.6f}"
            for o_serial, h1_serial, h2_serial in serial_triplets:
                o = coords[o_serial - 1]
                h1 = coords[h1_serial - 1]
                h2 = coords[h2_serial - 1]
                handles[o_serial].write(
                    f"{rel_frame} {step_s} {time_s} "
                    f"{o[0]:.8f} {o[1]:.8f} {o[2]:.8f} "
                    f"{h1[0]:.8f} {h1[1]:.8f} {h1[2]:.8f} "
                    f"{h2[0]:.8f} {h2[1]:.8f} {h2[2]:.8f}\n"
                )
            written_frames += 1
            if written_frames % 5000 == 0:
                print(f"wrote {written_frames}/{selected_frames} frames", flush=True)

        if written_frames != selected_frames:
            raise ValueError(
                f"Expected {selected_frames} frames, wrote {written_frames} frames."
            )
    finally:
        for handle in handles.values():
            handle.close()

    coord_meta = {
        "source_xdatcar": str(args.xdatcar),
        "source_poscar": str(args.poscar),
        "analysis_metadata": str(args.analysis_metadata),
        "output_filename_per_O_folder": args.coord_filename,
        "selected_frames": selected_frames,
        "frame_numbering": "0_to_49999",
        "columns": [
            "frame_0_based",
            "step",
            "time_fs",
            "O_x",
            "O_y",
            "O_z",
            "H1_x",
            "H1_y",
            "H1_z",
            "H2_x",
            "H2_y",
            "H2_z",
        ],
        "coordinate_unit": "Angstrom",
        "coordinate_type": "Cartesian",
    }
    (args.output_dir / "water_xyz_cartesian_metadata.json").write_text(
        json.dumps(coord_meta, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"Done. Wrote {len(topology)} water coordinate files to {args.output_dir}")


if __name__ == "__main__":
    main()
