#!/usr/bin/env python3
"""Build mean-O-z metadata and z-first copies of per-water outputs."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

import numpy as np


NAME_RE = re.compile(r"^[0-9]+\.[0-9]{2}_O[0-9]+$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hbond-dir", default=Path("hbonds_last50000_by_O_serial"), type=Path
    )
    parser.add_argument(
        "--vdos-dir", default=Path("vdos_per_water_vaspkit_728"), type=Path
    )
    parser.add_argument(
        "--hbond-z-dir", default=Path("hbonds_last50000_z_named"), type=Path
    )
    parser.add_argument(
        "--vdos-z-dir",
        default=Path("vdos_per_water_vaspkit_728_z_named_txt"),
        type=Path,
    )
    args = parser.parse_args()

    folders = sorted(
        (path for path in args.hbond_dir.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    if not folders:
        raise ValueError(f"No numeric O folders found in {args.hbond_dir}")

    rows: list[dict[str, object]] = []
    for folder in folders:
        coords = np.loadtxt(folder / "water_xyz_cartesian.txt", comments="#")
        if coords.ndim != 2 or coords.shape[1] < 6:
            raise ValueError(f"Unexpected coordinate table in {folder}")
        z_values = coords[:, 5]
        o_serial = int(folder.name)
        directory = f"{float(np.mean(z_values)):.2f}_O{o_serial}"
        if NAME_RE.fullmatch(directory) is None:
            raise ValueError(f"Invalid z-first name: {directory}")
        rows.append(
            {
                "O_serial": o_serial,
                "mean_O_z_A": float(np.mean(z_values)),
                "min_O_z_A": float(np.min(z_values)),
                "max_O_z_A": float(np.max(z_values)),
                "n_frames": int(z_values.size),
                "directory": directory,
            }
        )

    csv_path = args.hbond_dir / "average_O_z_position.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    z_name_by_o = {int(row["O_serial"]): str(row["directory"]) for row in rows}

    if args.hbond_z_dir.exists():
        shutil.rmtree(args.hbond_z_dir)
    if args.vdos_z_dir.exists():
        shutil.rmtree(args.vdos_z_dir)
    args.hbond_z_dir.mkdir(parents=True)
    args.vdos_z_dir.mkdir(parents=True)

    for row in rows:
        o_serial = int(row["O_serial"])
        directory = str(row["directory"])
        shutil.copytree(args.hbond_dir / str(o_serial), args.hbond_z_dir / directory)
        source_vdos = args.vdos_dir / f"O{o_serial}.txt"
        shutil.copy2(source_vdos, args.vdos_z_dir / f"{directory}.txt")

    for source_name in (
        "kept_hbonds_strict_ge15pct.csv",
        "removed_hbonds_strict_lt15pct.csv",
    ):
        source = args.hbond_dir / source_name
        if not source.exists():
            continue
        with source.open(newline="") as handle:
            strict_rows = list(csv.DictReader(handle))
        output = args.hbond_z_dir / source_name.replace(".csv", "_z_named.csv")
        fieldnames = ["z_named_O"] + list(strict_rows[0]) if strict_rows else ["z_named_O"]
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for strict_row in strict_rows:
                writer.writerow(
                    {
                        "z_named_O": z_name_by_o[int(strict_row["O_folder"])],
                        **strict_row,
                    }
                )

    print(
        f"Wrote {len(rows)} z rows, {len(rows)} H-bond directories, "
        f"and {len(rows)} VDOS txt files"
    )


if __name__ == "__main__":
    main()
