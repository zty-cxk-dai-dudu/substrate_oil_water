#!/usr/bin/env python3
"""Remove H-bond distance files whose stable-frame count is below a threshold."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def stable_frame_count(path: Path, cutoff: float) -> tuple[int, int]:
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"Expected two columns in {path}")
    distances = data[:, 1]
    return int(np.count_nonzero(distances <= cutoff)), int(distances.size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="hbonds_last50000_by_O_serial", type=Path)
    parser.add_argument("--cutoff", default=2.45, type=float)
    parser.add_argument("--min-stable-frames", default=100, type=int)
    args = parser.parse_args()

    folders = sorted(
        [p for p in args.input_dir.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )

    removed_rows: list[dict[str, object]] = []
    kept_rows: list[dict[str, object]] = []

    for folder in folders:
        for stale_name in ("hbond_stability.svg", "hbond_stability.png", "hbond_stability.csv"):
            stale_path = folder / stale_name
            if stale_path.exists():
                stale_path.unlink()

        for path in sorted(folder.glob("H*_O*.txt")):
            stable_count, total_count = stable_frame_count(path, args.cutoff)
            row = {
                "O_folder": folder.name,
                "hbond": path.stem,
                "stable_frames": stable_count,
                "total_frames": total_count,
                "stable_fraction": stable_count / total_count if total_count else 0.0,
            }
            if stable_count < args.min_stable_frames:
                path.unlink()
                removed_rows.append(row)
            else:
                kept_rows.append(row)

    for stale_name in ("hbond_stability_all.csv",):
        stale_path = args.input_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    with (args.input_dir / "removed_hbonds_lt100_frames.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["O_folder", "hbond", "stable_frames", "total_frames", "stable_fraction"],
        )
        writer.writeheader()
        writer.writerows(removed_rows)

    with (args.input_dir / "kept_hbonds_ge100_frames.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["O_folder", "hbond", "stable_frames", "total_frames", "stable_fraction"],
        )
        writer.writeheader()
        writer.writerows(kept_rows)

    with (args.input_dir / "summary_after_filter.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["O_serial", "kept_hbond_files"])
        for folder in folders:
            writer.writerow([folder.name, len(list(folder.glob("H*_O*.txt")))])

    print(
        f"Done. Removed {len(removed_rows)} files with stable_frames < "
        f"{args.min_stable_frames}; kept {len(kept_rows)} files."
    )


if __name__ == "__main__":
    main()
