#!/usr/bin/env python3
"""Filter reformatted H-bond files by strict per-frame hbond occupancy."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_metadata(path: Path) -> tuple[int, list[int]]:
    metadata = json.loads(path.read_text())
    selected_frames = int(metadata["selected_frames"])
    o_serials = [int(row["O_serial"]) for row in metadata["water_topology"]]
    return selected_frames, o_serials


def collect_strict_counts(detail_dir: Path, o_serials: list[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for o_serial in o_serials:
        folder = detail_dir / f"O_{o_serial}"
        # Use only the acceptor-side file. The donor-side files contain the same
        # H...O pair again, so reading both would double-count strict frames.
        path = folder / "O_acceptor_hbonds.txt"
        with path.open("r", errors="ignore") as handle:
            for line in handle:
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 9:
                    continue
                donor_h = int(parts[4])
                acceptor_o = int(parts[5])
                is_hbond = int(parts[8])
                key = f"H{donor_h}_O{acceptor_o}.txt"
                counts[key] = counts.get(key, 0) + is_hbond
    return counts


def remove_stale_outputs(base: Path) -> None:
    for path in base.glob("*/hbond_stability.csv"):
        path.unlink()
    for path in base.glob("*/hbond_stability.svg"):
        path.unlink()
    for path in base.glob("*/hbond_stability.png"):
        path.unlink()
    for name in ("hbond_stability_all.csv",):
        path = base / name
        if path.exists():
            path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detail-dir", default="hbonds_last50000", type=Path)
    parser.add_argument("--reformatted-dir", default="hbonds_last50000_by_O_serial", type=Path)
    parser.add_argument("--min-fraction", default=0.15, type=float)
    args = parser.parse_args()

    selected_frames, o_serials = load_metadata(args.detail_dir / "metadata.json")
    strict_counts = collect_strict_counts(args.detail_dir, o_serials)
    min_frames = int(args.min_fraction * selected_frames)

    removed: list[dict[str, object]] = []
    kept: list[dict[str, object]] = []
    for folder in sorted(
        [p for p in args.reformatted_dir.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    ):
        for path in sorted(folder.glob("H*_O*.txt")):
            strict_frames = strict_counts.get(path.name, 0)
            row = {
                "O_folder": folder.name,
                "hbond": path.stem,
                "strict_hbond_frames": strict_frames,
                "total_frames": selected_frames,
                "strict_hbond_fraction": strict_frames / selected_frames,
            }
            if strict_frames < min_frames:
                path.unlink()
                removed.append(row)
            else:
                kept.append(row)

    remove_stale_outputs(args.reformatted_dir)

    for filename, rows in (
        ("removed_hbonds_strict_lt15pct.csv", removed),
        ("kept_hbonds_strict_ge15pct.csv", kept),
    ):
        with (args.reformatted_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "O_folder",
                    "hbond",
                    "strict_hbond_frames",
                    "total_frames",
                    "strict_hbond_fraction",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    with (args.reformatted_dir / "summary_after_strict_15pct_filter.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["O_serial", "kept_hbond_files"])
        for o_serial in o_serials:
            folder = args.reformatted_dir / str(o_serial)
            writer.writerow([o_serial, len(list(folder.glob("H*_O*.txt")))])

    print(
        f"Done. Strict criterion: H...O<=2.45 A, O...O<=3.50 A, angle>=150 deg. "
        f"Removed {len(removed)} files with fraction < {args.min_fraction:.2f} "
        f"({min_frames}/{selected_frames} frames); kept {len(kept)} files."
    )


if __name__ == "__main__":
    main()
