#!/usr/bin/env python3
"""
Reformat hydrogen-bond output into one distance-vs-time file per H...O pair.

Input is the detailed output from analyze_hbonds_last50000.py. Output folders
are named only by the water oxygen atom serial, and each txt file contains only:

  frame_0_based distance_A
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def load_metadata(path: Path) -> dict:
    return json.loads(path.read_text())


def water_serials(metadata: dict) -> list[int]:
    return [int(item["O_serial"]) for item in metadata["water_topology"]]


def reformat_file(
    input_path: Path,
    output_folder: Path,
    start_frame: int,
    selected_frames: int,
    expected_this_o: int,
    pair_counts: dict[str, int],
) -> None:
    handles: dict[str, object] = {}
    seen_frames: dict[str, int] = {}
    try:
        with input_path.open("r", errors="ignore") as handle:
            for line in handle:
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 9:
                    continue
                frame = int(parts[0]) - start_frame
                if frame < 0 or frame >= selected_frames:
                    raise ValueError(
                        f"Relative frame outside 0-{selected_frames - 1}: {line!r}"
                    )
                donor_h = int(parts[4])
                acceptor_o = int(parts[5])
                distance = parts[6]

                # Filename is based only on atom serials and direction:
                # donor H atom -> acceptor O atom.
                name = f"H{donor_h}_O{acceptor_o}.txt"
                if name not in handles:
                    handles[name] = (output_folder / name).open("w")
                    seen_frames[name] = 0
                    pair_counts[name] = pair_counts.get(name, 0) + 1
                handles[name].write(f"{frame} {distance}\n")
                seen_frames[name] += 1

        bad = {
            name: count for name, count in seen_frames.items() if count != selected_frames
        }
        if bad:
            raise ValueError(
                f"{input_path} did not provide {selected_frames} rows for all pairs: "
                f"{bad}"
            )
    finally:
        for out in handles.values():
            out.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="hbonds_last50000", type=Path)
    parser.add_argument("--output-dir", default="hbonds_last50000_by_O_serial", type=Path)
    args = parser.parse_args()

    metadata = load_metadata(args.input_dir / "metadata.json")
    start_frame = int(metadata["start_frame_index_0_based"])
    selected_frames = int(metadata["selected_frames"])
    o_serials = water_serials(metadata)

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    summary_rows: list[dict[str, int]] = []
    total_files = 0
    for o_serial in o_serials:
        source = args.input_dir / f"O_{o_serial}"
        target = args.output_dir / str(o_serial)
        target.mkdir(parents=True)

        pair_counts: dict[str, int] = {}
        for filename in (
            "O_acceptor_hbonds.txt",
            "H1_donor_hbonds.txt",
            "H2_donor_hbonds.txt",
        ):
            reformat_file(
                source / filename,
                target,
                start_frame,
                selected_frames,
                o_serial,
                pair_counts,
            )

        files = sorted(target.glob("*.txt"))
        total_files += len(files)
        summary_rows.append(
            {
                "O_serial": o_serial,
                "formed_hbond_files": len(files),
                "duplicate_pair_files_seen": sum(
                    1 for count in pair_counts.values() if count > 1
                ),
            }
        )

    out_meta = {
        "source_dir": str(args.input_dir),
        "selected_frames": selected_frames,
        "frame_numbering": "0_to_49999",
        "columns_in_each_txt": ["frame_0_based", "distance_A"],
        "filename_rule": "H<donor_H_atom_serial>_O<acceptor_O_atom_serial>.txt",
        "directory_rule": "<water_O_atom_serial>",
        "total_water_O_directories": len(o_serials),
        "total_distance_txt_files": total_files,
        "note": (
            "Each O directory contains all hydrogen bonds involving that water: "
            "O as acceptor plus its two H atoms as donors. Each pair was retained "
            "because it formed a hydrogen bond at least once in the 50000-frame window."
        ),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(out_meta, indent=2, ensure_ascii=False) + "\n"
    )

    with (args.output_dir / "summary.csv").open("w") as handle:
        handle.write("O_serial,formed_hbond_files,duplicate_pair_files_seen\n")
        for row in summary_rows:
            handle.write(
                f"{row['O_serial']},{row['formed_hbond_files']},"
                f"{row['duplicate_pair_files_seen']}\n"
            )

    print(
        f"Done. Output: {args.output_dir} "
        f"({len(o_serials)} O folders, {total_files} distance txt files)"
    )


if __name__ == "__main__":
    main()
