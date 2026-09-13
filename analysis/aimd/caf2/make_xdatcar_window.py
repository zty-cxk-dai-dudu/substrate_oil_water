#!/usr/bin/env python3
"""Write a renumbered XDATCAR window from the current XDATCAR."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


CONFIG_RE = re.compile(r"Direct\s+configuration\s*=\s*-?\d+", re.I)


def natoms_from_header(header: list[str]) -> int:
    return sum(int(x) for x in header[6].split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="XDATCAR", type=Path)
    parser.add_argument("--output-dir", default="last50000_xdatcar", type=Path)
    parser.add_argument("--last-frames", default=50000, type=int)
    parser.add_argument("--renumber", action="store_true")
    args = parser.parse_args()

    with args.input.open("r", errors="ignore") as handle:
        header = [handle.readline() for _ in range(7)]
        natoms = natoms_from_header(header)
        nlines = sum(1 for _ in handle)
    total_frames = nlines // (natoms + 1)
    start = max(0, total_frames - args.last_frames)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "XDATCAR"
    with args.input.open("r", errors="ignore") as src, out_path.open("w") as out:
        header = [src.readline() for _ in range(7)]
        out.writelines(header)
        for frame in range(total_frames):
            config = src.readline()
            coords = [src.readline() for _ in range(natoms)]
            if frame < start:
                continue
            if args.renumber:
                rel = frame - start + 1
                config = CONFIG_RE.sub(f"Direct configuration= {rel:6d}", config)
            out.write(config)
            out.writelines(coords)

    meta = args.output_dir / "window_meta.txt"
    meta.write_text(
        "\n".join(
            [
                f"source={args.input}",
                f"natoms={natoms}",
                f"total_frames={total_frames}",
                f"start_frame_0_based={start}",
                f"written_frames={total_frames - start}",
                f"renumber={args.renumber}",
            ]
        )
        + "\n"
    )
    print(meta.read_text(), end="")


if __name__ == "__main__":
    main()
