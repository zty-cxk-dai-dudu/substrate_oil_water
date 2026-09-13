#!/usr/bin/env python3
"""Write the latest complete XDATCAR frames without loading the file in memory."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--natoms", type=int, required=True)
    parser.add_argument("--last-frames", type=int, default=50000)
    args = parser.parse_args()

    lines_per_frame = args.natoms + 1
    keep_lines = args.last_frames * lines_per_frame
    with args.source.open("r", errors="strict") as source:
        header = [source.readline() for _ in range(7)]
        if any(not line for line in header):
            raise ValueError("XDATCAR header is incomplete")
        payload = deque(source, maxlen=keep_lines)

    if len(payload) != keep_lines:
        raise ValueError(
            f"Requested {args.last_frames} frames but only "
            f"{len(payload) // lines_per_frame} complete frames are available"
        )
    first = payload[0].lstrip().lower()
    if not first.startswith("direct configuration"):
        raise ValueError("Trimmed payload does not start at a frame boundary")

    with args.output.open("w") as output:
        output.writelines(header)
        output.writelines(payload)

    print(
        f"Wrote {args.last_frames} frames, {args.natoms} atoms/frame, "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
