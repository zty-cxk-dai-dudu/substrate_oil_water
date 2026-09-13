#!/usr/bin/env python3
"""Compress one completed raw CaF2 trajectory HDF5 segment off the MD path."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py


def copy_dataset(src, dst, name):
    s = src[name]
    if s.ndim == 0:
        d = dst.create_dataset(name, data=s[()])
    else:
        chunks = s.chunks
        if chunks is None:
            chunks = (min(128, s.shape[0]),) + s.shape[1:]
        d = dst.create_dataset(
            name, shape=s.shape, dtype=s.dtype, chunks=chunks,
            compression="gzip", compression_opts=1, shuffle=True,
        )
        block = max(1, min(128, s.shape[0]))
        for i in range(0, s.shape[0], block):
            d[i:i + block] = s[i:i + block]
    for key, value in s.attrs.items():
        d.attrs[key] = value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path, help="Completed raw HDF5 segment")
    parser.add_argument("final", type=Path, help="Compressed HDF5 destination")
    parser.add_argument("--keep-raw", action="store_true",
                        help="Keep the raw segment after successful compression")
    args = parser.parse_args()
    raw = args.raw.resolve()
    final = args.final.resolve()
    if raw == final:
        parser.error("RAW and FINAL must be different files")
    tmp = final.with_suffix(final.suffix + ".compressing")
    if tmp.exists():
        tmp.unlink()
    with h5py.File(raw, "r") as src, h5py.File(tmp, "w", libver="latest") as dst:
        for name in src.keys():
            copy_dataset(src, dst, name)
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        dst.attrs["compression_completed"] = True
        dst.flush()
    os.replace(tmp, final)
    if not args.keep_raw:
        raw.unlink()


if __name__ == "__main__":
    main()
