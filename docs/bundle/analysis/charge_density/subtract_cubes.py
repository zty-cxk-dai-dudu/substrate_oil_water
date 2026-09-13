#!/usr/bin/env python3
"""Subtract CP2K electron-density cubes and validate grids/integrals."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CALCS = ("ab_full", "a_sio2oil", "b_h2o")


def find_density_cube(directory: Path) -> Path:
    exact = directory / "electron_density.cube"
    if exact.is_file() and exact.stat().st_size:
        return exact
    candidates = sorted(directory.glob("*ELECTRON_DENSITY*.cube")) + sorted(directory.glob("*density*.cube"))
    candidates = [path for path in candidates if path.stat().st_size]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one electron-density cube in {directory}, found {candidates}")
    return candidates[0]


def read_cube_header(path: Path):
    handle = path.open()
    header = [next(handle), next(handle)]
    origin_line = next(handle)
    header.append(origin_line)
    origin_tokens = origin_line.split()
    natoms = abs(int(origin_tokens[0]))
    origin = tuple(float(x) for x in origin_tokens[1:4])
    shape = []
    axes = []
    for _ in range(3):
        line = next(handle)
        header.append(line)
        tokens = line.split()
        shape.append(abs(int(tokens[0])))
        axes.append(tuple(float(x) for x in tokens[1:4]))
    for _ in range(natoms):
        header.append(next(handle))
    return handle, header, natoms, tuple(shape), origin, tuple(axes)


def values(handle):
    for line in handle:
        for token in line.split():
            yield float(token)


def output_complete(path: Path) -> bool:
    text = path.read_text(errors="replace")
    return "SCF run converged" in text and "PROGRAM ENDED AT" in text


def total_energy(path: Path) -> float:
    matches = re.findall(
        r"ENERGY\| Total FORCE_EVAL \( QS \) energy \[(?:a\.u\.|hartree)\]\s*:?[ \t]+([-+0-9.Ee]+)",
        path.read_text(errors="replace"),
    )
    if not matches:
        raise RuntimeError(f"Missing final energy in {path}")
    return float(matches[-1])


def main():
    for calc in CALCS:
        if not output_complete(ROOT / calc / "output.log"):
            raise RuntimeError(f"Incomplete or unconverged CP2K output: {calc}")
    energies = {calc: total_energy(ROOT / calc / "output.log") for calc in CALCS}
    paths = [find_density_cube(ROOT / calc) for calc in CALCS]
    parsed = [read_cube_header(path) for path in paths]
    handles = [item[0] for item in parsed]
    headers = [item[1] for item in parsed]
    signatures = [(item[2], item[3], item[4], item[5]) for item in parsed]
    if len(set(signatures)) != 1:
        raise RuntimeError(f"Cube-grid mismatch: {signatures}")
    natoms, shape, origin, axes = signatures[0]
    expected = math.prod(shape)
    ax, ay, az = axes
    voxel_volume = abs(
        ax[0] * (ay[1] * az[2] - ay[2] * az[1])
        - ax[1] * (ay[0] * az[2] - ay[2] * az[0])
        + ax[2] * (ay[0] * az[1] - ay[1] * az[0])
    )
    sums = [0.0, 0.0, 0.0]
    diff_sum = 0.0
    minimum = float("inf")
    maximum = float("-inf")
    count = 0
    out_path = ROOT / "CHGDIFF_CP2K.cube"
    with out_path.open("w") as out:
        out.write("CP2K electron-density difference\n")
        out.write("rho(full)-rho(SiO2+oil)-rho(H2O); ghost basis retained\n")
        out.writelines(headers[0][2:])
        iterators = [values(handle) for handle in handles]
        for triple in itertools.zip_longest(*iterators):
            if any(value is None for value in triple):
                raise RuntimeError("Cube data lengths differ")
            count += 1
            for i, value in enumerate(triple):
                sums[i] += value
            diff = triple[0] - triple[1] - triple[2]
            diff_sum += diff
            minimum = min(minimum, diff)
            maximum = max(maximum, diff)
            out.write(f" {diff:12.5E}")
            if count % 6 == 0:
                out.write("\n")
        if count % 6:
            out.write("\n")
    for handle in handles:
        handle.close()
    if count != expected:
        raise RuntimeError(f"Truncated cube: {count}/{expected}")

    integrals = [value * voxel_volume for value in sums]
    diff_integral = diff_sum * voxel_volume
    if abs(diff_integral) >= 1.0e-3:
        raise RuntimeError(f"Integrated difference is not near zero: {diff_integral} e")
    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    summary = {
        "definition": "rho(ab_full)-rho(a_sio2oil)-rho(b_h2o)",
        "ghost_basis_retained": True,
        "grid_shape": list(shape),
        "grid_values": expected,
        "voxel_volume_bohr3": voxel_volume,
        "source_density_integrals_e": dict(zip(CALCS, integrals)),
        "integrated_difference_e": diff_integral,
        "difference_min_e_per_bohr3": minimum,
        "difference_max_e_per_bohr3": maximum,
        "component_total_energies_Ha": energies,
        "chgdiff_size_bytes": out_path.stat().st_size,
        "chgdiff_sha256": digest,
        "status": "ok",
    }
    (ROOT / "charge_difference_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (ROOT / "CHGDIFF_CP2K.sha256").write_text(f"{digest}  CHGDIFF_CP2K.cube\n")
    (ROOT / "COMPLETE").write_text("validated\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
