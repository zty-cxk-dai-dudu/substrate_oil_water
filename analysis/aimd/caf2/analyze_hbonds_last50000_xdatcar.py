#!/usr/bin/env python3
"""
Analyze water hydrogen bonds in the latest frames of a VASP XDATCAR trajectory.

This follows the same output contract as analyze_hbonds_last50000.py used in
the sibling workflow: keep donor-H / acceptor-O pairs that form a hydrogen bond
at least once in the selected window, then write their H...O distance, angle and
instantaneous state for every selected frame.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


CONFIG_RE = re.compile(r"configuration\s*=\s*(?P<step>-?\d+)", re.I)


def parse_poscar(path: Path) -> tuple[np.ndarray, list[str], list[int], list[str]]:
    lines = path.read_text().splitlines()
    if len(lines) < 8:
        raise ValueError(f"POSCAR is too short: {path}")
    scale = float(lines[1].split()[0])
    cell = np.array([[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)])
    cell *= scale
    species = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    if len(species) != len(counts):
        raise ValueError("POSCAR symbol/count lines have different lengths.")

    symbols: list[str] = []
    for sym, count in zip(species, counts):
        symbols.extend([sym] * count)
    return cell, species, counts, symbols


def xdatcar_frame_count(path: Path, natoms: int) -> int:
    nlines = sum(1 for _ in path.open("r", errors="ignore"))
    lines_per_frame = natoms + 1
    payload = nlines - 7
    if payload < 0 or payload % lines_per_frame != 0:
        raise ValueError(
            f"{path} has {nlines} lines, not compatible with {natoms} atoms."
        )
    return payload // lines_per_frame


def read_xdatcar_header(path: Path) -> list[str]:
    with path.open("r", errors="ignore") as handle:
        return [handle.readline() for _ in range(7)]


def parse_step(line: str) -> int | None:
    match = CONFIG_RE.search(line)
    return None if match is None else int(match.group("step"))


def iter_xdatcar_frames(
    path: Path,
    symbols: list[str],
    cell: np.ndarray,
    start_frame: int,
):
    natoms = len(symbols)
    with path.open("r", errors="ignore") as handle:
        for _ in range(7):
            handle.readline()
        frame_no = 0
        while True:
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
                yield frame_no, parse_step(config_line), None, symbols, frac @ cell
            frame_no += 1


def minimum_image(delta: np.ndarray, cell: np.ndarray, inv_cell: np.ndarray) -> np.ndarray:
    frac = delta @ inv_cell
    frac -= np.rint(frac)
    return frac @ cell


def vector_norm(values: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(values * values, axis=-1))


def assign_water_hydrogens(
    symbols: list[str],
    coords: np.ndarray,
    cell: np.ndarray,
    o_count: int,
    h_count: int,
    covalent_cutoff: float,
) -> tuple[np.ndarray, np.ndarray]:
    inv_cell = np.linalg.inv(cell)
    o_indices = np.array([i for i, sym in enumerate(symbols) if sym == "O"], dtype=np.int64)
    h_indices = np.array([i for i, sym in enumerate(symbols) if sym == "H"], dtype=np.int64)
    if o_indices.size < o_count:
        raise ValueError(f"Found {o_indices.size} O atoms, but POSCAR needs {o_count}.")
    if h_indices.size < h_count:
        raise ValueError(f"Found {h_indices.size} H atoms, but POSCAR needs {h_count}.")

    water_o = o_indices[:o_count]
    all_h = h_indices[:h_count]
    delta = coords[water_o, None, :] - coords[all_h][None, :, :]
    dist = vector_norm(minimum_image(delta, cell, inv_cell))

    candidates: list[tuple[float, int, int]] = []
    for oi in range(o_count):
        for hi in np.argsort(dist[oi])[:8]:
            if dist[oi, hi] <= covalent_cutoff:
                candidates.append((float(dist[oi, hi]), oi, int(hi)))
    candidates.sort()

    assigned_h: list[list[int]] = [[] for _ in range(o_count)]
    used_h: set[int] = set()
    for _distance, oi, hi in candidates:
        if hi in used_h or len(assigned_h[oi]) >= 2:
            continue
        assigned_h[oi].append(int(all_h[hi]))
        used_h.add(hi)

    missing = [int(water_o[i]) + 1 for i, hs in enumerate(assigned_h) if len(hs) != 2]
    if missing:
        raise ValueError(
            "Could not assign two covalent H atoms to these O serials with "
            f"cutoff {covalent_cutoff} A: {missing[:20]}"
        )

    return water_o, np.array(assigned_h, dtype=np.int64)


def hbond_arrays(
    coords: np.ndarray,
    cell: np.ndarray,
    inv_cell: np.ndarray,
    water_o: np.ndarray,
    water_h: np.ndarray,
    oh_cutoff: float,
    oo_cutoff: float,
    angle_cutoff_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    o_xyz = coords[water_o]
    h_xyz = coords[water_h]
    nwater = water_o.size

    h_to_acceptor = minimum_image(
        o_xyz[None, None, :, :] - h_xyz[:, :, None, :],
        cell,
        inv_cell,
    )
    h_o_dist = vector_norm(h_to_acceptor)

    donor_o_to_acceptor_o = minimum_image(
        o_xyz[None, :] - o_xyz[:, None],
        cell,
        inv_cell,
    )
    oo_dist = vector_norm(donor_o_to_acceptor_o)[:, None, :]

    h_to_donor_o = minimum_image(
        o_xyz[:, None, :] - h_xyz,
        cell,
        inv_cell,
    )[:, :, None, :]
    oh_bond_dist = vector_norm(h_to_donor_o)

    dot = np.sum(h_to_donor_o * h_to_acceptor, axis=-1)
    denom = np.maximum(oh_bond_dist * h_o_dist, 1.0e-12)
    angle = np.degrees(np.arccos(np.clip(dot / denom, -1.0, 1.0)))

    not_self = ~np.eye(nwater, dtype=bool)
    is_hbond = (
        (h_o_dist <= oh_cutoff)
        & (oo_dist <= oo_cutoff)
        & (angle >= angle_cutoff_deg)
        & not_self[:, None, :]
    )
    return h_o_dist, oo_dist[:, 0, :], angle, is_hbond


def write_headers(
    output_dir: Path,
    water_o: np.ndarray,
    water_h: np.ndarray,
    acceptor_pairs: list[list[tuple[int, int]]],
    donor_pairs: list[list[list[int]]],
) -> dict[tuple[int, int], object]:
    handles: dict[tuple[int, int], object] = {}
    for wi, o_idx in enumerate(water_o):
        folder = output_dir / f"O_{o_idx + 1}"
        folder.mkdir(parents=True, exist_ok=True)

        handle = (folder / "O_acceptor_hbonds.txt").open("w")
        handle.write(
            "# This oxygen as H-bond acceptor. Distances are donor_H...this_O.\n"
            "# Only donor-H pairs that formed an H bond at least once are retained.\n"
            "# columns: frame_index step time_fs donor_O_serial donor_H_serial "
            "acceptor_O_serial distance_A angle_deg is_hbond\n"
        )
        handle.write(
            "# retained_pairs = "
            + json.dumps(
                [
                    {
                        "donor_O_serial": int(water_o[d] + 1),
                        "donor_H_serial": int(water_h[d, h] + 1),
                        "acceptor_O_serial": int(o_idx + 1),
                    }
                    for d, h in acceptor_pairs[wi]
                ],
                ensure_ascii=False,
            )
            + "\n"
        )
        handles[(wi, -1)] = handle

        for hi in range(2):
            handle = (folder / f"H{hi + 1}_donor_hbonds.txt").open("w")
            handle.write(
                f"# H{hi + 1} of this water as H-bond donor. "
                f"Donor H serial = {int(water_h[wi, hi] + 1)}.\n"
                "# Only acceptor-O pairs that formed an H bond at least once are retained.\n"
                "# columns: frame_index step time_fs donor_O_serial donor_H_serial "
                "acceptor_O_serial distance_A angle_deg is_hbond\n"
            )
            handle.write(
                "# retained_pairs = "
                + json.dumps(
                    [
                        {
                            "donor_O_serial": int(o_idx + 1),
                            "donor_H_serial": int(water_h[wi, hi] + 1),
                            "acceptor_O_serial": int(water_o[a] + 1),
                        }
                        for a in donor_pairs[wi][hi]
                    ],
                    ensure_ascii=False,
                )
                + "\n"
            )
            handles[(wi, hi)] = handle
    return handles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xdatcar", default="XDATCAR", type=Path)
    parser.add_argument("--poscar", default="POSCAR", type=Path)
    parser.add_argument("--output-dir", default="hbonds_last50000", type=Path)
    parser.add_argument("--last-frames", default=50000, type=int)
    parser.add_argument("--oh-cutoff", default=2.45, type=float)
    parser.add_argument("--oo-cutoff", default=3.50, type=float)
    parser.add_argument("--angle-cutoff", default=150.0, type=float)
    parser.add_argument("--covalent-oh-cutoff", default=1.25, type=float)
    args = parser.parse_args()

    cell, species, counts, symbols = parse_poscar(args.poscar)
    count_by_species = dict(zip(species, counts))
    if "O" not in count_by_species or "H" not in count_by_species:
        raise ValueError("POSCAR must contain O and H species.")
    o_count = count_by_species["O"]
    h_count = count_by_species["H"]
    natoms = len(symbols)

    xdat_header = read_xdatcar_header(args.xdatcar)
    total_frames = xdatcar_frame_count(args.xdatcar, natoms)
    start_frame = max(0, total_frames - args.last_frames)
    selected_frames = total_frames - start_frame

    first = next(iter_xdatcar_frames(args.xdatcar, symbols, cell, 0))
    _frame_no, _step, _time_fs, _symbols, coords = first
    water_o, water_h = assign_water_hydrogens(
        symbols, coords, cell, o_count, h_count, args.covalent_oh_cutoff
    )
    nwater = water_o.size
    inv_cell = np.linalg.inv(cell)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "trajectory_file": str(args.xdatcar),
        "poscar_file": str(args.poscar),
        "xdatcar_header": [line.rstrip("\n") for line in xdat_header],
        "total_frames": total_frames,
        "start_frame_index_0_based": start_frame,
        "selected_frames": selected_frames,
        "natoms": natoms,
        "water_count": nwater,
        "criteria": {
            "donorH_acceptorO_distance_A_max": args.oh_cutoff,
            "donorO_acceptorO_distance_A_max": args.oo_cutoff,
            "O_H_O_angle_deg_min": args.angle_cutoff,
            "covalent_OH_assignment_cutoff_A": args.covalent_oh_cutoff,
        },
        "water_topology": [
            {
                "water_index_1_based": i + 1,
                "O_serial": int(water_o[i] + 1),
                "H1_serial": int(water_h[i, 0] + 1),
                "H2_serial": int(water_h[i, 1] + 1),
            }
            for i in range(nwater)
        ],
    }

    ever = np.zeros((nwater, 2, nwater), dtype=bool)
    frame_counter = 0
    for _frame_no, _step, _time_fs, _symbols, frame_coords in iter_xdatcar_frames(
        args.xdatcar, symbols, cell, start_frame
    ):
        _h_o_dist, _oo_dist, _angle, is_hbond = hbond_arrays(
            frame_coords,
            cell,
            inv_cell,
            water_o,
            water_h,
            args.oh_cutoff,
            args.oo_cutoff,
            args.angle_cutoff,
        )
        ever |= is_hbond
        frame_counter += 1
        if frame_counter % 5000 == 0:
            print(f"pass1 {frame_counter}/{selected_frames} frames", flush=True)

    acceptor_pairs: list[list[tuple[int, int]]] = []
    donor_pairs: list[list[list[int]]] = []
    for wi in range(nwater):
        acceptor_pairs.append([(int(d), int(h)) for d, h in np.argwhere(ever[:, :, wi])])
        donor_pairs.append(
            [[int(a) for a in np.flatnonzero(ever[wi, hi, :])] for hi in range(2)]
        )

    metadata["retained_pair_counts"] = {
        "total_donorH_acceptorO_pairs": int(ever.sum()),
        "by_acceptor_O": {
            str(int(water_o[i] + 1)): len(acceptor_pairs[i]) for i in range(nwater)
        },
        "by_donor_H1": {
            str(int(water_o[i] + 1)): len(donor_pairs[i][0]) for i in range(nwater)
        },
        "by_donor_H2": {
            str(int(water_o[i] + 1)): len(donor_pairs[i][1]) for i in range(nwater)
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )

    handles = write_headers(args.output_dir, water_o, water_h, acceptor_pairs, donor_pairs)
    try:
        frame_counter = 0
        for frame_no, step, time_fs, _symbols, frame_coords in iter_xdatcar_frames(
            args.xdatcar, symbols, cell, start_frame
        ):
            h_o_dist, _oo_dist, angle, is_hbond = hbond_arrays(
                frame_coords,
                cell,
                inv_cell,
                water_o,
                water_h,
                args.oh_cutoff,
                args.oo_cutoff,
                args.angle_cutoff,
            )
            step_s = "" if step is None else str(step)
            time_s = step_s if time_fs is None else f"{time_fs:.6f}"

            for acceptor_wi, pairs in enumerate(acceptor_pairs):
                handle = handles[(acceptor_wi, -1)]
                acceptor_serial = int(water_o[acceptor_wi] + 1)
                for donor_wi, donor_hi in pairs:
                    handle.write(
                        f"{frame_no} {step_s} {time_s} "
                        f"{int(water_o[donor_wi] + 1)} "
                        f"{int(water_h[donor_wi, donor_hi] + 1)} "
                        f"{acceptor_serial} "
                        f"{h_o_dist[donor_wi, donor_hi, acceptor_wi]:.8f} "
                        f"{angle[donor_wi, donor_hi, acceptor_wi]:.4f} "
                        f"{int(is_hbond[donor_wi, donor_hi, acceptor_wi])}\n"
                    )

            for donor_wi in range(nwater):
                for donor_hi in range(2):
                    handle = handles[(donor_wi, donor_hi)]
                    donor_o_serial = int(water_o[donor_wi] + 1)
                    donor_h_serial = int(water_h[donor_wi, donor_hi] + 1)
                    for acceptor_wi in donor_pairs[donor_wi][donor_hi]:
                        handle.write(
                            f"{frame_no} {step_s} {time_s} "
                            f"{donor_o_serial} {donor_h_serial} "
                            f"{int(water_o[acceptor_wi] + 1)} "
                            f"{h_o_dist[donor_wi, donor_hi, acceptor_wi]:.8f} "
                            f"{angle[donor_wi, donor_hi, acceptor_wi]:.4f} "
                            f"{int(is_hbond[donor_wi, donor_hi, acceptor_wi])}\n"
                        )

            frame_counter += 1
            if frame_counter % 5000 == 0:
                print(f"pass2 {frame_counter}/{selected_frames} frames", flush=True)
    finally:
        for handle in handles.values():
            handle.close()

    print(
        f"Done. Output: {args.output_dir} "
        f"({int(ever.sum())} retained donor-H/acceptor-O pairs)"
    )


if __name__ == "__main__":
    main()
