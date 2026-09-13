#!/usr/bin/env python3
"""Extract per-C6 oil-chain COM fluctuation data from the three 500 ps systems.

The H5 trajectories are sampled at 1 fs and the SiO2 LAMMPS position dump at
0.1 ps.  For a like-for-like comparison, all output rows use a 100 fs stride.
Oil chains are reconstructed from C--C connectivity (six C atoms per
component) and the attached H atoms within 1.30 A of those carbons.  The
CaF2/oil trajectories contain C6H14; the current SiO2 trajectory contains
four C6 chains with 13 attached H each (C6H13), so the raw composition is
reported instead of silently forcing C6H14.  The output is
raw per-molecule COM data plus non-overlapping 10 ps fluctuation summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


MASS = {1: 1.008, 6: 12.011}


def min_image(delta: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Apply the minimum-image convention for a (possibly triclinic) cell."""
    delta = np.asarray(delta, dtype=float)
    inv = np.linalg.inv(cell)
    frac = delta @ inv
    frac -= np.rint(frac)
    return frac @ cell


def wrap_cartesian(position: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Wrap a Cartesian position into an orthorhombic cell at the origin."""
    length = np.diag(cell)
    return np.mod(position, length)


def connected_components(carbon_indices: np.ndarray, positions: np.ndarray,
                         cell: np.ndarray, cutoff: float = 1.80) -> list[list[int]]:
    """Return C--C connected components using a covalent-distance cutoff."""
    cpos = positions[carbon_indices]
    distance = np.linalg.norm(
        min_image(cpos[:, None, :] - cpos[None, :, :], cell), axis=-1
    )
    np.fill_diagonal(distance, np.inf)
    seen: set[int] = set()
    components: list[list[int]] = []
    for i in range(len(carbon_indices)):
        if i in seen:
            continue
        stack = [i]
        seen.add(i)
        component: list[int] = []
        while stack:
            j = stack.pop()
            component.append(int(carbon_indices[j]))
            for k in np.flatnonzero(distance[j] <= cutoff):
                k = int(k)
                if k not in seen:
                    seen.add(k)
                    stack.append(k)
        components.append(sorted(component))
    components.sort(key=lambda x: (x[0], len(x)))
    return components


def reconstruct_hexanes(atomic_numbers: np.ndarray, positions: np.ndarray,
                        cell: np.ndarray) -> list[np.ndarray]:
    """Build atom-index arrays for each six-carbon oil chain in the first frame."""
    atomic_numbers = np.asarray(atomic_numbers, dtype=int)
    carbon = np.flatnonzero(atomic_numbers == 6)
    hydrogen = np.flatnonzero(atomic_numbers == 1)
    components = connected_components(carbon, positions, cell)
    if not components or any(len(c) != 6 for c in components):
        raise RuntimeError(
            f"C--C reconstruction did not produce C6 components: "
            f"{[len(c) for c in components]}"
        )

    cpos = positions[carbon]
    hpos = positions[hydrogen]
    h_to_c = min_image(hpos[:, None, :] - cpos[None, :, :], cell)
    h_distance = np.linalg.norm(h_to_c, axis=-1)
    nearest = np.argmin(h_distance, axis=1)
    nearest_distance = h_distance[np.arange(len(hydrogen)), nearest]
    h_to_component: dict[int, list[int]] = {i: [] for i in range(len(components))}
    carbon_to_component = {
        carbon_index: component_index
        for component_index, component in enumerate(components)
        for carbon_index in component
    }
    for h_atom, c_local, distance in zip(hydrogen, nearest, nearest_distance):
        if distance <= 1.30:
            c_atom = int(carbon[c_local])
            h_to_component[carbon_to_component[c_atom]].append(int(h_atom))

    groups: list[np.ndarray] = []
    for component_index, component in enumerate(components):
        h_atoms = sorted(h_to_component[component_index])
        # The CaF2/oil controls are C6H14.  In the current SiO2 trajectory
        # each adsorbed six-carbon chain is C6H13, so accept the observed
        # C6H13/C6H14 forms while still rejecting a wrong molecule grouping.
        if len(h_atoms) not in (13, 14):
            raise RuntimeError(
                f"C6 group {component_index + 1} has {len(h_atoms)} attached H, "
                f"expected 13 or 14; carbon IDs={component}"
            )
        groups.append(np.asarray(component + h_atoms, dtype=int))
    return groups


def group_com(position: np.ndarray, group: np.ndarray, atomic_numbers: np.ndarray,
              cell: np.ndarray) -> np.ndarray:
    """Mass-weighted molecular COM, unwrapping atoms around the first carbon."""
    carbon = group[atomic_numbers[group] == 6]
    reference = position[carbon[0]]
    local = reference + min_image(position[group] - reference, cell)
    weights = np.asarray([MASS[int(z)] for z in atomic_numbers[group]], dtype=float)
    return np.sum(local * weights[:, None], axis=0) / np.sum(weights)


def unwrap_com(wrapped: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Continuously unwrap a molecular COM through periodic boundaries."""
    output = np.empty_like(wrapped, dtype=float)
    output[0] = wrapped[0]
    for frame in range(1, len(wrapped)):
        output[frame] = output[frame - 1] + min_image(
            wrapped[frame] - wrapped[frame - 1], cell
        )
    return output


def read_h5_system(system: str, directory: Path, sample_stride: int = 100,
                   max_samples: int = 5000):
    import h5py

    files = sorted(directory.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No H5 files in {directory}")
    with h5py.File(files[0], "r") as handle:
        atomic_numbers = np.asarray(handle["atomic_numbers"][:], dtype=int)
        first_position = np.asarray(handle["positions_A"][0], dtype=float)
        cell = np.asarray(handle["cell_A"][:], dtype=float)
        groups = reconstruct_hexanes(atomic_numbers, first_position, cell)

    wrapped_frames: list[np.ndarray] = []
    times: list[float] = []
    source_steps: list[int] = []
    for path in files:
        with h5py.File(path, "r") as handle:
            positions = handle["positions_A"]
            time_ps = handle.get("time_ps")
            steps = handle.get("step")
            indices = range(0, positions.shape[0], sample_stride)
            for index in indices:
                if len(times) >= max_samples:
                    break
                frame_position = np.asarray(positions[index], dtype=float)
                frame_com = np.asarray(
                    [group_com(frame_position, group, atomic_numbers, cell)
                     for group in groups]
                )
                wrapped_frames.append(wrap_cartesian(frame_com, cell))
                times.append(float(time_ps[index]) if time_ps is not None
                             else float(len(times) * 0.1))
                source_steps.append(int(steps[index]) if steps is not None
                                     else len(times))
        if len(times) >= max_samples:
            break
    wrapped = np.asarray(wrapped_frames, dtype=float)
    unwrapped = np.asarray([unwrap_com(wrapped[:, i, :], cell)
                            for i in range(len(groups))]).transpose(1, 0, 2)
    times = np.asarray(times, dtype=float)
    times -= times[0]
    return {
        "system": system,
        "times_ps": times,
        "source_steps": np.asarray(source_steps, dtype=np.int64),
        "wrapped_A": wrapped,
        "unwrapped_A": unwrapped,
        "groups": groups,
        "cell_A": cell,
        "source": str(directory),
    }


def read_dump_header(stream):
    if stream.readline().strip() != "ITEM: TIMESTEP":
        raise RuntimeError("Invalid LAMMPS timestep header")
    step = int(stream.readline().strip())
    if stream.readline().strip() != "ITEM: NUMBER OF ATOMS":
        raise RuntimeError("Invalid LAMMPS atom-count header")
    natoms = int(stream.readline().strip())
    box_header = stream.readline().strip()
    if not box_header.startswith("ITEM: BOX BOUNDS"):
        raise RuntimeError("Invalid LAMMPS box header")
    bounds = []
    for _ in range(3):
        values = [float(x) for x in stream.readline().split()]
        bounds.append(values[:2])
    atom_header = stream.readline().strip()
    if not atom_header.startswith("ITEM: ATOMS"):
        raise RuntimeError("Invalid LAMMPS atom header")
    return step, natoms, np.asarray(bounds, dtype=float), atom_header


def read_lammps_system(system: str, dump_path: Path, data_path: Path,
                       nframes: int = 5000):
    # The data file provides the atom types and box; the trajectory provides
    # the current coordinates.  The first dump frame determines C6 groups.
    with dump_path.open() as stream:
        step, natoms, bounds, atom_header = read_dump_header(stream)
        first_records: dict[int, tuple[int, np.ndarray]] = {}
        for _ in range(natoms):
            fields = stream.readline().split()
            atom_id = int(fields[0])
            atom_type = int(fields[1])
            if atom_type in (3, 4):
                first_records[atom_id] = (atom_type, np.asarray(
                    [float(fields[-3]), float(fields[-2]), float(fields[-1])]
                ))
        if len(first_records) == 0:
            raise RuntimeError("No hexane C/H atoms found in first LAMMPS frame")
        atom_ids = sorted(first_records)
        atomic_numbers = np.asarray(
            [1 if first_records[atom_id][0] == 3 else 6 for atom_id in atom_ids],
            dtype=int,
        )
        first_position = np.asarray([first_records[atom_id][1] for atom_id in atom_ids])
        lower = bounds[:, 0]
        cell = np.diag(bounds[:, 1] - bounds[:, 0])
        groups_local = reconstruct_hexanes(atomic_numbers, first_position - lower, cell)
        groups = [np.asarray([atom_ids[i] for i in group], dtype=int)
                  for group in groups_local]
        wanted = {atom_id: i for i, atom_id in enumerate(atom_ids)}

        wrapped_frames = [
            wrap_cartesian(
                np.asarray([group_com(first_position - lower, group, atomic_numbers, cell)
                            for group in groups_local]), cell
            )
        ]
        times = [0.0]
        source_steps = [step]
        for frame in range(1, nframes):
            next_step, next_natoms, next_bounds, _ = read_dump_header(stream)
            if next_natoms != natoms:
                raise RuntimeError(f"Atom count changed at frame {frame}: {next_natoms}")
            frame_position = np.empty((len(atom_ids), 3), dtype=float)
            found = 0
            for _ in range(natoms):
                fields = stream.readline().split()
                atom_id = int(fields[0])
                local = wanted.get(atom_id)
                if local is not None:
                    frame_position[local] = [float(fields[-3]), float(fields[-2]),
                                             float(fields[-1])]
                    found += 1
            if found != len(atom_ids):
                raise RuntimeError(f"Missing C/H atoms at frame {frame}: {found}")
            current_lower = next_bounds[:, 0]
            current_cell = np.diag(next_bounds[:, 1] - next_bounds[:, 0])
            if not np.allclose(current_cell, cell, atol=1e-5):
                raise RuntimeError("Changing cell is not supported for this dump")
            current_position = frame_position - current_lower
            frame_com = np.asarray(
                [group_com(current_position, group_local, atomic_numbers, cell)
                 for group_local in groups_local]
            )
            wrapped_frames.append(wrap_cartesian(frame_com, cell))
            times.append(frame * 0.1)
            source_steps.append(next_step)

    wrapped = np.asarray(wrapped_frames, dtype=float)
    unwrapped = np.asarray([unwrap_com(wrapped[:, i, :], cell)
                            for i in range(len(groups))]).transpose(1, 0, 2)
    return {
        "system": system,
        "times_ps": np.asarray(times, dtype=float),
        "source_steps": np.asarray(source_steps, dtype=np.int64),
        "wrapped_A": wrapped,
        "unwrapped_A": unwrapped,
        "groups": groups,
        "cell_A": cell,
        "source": str(dump_path),
        "data_file": str(data_path),
    }


def write_csv(path: Path, fields: list[str], rows: list[dict]):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def derive_outputs(results: list[dict], output: Path, window_frames: int = 100):
    timeseries_fields = [
        "system", "hexane_index", "time_ps", "source_step",
        "com_x_wrapped_A", "com_y_wrapped_A", "com_z_wrapped_A",
        "com_x_unwrapped_A", "com_y_unwrapped_A", "com_z_unwrapped_A",
        "delta_x_A", "delta_y_A", "delta_z_A", "delta_3d_A",
    ]
    timeseries_rows: list[dict] = []
    block_fields = [
        "system", "hexane_index", "block_index", "start_time_ps", "end_time_ps",
        "n_frames", "mean_x_A", "mean_y_A", "mean_z_A", "rms_x_A", "rms_y_A",
        "rms_z_A", "rms_lateral_A", "rms_3d_A", "p95_radius_A",
        "mean_frame_to_frame_delta_A",
    ]
    block_rows: list[dict] = []
    summary_fields = [
        "system", "hexane_index", "n_frames", "duration_ps", "n_blocks_10ps",
        "mean_block_rms_3d_A", "median_block_rms_3d_A", "p95_block_rms_3d_A",
        "max_block_rms_3d_A", "mean_block_rms_lateral_A", "mean_block_rms_z_A",
        "mean_frame_to_frame_delta_A", "global_std_x_A", "global_std_y_A",
        "global_std_z_A", "global_std_3d_from_global_mean_A",
    ]
    summary_rows: list[dict] = []
    composition_fields = ["system", "hexane_index", "n_carbons", "n_hydrogens",
                          "composition", "atom_index_or_id_list"]
    composition_rows: list[dict] = []

    for result in results:
        system = result["system"]
        t = result["times_ps"]
        wrapped = result["wrapped_A"]
        unwrapped = result["unwrapped_A"]
        steps = result["source_steps"]
        nframes, nmolecules, _ = unwrapped.shape
        for molecule, group in enumerate(result["groups"], 1):
            n_hydrogen = len(group) - 6
            composition_rows.append({
                "system": system,
                "hexane_index": molecule,
                "n_carbons": 6,
                "n_hydrogens": n_hydrogen,
                "composition": f"C6H{n_hydrogen}",
                "atom_index_or_id_list": ";".join(str(int(x)) for x in group),
            })
        delta = np.full_like(unwrapped, np.nan)
        delta[1:] = unwrapped[1:] - unwrapped[:-1]
        for molecule in range(nmolecules):
            for frame in range(nframes):
                timeseries_rows.append({
                    "system": system,
                    "hexane_index": molecule + 1,
                    "time_ps": f"{t[frame]:.6f}",
                    "source_step": int(steps[frame]),
                    "com_x_wrapped_A": f"{wrapped[frame, molecule, 0]:.8f}",
                    "com_y_wrapped_A": f"{wrapped[frame, molecule, 1]:.8f}",
                    "com_z_wrapped_A": f"{wrapped[frame, molecule, 2]:.8f}",
                    "com_x_unwrapped_A": f"{unwrapped[frame, molecule, 0]:.8f}",
                    "com_y_unwrapped_A": f"{unwrapped[frame, molecule, 1]:.8f}",
                    "com_z_unwrapped_A": f"{unwrapped[frame, molecule, 2]:.8f}",
                    "delta_x_A": "" if frame == 0 else f"{delta[frame, molecule, 0]:.8f}",
                    "delta_y_A": "" if frame == 0 else f"{delta[frame, molecule, 1]:.8f}",
                    "delta_z_A": "" if frame == 0 else f"{delta[frame, molecule, 2]:.8f}",
                    "delta_3d_A": "" if frame == 0 else f"{np.linalg.norm(delta[frame, molecule]):.8f}",
                })

            block_values = []
            for block_index, start in enumerate(range(0, nframes, window_frames), 1):
                stop = min(start + window_frames, nframes)
                values = unwrapped[start:stop, molecule]
                centered = values - np.mean(values, axis=0)
                rms = np.sqrt(np.mean(centered ** 2, axis=0))
                radius = np.linalg.norm(centered, axis=1)
                frame_delta = delta[start + 1:stop, molecule]
                mean_delta = (float(np.mean(np.linalg.norm(frame_delta, axis=1)))
                              if len(frame_delta) else float("nan"))
                rms_lateral = float(np.sqrt(rms[0] ** 2 + rms[1] ** 2))
                rms_3d = float(np.linalg.norm(rms))
                block_values.append(rms_3d)
                block_rows.append({
                    "system": system,
                    "hexane_index": molecule + 1,
                    "block_index": block_index,
                    "start_time_ps": f"{t[start]:.6f}",
                    "end_time_ps": f"{t[stop - 1]:.6f}",
                    "n_frames": stop - start,
                    "mean_x_A": f"{np.mean(values[:, 0]):.8f}",
                    "mean_y_A": f"{np.mean(values[:, 1]):.8f}",
                    "mean_z_A": f"{np.mean(values[:, 2]):.8f}",
                    "rms_x_A": f"{rms[0]:.8f}",
                    "rms_y_A": f"{rms[1]:.8f}",
                    "rms_z_A": f"{rms[2]:.8f}",
                    "rms_lateral_A": f"{rms_lateral:.8f}",
                    "rms_3d_A": f"{rms_3d:.8f}",
                    "p95_radius_A": f"{np.percentile(radius, 95):.8f}",
                    "mean_frame_to_frame_delta_A": f"{mean_delta:.8f}",
                })

            block_array = np.asarray(block_values, dtype=float)
            global_centered = unwrapped[:, molecule] - np.mean(unwrapped[:, molecule], axis=0)
            global_std = np.std(global_centered, axis=0)
            all_delta = delta[1:, molecule]
            summary_rows.append({
                "system": system,
                "hexane_index": molecule + 1,
                "n_frames": nframes,
                "duration_ps": f"{t[-1] - t[0]:.6f}",
                "n_blocks_10ps": len(block_array),
                "mean_block_rms_3d_A": f"{np.mean(block_array):.8f}",
                "median_block_rms_3d_A": f"{np.median(block_array):.8f}",
                "p95_block_rms_3d_A": f"{np.percentile(block_array, 95):.8f}",
                "max_block_rms_3d_A": f"{np.max(block_array):.8f}",
                "mean_block_rms_lateral_A": f"{np.mean([float(r['rms_lateral_A']) for r in block_rows[-len(block_array):]]):.8f}",
                "mean_block_rms_z_A": f"{np.mean([float(r['rms_z_A']) for r in block_rows[-len(block_array):]]):.8f}",
                "mean_frame_to_frame_delta_A": f"{np.mean(np.linalg.norm(all_delta, axis=1)):.8f}",
                "global_std_x_A": f"{global_std[0]:.8f}",
                "global_std_y_A": f"{global_std[1]:.8f}",
                "global_std_z_A": f"{global_std[2]:.8f}",
                "global_std_3d_from_global_mean_A": f"{np.linalg.norm(global_std):.8f}",
            })

    write_csv(output / "hexane_com_timeseries_500ps_stride100fs.csv",
              timeseries_fields, timeseries_rows)
    write_csv(output / "hexane_fluctuation_10ps_blocks.csv", block_fields, block_rows)
    write_csv(output / "hexane_fluctuation_summary.csv", summary_fields, summary_rows)
    write_csv(output / "hexane_composition.csv", composition_fields, composition_rows)

    system_fields = [
        "system", "n_hexane", "mean_over_hexanes_block_rms_3d_A",
        "std_over_hexanes_block_rms_3d_A", "mean_over_hexanes_block_rms_lateral_A",
        "mean_over_hexanes_block_rms_z_A", "mean_over_hexanes_frame_delta_A",
        "max_hexane_p95_block_rms_3d_A",
    ]
    system_rows: list[dict] = []
    for system in [r["system"] for r in results]:
        rows = [row for row in summary_rows if row["system"] == system]
        rms3d = np.asarray([float(row["mean_block_rms_3d_A"]) for row in rows])
        lateral = np.asarray([float(row["mean_block_rms_lateral_A"]) for row in rows])
        vertical = np.asarray([float(row["mean_block_rms_z_A"]) for row in rows])
        delta_values = np.asarray([float(row["mean_frame_to_frame_delta_A"]) for row in rows])
        p95_values = np.asarray([float(row["p95_block_rms_3d_A"]) for row in rows])
        system_rows.append({
            "system": system,
            "n_hexane": len(rows),
            "mean_over_hexanes_block_rms_3d_A": f"{np.mean(rms3d):.8f}",
            "std_over_hexanes_block_rms_3d_A": f"{np.std(rms3d):.8f}",
            "mean_over_hexanes_block_rms_lateral_A": f"{np.mean(lateral):.8f}",
            "mean_over_hexanes_block_rms_z_A": f"{np.mean(vertical):.8f}",
            "mean_over_hexanes_frame_delta_A": f"{np.mean(delta_values):.8f}",
            "max_hexane_p95_block_rms_3d_A": f"{np.max(p95_values):.8f}",
        })
    write_csv(output / "hexane_fluctuation_system_summary.csv", system_fields, system_rows)

    conclusion_lines = [
        "# 三体系正己烷（六碳链）波动比较",
        "",
        "波动幅度定义：每个 10 ps block 内，单条六碳链质量加权质心相对于该 block 平均质心的 RMS。",
        "这是质心位置波动，不是水分子波动，也不是把两个相邻六碳链合并后的油副本。",
        "",
        "| 体系 | 六碳链数 | 平均三维 RMS (Å) | 平均横向 RMS (Å) | 平均 z RMS (Å) | 100 fs 平均相邻帧位移 (Å) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in system_rows:
        conclusion_lines.append(
            f"| {row['system']} | {row['n_hexane']} | "
            f"{row['mean_over_hexanes_block_rms_3d_A']} | "
            f"{row['mean_over_hexanes_block_rms_lateral_A']} | "
            f"{row['mean_over_hexanes_block_rms_z_A']} | "
            f"{row['mean_over_hexanes_frame_delta_A']} |"
        )
    conclusion_lines.extend([
        "",
        "按该 10 ps 质心 RMS，oil/water 最大，CaF2 居中，SiO2 最小。当前 SiO2 源轨迹的四条六碳链首帧均为 C6H13；CaF2 和 oil/water 源轨迹各两条 C6H14。",
    ])
    (output / "comparison_conclusion_CN.md").write_text("\n".join(conclusion_lines) + "\n")

    metadata = {
        "status": "complete",
        "systems": [r["system"] for r in results],
        "sampling": "100 fs common stride; source H5 trajectories are 1 fs and SiO2 dump is 100 fs",
        "source_duration_ps": 500.0,
        "frames_per_system": {r["system"]: int(len(r["times_ps"])) for r in results},
        "hexanes_per_system": {r["system"]: int(r["unwrapped_A"].shape[1]) for r in results},
        "composition_per_system": {
            r["system"]: [f"C6H{len(group) - 6}" for group in r["groups"]]
            for r in results
        },
        "fluctuation_window_ps": 10.0,
        "molecular_definition": "C-C connected components of six carbons; attached H within 1.30 A; mass-weighted C6H13/C6H14 COM as present in each source",
        "periodic_boundary_handling": "minimum-image atom grouping and continuous COM unwrapping",
        "sources": {r["system"]: r["source"] for r in results},
    }
    (output / "analysis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )

    # A compact Chinese readme makes the raw-data meaning explicit.
    lines = [
        "# 三体系正己烷波动原始数据",
        "",
        "这里按每一个六碳正己烷碳骨架输出，不是水分子，也不是把两个正己烷合并后的油副本。CaF2/油水源轨迹为 C6H14；当前 SiO2 源轨迹的四条六碳链各有 13 个 attached H（C6H13），因此保留源数据本身的组成。",
        "所有体系按 100 fs 共同采样；每个 10 ps block 内以该 block 的分子质心均值为中心计算 RMS。",
        "",
        "- `hexane_com_timeseries_500ps_stride100fs.csv`: 每个正己烷、每一帧的包裹/连续展开质心坐标和相邻帧位移。",
        "- `hexane_fluctuation_10ps_blocks.csv`: 每个正己烷的 10 ps x/y/z、横向、三维波动幅度。",
        "- `hexane_fluctuation_summary.csv`: 500 ps 汇总，包括平均/中位数/P95 的 10 ps 三维 RMS。",
        "- `hexane_fluctuation_system_summary.csv`: 按体系平均后的三体系比较表。",
        "- `hexane_composition.csv`: 每条六碳链的 C/H 数量和首帧原子索引/ID。",
        "",
        "`rms_3d_A` 是该 10 ps 时间块内正己烷质心相对其块平均位置的三维 RMS；它用于比较油分子的局部波动幅度。",
    ]
    (output / "README_CN.md").write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--caf2-h5-dir", type=Path, required=True)
    parser.add_argument("--oil-h5-dir", type=Path, required=True)
    parser.add_argument("--sio2-dump", type=Path, required=True)
    parser.add_argument("--sio2-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = [
        read_h5_system("CaF2", args.caf2_h5_dir),
        read_h5_system("oil", args.oil_h5_dir),
        read_lammps_system("SiO2", args.sio2_dump, args.sio2_data),
    ]
    derive_outputs(results, args.output)
    print(json.dumps({
        "output": str(args.output),
        "frames": {r["system"]: len(r["times_ps"]) for r in results},
        "hexanes": {r["system"]: int(r["unwrapped_A"].shape[1]) for r in results},
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
