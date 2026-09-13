#!/usr/bin/env python3
"""Derive z-direction RMS and MSD from exported hexane COM time series."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_rows(path: Path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, fields, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def derive(source: Path, output: Path, block_ps: float = 10.0):
    rows = read_rows(source / "hexane_com_timeseries_500ps_stride100fs.csv")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["system"], int(row["hexane_index"]))].append(row)
    block_rows = []
    summary_rows = []
    for (system, molecule), values in sorted(grouped.items()):
        values.sort(key=lambda r: float(r["time_ps"]))
        t = np.asarray([float(r["time_ps"]) for r in values])
        z = np.asarray([float(r["com_z_unwrapped_A"]) for r in values])
        z0 = z[0]
        block_size = max(1, int(round(block_ps / np.median(np.diff(t)))) )
        z_rms_all = []
        z_msd_all = []
        for block_index, start in enumerate(range(0, len(z), block_size), 1):
            stop = min(start + block_size, len(z))
            zb = z[start:stop]
            z_rms = float(np.sqrt(np.mean((zb - np.mean(zb)) ** 2)))
            z_msd = float(np.mean((zb - z0) ** 2))
            z_rms_all.append(z_rms)
            z_msd_all.append(z_msd)
            block_rows.append({
                "system": system,
                "hexane_index": molecule,
                "block_index": block_index,
                "start_time_ps": f"{t[start]:.6f}",
                "end_time_ps": f"{t[stop - 1]:.6f}",
                "n_frames": stop - start,
                "z_mean_A": f"{np.mean(zb):.8f}",
                "z_rms_A": f"{z_rms:.8f}",
                "z_msd_from_initial_A2": f"{z_msd:.8f}",
                "z_displacement_from_initial_A": f"{np.mean(zb) - z0:.8f}",
            })
        # Also retain the per-frame cumulative MSD, which is the standard
        # one-origin MSD for the z coordinate.
        for index, row in enumerate(values):
            row["z_msd_from_initial_A2"] = f"{(z[index] - z0) ** 2:.8f}"
        summary_rows.append({
            "system": system,
            "hexane_index": molecule,
            "n_frames": len(z),
            "duration_ps": f"{t[-1] - t[0]:.6f}",
            "n_blocks_10ps": len(z_rms_all),
            "mean_10ps_z_rms_A": f"{np.mean(z_rms_all):.8f}",
            "median_10ps_z_rms_A": f"{np.median(z_rms_all):.8f}",
            "p95_10ps_z_rms_A": f"{np.percentile(z_rms_all, 95):.8f}",
            "mean_10ps_z_msd_A2": f"{np.mean(z_msd_all):.8f}",
            "final_z_msd_A2": f"{(z[-1] - z0) ** 2:.8f}",
            "max_z_msd_A2": f"{np.max((z - z0) ** 2):.8f}",
            "global_z_std_A": f"{np.std(z - np.mean(z)):.8f}",
        })

    frame_fields = list(rows[0])
    block_fields = list(block_rows[0])
    summary_fields = list(summary_rows[0])
    write_csv(output / "hexane_z_rms_msd_timeseries.csv", frame_fields, rows)
    write_csv(output / "hexane_z_rms_msd_10ps_blocks.csv", block_fields, block_rows)
    write_csv(output / "hexane_z_rms_msd_summary.csv", summary_fields, summary_rows)

    system_rows = []
    for system in sorted({r["system"] for r in summary_rows}):
        selected = [r for r in summary_rows if r["system"] == system]
        system_rows.append({
            "system": system,
            "n_hexane": len(selected),
            "mean_10ps_z_rms_A": f"{np.mean([float(r['mean_10ps_z_rms_A']) for r in selected]):.8f}",
            "mean_10ps_z_msd_A2": f"{np.mean([float(r['mean_10ps_z_msd_A2']) for r in selected]):.8f}",
            "final_z_msd_A2_mean_hexanes": f"{np.mean([float(r['final_z_msd_A2']) for r in selected]):.8f}",
            "max_z_msd_A2_across_hexanes": f"{np.max([float(r['max_z_msd_A2']) for r in selected]):.8f}",
            "global_z_std_A_mean_hexanes": f"{np.mean([float(r['global_z_std_A']) for r in selected]):.8f}",
        })
    write_csv(output / "hexane_z_rms_msd_system_summary.csv", list(system_rows[0]), system_rows)

    readme = """# 正己烷 z 向 RMS 和 MSD

- `z_rms_A`：每个 10 ps 窗口内，正己烷质心 z 坐标相对该窗口平均值的 RMS，表示局部 z 向波动幅度。
- `z_msd_from_initial_A2`：z 坐标相对该分子 500 ps 起始位置的平方位移平均，即 `<[z(t)-z(0)]^2>`，单位 Å²。
- 轨迹按 100 fs 共同比较采样；每个体系均为 5000 帧。
"""
    (output / "README_z_RMS_MSD_CN.md").write_text(readme)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    derive(args.source, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
