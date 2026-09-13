#!/usr/bin/env python3
"""Analyze trajectories and dipole/radial orientation of selected waters.

The oxygen trajectory is continuously unwrapped with minimum-image fractional
increments.  Each H atom is rebuilt in the nearest periodic image around its O.
The molecular orientation vector is O -> midpoint(H1,H2).  The instantaneous
radial direction is the minimum-image in-plane vector from the xy cell center
to the wrapped oxygen position.  theta_radial is arccos(mu_hat dot r_hat), in
the range 0--180 degrees.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


POTIM_FS = 1.0


def read_selected(path: Path):
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    return [
        {
            "rank": int(r["rank"]),
            "O": int(r["O_serial"]),
            "H1": int(r["H1_serial"]),
            "H2": int(r["H2_serial"]),
            "label": f'O{r["O_serial"]}-H{r["H1_serial"]}-H{r["H2_serial"]}',
        }
        for r in rows
    ]


def parse_xdatcar(path: Path, selected):
    wanted = sorted({x[k] for x in selected for k in ("O", "H1", "H2")})
    wanted_zero = {i - 1: i for i in wanted}
    frames = []
    with path.open() as fh:
        title = fh.readline().rstrip("\n")
        scale = float(fh.readline().split()[0])
        cell = np.array([[float(v) for v in fh.readline().split()[:3]] for _ in range(3)]) * scale
        species = fh.readline().split()
        counts = [int(v) for v in fh.readline().split()]
        natoms = sum(counts)
        frame_re = re.compile(r"(?:Direct|Cartesian) configuration=\s*(\d+)")
        while True:
            line = fh.readline()
            if not line:
                break
            m = frame_re.search(line)
            if not m:
                continue
            config = int(m.group(1))
            coord_mode = "Cartesian" if line.lstrip().startswith("Cartesian") else "Direct"
            chosen = {}
            for idx in range(natoms):
                vals = fh.readline().split()
                if len(vals) < 3:
                    raise RuntimeError(f"Truncated coordinates at configuration {config}, atom {idx+1}")
                if idx in wanted_zero:
                    chosen[wanted_zero[idx]] = np.array([float(vals[0]), float(vals[1]), float(vals[2])])
            if coord_mode == "Cartesian":
                inv = np.linalg.inv(cell)
                chosen = {serial: xyz @ inv for serial, xyz in chosen.items()}
            if len(chosen) != len(wanted):
                raise RuntimeError(f"Missing selected atoms at configuration {config}")
            frames.append((config, chosen))
    if not frames:
        raise RuntimeError("No XDATCAR frames found")
    return title, cell, species, counts, frames


def frac_minimum_image(delta):
    return delta - np.round(delta)


def analyze_one(water, frames, cell, potim_fs=POTIM_FS):
    rows = []
    prev_o = None
    unwrapped_o = None
    for frame_index, (config, coords) in enumerate(frames, start=1):
        o = coords[water["O"]]
        h1 = coords[water["H1"]]
        h2 = coords[water["H2"]]
        if prev_o is None:
            unwrapped_o = o.copy()
        else:
            unwrapped_o = unwrapped_o + frac_minimum_image(o - prev_o)
        prev_o = o.copy()

        h1_rel_frac = frac_minimum_image(h1 - o)
        h2_rel_frac = frac_minimum_image(h2 - o)
        o_cart = o @ cell
        ou_cart = unwrapped_o @ cell
        h1_cart = ou_cart + h1_rel_frac @ cell
        h2_cart = ou_cart + h2_rel_frac @ cell
        mu = 0.5 * (h1_rel_frac + h2_rel_frac) @ cell
        mu_norm = float(np.linalg.norm(mu))
        mu_hat = mu / mu_norm

        # Minimum-image xy direction from lateral cell center to wrapped O.
        r_frac = frac_minimum_image(o - np.array([0.5, 0.5, o[2]]))
        r_frac[2] = 0.0
        radial = r_frac @ cell
        radial_norm = float(np.linalg.norm(radial))
        if radial_norm < 1.0e-12:
            theta = float("nan")
        else:
            cosang = float(np.clip(np.dot(mu_hat, radial / radial_norm), -1.0, 1.0))
            theta = math.degrees(math.acos(cosang))
        theta_z = math.degrees(math.acos(float(np.clip(mu_hat[2], -1.0, 1.0))))
        phi_xy = math.degrees(math.atan2(mu_hat[1], mu_hat[0]))

        rows.append({
            "water_rank": water["rank"],
            "water": water["label"],
            "O_serial": water["O"],
            "H1_serial": water["H1"],
            "H2_serial": water["H2"],
            "frame_index": frame_index,
            "configuration": config,
            "time_ps_relative": (frame_index - 1) * potim_fs / 1000.0,
            "O_x_A": ou_cart[0], "O_y_A": ou_cart[1], "O_z_A": ou_cart[2],
            "O_x_wrapped_A": o_cart[0], "O_y_wrapped_A": o_cart[1], "O_z_wrapped_A": o_cart[2],
            "H1_x_A": h1_cart[0], "H1_y_A": h1_cart[1], "H1_z_A": h1_cart[2],
            "H2_x_A": h2_cart[0], "H2_y_A": h2_cart[1], "H2_z_A": h2_cart[2],
            "mu_x": mu_hat[0], "mu_y": mu_hat[1], "mu_z": mu_hat[2],
            "radial_x": radial[0] / radial_norm if radial_norm else float("nan"),
            "radial_y": radial[1] / radial_norm if radial_norm else float("nan"),
            "theta_radial_deg": theta,
            "theta_z_deg": theta_z,
            "phi_xy_deg": phi_xy,
        })
    return rows


def write_csv(path, rows, delimiter=","):
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def plot_trajectories(all_rows, waters, out):
    fig, axes = plt.subplots(3, 2, figsize=(11.0, 13.0), constrained_layout=True)
    for i, water in enumerate(waters):
        rows = [r for r in all_rows if r["water_rank"] == water["rank"]]
        x = np.array([r["O_x_A"] for r in rows])
        y = np.array([r["O_y_A"] for r in rows])
        z = np.array([r["O_z_A"] for r in rows])
        t = np.array([r["time_ps_relative"] for r in rows])
        mux = np.array([r["mu_x"] for r in rows])
        muy = np.array([r["mu_y"] for r in rows])
        muz = np.array([r["mu_z"] for r in rows])
        for ax, xx, yy, uu, vv, xlabel, ylabel, projection in [
            (axes[i, 0], x, y, mux, muy, "x (Å, unwrapped)", "y (Å, unwrapped)", "XY"),
            (axes[i, 1], y, z, muy, muz, "y (Å, unwrapped)", "z (Å, unwrapped)", "YZ"),
        ]:
            sc = ax.scatter(xx, yy, c=t, s=2.0, cmap="viridis", rasterized=True)
            stride = max(1, len(rows) // 35)
            ids = np.arange(0, len(rows), stride)
            ax.quiver(xx[ids], yy[ids], uu[ids], vv[ids], color="black", alpha=0.42,
                      angles="xy", scale_units="xy", scale=3.0, width=0.0025)
            ax.scatter(xx[0], yy[0], s=28, marker="o", facecolor="white", edgecolor="black", label="start")
            ax.scatter(xx[-1], yy[-1], s=35, marker="X", color="#d62728", label="end")
            ax.set(xlabel=xlabel, ylabel=ylabel, title=f'{water["label"]}: {projection} trajectory')
            ax.grid(alpha=0.2)
            ax.legend(frameon=False, fontsize=8)
            cb = fig.colorbar(sc, ax=ax, pad=0.01)
            cb.set_label("time from first stored frame (ps)")
    for ext in ("png", "svg"):
        fig.savefig(out / f"three_water_XY_YZ_trajectories.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)


def plot_angles(all_rows, waters, out):
    fig, axes = plt.subplots(3, 1, figsize=(11.0, 9.0), sharex=True, constrained_layout=True)
    for ax, water in zip(axes, waters):
        rows = [r for r in all_rows if r["water_rank"] == water["rank"]]
        t = np.array([r["time_ps_relative"] for r in rows])
        theta = np.array([r["theta_radial_deg"] for r in rows])
        # 101-frame centered moving average for readability (~0.101 ps).
        kernel = np.ones(101) / 101.0
        smooth = np.convolve(np.pad(theta, (50, 50), mode="edge"), kernel, mode="valid")
        ax.plot(t, theta, color="#4c78a8", lw=0.35, alpha=0.42, label="instantaneous")
        ax.plot(t, smooth, color="#e45756", lw=1.0, label="101-frame mean")
        ax.set(ylabel=r"$\theta_r$ (deg)", title=water["label"], ylim=(0, 180))
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, ncol=2, fontsize=8)
    axes[-1].set_xlabel("time from first stored frame (ps)")
    for ext in ("png", "svg"):
        fig.savefig(out / f"three_water_radial_orientation_vs_time.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xdatcar", type=Path, required=True, help="XDATCAR with one stored frame per MD step")
    parser.add_argument("--selected", type=Path, required=True, help="TSV with rank, O_serial, H1_serial and H2_serial for three waters")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--potim-fs", type=float, default=POTIM_FS, help="MD timestep in fs (default: 1.0)")
    args = parser.parse_args()
    if not math.isfinite(args.potim_fs) or args.potim_fs <= 0:
        parser.error("--potim-fs must be positive and finite")
    waters = read_selected(args.selected)
    if len(waters) != 3:
        parser.error("--selected must contain exactly three waters")
    title, cell, species, counts, frames = parse_xdatcar(args.xdatcar, waters)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summaries = []
    for water in waters:
        rows = analyze_one(water, frames, cell, args.potim_fs)
        all_rows.extend(rows)
        write_csv(out / f'{water["label"]}_trajectory_orientation.csv', rows)
        theta = np.array([r["theta_radial_deg"] for r in rows])
        xyz = np.array([[r["O_x_A"], r["O_y_A"], r["O_z_A"]] for r in rows])
        path_length = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
        summaries.append({
            "water_rank": water["rank"], "water": water["label"],
            "n_frames": len(rows), "time_span_ps_relative": rows[-1]["time_ps_relative"],
            "mean_theta_radial_deg": float(np.nanmean(theta)),
            "std_theta_radial_deg": float(np.nanstd(theta)),
            "min_theta_radial_deg": float(np.nanmin(theta)),
            "max_theta_radial_deg": float(np.nanmax(theta)),
            "O_path_length_A": path_length,
        })
    write_csv(out / "source_data_all_waters.csv", all_rows)
    write_csv(out / "orientation_summary.tsv", summaries, delimiter="\t")
    plot_trajectories(all_rows, waters, out)
    plot_angles(all_rows, waters, out)
    metadata = {
        "source_xdatcar": str(args.xdatcar.resolve()), "source_selected_waters": str(args.selected.resolve()),
        "title": title, "species": species, "counts": counts,
        "cell_A": cell.tolist(), "n_frames": len(frames),
        "first_configuration": frames[0][0], "last_configuration": frames[-1][0],
        "POTIM_fs": args.potim_fs, "frame_stride_steps": 1,
        "relative_time_definition": "(frame_index - 1) * POTIM / 1000 ps",
        "oxygen_unwrapping": "minimum-image fractional displacement between consecutive frames",
        "hydrogen_reconstruction": "each H in nearest periodic image around its O in every frame",
        "orientation_vector": "unit vector from O to midpoint(H1,H2)",
        "radial_vector": "minimum-image in-plane vector from xy cell center (fractional 0.5,0.5) to wrapped O",
        "theta_radial_definition": "arccos(mu_hat dot radial_hat), degrees, range 0-180",
        "trajectory_plot": "continuously unwrapped O path; arrows show projected O-to-H-midpoint unit vector",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    (out / "README.md").write_text(
        "# Three middle-water trajectory and radial-orientation analysis\n\n"
        "The oxygen trajectory is continuously unwrapped using minimum-image fractional increments. "
        "Each hydrogen is rebuilt in the nearest image around its oxygen. The molecular orientation "
        "vector is O to the midpoint of H1/H2. The radial vector is the shortest in-plane vector from "
        "the xy cell center to the oxygen. `theta_radial_deg = acos(mu_hat dot radial_hat)`; 0 degrees "
        "points outward and 180 degrees points inward. Time is relative to the first stored XDATCAR frame.\n"
    )
    print(json.dumps({"out": str(out), "n_frames": len(frames), "waters": waters}, ensure_ascii=False))


if __name__ == "__main__":
    main()
