#!/usr/bin/env python3
"""Water VDOS in fixed 4 A layers measured upward from an interface GDS."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import h5py
import numpy as np


CM_PER_PS = 33.3564095198152


def water_topology(symbols: np.ndarray, positions: np.ndarray,
                   cell: np.ndarray) -> np.ndarray:
    oxygen = np.flatnonzero(symbols == "O")
    hydrogen = np.flatnonzero(symbols == "H")
    box = np.diag(cell)
    candidates: list[tuple[float, int, int]] = []
    for o in oxygen:
        delta = positions[hydrogen] - positions[o]
        delta -= box * np.rint(delta / box)
        candidates.extend(
            (float(distance), int(o), int(h))
            for h, distance in zip(hydrogen, np.linalg.norm(delta, axis=1))
            if distance <= 1.30
        )
    candidates.sort()
    assigned = {int(o): [] for o in oxygen}
    used: set[int] = set()
    for _, o, h in candidates:
        if len(assigned[o]) < 2 and h not in used:
            assigned[o].append(h)
            used.add(h)
    waters = [(o, *assigned[int(o)]) for o in oxygen
              if len(assigned[int(o)]) == 2]
    if not waters:
        raise RuntimeError("No complete H2O topology found")
    return np.asarray(waters, dtype=int)


def gaussian(values: np.ndarray, frequency: np.ndarray,
             sigma_cm: float) -> np.ndarray:
    spacing = float(np.median(np.diff(frequency)))
    radius = max(2, int(np.ceil(4.0 * sigma_cm / spacing)))
    x = np.arange(-radius, radius + 1, dtype=float) * spacing
    kernel = np.exp(-0.5 * (x / sigma_cm) ** 2)
    kernel /= kernel.sum()
    return np.convolve(np.pad(values, radius, mode="edge"), kernel, mode="valid")


def block_psd(velocity: np.ndarray, dt_fs: float) -> tuple[np.ndarray, np.ndarray]:
    velocity = velocity - velocity.mean(axis=0, keepdims=True)
    window = np.blackman(len(velocity))
    nfft = 1 << (4 * len(velocity) - 1).bit_length()
    transformed = np.fft.rfft(
        velocity * window[:, None, None], n=nfft, axis=0
    )
    frequency = (
        np.fft.rfftfreq(nfft, d=dt_fs * 1.0e-3) * CM_PER_PS
    )
    psd = (np.abs(transformed) ** 2).mean(axis=(1, 2)) / np.sum(window ** 2)
    return frequency, psd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path,
                        help="Directory containing normalized segment_*.h5 files")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--z-origin", required=True, type=float,
                        help="Oil/water Gibbs dividing surface in angstrom")
    parser.add_argument("--layer-width", type=float, default=4.0)
    parser.add_argument("--block-ps", type=float, default=10.0)
    parser.add_argument("--smooth-sigma-cm", type=float, default=15.0)
    parser.add_argument("--system-label", default="Oil/water")
    args = parser.parse_args()
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)

    segments = sorted(args.input.glob("segment_*.h5"))
    if not segments or len(segments) % 2:
        raise RuntimeError("Expected an even, nonzero set of segment_*.h5 files")
    with h5py.File(segments[0], "r") as handle:
        symbols = np.asarray(handle["symbols"]).astype(str)
        positions0 = np.asarray(handle["positions_A"][0], dtype=float)
        cell = np.asarray(handle["cell_A"], dtype=float)
        dt_fs = float(handle.attrs["output_interval_fs"])
        frames_per_segment = int(handle["positions_A"].shape[0])
        first_time_ps = float(handle["time_ps"][0])
    if not np.allclose(cell, np.diag(np.diag(cell)), atol=1.0e-7):
        raise RuntimeError("Orthorhombic cell required")
    box = np.diag(cell)
    waters = water_topology(symbols, positions0, cell)
    oxygen = waters[:, 0]

    z_sum = np.zeros(len(waters), dtype=float)
    total_frames = 0
    last_time_ps = first_time_ps
    for path in segments:
        with h5py.File(path, "r") as handle:
            if not bool(handle.attrs.get("complete", False)):
                raise RuntimeError(f"Incomplete segment: {path}")
            z = np.asarray(handle["positions_A"][:, oxygen, 2], dtype=float) % box[2]
            z_sum += z.sum(axis=0)
            total_frames += len(z)
            last_time_ps = float(handle["time_ps"][-1])
    mean_z = z_sum / total_frames
    relative_z = mean_z - args.z_origin
    layer_id = np.floor(relative_z / args.layer_width).astype(int)
    valid = layer_id >= 0
    excluded = np.flatnonzero(~valid)
    layer_records = []
    for index in range(int(layer_id[valid].max()) + 1):
        water_indices = np.flatnonzero(valid & (layer_id == index))
        if not len(water_indices):
            continue
        atoms = np.sort(waters[water_indices].ravel())
        rel_lo = index * args.layer_width
        rel_hi = rel_lo + args.layer_width
        layer_records.append({
            "index": index,
            "column": f"L{index + 1}_{rel_lo:g}_{rel_hi:g}A",
            "relative_lo": rel_lo,
            "relative_hi": rel_hi,
            "absolute_lo": args.z_origin + rel_lo,
            "absolute_hi": args.z_origin + rel_hi,
            "water_indices": water_indices,
            "atoms": atoms,
        })
    if not layer_records:
        raise RuntimeError("No waters lie above the requested z origin")
    print(json.dumps({
        "water_molecules_total": int(len(waters)),
        "waters_assigned_above_GDS": int(valid.sum()),
        "waters_excluded_below_GDS_by_mean_z": int(len(excluded)),
        "layers": [
            {
                "relative_z_A": [record["relative_lo"], record["relative_hi"]],
                "absolute_z_A": [record["absolute_lo"], record["absolute_hi"]],
                "water_count": int(len(record["water_indices"])),
            }
            for record in layer_records
        ],
    }), flush=True)

    frames_per_block = int(round(args.block_ps * 1000.0 / dt_fs))
    segments_per_block = frames_per_block // frames_per_segment
    if segments_per_block * frames_per_segment != frames_per_block:
        raise RuntimeError("Block length is not an integer number of segments")
    if len(segments) % segments_per_block:
        raise RuntimeError("Segments do not fill complete spectral blocks")
    blocks = len(segments) // segments_per_block
    sums = [None] * len(layer_records)
    sums_sq = [None] * len(layer_records)
    frequency = None
    for block in range(blocks):
        paths = segments[
            block * segments_per_block:(block + 1) * segments_per_block
        ]
        parts = []
        for path in paths:
            with h5py.File(path, "r") as handle:
                parts.append(np.asarray(handle["velocities_A_per_fs"], dtype=float))
        velocities = np.concatenate(parts, axis=0)
        for k, record in enumerate(layer_records):
            frequency, psd = block_psd(velocities[:, record["atoms"], :], dt_fs)
            if sums[k] is None:
                sums[k] = np.zeros_like(psd)
                sums_sq[k] = np.zeros_like(psd)
            sums[k] += psd
            sums_sq[k] += psd ** 2
        if block == 0 or (block + 1) % 10 == 0:
            print(json.dumps({"blocks_done": block + 1, "blocks_total": blocks}),
                  flush=True)

    export_mask = (frequency >= 0.0) & (frequency <= 4000.0)
    plot_mask = (frequency >= 1200.0) & (frequency <= 4000.0)
    oh_mask = (frequency >= 3000.0) & (frequency <= 4000.0)
    bend_mask = (frequency >= 1400.0) & (frequency <= 1800.0)
    spectra = []
    spectra_sem = []
    raw_spectra = []
    peak_rows = []
    for k, record in enumerate(layer_records):
        mean = sums[k] / blocks
        variance = np.maximum(
            (sums_sq[k] - blocks * mean ** 2) / (blocks - 1), 0.0
        )
        sem = np.sqrt(variance / blocks)
        smooth = gaussian(mean, frequency, args.smooth_sigma_cm)
        smooth_sem = gaussian(sem, frequency, args.smooth_sigma_cm)
        scale = max(float(smooth[plot_mask].max()), 1.0e-30)
        raw_spectra.append(mean / scale)
        spectra.append(smooth / scale)
        spectra_sem.append(smooth_sem / scale)
        oh_index = np.flatnonzero(oh_mask)[np.argmax(smooth[oh_mask])]
        bend_index = np.flatnonzero(bend_mask)[np.argmax(smooth[bend_mask])]
        peak_rows.append([
            record["index"] + 1, record["relative_lo"], record["relative_hi"],
            record["absolute_lo"], record["absolute_hi"],
            len(record["water_indices"]), frequency[bend_index],
            frequency[oh_index], smooth[oh_index] / scale,
        ])
    spectra = np.asarray(spectra)
    spectra_sem = np.asarray(spectra_sem)
    raw_spectra = np.asarray(raw_spectra)

    with (args.output / "layer_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "layer_index", "relative_z_min_A", "relative_z_max_A",
            "absolute_z_min_A", "absolute_z_max_A", "water_count",
            "atom_count", "O_serials_1based", "source_data_column",
        ])
        for record in layer_records:
            wi = record["water_indices"]
            writer.writerow([
                record["index"] + 1, record["relative_lo"], record["relative_hi"],
                record["absolute_lo"], record["absolute_hi"], len(wi),
                len(record["atoms"]), ";".join(map(str, (oxygen[wi] + 1).tolist())),
                record["column"],
            ])
    with (args.output / "layer_peak_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "layer_index", "relative_z_min_A", "relative_z_max_A",
            "absolute_z_min_A", "absolute_z_max_A", "water_count",
            "bend_peak_cm-1", "OH_stretch_peak_3000_4000_cm-1",
            "OH_peak_normalized_amplitude",
        ])
        writer.writerows(peak_rows)
    with (args.output / "water_mean_z.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "water_index", "O_serial_1based", "mean_O_z_A",
            "z_minus_GDS_A", "layer_index",
        ])
        for index, (o, z, rel, lid) in enumerate(
            zip(oxygen, mean_z, relative_z, layer_id), start=1
        ):
            writer.writerow([index, o + 1, z, rel, lid + 1 if lid >= 0 else "excluded"])

    with (args.output / "vdos_layers_4A_source_data.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        header = ["wavenumber_cm-1"]
        for record in layer_records:
            header.extend([
                record["column"], record["column"] + "_block_SEM"
            ])
        writer.writerow(header)
        for j in np.flatnonzero(export_mask):
            row = [frequency[j]]
            for k in range(len(layer_records)):
                row.extend([spectra[k, j], spectra_sem[k, j]])
            writer.writerow(row)
    for k, record in enumerate(layer_records):
        with (args.output / f"vdos_{record['column']}.csv").open(
            "w", newline=""
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow([
                "wavenumber_cm-1", "VDOS_raw_normalized",
                "VDOS_smoothed_normalized", "VDOS_smoothed_block_SEM",
            ])
            for j in np.flatnonzero(export_mask):
                writer.writerow([
                    frequency[j], raw_spectra[k, j], spectra[k, j],
                    spectra_sem[k, j],
                ])

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.size": 7,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False,
    })
    colors = plt.get_cmap("cividis")(
        np.linspace(0.12, 0.88, len(layer_records))
    )
    fig, ax = plt.subplots(figsize=(3.6, 4.7), constrained_layout=True)
    offset = 1.14
    for k, (record, color) in enumerate(zip(layer_records, colors)):
        base = k * offset
        ax.fill_between(
            frequency[plot_mask], base,
            base + spectra[k, plot_mask], color=color, alpha=0.22,
            linewidth=0,
        )
        ax.plot(frequency[plot_mask], base + spectra[k, plot_mask],
                color=color, lw=1.0)
        ax.text(
            3980, base + 0.55,
            f"{record['relative_lo']:g}–{record['relative_hi']:g} Å (n={len(record['water_indices'])})",
            ha="right", va="center", fontsize=5.9, color=color,
        )
    ax.axvspan(1550, 1750, color="0.5", alpha=0.05, linewidth=0)
    ax.axvspan(3000, 3700, color="#E69F00", alpha=0.045, linewidth=0)
    ax.set(
        xlim=(1200, 4000),
        ylim=(-0.08, (len(layer_records) - 1) * offset + 1.08),
        xlabel=r"Wavenumber (cm$^{-1}$)",
        ylabel="Normalized water VDOS (offset)",
        title=f"{args.system_label}: 4 Å layers above oil",
    )
    ax.set_yticks([])
    for ext in ("png", "pdf", "svg"):
        fig.savefig(args.output / f"vdos_water_4A_from_oil_1200_4000.{ext}",
                    dpi=600 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
    oh_plot_mask = (frequency >= 2700.0) & (frequency <= 4000.0)
    for k, (record, color, peak) in enumerate(
        zip(layer_records, colors, peak_rows)
    ):
        ax.plot(frequency[oh_plot_mask], spectra[k, oh_plot_mask],
                color=color, lw=1.0,
                label=f"{record['relative_lo']:g}–{record['relative_hi']:g} Å")
        ax.plot(peak[7], peak[8], marker="o", ms=2.5, color=color)
    ax.set(
        xlim=(2700, 4000),
        xlabel=r"Wavenumber (cm$^{-1}$)", ylabel="Normalized water VDOS",
        title=f"{args.system_label}: O–H stretch by 4 Å layer",
    )
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=5.8, ncol=2)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(args.output / f"vdos_water_4A_from_oil_OH_2700_4000.{ext}",
                    dpi=600 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "status": "complete",
        "method": "one-sided Cartesian velocity PSD averaged over water atoms/components and 10 ps Blackman-tapered blocks",
        "trajectory_segments": len(segments),
        "trajectory_frames": total_frames,
        "trajectory_time_ps": [first_time_ps, last_time_ps],
        "sampling_interval_fs": dt_fs,
        "water_molecules_total": int(len(waters)),
        "waters_assigned_above_GDS": int(valid.sum()),
        "waters_excluded_below_GDS_by_mean_z": int(len(excluded)),
        "z_origin_GDS_A": args.z_origin,
        "layer_width_A": args.layer_width,
        "layer_membership": "fixed by each water oxygen mean wrapped Cartesian z over the full 500 ps trajectory",
        "layers": len(layer_records),
        "spectral_blocks": blocks,
        "spectral_block_ps": args.block_ps,
        "window": "Blackman",
        "zero_padding": "next power of two at least four times the 10 ps block length",
        "display_gaussian_sigma_cm-1": args.smooth_sigma_cm,
        "normalization": "each layer divided by its own smoothed maximum over 1200-4000 cm-1",
        "export_frequency_range_cm-1": [0.0, 4000.0],
        "OH_peak_search_range_cm-1": [3000.0, 4000.0],
        "system_label": args.system_label,
        "elapsed_s": time.time() - started,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
