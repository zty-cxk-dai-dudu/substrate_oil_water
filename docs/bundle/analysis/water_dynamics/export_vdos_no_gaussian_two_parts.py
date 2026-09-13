#!/usr/bin/env python3
"""Export the two requested 4 A-layer VDOS sets without Gaussian smoothing."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "vdos_no_gaussian_two_parts_20260810"
SYSTEMS = (
    {
        "source": ROOT / "caf2_ax2_prl_dipole_vdos_500ps_mloong_20260809" / "vdos_4A_from_interface",
        "output": OUT_ROOT / "caf2",
        "frequency_shift_cm-1": 0.0,
        "xmax": 4000.0,
    },
    {
        "source": ROOT / "oil_water_b2_pc_angew_S6_strict_500ps_20260808" / "vdos_4A_from_oil_500ps",
        "output": OUT_ROOT / "oil_water",
        "frequency_shift_cm-1": 80.0,
        "xmax": 4080.0,
    },
)


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    for suffix, kwargs in (
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": 600}),
        ("tiff", {"dpi": 600}),
    ):
        fig.savefig(stem.with_suffix(f".{suffix}"), bbox_inches="tight", **kwargs)


def export_system(config: dict) -> None:
    source: Path = config["source"]
    output: Path = config["output"]
    shift = float(config["frequency_shift_cm-1"])
    xmax = float(config["xmax"])
    output.mkdir(parents=True, exist_ok=True)

    layer_rows = list(csv.DictReader((source / "layer_summary.csv").open()))
    spectra = []
    for row in layer_rows:
        column = row["source_data_column"]
        data = np.genfromtxt(
            source / f"vdos_{column}.csv", delimiter=",", names=True
        )
        frequency = np.asarray(data["wavenumber_cm1"], dtype=float) + shift
        raw = np.asarray(data["VDOS_raw_normalized"], dtype=float)
        normalization_mask = (frequency >= 1200.0) & (frequency <= xmax)
        scale = max(float(np.nanmax(raw[normalization_mask])), 1.0e-30)
        raw = raw / scale
        spectra.append((column, frequency, raw))

        with (output / f"vdos_{column}.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["wavenumber_cm-1", "VDOS_no_gaussian_normalized"])
            for x, y in zip(frequency, raw):
                if 0.0 <= x <= xmax:
                    writer.writerow([x, y])

    with (output / "vdos_layers_source_data.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["wavenumber_cm-1"] + [item[0] for item in spectra])
        reference_x = spectra[0][1]
        export_mask = (reference_x >= 0.0) & (reference_x <= xmax)
        for index in np.flatnonzero(export_mask):
            writer.writerow(
                [reference_x[index]] + [item[2][index] for item in spectra]
            )

    colors = plt.get_cmap("cividis")(
        np.linspace(0.12, 0.88, len(layer_rows))
    )
    offset = 1.14
    fig, ax = plt.subplots(figsize=(3.6, 4.7), constrained_layout=True)
    for index, (row, spectrum, color) in enumerate(
        zip(layer_rows, spectra, colors)
    ):
        _, frequency, raw = spectrum
        mask = (frequency >= 1200.0) & (frequency <= xmax)
        base = index * offset
        ax.fill_between(
            frequency[mask], base, base + raw[mask], color=color,
            alpha=0.22, linewidth=0,
        )
        ax.plot(frequency[mask], base + raw[mask], color=color, lw=0.8)
        ax.text(
            2500.0, base + 0.55,
            f"{float(row['relative_z_min_A']):g}–{float(row['relative_z_max_A']):g} Å "
            f"(n={row['water_count']})",
            ha="center", va="center", fontsize=5.9, color=color,
        )
    ax.set(
        xlim=(1200.0, xmax),
        ylim=(-0.08, (len(layer_rows) - 1) * offset + 1.08),
        xlabel=r"Wavenumber (cm$^{-1}$)",
        ylabel="Normalized water VDOS (offset)",
    )
    ax.set_yticks([])
    save_figure(fig, output / "vdos_layers_1200_end")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
    for row, spectrum, color in zip(layer_rows, spectra, colors):
        _, frequency, raw = spectrum
        mask = (frequency >= 2700.0) & (frequency <= xmax)
        ax.plot(
            frequency[mask], raw[mask], color=color, lw=0.8,
            label=(
                f"{float(row['relative_z_min_A']):g}–"
                f"{float(row['relative_z_max_A']):g} Å"
            ),
        )
    ax.set(
        xlim=(2700.0, xmax),
        ylim=(0.0, None),
        xlabel=r"Wavenumber (cm$^{-1}$)",
        ylabel="Normalized water VDOS",
    )
    ax.legend(frameon=False, fontsize=5.8, ncol=2)
    save_figure(fig, output / "vdos_oh_2700_end")
    plt.close(fig)

    metadata = {
        "source_directory": str(source.relative_to(ROOT)),
        "spectrum": "50-block mean Blackman velocity PSD without Gaussian smoothing",
        "normalization": "each layer divided by its own unsmoothed maximum over the plotted 1200-end range",
        "frequency_shift_cm-1": shift,
        "frequency_shift_is_not_annotated_in_figures": True,
        "peak_markers": False,
        "layers": len(layer_rows),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    for image_path in output.glob("*.png"):
        with Image.open(image_path) as image:
            assert image.width > 1000 and image.height > 1000
    for table_path in output.glob("*.csv"):
        table = np.genfromtxt(table_path, delimiter=",", names=True)
        for name in table.dtype.names or ():
            if np.issubdtype(table[name].dtype, np.number):
                assert np.all(np.isfinite(table[name]))


def main() -> None:
    for config in SYSTEMS:
        export_system(config)
    print(OUT_ROOT)


if __name__ == "__main__":
    main()
