#!/usr/bin/env python3
"""Compare partial RDFs from the final DeepMD ML trajectory and CP2K AIMD.

The two trajectories have the same 410-atom AX1 topology and fixed
orthorhombic cell.  Frames are sampled uniformly, and g_ab(r) is normalized
to the instantaneous box volume with the usual pair-count convention.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
ML_PATH = ROOT / "md/1ns/ax1_100ps_sio2fixed_newmodel_chcorrect50k_unstable_scan_20260811/positions.100ps.ax1.sio2fixed.chcorrect50k.1fs.lammpstrj"
AIMD_PATH = ROOT / "source_raw/1-pos-1.xyz"
CELL_PATH = ROOT / "source_raw/1.cell"
MODEL_PATH = ROOT / "active_learning_chcorrect_aimd500_continue50k_w0p2_20260810/training_50k/selected_model/graph-compress-selected.pb"
SELECTION_JSON = ROOT / "active_learning_chcorrect_aimd500_continue50k_w0p2_20260810/training_50k/model_selection.json"
NEW_TEST_LOG = ROOT / "active_learning_chcorrect_aimd500_continue50k_w0p2_20260810/training_50k/selected_model/test_ch_compressed.log"
OLD_TEST_LOG = ROOT / "active_learning_chcorrect_aimd500_continue50k_w0p2_20260810/training_50k/test_final_old.log"

NATOMS = 410
BOX = np.array([9.9130, 8.5048, 73.7289], dtype=float)
PAIRS = (("Si", "O"), ("O", "O"), ("O", "H"))
RMAX = 4.2
DR = 0.01
NBINS = int(round(RMAX / DR))
EDGES = np.linspace(0.0, RMAX, NBINS + 1)
R = 0.5 * (EDGES[:-1] + EDGES[1:])
SHELL = (4.0 * np.pi / 3.0) * (EDGES[1:] ** 3 - EDGES[:-1] ** 3)


def _sample_indices(nframes: int, nmax: int) -> set[int]:
    if nframes <= nmax:
        return set(range(nframes))
    return set(np.linspace(0, nframes - 1, nmax, dtype=int).tolist())


def count_xyz_frames(path: Path) -> int:
    lines = sum(1 for _ in path.open("r"))
    if lines % (NATOMS + 2) != 0:
        raise ValueError(f"XYZ line count is not divisible by {NATOMS + 2}: {lines}")
    return lines // (NATOMS + 2)


def count_lammps_frames(path: Path) -> int:
    # Each dump frame is 9 header lines plus NATOMS atom lines for this dump style.
    lines = sum(1 for _ in path.open("r"))
    block = NATOMS + 9
    if lines % block != 0:
        raise ValueError(f"LAMMPS line count is not divisible by {block}: {lines}")
    return lines // block


def iter_xyz(path: Path, selected: set[int]):
    with path.open("r") as fh:
        for iframe in range(max(selected) + 1):
            nline = fh.readline()
            if not nline:
                break
            n = int(nline.strip())
            if n != NATOMS:
                raise ValueError(f"Unexpected XYZ atom count {n} at frame {iframe}")
            comment = fh.readline().strip()
            if iframe not in selected:
                for _ in range(n):
                    fh.readline()
                continue
            symbols = []
            xyz = np.empty((n, 3), dtype=float)
            for i in range(n):
                fields = fh.readline().split()
                symbols.append(fields[0])
                xyz[i] = [float(fields[1]), float(fields[2]), float(fields[3])]
            step_match = re.search(r"i\s*=\s*(\d+)", comment)
            step = int(step_match.group(1)) if step_match else iframe
            yield iframe, step, np.asarray(symbols), xyz


def iter_lammps(path: Path, selected: set[int]):
    with path.open("r") as fh:
        for iframe in range(max(selected) + 1):
            if fh.readline().strip() != "ITEM: TIMESTEP":
                raise ValueError(f"Unexpected LAMMPS header at frame {iframe}")
            step = int(fh.readline().strip())
            if fh.readline().strip() != "ITEM: NUMBER OF ATOMS":
                raise ValueError("Unexpected LAMMPS number-of-atoms header")
            n = int(fh.readline().strip())
            if n != NATOMS:
                raise ValueError(f"Unexpected LAMMPS atom count {n} at frame {iframe}")
            box_header = fh.readline().strip()
            if not box_header.startswith("ITEM: BOX BOUNDS"):
                raise ValueError("Unexpected LAMMPS box header")
            bounds = np.array([[float(x) for x in fh.readline().split()[:2]] for _ in range(3)])
            if fh.readline().strip() != "ITEM: ATOMS id type element x y z":
                raise ValueError("Unexpected LAMMPS atom header")
            if iframe not in selected:
                for _ in range(n):
                    fh.readline()
                continue
            symbols = []
            xyz = np.empty((n, 3), dtype=float)
            for i in range(n):
                fields = fh.readline().split()
                symbols.append(fields[2])
                xyz[i] = [float(fields[3]), float(fields[4]), float(fields[5])]
            box = bounds[:, 1] - bounds[:, 0]
            if not np.allclose(box, BOX, atol=2e-5):
                raise ValueError(f"Unexpected ML cell {box}")
            yield iframe, step, np.asarray(symbols), xyz


def pair_hist(xyz: np.ndarray, symbols: np.ndarray):
    out = {}
    for a, b in PAIRS:
        ia = np.flatnonzero(symbols == a)
        ib = np.flatnonzero(symbols == b)
        if a == b:
            ii, jj = np.triu_indices(len(ia), k=1)
            delta = xyz[ia[ii]] - xyz[ia[jj]]
            n_pairs = len(ia) * (len(ia) - 1) / 2.0
        else:
            delta = xyz[ia][:, None, :] - xyz[ib][None, :, :]
            delta = delta.reshape(-1, 3)
            n_pairs = float(len(ia) * len(ib))
        delta -= BOX * np.rint(delta / BOX)
        dist = np.sqrt(np.einsum("ij,ij->i", delta, delta))
        hist, _ = np.histogram(dist, bins=EDGES)
        out[f"{a}-{b}"] = hist.astype(float) / (n_pairs * SHELL / np.prod(BOX))
    return out


def analyze(label: str, frames, nframes: int, selected: set[int]):
    sums = {f"{a}-{b}": np.zeros(NBINS) for a, b in PAIRS}
    sums2 = {f"{a}-{b}": np.zeros(NBINS) for a, b in PAIRS}
    seen = 0
    first_step = last_step = None
    for _, step, _, xyz in frames:
        vals = pair_hist(xyz, symbols_current[0])
        for key, arr in vals.items():
            sums[key] += arr
            sums2[key] += arr * arr
        first_step = step if first_step is None else first_step
        last_step = step
        seen += 1
    if seen == 0:
        raise RuntimeError(f"No selected frames read for {label}")
    result = {"n_frames": seen, "first_step": first_step, "last_step": last_step, "curves": {}}
    for key in sums:
        mean = sums[key] / seen
        sem = np.sqrt(np.maximum(sums2[key] / seen - mean * mean, 0.0) / seen)
        result["curves"][key] = {"g": mean, "sem": sem}
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n_ml = count_lammps_frames(ML_PATH)
    n_aimd = count_xyz_frames(AIMD_PATH)
    # Exclude the first 20 ps of the ML run as equilibration; AIMD source is already
    # a continuous equilibrated CP2K segment (12.031--62.460 ps).
    ml_steps = np.arange(n_ml)
    ml_selected = set(np.linspace(int(20.0 / 0.0005 / 0.001 * 1), n_ml - 1, 1000, dtype=int))
    # Dump interval is 0.001 ps (1 fs), so 20 ps corresponds to frame 20000.
    ml_selected = set(np.linspace(20000, n_ml - 1, 1000, dtype=int).tolist())
    aimd_selected = _sample_indices(n_aimd, 1000)

    global symbols_current
    symbols_current = [None]

    def ml_frames():
        for item in iter_lammps(ML_PATH, ml_selected):
            symbols_current[0] = item[2]
            yield item

    def aimd_frames():
        for item in iter_xyz(AIMD_PATH, aimd_selected):
            symbols_current[0] = item[2]
            yield item

    # analyze() uses the symbols from the current frame; all frames have the same order.
    ml_res = analyze("ML", ml_frames(), n_ml, ml_selected)
    aimd_res = analyze("AIMD", aimd_frames(), n_aimd, aimd_selected)

    for res in (ml_res, aimd_res):
        for curve in res["curves"].values():
            curve["g"] = np.asarray(curve["g"]).tolist()
            curve["sem"] = np.asarray(curve["sem"]).tolist()

    data = {
        "system": "410-atom AX1 SiO2/oil/water interface",
        "model": str(MODEL_PATH),
        "cell_A": BOX.tolist(),
        "rmax_A": RMAX,
        "dr_A": DR,
        "normalization": "3D periodic shell-volume normalization; unique pairs for same species and all cross pairs for unlike species",
        "pairs": [f"{a}-{b}" for a, b in PAIRS],
        "ML": {"trajectory": str(ML_PATH), "all_frames": n_ml, "sample_rule": "1000 uniformly spaced frames from 20--100 ps", **ml_res},
        "AIMD": {"trajectory": str(AIMD_PATH), "all_frames": n_aimd, "sample_rule": "1000 uniformly spaced frames from source (12.031--62.460 ps)", **aimd_res},
    }
    (OUT / "rdf_ml_aimd_summary.json").write_text(json.dumps(data, indent=2))

    csv = OUT / "rdf_ml_aimd_curves.csv"
    with csv.open("w") as fh:
        fh.write("r_A,pair,ML_g,ML_sem,AIMD_g,AIMD_sem,difference_ML_minus_AIMD\n")
        for i, rr in enumerate(R):
            for a, b in PAIRS:
                key = f"{a}-{b}"
                mg = ml_res["curves"][key]["g"][i]
                ag = aimd_res["curves"][key]["g"][i]
                ms = ml_res["curves"][key]["sem"][i]
                ass = aimd_res["curves"][key]["sem"][i]
                fh.write(f"{rr:.5f},{key},{mg:.10g},{ms:.10g},{ag:.10g},{ass:.10g},{mg-ag:.10g}\n")
    with (OUT / "rdf_peak_comparison.csv").open("w") as fh:
        fh.write("pair,ML_peak_r_A,ML_peak_g,AIMD_peak_r_A,AIMD_peak_g,curve_RMSE_0_4p2\n")
        peak_mask = (R >= 0.5) & (R <= 4.0)
        for a, b in PAIRS:
            key = f"{a}-{b}"
            mg = np.asarray(ml_res["curves"][key]["g"])
            ag = np.asarray(aimd_res["curves"][key]["g"])
            mi = np.flatnonzero(peak_mask)[np.argmax(mg[peak_mask])]
            ai = np.flatnonzero(peak_mask)[np.argmax(ag[peak_mask])]
            fh.write(f"{key},{R[mi]:.3f},{mg[mi]:.6f},{R[ai]:.3f},{ag[ai]:.6f},{np.sqrt(np.mean((mg-ag)**2)):.6f}\n")

    # Publication-style quantitative grid: same y-scale and direct curve labels.
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "svg.fonttype": "none", "pdf.fonttype": 42, "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.8})
    colors = {"ML": "#1f77b4", "AIMD": "#333333"}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), sharey=True)
    global_ymax = 1.08 * max(
        max(np.max(np.asarray(res["curves"][f"{a}-{b}"]["g"])) for res in (ml_res, aimd_res))
        for a, b in PAIRS
    )
    for ax, (a, b) in zip(axes, PAIRS):
        key = f"{a}-{b}"
        for label, res in (("AIMD", aimd_res), ("ML", ml_res)):
            g = np.asarray(res["curves"][key]["g"])
            sem = np.asarray(res["curves"][key]["sem"])
            ax.plot(R, g, lw=1.25, color=colors[label], label=label)
            ax.fill_between(R, g - sem, g + sem, color=colors[label], alpha=0.12, linewidth=0)
        ax.set_title(key, fontsize=9, pad=5)
        ax.set_xlabel(r"Distance $r$ (Å)")
        ax.set_xlim(0, RMAX)
        ax.set_ylim(0, global_ymax)
        ax.axhline(1.0, color="#999999", lw=0.55, ls=(0, (2, 2)), zorder=0)
        ax.tick_params(width=0.7, length=3)
    axes[0].set_ylabel(r"$g_{ab}(r)$")
    axes[-1].legend(frameon=False, loc="upper right", handlelength=2.2)
    fig.suptitle("Final-model ML versus CP2K AIMD partial RDFs", fontsize=10, y=1.02)
    fig.tight_layout(pad=0.7, w_pad=0.8)
    stem = OUT / "rdf_ml_aimd_final_model"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    # Validation metrics: direct dp test logs are the primary reported values;
    # the selection record is retained as the auditable candidate-selection context.
    selection = json.loads(SELECTION_JSON.read_text())
    def parse_dp_test(path: Path):
        text = path.read_text()
        def grab(pattern):
            match = re.search(pattern, text)
            if not match:
                raise RuntimeError(f"Could not parse {pattern} from {path}")
            return float(match.group(1))
        return {
            "energy_rmse_eV_total": grab(r"Energy RMSE\s+:\s+([0-9.eE+-]+) eV"),
            "energy_rmse_eV_per_atom": grab(r"Energy RMSE/Natoms\s+:\s+([0-9.eE+-]+) eV"),
            "force_rmse_eV_per_A": grab(r"Force\s+RMSE\s+:\s+([0-9.eE+-]+) eV/A"),
            "virial_rmse_eV_total": grab(r"Virial RMSE\s+:\s+([0-9.eE+-]+) eV"),
            "virial_rmse_eV_per_atom": grab(r"Virial RMSE/Natoms\s+:\s+([0-9.eE+-]+) eV"),
        }
    metrics = {
        "selected_step": selection["selected_step"],
        "model": str(MODEL_PATH),
        "direct_dp_test": {"new_CH_validation": parse_dp_test(NEW_TEST_LOG), "old_validation": parse_dp_test(OLD_TEST_LOG)},
        "new_CH_validation": {k: v for k, v in selection["selected_metrics"].items() if k.startswith("ch_")},
        "old_validation": {k: v for k, v in selection["selected_metrics"].items() if k.startswith("old_")},
        "selected_model_sha256": (MODEL_PATH.parent / "model.sha256").read_text().strip(),
    }
    (OUT / "final_model_validation_rmse.json").write_text(json.dumps(metrics, indent=2))

    # DFT-versus-ML validation parity curves (main 1668-frame validation set).
    energy_file = ROOT / "active_learning_chcorrect_aimd500_continue50k_w0p2_20260810/training_50k/final_old.e.out"
    force_file = ROOT / "active_learning_chcorrect_aimd500_continue50k_w0p2_20260810/training_50k/final_old.f.out"
    energy = np.loadtxt(energy_file, comments="#") / NATOMS
    force = np.loadtxt(force_file, comments="#").reshape(-1, 6)

    def parity_plot(ref, pred, xlabel, ylabel, title, stem_name, unit, point_limit=None):
        ref = np.asarray(ref).reshape(-1)
        pred = np.asarray(pred).reshape(-1)
        slope, intercept = np.polyfit(ref, pred, 1)
        fitted = slope * ref + intercept
        ss_res = np.sum((pred - fitted) ** 2)
        ss_tot = np.sum((pred - np.mean(pred)) ** 2)
        r2 = 1.0 - ss_res / ss_tot
        rmse = np.sqrt(np.mean((pred - ref) ** 2))
        bias = np.mean(pred - ref)
        lo = min(ref.min(), pred.min())
        hi = max(ref.max(), pred.max())
        pad = 0.04 * (hi - lo if hi > lo else 1.0)
        lim = (lo - pad, hi + pad)
        if point_limit and len(ref) > point_limit:
            idx = np.linspace(0, len(ref) - 1, point_limit, dtype=int)
        else:
            idx = np.arange(len(ref))
        fig, ax = plt.subplots(figsize=(3.35, 3.05))
        ax.scatter(ref[idx], pred[idx], s=7, alpha=0.28, color="#2878B5", edgecolors="none", rasterized=True)
        line = np.array(lim)
        ax.plot(line, line, color="#222222", lw=1.0, label="Ideal: y = x")
        ax.plot(line, slope * line + intercept, color="#C4473A", lw=1.0, ls=(0, (4, 2)), label="Linear fit")
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title, xlim=lim, ylim=lim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color="#D9D9D9", lw=0.45, alpha=0.65)
        ax.legend(frameon=False, fontsize=6.5, loc="lower right")
        ax.text(0.04, 0.96, f"$R^2$ = {r2:.5f}\nRMSE = {rmse:.4g} {unit}\nFit: y = {slope:.4f}x {intercept:+.4g}", transform=ax.transAxes, va="top", fontsize=6.8, bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "linewidth": 0.5, "alpha": 0.9, "pad": 3})
        fig.tight_layout(pad=0.5)
        stem = OUT / stem_name
        fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        np.savetxt(OUT / f"{stem_name}_data.txt", np.column_stack([ref, pred]), header=f"DFT_{unit.replace('/', '_per_')} ML_{unit.replace('/', '_per_')}")
        return {"n": int(len(ref)), "r2": float(r2), "rmse": float(rmse), "bias": float(bias), "slope": float(slope), "intercept": float(intercept)}

    # Subtract one common DFT mean only for readable axes; this preserves every
    # prediction error, RMSE, slope, R2, and the visible ML energy bias.
    energy_zero = float(np.mean(energy[:, 0]))
    energy_bias = float(np.mean(energy[:, 1] - energy[:, 0]))
    energy_ml_aligned = energy[:, 1] - energy_bias
    parity_metrics = {
        "energy_per_atom_offset_corrected": parity_plot(energy[:, 0] - energy_zero, energy_ml_aligned - energy_zero, "DFT energy (eV/atom; common zero)", "Offset-corrected ML energy (eV/atom)", "Energy parity — offset corrected", "energy_linear_parity_validation_ev_per_atom", "eV/atom"),
        "force_components": parity_plot(force[:, :3].reshape(-1), force[:, 3:].reshape(-1), "DFT force (eV/Å)", "ML force (eV/Å)", "Force parity — validation", "force_linear_parity_validation_ev_per_A", "eV/Å", point_limit=30000),
    }
    parity_metrics["energy_per_atom_offset_corrected"].update({
        "common_plot_zero_eV_per_atom": energy_zero,
        "ML_minus_DFT_bias_removed_eV_per_atom": energy_bias,
        "correction_applied": "E_ML_corrected = E_ML_raw - mean(E_ML_raw - E_DFT)",
    })
    (OUT / "linear_parity_metrics.json").write_text(json.dumps(parity_metrics, indent=2))
    print(json.dumps({"ML_frames": n_ml, "AIMD_frames": n_aimd, "ML_sampled": len(ml_selected), "AIMD_sampled": len(aimd_selected), "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
