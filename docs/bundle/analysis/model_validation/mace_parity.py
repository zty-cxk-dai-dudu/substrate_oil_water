#!/usr/bin/env python3
"""Latest Youshui/CaF2 MACE validation parity plots and trajectory audit.

Energy is plotted after removing only the constant mean (MACE-DFT) offset;
forces are plotted without any correction.  All metrics are evaluated on the
complete validation sets.  The script deliberately does not infer an RDF from
an unreadable trajectory file: the CaF2 mloong HDF5 payload is audited and the
failure is written to trajectory_provenance.json instead.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
import h5py


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "mace_youshui_caf2_latest_validation_parity_20260813"

SYSTEMS = {
    "youshui": {
        "label": "Youshui",
        "model": ROOT / "youshui_mace_all_current_20260805/youshui_mace_all_current_zbl_20260805.model",
        "validation": ROOT / "youshui_mace_all_current_20260805/dataset/valid.xyz",
        "trajectory": ROOT / "youshui_mace_all_current_20260805/md_500ps_300K_velocity_1fs_h5_epoch76_20260806",
        "segments": ROOT / "youshui_mace_all_current_20260805/md_500ps_300K_velocity_1fs_h5_epoch76_20260806/velocity_h5_1fs",
        "complete_marker": ROOT / "youshui_mace_all_current_20260805/md_500ps_300K_velocity_1fs_h5_epoch76_20260806/MD500PS_COMPLETE",
        "color": "#0072B2",
    },
    "caf2": {
        "label": "CaF$_2$",
        "model": ROOT / "caf2_mace_local_a2_vacuum5A_500ps_20260807/caf2_mace_weight0p25_seed20260805.model",
        "validation": ROOT / "caf2_nequip_allegro_dataset_20260805/validation_base403.xyz",
        "trajectory": ROOT / "caf2_mace_mloong_20260805/md_500ps_vdos_1fs",
        "segments": ROOT / "caf2_mace_mloong_20260805/md_500ps_vdos_1fs/segments",
        "complete_marker": ROOT / "caf2_mace_mloong_20260805/md_500ps_vdos_1fs/md500ps.complete",
        "color": "#D55E00",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite_stats(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    return np.asarray(x)[mask], np.asarray(y)[mask]


def fit_metrics(ref: np.ndarray, pred: np.ndarray) -> dict:
    ref, pred = finite_stats(ref, pred)
    slope, intercept = np.polyfit(ref, pred, 1)
    fitted = slope * ref + intercept
    ss_res = float(np.sum((pred - fitted) ** 2))
    ss_tot = float(np.sum((pred - pred.mean()) ** 2))
    return {
        "n": int(ref.size),
        "rmse": float(np.sqrt(np.mean((pred - ref) ** 2))),
        "mae": float(np.mean(np.abs(pred - ref))),
        "r2_linear_fit": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "fit_slope": float(slope),
        "fit_intercept": float(intercept),
        "ref_min": float(ref.min()),
        "ref_max": float(ref.max()),
        "pred_min": float(pred.min()),
        "pred_max": float(pred.max()),
    }


def read_reference(atoms):
    # Youshui stores REF_* directly.  CaF2 validation_base403 is an extxyz
    # SinglePointCalculator with energy/forces, so capture those before replacing
    # the calculator with MACE.
    if "REF_energy" in atoms.info:
        e = float(atoms.info["REF_energy"])
    else:
        e = float(atoms.get_potential_energy())
    if "REF_forces" in atoms.arrays:
        f = np.asarray(atoms.arrays["REF_forces"], dtype=float)
    else:
        f = np.asarray(atoms.get_forces(), dtype=float)
    return e, f


def evaluate_system(key: str, spec: dict) -> dict:
    from mace.calculators import MACECalculator

    frames = read(spec["validation"], index=":")
    calc = MACECalculator(model_paths=str(spec["model"]), device="cuda", default_dtype="float32")
    ref_e, pred_e = [], []
    ref_f, pred_f = [], []
    for i, atoms in enumerate(frames):
        e_ref, f_ref = read_reference(atoms)
        atoms.calc = calc
        e_pred = float(atoms.get_potential_energy())
        f_pred = np.asarray(atoms.get_forces(), dtype=float)
        ref_e.append(e_ref)
        pred_e.append(e_pred)
        ref_f.append(f_ref)
        pred_f.append(f_pred)
        if (i + 1) % 50 == 0 or i + 1 == len(frames):
            print(f"{key}: evaluated {i + 1}/{len(frames)}", flush=True)
    ref_e = np.asarray(ref_e, dtype=float)
    pred_e = np.asarray(pred_e, dtype=float)
    ref_f = np.asarray(ref_f, dtype=float)
    pred_f = np.asarray(pred_f, dtype=float)
    bias = float(np.mean(pred_e - ref_e))
    pred_e_corr = pred_e - bias
    metrics = {
        "system": key,
        "label": spec["label"],
        "validation": str(spec["validation"]),
        "validation_frames": int(len(frames)),
        "atoms_per_frame": int(len(frames[0])),
        "model": str(spec["model"]),
        "model_sha256": sha256(spec["model"]),
        "energy_bias_raw_mace_minus_dft_eV": bias,
        "energy": {
            "raw_total_eV": fit_metrics(ref_e, pred_e),
            "corrected_total_eV": fit_metrics(ref_e, pred_e_corr),
            "raw_per_atom_meV": fit_metrics(ref_e / len(frames[0]) * 1000.0, pred_e / len(frames[0]) * 1000.0),
            "corrected_per_atom_meV": fit_metrics(ref_e / len(frames[0]) * 1000.0, pred_e_corr / len(frames[0]) * 1000.0),
            "correction": "predicted total energy minus mean(predicted total energy - DFT total energy); no slope rescaling",
        },
        "force": fit_metrics(ref_f.ravel(), pred_f.ravel()),
    }
    np.savez_compressed(
        OUT / f"{key}_parity_source_data.npz",
        energy_ref_eV=ref_e,
        energy_mace_raw_eV=pred_e,
        energy_mace_corrected_eV=pred_e_corr,
        forces_ref_eV_A=ref_f,
        forces_mace_eV_A=pred_f,
    )
    return {"metrics": metrics, "energy_ref": ref_e, "energy_pred": pred_e_corr,
            "force_ref": ref_f.ravel(), "force_pred": pred_f.ravel()}


def set_axes(ax, ref, pred, xlabel, ylabel, title, color):
    lo = float(min(np.min(ref), np.min(pred)))
    hi = float(max(np.max(ref), np.max(pred)))
    pad = 0.04 * (hi - lo if hi > lo else 1.0)
    lim = (lo - pad, hi + pad)
    ax.plot(lim, lim, color="0.25", lw=1.0, ls="--", label="y = x")
    slope, intercept = np.polyfit(ref, pred, 1)
    xx = np.linspace(*lim, 200)
    ax.plot(xx, slope * xx + intercept, color=color, lw=1.3, label="linear fit")
    if ref.size > 100000:
        rng = np.random.default_rng(20260813)
        idx = rng.choice(ref.size, 100000, replace=False)
        xxp, yyp = ref[idx], pred[idx]
    else:
        xxp, yyp = ref, pred
    ax.scatter(xxp, yyp, s=3.0 if ref.size > 10000 else 8.0, alpha=0.22, color=color, linewidths=0)
    ax.set(xlim=lim, ylim=lim, xlabel=xlabel, ylabel=ylabel, title=title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="0.92", lw=0.6)
    ax.legend(frameon=False, fontsize=8, loc="upper left")


def plot_single(key, result, kind):
    spec = SYSTEMS[key]
    if kind == "energy":
        x, y = result["energy_ref"], result["energy_pred"]
        xlabel = "DFT energy (eV)"
        ylabel = "MACE energy (eV), bias-corrected"
        title = f"{spec['label']}: energy parity"
        stem = f"{key}_energy_parity"
    else:
        x, y = result["force_ref"], result["force_pred"]
        xlabel = r"DFT force component (eV Å$^{-1}$)"
        ylabel = r"MACE force component (eV Å$^{-1}$)"
        title = f"{spec['label']}: force parity"
        stem = f"{key}_force_parity"
    fig, ax = plt.subplots(figsize=(4.0, 3.8), constrained_layout=True)
    set_axes(ax, x, y, xlabel, ylabel, title, spec["color"])
    for ext, kw in (("png", {"dpi": 600}), ("tiff", {"dpi": 600}), ("pdf", {}), ("svg", {})):
        fig.savefig(OUT / f"{stem}.{ext}", **kw)
    plt.close(fig)


def plot_combined(results):
    fig, axs = plt.subplots(2, 2, figsize=(8.0, 7.2), constrained_layout=True)
    for col, key in enumerate(("youshui", "caf2")):
        spec = SYSTEMS[key]
        r = results[key]
        set_axes(axs[0, col], r["energy_ref"], r["energy_pred"], "DFT energy (eV)", "MACE energy (eV), corrected", f"{spec['label']}: energy", spec["color"])
        set_axes(axs[1, col], r["force_ref"], r["force_pred"], r"DFT force (eV Å$^{-1}$)", r"MACE force (eV Å$^{-1}$)", f"{spec['label']}: force", spec["color"])
    for ext, kw in (("png", {"dpi": 600}), ("tiff", {"dpi": 600}), ("pdf", {}), ("svg", {})):
        fig.savefig(OUT / f"mace_youshui_caf2_energy_force_parity.{ext}", **kw)
    plt.close(fig)


def natural_segment_key(p: Path):
    m = re.search(r"segment[_-](\d+)", p.name)
    return int(m.group(1)) if m else p.name


def audit_trajectory(key, spec):
    segs = sorted(spec["segments"].glob("*.h5"), key=natural_segment_key)
    out = {
        "system": key,
        "trajectory": str(spec["trajectory"]),
        "complete_marker": str(spec["complete_marker"]),
        "complete_marker_present": spec["complete_marker"].exists(),
        "segment_count": len(segs),
        "segments": [str(x) for x in segs],
        "hdf5_read_test": {},
    }
    for p in ([segs[0], segs[-1]] if len(segs) > 1 else segs):
        try:
            with h5py.File(p, "r") as h:
                keys = list(h.keys())
                # Access positions explicitly; this is the critical trajectory payload.
                shape = tuple(h["positions_A"].shape)
                first = np.asarray(h["positions_A"][0, 0, :], dtype=float).tolist()
                out["hdf5_read_test"][str(p)] = {"status": "readable", "keys": keys, "positions_shape": shape, "first_position": first}
        except Exception as exc:
            out["hdf5_read_test"][str(p)] = {"status": "unreadable", "error": f"{type(exc).__name__}: {exc}"}
    out["audit_note"] = (
        "CaF2 provenance is the latest mloong AX2 500 ps continuation (absolute 100-600 ps, "
        "500000 frames) as recorded in md500ps.complete/audit_final.json. Local HDF5 payload "
        "readability is reported verbatim; no substitute local trajectory is silently used."
        if key == "caf2" else "Latest Youshui epoch76 500 ps trajectory."
    )
    return out


def main():
    global ROOT, OUT, SYSTEMS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="Directory containing the original MACE validation and trajectory subdirectories")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--oil-model", type=Path, help="Override the oil/water model path")
    parser.add_argument("--caf2-model", type=Path, help="Override the CaF2 model path")
    parser.add_argument("--oil-validation", type=Path, help="Override the oil/water validation extxyz path")
    parser.add_argument("--caf2-validation", type=Path, help="Override the CaF2 validation extxyz path")
    args = parser.parse_args()
    old_root = ROOT
    ROOT = args.source_root.resolve()
    OUT = args.out.resolve()
    SYSTEMS = {
        key: {name: ROOT / value.relative_to(old_root) if isinstance(value, Path) else value for name, value in spec.items()}
        for key, spec in SYSTEMS.items()
    }
    for key, prefix in (("youshui", "oil"), ("caf2", "caf2")):
        for name in ("model", "validation"):
            override = getattr(args, f"{prefix}_{name}")
            if override is not None:
                SYSTEMS[key][name] = override.resolve()
            if not SYSTEMS[key][name].is_file():
                parser.error(f"missing {key} {name}: {SYSTEMS[key][name]}")
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for key, spec in SYSTEMS.items():
        results[key] = evaluate_system(key, spec)
        plot_single(key, results[key], "energy")
        plot_single(key, results[key], "force")
    plot_combined(results)
    summary = {k: v["metrics"] for k, v in results.items()}
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (OUT / "metrics_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "energy_bias_eV", "energy_rmse_total_eV_corrected", "energy_rmse_meV_atom_corrected", "force_rmse_eV_A", "energy_r2", "force_r2"])
        for key, m in summary.items():
            w.writerow([key, m["energy_bias_raw_mace_minus_dft_eV"], m["energy"]["corrected_total_eV"]["rmse"], m["energy"]["corrected_per_atom_meV"]["rmse"], m["force"]["rmse"], m["energy"]["corrected_total_eV"]["r2_linear_fit"], m["force"]["r2_linear_fit"]])
    provenance = {k: audit_trajectory(k, spec) for k, spec in SYSTEMS.items()}
    (OUT / "trajectory_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (OUT / "README_CN.md").write_text(
        "# 最新 Youshui / CaF2 MACE parity\n\n"
        "- Energy: total DFT vs MACE energy; only the constant mean (MACE−DFT) offset is removed.\n"
        "- Force: all Cartesian components, no correction. Metrics use every validation frame/component.\n"
        "- CaF2 trajectory provenance: mloong AX2 500 ps continuation under `caf2_mace_mloong_20260805/md_500ps_vdos_1fs`; local HDF5 readability is recorded in `trajectory_provenance.json`.\n"
        "- Source arrays are in the two `*_parity_source_data.npz` files; plots are exported as PNG/TIFF/PDF/SVG.\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
