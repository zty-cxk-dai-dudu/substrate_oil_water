#!/usr/bin/env python3
"""Restartable 300 K MACE NVT with 1 fs gzip-compressed velocity output.

The HDF5 segments are directly usable for velocity-FFT DOS and can be converted
without loss for SFG programs that require text trajectories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
from ase import units
from ase.io import read, write
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.bussi import Bussi
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

DT_FS = 0.5
WRITE_EVERY = 2                 # 1 fs velocity sampling
TARGET_TEMPERATURE_K = 300.0


class GuardStop(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--target-ps", type=float, default=500.0)
    p.add_argument("--segment-ps", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=20260806)
    p.add_argument("--ensemble", choices=("nhc", "csvr"), default="nhc")
    return p.parse_args()


def save_state(path: Path, atoms, dyn, steps: int, frames: int) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as handle:
        payload = dict(positions_A=atoms.positions, momenta=atoms.get_momenta(),
                       cell_A=atoms.cell.array, numbers=atoms.numbers,
                       steps=steps, frames=frames)
        if hasattr(dyn, "_thermostat"):
            payload.update(eta=dyn._thermostat._eta, p_eta=dyn._thermostat._p_eta,
                           thermostat_state="nose_hoover_chain")
        else:
            payload.update(thermostat_state="csvr_positions_momenta_only")
        np.savez_compressed(handle, **payload)
    os.replace(tmp, path)


def restore_state(path: Path, atoms, dyn):
    with np.load(path) as state:
        if not np.array_equal(state["numbers"], atoms.numbers):
            raise RuntimeError("Restart species order differs from source")
        atoms.set_cell(state["cell_A"])
        atoms.set_positions(state["positions_A"])
        atoms.set_momenta(state["momenta"])
        if dyn is not None:
            dyn._q = atoms.positions.copy()
            dyn._p = atoms.get_momenta().copy()
            if "eta" in state and hasattr(dyn, "_thermostat"):
                dyn._thermostat._eta[:] = state["eta"]
                dyn._thermostat._p_eta[:] = state["p_eta"]
        return int(state["steps"]), int(state["frames"])


def minimum_distance(atoms) -> float:
    pos = atoms.get_positions(wrap=True)
    cell = atoms.cell.array
    inv_cell = np.linalg.inv(cell)
    delta = pos[:, None, :] - pos[None, :, :]
    frac = delta @ inv_cell
    frac -= np.rint(frac)
    dist = np.linalg.norm(frac @ cell, axis=2)
    np.fill_diagonal(dist, np.inf)
    return float(dist.min())


def open_segment(path: Path, nframes: int, atoms, first_step: int):
    h = h5py.File(path, "w", libver="latest")
    n = len(atoms)
    chunks = (min(128, nframes), n, 3)
    for key in ("positions_A", "velocities_A_per_fs"):
        h.create_dataset(key, (nframes, n, 3), dtype="f4", chunks=chunks,
                         compression="gzip", compression_opts=1, shuffle=True)
    h.create_dataset("step", (nframes,), dtype="i8", compression="gzip", compression_opts=1)
    h.create_dataset("time_ps", (nframes,), dtype="f8", compression="gzip", compression_opts=1)
    h.create_dataset("cell_A", data=atoms.cell.array)
    h.create_dataset("atomic_numbers", data=atoms.numbers.astype(np.int16))
    h.create_dataset("symbols", data=np.asarray(atoms.get_chemical_symbols(), dtype="S2"))
    h.attrs.update(dt_fs=DT_FS, output_interval_fs=DT_FS * WRITE_EVERY,
                   temperature_target_K=TARGET_TEMPERATURE_K, first_step=first_step,
                   positions_unit="angstrom", velocities_unit="angstrom/fs")
    return h


def main(enable_cueq=False):
    args = parse_args()
    from mace.calculators import MACECalculator
    output = args.output.resolve()
    segments = output / "velocity_h5_1fs"
    checkpoints = output / "checkpoints"
    segments.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    target_steps = int(round(args.target_ps * 1000.0 / DT_FS))
    segment_steps = int(round(args.segment_ps * 1000.0 / DT_FS))
    if target_steps <= 0 or segment_steps <= 0 or target_steps % WRITE_EVERY or segment_steps % WRITE_EVERY:
        raise SystemExit("target and segment lengths must be positive multiples of 1 fs")
    atoms = read(args.source)
    calculator_options = {"enable_cueq": True} if enable_cueq else {}
    atoms.calc = MACECalculator(model_paths=str(args.model), device="cuda", default_dtype="float32",
                                **calculator_options)
    state = checkpoints / "latest_state.npz"
    if state.exists():
        steps, frames = restore_state(state, atoms, None)
        restart_mode = ("exact_positions_momenta_and_thermostat_restart" if args.ensemble == "nhc"
                        else "positions_momenta_restart_new_csvr_rng")
    else:
        np.random.seed(args.seed)
        MaxwellBoltzmannDistribution(atoms, temperature_K=TARGET_TEMPERATURE_K, force_temp=True)
        Stationary(atoms)
        steps, frames = 0, 0
        restart_mode = "fresh_maxwell_boltzmann_300K"
    # ASE integrators read the initial momenta during construction.
    if args.ensemble == "csvr":
        dyn = Bussi(atoms, timestep=DT_FS * units.fs, temperature_K=TARGET_TEMPERATURE_K,
                    taut=100 * units.fs, rng=np.random.RandomState(args.seed))
        ensemble_label = "NVT CSVR (Bussi velocity rescaling)"
    else:
        dyn = NoseHooverChainNVT(atoms, timestep=DT_FS * units.fs,
                                 temperature_K=TARGET_TEMPERATURE_K, tdamp=100 * units.fs)
        ensemble_label = "NVT Nose-Hoover chain"
    if state.exists():
        steps, frames = restore_state(state, atoms, dyn)
    write_json(output / "metadata.json", {
        "model": str(args.model), "model_sha256": sha256(args.model),
        "source": str(args.source), "source_sha256": sha256(args.source),
        "atoms": len(atoms), "species_order": atoms.get_chemical_symbols(),
        "ensemble": ensemble_label, "thermostat_tau_fs": 100.0,
        "temperature_K": TARGET_TEMPERATURE_K,
        "dt_fs": DT_FS, "velocity_sampling_fs": DT_FS * WRITE_EVERY,
        "target_ps": args.target_ps, "segment_ps": args.segment_ps,
        "restart_mode": restart_mode, "format": "gzip-compressed HDF5 segments",
    })
    try:
        while steps < target_steps:
            this_steps = min(segment_steps, target_steps - steps)
            nframes = this_steps // WRITE_EVERY
            index = steps // segment_steps
            first_step = steps + WRITE_EVERY
            final_step = steps + this_steps
            partial = segments / f"velocity_1fs_segment_{index:04d}_step{first_step:07d}-{final_step:07d}.partial.h5"
            final = partial.with_name(partial.name.replace(".partial.h5", ".h5"))
            h = open_segment(partial, nframes, atoms, first_step)
            try:
                for i in range(nframes):
                    dyn.run(WRITE_EVERY)
                    steps += WRITE_EVERY
                    temp = float(atoms.get_temperature())
                    force = np.asarray(atoms.get_forces())
                    dmin = minimum_distance(atoms)
                    if not (np.isfinite(temp) and np.isfinite(dmin) and np.isfinite(force).all()):
                        raise GuardStop(f"NaN/Inf at step {steps}")
                    if temp > 1200.0 or float(np.linalg.norm(force, axis=1).max()) > 200.0 or dmin < 0.60:
                        raise GuardStop(f"geometry/temperature guard at step {steps}: T={temp:.1f} K, dmin={dmin:.3f} A")
                    h["step"][i] = steps
                    h["time_ps"][i] = steps * DT_FS / 1000.0
                    h["positions_A"][i] = atoms.get_positions(wrap=True).astype("f4")
                    h["velocities_A_per_fs"][i] = (atoms.get_velocities() * units.fs).astype("f4")
                    if (i + 1) % 128 == 0 or i + 1 == nframes:
                        h.attrs["completed_frames"] = i + 1
                        h.flush()
                        write_json(output / "progress.json", {"status": "running", "steps": steps,
                                   "time_ps": steps * DT_FS / 1000.0, "frames": frames + i + 1,
                                   "temperature_K": temp, "minimum_distance_A": dmin})
                h.attrs["complete"] = True
                h.flush()
            finally:
                h.close()
            os.replace(partial, final)
            frames += nframes
            save_state(state, atoms, dyn, steps, frames)
            write(checkpoints / "latest_structure_with_momenta.extxyz", atoms, format="extxyz")
            write_json(output / "progress.json", {"status": "running", "steps": steps,
                       "time_ps": steps * DT_FS / 1000.0, "frames": frames})
        write_json(output / "audit_final.json", {"status": "complete", "steps": steps,
                   "time_ps": steps * DT_FS / 1000.0, "frames": frames})
        (output / "MD500PS_COMPLETE").write_text("passed\n")
    except Exception as exc:
        write_json(output / "progress.json", {"status": "failed", "steps": steps,
                   "time_ps": steps * DT_FS / 1000.0, "frames": frames,
                   "error": f"{type(exc).__name__}: {exc}"})
        if isinstance(exc, GuardStop):
            (output / "SCIENTIFIC_GUARD_STOP").write_text(str(exc) + "\n")
            raise SystemExit(42)
        raise


if __name__ == "__main__":
    main()
