#!/usr/bin/env python3
"""Restartable MACE NVT continuation with 1 fs velocity/position HDF5 output."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import h5py
import numpy as np
from ase import units
from ase.io import read, write
from ase.md.nose_hoover_chain import NoseHooverChainNVT


SOURCE_STEP = 200_000
DT_FS = 0.5
OUTPUT_EVERY_STEPS = 2
Z_TO_TYPE = {1: 1, 6: 2, 8: 3, 9: 4, 20: 5}
TYPE_TO_NAME = {1: "H", 2: "C", 3: "O", 4: "F", 5: "Ca"}
PAIR_LIMITS = {
    (1, 1): 1.00, (1, 2): 0.75, (1, 3): 0.75, (1, 4): 1.20,
    (1, 5): 1.50, (2, 2): 1.15, (2, 3): 1.50, (2, 4): 1.50,
    (2, 5): 1.80, (3, 3): 1.80, (3, 4): 1.50, (3, 5): 1.80,
    (4, 4): 2.00, (4, 5): 1.70, (5, 5): 2.80,
}


class GuardViolation(RuntimeError):
    """Scientific guard failure that must not be auto-restarted."""


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--target-ps", type=float, default=500.0)
    p.add_argument("--segment-ps", type=float, default=5.0)
    return p.parse_args()


def atomic_json(path: Path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_checkpoint(path: Path, dyn, continuation_steps: int, frames: int):
    tmp = path.with_suffix(".npz.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(
            f,
            positions_A=dyn.atoms.get_positions(),
            momenta=dyn.atoms.get_momenta(),
            eta=dyn._thermostat._eta,
            p_eta=dyn._thermostat._p_eta,
            continuation_steps=np.int64(continuation_steps),
            frames=np.int64(frames),
            numbers=dyn.atoms.numbers,
            cell_A=dyn.atoms.cell.array,
        )
    os.replace(tmp, path)


def restore_checkpoint(path: Path, atoms, dyn):
    with np.load(path) as d:
        if not np.array_equal(d["numbers"], atoms.numbers):
            raise RuntimeError("Checkpoint ordered species differ from source")
        atoms.set_cell(d["cell_A"])
        atoms.set_positions(d["positions_A"])
        atoms.set_momenta(d["momenta"])
        dyn._q = atoms.get_positions().copy()
        dyn._p = atoms.get_momenta().copy()
        dyn._thermostat._eta[:] = d["eta"]
        dyn._thermostat._p_eta[:] = d["p_eta"]
        return int(d["continuation_steps"]), int(d["frames"])


class Audit:
    def __init__(self, numbers):
        self.expected_numbers = numbers.copy()
        self.temp_min = math.inf
        self.temp_max = -math.inf
        self.pe_min = math.inf
        self.pe_max = -math.inf
        self.max_force = 0.0
        self.minima = {pair: [math.inf, None, None] for pair in PAIR_LIMITS}
        self.violations = []

    def restore(self, payload):
        self.temp_min = payload["temperature_K"]["min"]
        self.temp_max = payload["temperature_K"]["max"]
        self.pe_min = payload["potential_eV"]["min"]
        self.pe_max = payload["potential_eV"]["max"]
        self.max_force = payload["max_force_eV_per_A"]
        for (a, b), value in self.minima.items():
            old = payload["pair_minima"][f"{TYPE_TO_NAME[a]}-{TYPE_TO_NAME[b]}"]
            value[:] = [old["distance_A"], old["absolute_step"], old["atom_ids"]]
        self.violations = list(payload.get("violations", []))

    def update(self, atoms, absolute_step):
        if len(atoms) != 262 or not np.array_equal(atoms.numbers, self.expected_numbers):
            raise GuardViolation("Atom count or ordered species changed")
        pe = float(atoms.get_potential_energy())
        temp = float(atoms.get_temperature())
        force = np.asarray(atoms.get_forces())
        fmax = float(np.linalg.norm(force, axis=1).max())
        if not (np.isfinite([pe, temp, fmax]).all() and np.isfinite(force).all()):
            raise GuardViolation(f"NaN/Inf at absolute step {absolute_step}")
        self.temp_min = min(self.temp_min, temp)
        self.temp_max = max(self.temp_max, temp)
        self.pe_min = min(self.pe_min, pe)
        self.pe_max = max(self.pe_max, pe)
        self.max_force = max(self.max_force, fmax)
        pos = atoms.get_positions(wrap=True)
        cell = atoms.cell.array
        lengths = np.diag(cell)
        if not np.allclose(cell, np.diag(lengths), atol=1e-8):
            raise GuardViolation("Expected the verified orthorhombic cell")
        delta = pos[:, None, :] - pos[None, :, :]
        delta -= np.rint(delta / lengths) * lengths
        dist = np.linalg.norm(delta, axis=2)
        np.fill_diagonal(dist, np.inf)
        types = np.array([Z_TO_TYPE[int(z)] for z in atoms.numbers])
        frame_bad = []
        for pair, limit in PAIR_LIMITS.items():
            a, b = pair
            mask = (types[:, None] == a) & (types[None, :] == b)
            if a != b:
                mask |= (types[:, None] == b) & (types[None, :] == a)
            values = np.where(mask, dist, np.inf)
            ij = np.unravel_index(np.argmin(values), values.shape)
            value = float(values[ij])
            ids = [int(ij[0] + 1), int(ij[1] + 1)]
            if value < self.minima[pair][0]:
                self.minima[pair] = [value, absolute_step, ids]
            if value < limit:
                frame_bad.append({"pair": f"{TYPE_TO_NAME[a]}-{TYPE_TO_NAME[b]}",
                                  "distance_A": value, "limit_A": limit,
                                  "atom_ids": ids})
        if frame_bad:
            item = {"absolute_step": absolute_step, "violations": frame_bad}
            self.violations.append(item)
            raise GuardViolation(f"Severe short-contact violation: {item}")
        if temp > 1200.0:
            raise GuardViolation(f"Temperature guard exceeded: {temp} K")
        if fmax > 200.0:
            raise GuardViolation(f"Force guard exceeded: {fmax} eV/A")
        return pe, temp, fmax

    def payload(self, continuation_steps, frames, status, error=None):
        return {
            "status": status,
            "error": error,
            "continuation_steps_completed": continuation_steps,
            "continuation_time_ps": continuation_steps * DT_FS / 1000.0,
            "absolute_step": SOURCE_STEP + continuation_steps,
            "absolute_time_ps": (SOURCE_STEP + continuation_steps) * DT_FS / 1000.0,
            "frames_completed": frames,
            "temperature_K": {"min": self.temp_min, "max": self.temp_max},
            "potential_eV": {"min": self.pe_min, "max": self.pe_max},
            "max_force_eV_per_A": self.max_force,
            "ordered_species_preserved": True,
            "pair_minima": {
                f"{TYPE_TO_NAME[a]}-{TYPE_TO_NAME[b]}": {
                    "distance_A": v[0], "absolute_step": v[1], "atom_ids": v[2]
                } for (a, b), v in self.minima.items()
            },
            "violations": self.violations[:20],
        }


def create_h5(path, nframes, atoms, segment_index, first_abs_step):
    h = h5py.File(path, "w", libver="latest")
    c = min(128, nframes)
    h.create_dataset("step", (nframes,), dtype="i8", chunks=(c,),
                     compression="gzip", compression_opts=1, shuffle=True)
    h.create_dataset("time_ps", (nframes,), dtype="f8", chunks=(c,),
                     compression="gzip", compression_opts=1, shuffle=True)
    h.create_dataset("positions_A", (nframes, len(atoms), 3), dtype="f4",
                     chunks=(c, len(atoms), 3), compression="gzip",
                     compression_opts=1, shuffle=True)
    h.create_dataset("velocities_A_per_fs", (nframes, len(atoms), 3), dtype="f4",
                     chunks=(c, len(atoms), 3), compression="gzip",
                     compression_opts=1, shuffle=True)
    h.create_dataset("cell_A", data=np.asarray(atoms.cell.array, dtype=np.float64))
    h.create_dataset("symbols", data=np.asarray(atoms.get_chemical_symbols(), dtype="S2"))
    h.create_dataset("type", data=np.asarray([Z_TO_TYPE[int(z)] for z in atoms.numbers], dtype=np.int8))
    h.create_dataset("atomic_numbers", data=np.asarray(atoms.numbers, dtype=np.int16))
    h.attrs["segment_index"] = segment_index
    h.attrs["first_absolute_step"] = first_abs_step
    h.attrs["dt_fs"] = DT_FS
    h.attrs["output_interval_steps"] = OUTPUT_EVERY_STEPS
    h.attrs["output_interval_fs"] = DT_FS * OUTPUT_EVERY_STEPS
    h.attrs["positions_unit"] = "angstrom"
    h.attrs["velocities_unit"] = "angstrom/fs"
    h.attrs["time_unit"] = "ps"
    h.attrs["completed_frames"] = 0
    return h


def main():
    args = arguments()
    from mace.calculators import MACECalculator
    output = Path(args.output).resolve()
    segments = output / "segments"
    checkpoints = output / "checkpoints"
    segments.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    target_steps = int(round(args.target_ps * 1000.0 / DT_FS))
    segment_steps = int(round(args.segment_ps * 1000.0 / DT_FS))
    if target_steps % 2 or segment_steps % 2 or target_steps <= 0 or segment_steps <= 0:
        raise SystemExit("Target and segment lengths must be positive multiples of 1 fs")
    target_frames = target_steps // OUTPUT_EVERY_STEPS
    source = Path(args.source).resolve()
    model = Path(args.model).resolve()
    atoms = read(source)
    if not atoms.has("momenta"):
        raise SystemExit("Source does not contain momenta; refusing to resample velocities")
    source_numbers = atoms.numbers.copy()
    source_temperature = float(atoms.get_temperature())
    atoms.calc = MACECalculator(model_paths=str(model), device="cuda", default_dtype="float32")
    dyn = NoseHooverChainNVT(atoms, timestep=DT_FS * units.fs,
                             temperature_K=300.0, tdamp=100.0 * units.fs)
    checkpoint = checkpoints / "latest_state.npz"
    continuation_steps = 0
    frames_done = 0
    thermostat_origin = "reset_zero_at_verified_100ps_endpoint"
    if checkpoint.exists():
        continuation_steps, frames_done = restore_checkpoint(checkpoint, atoms, dyn)
        thermostat_origin = "restored_eta_and_p_eta_from_segment_checkpoint"
    if frames_done != continuation_steps // OUTPUT_EVERY_STEPS:
        raise RuntimeError("Checkpoint frame/step count mismatch")
    for partial in segments.glob("*.partial.h5"):
        abandoned = partial.with_suffix(partial.suffix + f".abandoned.{int(time.time())}")
        os.replace(partial, abandoned)
    completed = sorted(segments.glob("segment_*.h5"))
    expected_completed = frames_done // (segment_steps // OUTPUT_EVERY_STEPS)
    if len(completed) > expected_completed:
        for orphan in completed[expected_completed:]:
            os.replace(orphan, orphan.with_suffix(orphan.suffix + f".orphaned.{int(time.time())}"))
        completed = completed[:expected_completed]
    if len(completed) != expected_completed:
        raise RuntimeError("Complete segment count does not match checkpoint")
    audit = Audit(source_numbers)
    audit_checkpoint = checkpoints / "audit_at_latest_state.json"
    if checkpoint.exists() and audit_checkpoint.exists():
        audit.restore(json.loads(audit_checkpoint.read_text()))
    metadata = {
        "model": str(model), "model_sha256": sha256(model),
        "source": str(source), "source_sha256": sha256(source),
        "source_step": SOURCE_STEP, "source_time_ps": 100.0,
        "source_contains_momenta": True,
        "source_temperature_K": source_temperature,
        "thermostat_state_note": (
            "The 100 ps endpoint preserved positions and momenta but the prior script did not "
            "serialize Nose-Hoover eta/p_eta. The continuation reuses the exact endpoint "
            "positions and velocities and initializes a new 300 K chain at zero. All later "
            "segment restarts restore eta/p_eta exactly."
        ),
        "restart_origin": thermostat_origin,
        "ensemble": "NVT Nose-Hoover chain", "temperature_K": 300.0,
        "tdamp_fs": 100.0, "dt_fs": DT_FS, "output_every_steps": 2,
        "output_interval_fs": 1.0, "target_continuation_ps": args.target_ps,
        "target_frames": target_frames, "segment_ps": args.segment_ps,
        "compression": "gzip level 1 with shuffle", "float_dtype": "float32",
        "datasets": {
            "step": "absolute integer MD step since original trajectory start",
            "time_ps": "absolute time in ps since original trajectory start",
            "positions_A": "Cartesian positions, angstrom",
            "velocities_A_per_fs": "Cartesian velocities, angstrom/fs",
            "cell_A": "constant cell matrix, angstrom",
            "symbols": "ordered element symbols", "type": "H=1,C=2,O=3,F=4,Ca=5",
        },
    }
    atomic_json(output / "metadata.json", metadata)
    start_wall = time.time()
    try:
        while continuation_steps < target_steps:
            segment_index = continuation_steps // segment_steps
            this_steps = min(segment_steps, target_steps - continuation_steps)
            nframes = this_steps // OUTPUT_EVERY_STEPS
            first_abs = SOURCE_STEP + continuation_steps + OUTPUT_EVERY_STEPS
            final_abs = SOURCE_STEP + continuation_steps + this_steps
            base = f"segment_{segment_index:04d}_step{first_abs:07d}-{final_abs:07d}"
            partial = segments / f"{base}.partial.h5"
            final = segments / f"{base}.h5"
            h = create_h5(partial, nframes, atoms, segment_index, first_abs)
            try:
                for i in range(nframes):
                    dyn.run(OUTPUT_EVERY_STEPS)
                    continuation_steps += OUTPUT_EVERY_STEPS
                    absolute_step = SOURCE_STEP + continuation_steps
                    pe, temp, fmax = audit.update(atoms, absolute_step)
                    h["step"][i] = absolute_step
                    h["time_ps"][i] = absolute_step * DT_FS / 1000.0
                    h["positions_A"][i] = atoms.get_positions(wrap=True).astype(np.float32)
                    h["velocities_A_per_fs"][i] = (atoms.get_velocities() * units.fs).astype(np.float32)
                    if (i + 1) % 128 == 0 or i + 1 == nframes:
                        h.attrs["completed_frames"] = i + 1
                        h.flush()
                        atomic_json(output / "progress.json", audit.payload(
                            continuation_steps, frames_done + i + 1, "running"))
                h.attrs["complete"] = True
                h.flush()
            finally:
                h.close()
            os.replace(partial, final)
            frames_done += nframes
            save_checkpoint(checkpoint, dyn, continuation_steps, frames_done)
            write(checkpoints / "latest_structure_with_momenta.xyz", atoms, format="extxyz")
            record = {
                "segment_index": segment_index, "file": str(final),
                "sha256": sha256(final), "frames": nframes,
                "first_absolute_step": first_abs, "last_absolute_step": final_abs,
                "first_time_ps": first_abs * DT_FS / 1000.0,
                "last_time_ps": final_abs * DT_FS / 1000.0,
            }
            manifest_path = output / "segments_manifest.json"
            records = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
            records.append(record)
            atomic_json(manifest_path, records)
            progress = audit.payload(continuation_steps, frames_done, "running")
            progress["elapsed_wall_seconds"] = time.time() - start_wall
            progress["last_complete_segment"] = record
            atomic_json(output / "progress.json", progress)
            atomic_json(audit_checkpoint, progress)
        final_progress = audit.payload(continuation_steps, frames_done, "complete")
        final_progress["elapsed_wall_seconds"] = time.time() - start_wall
        atomic_json(output / "audit_final.json", final_progress)
        (output / f"md{int(args.target_ps)}ps.complete").write_text("passed\n")
        with (output / "SHA256SUMS").open("w") as out:
            for path in sorted(segments.glob("segment_*.h5")):
                out.write(f"{sha256(path)}  {path.relative_to(output)}\n")
    except Exception as exc:
        atomic_json(output / "progress.json", audit.payload(
            continuation_steps, frames_done, "failed", f"{type(exc).__name__}: {exc}"))
        if isinstance(exc, GuardViolation):
            (output / "SCIENTIFIC_GUARD_STOP").write_text(f"{type(exc).__name__}: {exc}\n")
            raise SystemExit(42)
        raise


if __name__ == "__main__":
    main()
