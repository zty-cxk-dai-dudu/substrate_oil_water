#!/usr/bin/env python3
"""Guarded 300 K NVT validation of the trained CaF2 MACE model.

The protocol intentionally mirrors the accepted DeePMD validation: metal-like
eV/A/fs units, periodic 262-atom cell, 300 K NVT, 0.5 fs timestep, 100 fs
thermostat time, and trajectory output every 5 fs.  It gates production at
1 ps and 5 ps before completing 100 ps.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from ase import units
from ase.io import read, write
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary


Z_TO_TYPE = {1: 1, 6: 2, 8: 3, 9: 4, 20: 5}
TYPE_TO_NAME = {1: "H", 2: "C", 3: "O", 4: "F", 5: "Ca"}
Z_OF_TYPE = {v: k for k, v in Z_TO_TYPE.items()}
PAIR_LIMITS = {
    (1, 1): 1.00, (1, 2): 0.75, (1, 3): 0.75, (1, 4): 1.20,
    (1, 5): 1.50, (2, 2): 1.15, (2, 3): 1.50, (2, 4): 1.50,
    (2, 5): 1.80, (3, 3): 1.80, (3, 4): 1.50, (3, 5): 1.80,
    (4, 4): 2.00, (4, 5): 1.70, (5, 5): 2.80,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--structure", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=20260805)
    return p.parse_args()


class GuardedRun:
    def __init__(self, atoms, output: Path):
        self.atoms = atoms
        self.output = output
        self.step = 0
        self.expected_numbers = atoms.numbers.copy()
        self.minima = {pair: [math.inf, None, None] for pair in PAIR_LIMITS}
        self.violations = []
        self.temp_min = math.inf
        self.temp_max = -math.inf
        self.pe_min = math.inf
        self.pe_max = -math.inf
        self.max_force = 0.0
        self.dump_handle = (output / "trajectory_100ps.lammpstrj").open("w", buffering=1)
        self.thermo_handle = (output / "thermo.csv").open("w", newline="", buffering=1)
        self.thermo = csv.writer(self.thermo_handle)
        self.thermo.writerow(["step", "time_ps", "temperature_K", "potential_eV",
                              "kinetic_eV", "total_eV", "max_force_eV_per_A"])

    def close(self):
        self.dump_handle.close()
        self.thermo_handle.close()

    def _pair_geometry(self):
        xyz = self.atoms.get_positions(wrap=True)
        lengths = np.diag(self.atoms.cell.array)
        if np.any(lengths <= 0) or not np.allclose(
                self.atoms.cell.array, np.diag(lengths), atol=1e-8):
            raise RuntimeError("The audited production cell must be orthorhombic")
        delta = xyz[:, None, :] - xyz[None, :, :]
        delta -= np.rint(delta / lengths) * lengths
        dist = np.linalg.norm(delta, axis=2)
        np.fill_diagonal(dist, np.inf)
        types = np.array([Z_TO_TYPE[int(z)] for z in self.atoms.numbers])
        ids = np.arange(1, len(types) + 1)
        frame_bad = []
        for pair, limit in PAIR_LIMITS.items():
            a, b = pair
            mask = (types[:, None] == a) & (types[None, :] == b)
            if a != b:
                mask |= (types[:, None] == b) & (types[None, :] == a)
            values = np.where(mask, dist, np.inf)
            where = np.unravel_index(np.argmin(values), values.shape)
            value = float(values[where])
            atom_ids = [int(ids[where[0]]), int(ids[where[1]])]
            if value < self.minima[pair][0]:
                self.minima[pair] = [value, self.step, atom_ids]
            if value < limit:
                frame_bad.append({
                    "pair": f"{TYPE_TO_NAME[a]}-{TYPE_TO_NAME[b]}",
                    "distance_A": value,
                    "limit_A": limit,
                    "atom_ids": atom_ids,
                })
        return frame_bad

    def _write_dump(self):
        pos = self.atoms.get_positions(wrap=True)
        lengths = np.diag(self.atoms.cell.array)
        h = self.dump_handle
        h.write(f"ITEM: TIMESTEP\n{self.step}\n")
        h.write(f"ITEM: NUMBER OF ATOMS\n{len(self.atoms)}\n")
        h.write("ITEM: BOX BOUNDS pp pp pp\n")
        for length in lengths:
            h.write(f"0.0 {length:.12f}\n")
        h.write("ITEM: ATOMS id type element x y z\n")
        for i, (z, xyz) in enumerate(zip(self.atoms.numbers, pos), start=1):
            t = Z_TO_TYPE[int(z)]
            h.write(f"{i} {t} {TYPE_TO_NAME[t]} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}\n")

    def sample(self):
        if len(self.atoms) != 262 or not np.array_equal(self.atoms.numbers, self.expected_numbers):
            raise RuntimeError("Atom count or ordered species changed")
        pe = float(self.atoms.get_potential_energy())
        ke = float(self.atoms.get_kinetic_energy())
        temp = float(self.atoms.get_temperature())
        forces = np.asarray(self.atoms.get_forces())
        fmax = float(np.linalg.norm(forces, axis=1).max())
        finite = np.isfinite([pe, ke, temp, fmax]).all() and np.isfinite(forces).all()
        if not finite:
            raise RuntimeError(f"NaN/Inf detected at step {self.step}")
        self.temp_min = min(self.temp_min, temp)
        self.temp_max = max(self.temp_max, temp)
        self.pe_min = min(self.pe_min, pe)
        self.pe_max = max(self.pe_max, pe)
        self.max_force = max(self.max_force, fmax)
        frame_bad = self._pair_geometry()
        self._write_dump()
        if self.step % 100 == 0:
            self.thermo.writerow([self.step, self.step * 0.0005, temp, pe, ke,
                                  pe + ke, fmax])
        if frame_bad:
            item = {"step": self.step, "violations": frame_bad}
            self.violations.append(item)
            raise RuntimeError(f"Severe short-contact violation: {item}")
        if temp > 1200.0:
            raise RuntimeError(f"Temperature guard exceeded at step {self.step}: {temp} K")
        if fmax > 200.0:
            raise RuntimeError(f"Force guard exceeded at step {self.step}: {fmax} eV/A")

    def audit(self, stage: str, passed: bool, error: str | None = None):
        payload = {
            "stage": stage,
            "passed": bool(passed),
            "error": error,
            "last_step": self.step,
            "time_ps": self.step * 0.0005,
            "atom_count": len(self.atoms),
            "ordered_species_preserved": bool(np.array_equal(
                self.atoms.numbers, self.expected_numbers)),
            "temperature_K": {"min": self.temp_min, "max": self.temp_max},
            "potential_eV": {"min": self.pe_min, "max": self.pe_max},
            "max_force_eV_per_A": self.max_force,
            "pair_limits_A": {
                f"{TYPE_TO_NAME[a]}-{TYPE_TO_NAME[b]}": v
                for (a, b), v in PAIR_LIMITS.items()
            },
            "pair_minima": {
                f"{TYPE_TO_NAME[a]}-{TYPE_TO_NAME[b]}": {
                    "distance_A": v[0], "step": v[1], "atom_ids": v[2]
                } for (a, b), v in self.minima.items()
            },
            "violations": self.violations[:20],
        }
        path = self.output / f"audit_{stage}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        return path


def main():
    args = parse_args()
    from mace.calculators import MACECalculator
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    atoms = read(args.structure, format="lammps-data", style="atomic", Z_of_type=Z_OF_TYPE)
    atoms.pbc = True
    if len(atoms) != 262:
        raise SystemExit(f"Expected 262 atoms, found {len(atoms)}")
    calc = MACECalculator(model_paths=args.model, device="cuda", default_dtype="float32")
    atoms.calc = calc
    MaxwellBoltzmannDistribution(
        atoms, temperature_K=300.0, rng=np.random.default_rng(args.seed)
    )
    Stationary(atoms, preserve_temperature=True)
    dyn = NoseHooverChainNVT(
        atoms, timestep=0.5 * units.fs, temperature_K=300.0,
        tdamp=100.0 * units.fs,
    )
    guard = GuardedRun(atoms, output)
    dyn.attach(lambda: setattr(guard, "step", dyn.get_number_of_steps()), interval=10)
    dyn.attach(guard.sample, interval=10)
    guard.step = 0
    guard.sample()
    try:
        stages = [("preflight_1ps", 2000), ("preflight_5ps", 8000),
                  ("production_100ps", 190000)]
        for stage, steps in stages:
            try:
                dyn.run(steps)
                guard.step = dyn.get_number_of_steps()
                guard.audit(stage, True)
                write(output / f"final_{stage}.xyz", atoms, format="extxyz")
            except Exception as exc:
                guard.step = dyn.get_number_of_steps()
                guard.audit(stage, False, f"{type(exc).__name__}: {exc}")
                raise
        (output / "md100ps.complete").write_text("passed\n")
    finally:
        guard.close()


if __name__ == "__main__":
    main()
