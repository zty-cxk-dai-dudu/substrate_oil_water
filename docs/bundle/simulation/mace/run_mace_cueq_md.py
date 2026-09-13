#!/usr/bin/env python3
"""Run the colocated restartable MD driver with official MACE cuEq inference."""
from __future__ import annotations

import runpy
from pathlib import Path

import mace.calculators

_base = mace.calculators.MACECalculator


class CuEqMACECalculator(_base):
    def __init__(self, *args, **kwargs):
        kwargs["enable_cueq"] = True
        super().__init__(*args, **kwargs)


mace.calculators.MACECalculator = CuEqMACECalculator
runpy.run_path(str(Path(__file__).with_name("run_mace_nvt500ps_velocity_h5.py")), run_name="__main__")
