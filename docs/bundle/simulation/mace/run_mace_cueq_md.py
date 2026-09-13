#!/usr/bin/env python3
"""Run the colocated restartable MD driver with MACE cuEq inference."""

from run_mace_nvt500ps_velocity_h5 import main


if __name__ == "__main__":
    main(enable_cueq=True)
