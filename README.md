# substrate_oil_water

Computational scripts, source data, potential-energy models, representative structures and trajectories for **Buried substrates transmit dynamic constraints through oil into the aqueous interior**.

Tianyue Zhang, Xiaoke Chen, Jing Li and Xiao-Yu Yang

**Project website:** [https://zty-cxk-dai-dudu.github.io/substrate_oil_water/](https://zty-cxk-dai-dudu.github.io/substrate_oil_water/)

**Complete download:** [Code, source data, structures, trajectories and potential-energy models](https://github.com/zty-cxk-dai-dudu/substrate_oil_water/releases/download/v0.6.0/substrate_oil_water_complete_v0.6.0.zip)

## Contents

| Directory | Contents |
| --- | --- |
| `data/` | Figure source data in Excel and CSV formats |
| `simulation/mace/` | MACE training and molecular-dynamics scripts |
| `analysis/aimd/` | AIMD trajectory processing, hydrogen bonds, water orientation and VDOS |
| `analysis/water_dynamics/` | Water structure, displacement, orientational dynamics and collective spectra |
| `analysis/oil_motion/` | Oil-chain centre-of-mass motion, RMS displacement and MSD |
| `analysis/interaction_energy/` | Frozen-geometry input preparation and interaction-energy calculation |
| `analysis/charge_density/` | CP2K single-point preparation and charge-density subtraction |
| `analysis/model_validation/` | Energy/force comparison and radial distribution functions |
| `analysis/trajectory_reducers/` | C++ programs for reducing LAMMPS velocity trajectories |
| `analysis/trajectories/` | Representative trajectory export and frame indexing |
| `analysis/structures/` | Structure conversion to CIF |
| `input_files/` | Molecular-dynamics and potential-training inputs |
| `trajectories/` | Representative AIMD and MLMD trajectories and frame indices |
| `structures/` | Representative CIFs, arranged by system |
| `models/` | MACE and DeePMD potential-energy weights and loading instructions |

## Source data

The [data directory](data/README.md) contains 11 figure workbooks and companion numerical tables for water and oil dynamics, layer spectra, and potential validation. The [figure index](data/figure_index.csv) maps each table to its panels.

[Download source data](https://github.com/zty-cxk-dai-dudu/substrate_oil_water/releases/download/v0.6.0/substrate_oil_water_source_data_v0.6.0.zip) · [Data and code availability](DATA_AVAILABILITY.md)

## Structures

Each system has an AIMD model and a 500 ps MLMD snapshot. The CIFs contain the simulation cell and all atoms. Atom labels retain the source atom numbers; LAMMPS structures follow atom-ID order. Fractional coordinates are wrapped into the periodic cell.

| System | AIMD model | MLMD, 500 ps |
| --- | --- | --- |
| CaF2/oil/water | [184 atoms](structures/caf2/caf2_oil_water_aimd.cif) | [524 atoms](structures/caf2/caf2_oil_water_mlmd_500ps.cif) |
| SiO2/oil/water | [410 atoms](structures/sio2/sio2_oil_water_aimd.cif) | [820 atoms](structures/sio2/sio2_oil_water_mlmd_500ps.cif) |
| Oil/water | [260 atoms](structures/oil_water/oil_water_aimd.cif) | [500 atoms](structures/oil_water/oil_water_mlmd_500ps.cif) |

The [structure index](structures/index.csv) lists compositions and water counts. Open the CIF files in VESTA, OVITO or ASE. Simulation constraints are specified in the original inputs under `input_files/`.

## Representative trajectories

Four trajectory files cover CaF2-supported, SiO2-supported and unsupported oil/water. Each contains 201 regularly sampled frames with all atoms and the simulation cell. See [trajectories/README.md](trajectories/README.md) for selections, downloads and viewing instructions.

## Installation

Use Python 3.11 or later for the analysis scripts:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Simulation scripts use MACE, DeePMD-kit/LAMMPS, CP2K or VASP, according to the supplied input. The per-water VDOS wrappers use VASPKIT. Set the input, output, model and executable paths for the calculation being run. Scripts with command-line options list them with `--help`; file-based workflows specify paths near the top of the script.

## Usage

Convert an atomic structure to CIF:

```bash
python analysis/structures/export_cif.py \
  input_files/aimd/cafyoushui/H2O/POSCAR caf2_model.cif --format vasp
```

Calculate water VDOS from a CP2K velocity trajectory:

```bash
python analysis/aimd/cp2k/cp2k_velocity_vdos.py \
  --poscar POSCAR --velocity-xyz merged-vel-1.xyz \
  --last-frames 50000 --dt-fs 1.0 --layer-width 2.0
```

Calculate hydrogen bonds from a CaF2/oil/water AIMD trajectory:

```bash
python analysis/aimd/caf2/analyze_hbonds_last50000_xdatcar.py \
  --poscar POSCAR --xdatcar XDATCAR --output-dir hbonds
```

Calculate SiO2 water-layer structure:

```bash
python analysis/water_dynamics/analyze_sio2_best_contrast_and_four_waters.py \
  --mode structure --position-cache water_positions.npz \
  --z0 18 --layer-width 2 --n-layers 21 --output sio2_layers
```

Summarize completed interaction-energy calculations:

```bash
python analysis/interaction_energy/summarize_interactions.py --input interaction_calculations
```

The energy workflow uses `E_interaction = E_AB - E_A - E_B`, with component geometries taken from the same snapshot. The resulting table contains energies in eV and kJ mol-1.

The C++ reducers can be compiled with a C++17 compiler and zlib:

```bash
g++ -O3 -std=c++17 analysis/trajectory_reducers/reduce_lammps_water_com_velocity.cpp \
  -lz -o reduce_water_velocity
```

## Potential-energy models

See [models/README.md](models/README.md) for the three model files and loading examples. The complete download includes the model weights, code, source data, inputs, CIF structures and representative trajectories.
