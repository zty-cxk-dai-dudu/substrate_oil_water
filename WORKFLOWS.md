# Computational workflows

Run commands from the repository root, using paths to the corresponding calculation inputs. The complete release ZIP includes the model files in the directory layout specified by `models/index.json`. Install `requirements.txt` for analysis; use a separate MACE or DeePMD/LAMMPS environment for simulations.

## Input and output map

| Workflow | Inputs | Entry point | Outputs |
| --- | --- | --- | --- |
| MACE molecular dynamics | Atomic structure, matching MACE weight | `simulation/mace/run_mace_nvt500ps_velocity_h5.py` | Segmented positions, velocities, thermodynamic records and restart state |
| MACE cuEq molecular dynamics | Same inputs, MACE with cuEq support | `simulation/mace/run_mace_cueq_md.py` | Same output layout as the restartable driver |
| CaF2 supercell preparation | Atomic-style LAMMPS data, atom types H, C, O, F, Ca | `simulation/mace/prepare_caf2_ax2_vac5.py` | Structure replicated along a by two, with 5 Å added along c |
| MACE training | Training, validation and evaluation XYZ files | `simulation/mace/run_train.sh` | Model checkpoints and training records |
| SiO2 DeePMD molecular dynamics | `input_files/deepmd_sio2/conf_input_ax1.lmp`, `models/sio2/graph.pb`, DeePMD LAMMPS plugin | `input_files/deepmd_sio2/in.production.noch.lammps` | Position and velocity dumps, restart files and final structure |
| SiO2 DeePMD training | DeePMD datasets listed in the training configuration | `input_files/deepmd_sio2/input_50k.json` | Checkpoints and learning curve |
| Frozen-geometry interaction energies | Original-order CONTCAR, matching KPOINTS and an author-supplied POTCAR | `analysis/interaction_energy/{caf2,oil_water}/prepare_local_*_interaction.py` | Full-system and fragment input directories, run scripts and energy summaries |
| Charge-density difference | SiO2/oil/water POSCAR, CP2K basis and potential files | `analysis/charge_density/prepare_cp2k.py`, then `subtract_cubes.py` | Full/fragment inputs and difference-density cube |
| AIMD water orientation | XDATCAR and oxygen or water identifiers | `analysis/aimd/{caf2,oil_water}/xdatcar_water_orientation.py` | Orientation CSV files and plots |
| Water spectra and collective dynamics | Position and velocity time series, cell, atom mapping and frame interval | `analysis/water_dynamics/` | Layer-resolved spectra and dynamical statistics |

Training uses the original labelled datasets. Set the six dataset paths in `input_50k.json` for the DeePMD training/validation splits. For MACE, use the three XYZ filenames specified by `run_train.sh --help`. Trajectory analyses use the position or velocity stream and sampling interval specified by each command; preserve the original atom order when selecting individual water molecules.

## Interaction-energy inputs

Prepare the CaF2/oil/water three-water calculation:

```bash
python analysis/interaction_energy/caf2/prepare_local_cafyoushui_middle_water3_interaction.py \
  --source /path/to/CONTCAR --kpoints /path/to/KPOINTS \
  --potcar /path/to/POTCAR --output calculations/caf2_interaction
```

For unsupported oil/water, use `analysis/interaction_energy/oil_water/prepare_local_feiguding_middle_water3_interaction.py` with the same options. The two preparations use their original system-specific water selections. The common preparation module, execution scripts and finalizers are included under `analysis/interaction_energy/`. The generated run scripts accept the VASP executable and MPI settings through environment variables; consult their `--help` before starting a calculation.

## MACE simulation and training

List the restartable driver options:

```bash
python simulation/mace/run_mace_nvt500ps_velocity_h5.py --help
python simulation/mace/run_mace_cueq_md.py --help
```

Supply `--model`, `--source` and `--output` for a simulation. `--target-ps`, `--segment-ps`, `--seed` and `--ensemble` select duration, segmentation and thermostat settings. The CaF2-specific drivers use the expanded cell with all atoms mobile. The asynchronous driver uses the included `compress_caf2_h5_segment.py` to store HDF5 trajectory segments losslessly.

The simulation entry points were checked with MACE 0.3.16, PyTorch 2.6.0, ASE 3.29.0 and cuEquivariance 0.11.0. Use the corresponding CUDA build for GPU execution. The ordinary analysis environment also supports their `--help` commands.

Inspect the training command before running it:

```bash
bash simulation/mace/run_train.sh \
  --data-dir /path/to/training_xyz --output calculations/mace_training --dry-run
```

Use `--mace-run-train` to select the training executable. Omit `--dry-run` to start training in the specified output directory.

## SiO2 DeePMD simulation

The supplied 410-atom input is replicated by the production input to form the 820-atom system. Atom types follow O, Si, H, C. Set absolute paths so that output files can be written in a separate calculation directory:

```bash
lmp \
  -var deepmd_plugin /path/to/libdeepmd_lmp.so \
  -var structure /path/to/substrate_oil_water/input_files/deepmd_sio2/conf_input_ax1.lmp \
  -var model /path/to/substrate_oil_water/models/sio2/graph.pb \
  -in /path/to/substrate_oil_water/input_files/deepmd_sio2/in.production.noch.lammps
```

This input uses 298.15 K, a 0.5 fs timestep and 1,000,000 production steps. The published `graph.pb` is the uncompressed frozen DeePMD model. HDF5 or trajectory-file compression is separate from potential tabulation.

## Charge-density inputs

```bash
python analysis/charge_density/prepare_cp2k.py \
  --source /path/to/sio2_oil_water_POSCAR --out calculations/charge_density \
  --cp2k-data /path/to/cp2k/data
```

The source must retain the 820-atom ordering specified by the preparation script. After the full-system and fragment calculations finish, run:

```bash
python analysis/charge_density/subtract_cubes.py \
  --root calculations/charge_density --out results/charge_density
```

## Selected-water and spectrum analysis

For the three-water orientation workflow, provide a tab-separated selection table with columns `rank`, `O_serial`, `H1_serial` and `H2_serial` and one row per selected water:

```bash
python analysis/aimd/caf2/analyze_xin_three_water_orientation.py \
  --xdatcar /path/to/XDATCAR --selected /path/to/selected_waters.tsv \
  --out results/three_water_orientation --potim-fs 1.0
```

For the VDOS export, each input directory contains `layer_summary.csv` and the `vdos_<column>.csv` files named by its `source_data_column` field:

```bash
python analysis/water_dynamics/export_vdos_no_gaussian_two_parts.py \
  --caf2-source /path/to/caf2_vdos_layers \
  --oil-water-source /path/to/oil_water_vdos_layers --out results/vdos
```

The original export uses a 0 cm−1 display shift for CaF2 and +80 cm−1 for oil/water, recording the shift in its output metadata. The model-validation programs accept explicit model, validation-set and output paths; use `analysis/model_validation/mace_parity.py --help` or `analysis/model_validation/compute_rdf_compare.py --help` for the input list. The SiO2 RDF workflow retains its original 410-atom cell and 1 fs frame interval.
