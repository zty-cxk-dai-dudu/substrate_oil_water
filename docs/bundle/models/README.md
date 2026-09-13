# Potential-energy models

The model weights are arranged by system and retain their original filenames.

| System | Framework | Model file |
| --- | --- | --- |
| CaF2/oil/water | MACE | [`caf2/caf2_mace_weight0p25_seed20260805.model`](https://github.com/zty-cxk-dai-dudu/substrate_oil_water/releases/download/v0.4.0/caf2_mace_weight0p25_seed20260805.model) |
| SiO2/oil/water | DeePMD-kit | [`sio2/graph-compress-selected.pb`](https://github.com/zty-cxk-dai-dudu/substrate_oil_water/releases/download/v0.4.0/graph-compress-selected.pb) |
| Oil/water | MACE | [`oil_water/youshui_mace_all_current_zbl_20260805.model`](https://github.com/zty-cxk-dai-dudu/substrate_oil_water/releases/download/v0.4.0/youshui_mace_all_current_zbl_20260805.model) |

Use the MACE weights with the MACE/ASE simulation scripts in `simulation/mace/`.
The SiO2 `.pb` file is the compressed model used by the DeePMD/LAMMPS input in
`input_files/deepmd_sio2/`. Its atom-type order is **O, Si, H, C**.

## MACE example

Run this example from the `substrate_oil_water` directory in a MACE environment:

```python
from ase.io import read
from mace.calculators import MACECalculator

atoms = read("structures/caf2/caf2_oil_water_mlmd_500ps.cif")
atoms.calc = MACECalculator(
    model_paths="models/caf2/caf2_mace_weight0p25_seed20260805.model",
    device="cpu",
    default_dtype="float32",
)
energy_eV = atoms.get_potential_energy()
forces_eV_per_A = atoms.get_forces()
```

For the unsupported oil/water system, select its structure and model from
`structures/oil_water/` and `models/oil_water/`.

## DeePMD example

Use the following pair settings in a LAMMPS input whose atom types follow
O, Si, H, C. Set the model path relative to the calculation directory:

```text
pair_style deepmd models/sio2/graph-compress-selected.pb
pair_coeff * *
```

## Download and checksums

The complete ZIP contains all three weights. Individual files are also available
from the Models section of the project website. The small code-and-structures ZIP
includes this index; place separately downloaded weights in the directories above.

From the `models` directory, run `sha256sum -c SHA256SUMS.txt`.
