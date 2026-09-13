# Source data

Numerical data for **Buried substrates transmit dynamic constraints through oil into the aqueous interior**.

Each figure directory contains an Excel workbook and CSV tables. Column names specify observables and units; each workbook includes a short Readme. The [figure index](figure_index.csv) lists individual files and panels.

| Figure | Contents | Workbook |
| --- | --- | --- |
| 3c,d | CaF2-supported and unsupported water displacements over 500 ps | [Source_Data_Fig_3.xlsx](Fig_3/Source_Data_Fig_3.xlsx) |
| 4a–d | Collective translation, central-film block statistics and layer VDOS | [Source_Data_Fig_4.xlsx](Fig_4/Source_Data_Fig_4.xlsx) |
| S9 | CaF2 MACE energy and force validation | [Source_Data_Fig_S9.xlsx](Fig_S9/Source_Data_Fig_S9.xlsx) |
| S10 | Oil/water MACE energy and force validation | [Source_Data_Fig_S10.xlsx](Fig_S10/Source_Data_Fig_S10.xlsx) |
| S11 | CaF2 and oil/water radial distribution functions | [Source_Data_Fig_S11.xlsx](Fig_S11/Source_Data_Fig_S11.xlsx) |
| S12 | SiO2 DeePMD energy and force validation | [Source_Data_Fig_S12.xlsx](Fig_S12/Source_Data_Fig_S12.xlsx) |
| S13 | SiO2 radial distribution functions | [Source_Data_Fig_S13.xlsx](Fig_S13/Source_Data_Fig_S13.xlsx) |
| S14 | SiO2-supported water displacements over 500 ps | [Source_Data_Fig_S14.xlsx](Fig_S14/Source_Data_Fig_S14.xlsx) |
| S16 | Oil-chain motion normal to the interface | [Source_Data_Fig_S16.xlsx](Fig_S16/Source_Data_Fig_S16.xlsx) |
| S18 | Water hydrogen-bond coordination and partner statistics | [Source_Data_Fig_S18.xlsx](Fig_S18/Source_Data_Fig_S18.xlsx) |
| S20 | Tetrahedral and orientational order profiles | [Source_Data_Fig_S20.xlsx](Fig_S20/Source_Data_Fig_S20.xlsx) |

The full force-component pairs are in `force_parity.csv.gz` beside the S9, S10 and S12 workbooks (316,758, 411,750 and 2,051,640 pairs, respectively). These are losslessly compressed CSV tables. For example:

```python
import gzip
import numpy as np

with gzip.open("Fig_S12/force_parity.csv.gz", "rt") as stream:
    dft_force, ml_force = np.loadtxt(stream, delimiter=",", skiprows=1).T
```

Additional tables contain [water depth, orientation and coordination](water_dynamics/selected_water_descriptors.csv), [per-layer 10 ps block statistics](collective_blocks/), and [SiO2 layer spectra](spectra/). Distances are in angstrom, times in picoseconds, frequencies in cm−1, energies in eV/atom and forces in eV/angstrom unless specified in the column header. Relative depths and order parameters are dimensionless.

The VDOS tables retain the frequency convention of the original plotting exports. Figure 4 includes both plotted and calculated frequencies; the oil/water spectrum uses a +80 cm−1 display shift. The SiO2 spectrum directory contains the original and +80 cm−1 exports. The SiO2 slab bounds in `spectra/sio2_layers.csv` are absolute simulation-cell coordinates.

Representative [CIF structures](../structures/), [trajectories](../trajectories/) and [potential-energy models](../models/) are stored separately. The SiO2 model is the uncompressed frozen DeePMD file `graph.pb`.

[manifest.json](manifest.json) records numerical-file descriptions and source checksums. [SHA256SUMS.txt](SHA256SUMS.txt) contains checksums for the distributed files.
