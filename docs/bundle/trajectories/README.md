# Representative trajectories

Multi-frame extended XYZ files contain all atoms and the simulation cell. Coordinates are in angstrom. Atom identifiers and source frame/configuration numbers are retained in every frame; `time_ps`, when present, is in picoseconds.

| System | Dynamics | Atoms | Frames | Selection | Sampling | Trajectory |
| --- | --- | --- | --- | --- | --- | --- |
| CaF2-supported oil/water | AIMD | 184 | 201 | 30–50 ps | 0.1 ps | [caf2_aimd.extxyz](caf2/caf2_aimd.extxyz) |
| SiO2-supported oil/water | AIMD | 410 | 201 | source frames 30430–50430 | every 100 source frames | [sio2_aimd.extxyz](sio2/sio2_aimd.extxyz) |
| Oil/water | AIMD | 260 | 201 | 30.308–50.308 ps | 0.1 ps | [oil_water_aimd.extxyz](oil_water/oil_water_aimd.extxyz) |
| SiO2-supported oil/water | MLMD | 820 | 201 | 480–500 ps | 0.1 ps | [sio2_mlmd.extxyz](sio2/sio2_mlmd.extxyz) |

Open an `.extxyz` file in OVITO and use the animation controls, or load it with ASE:

```python
from ase.io import read
frames = read("trajectories/caf2/caf2_aimd.extxyz", index=":")
```

Each trajectory has a corresponding `_frames.csv` file mapping exported frames to the source. [index.json](index.json) records source hashes and selections.

The export script is [analysis/trajectories/export_representative.py](../analysis/trajectories/export_representative.py). For example, export source frames 30000–50000 at a stride of 100:

```bash
python analysis/trajectories/export_representative.py XDATCAR caf2_aimd.extxyz \
  --format xdatcar --start-frame 30000 --end-frame 50000 --stride 100 --timestep-fs 1
```
