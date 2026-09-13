#!/usr/bin/env python3
"""Prepare local interaction energies for three middle waters in feigudingyoushui."""

from pathlib import Path
import argparse
import hashlib
import math
import shutil
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
import prepare_remote_every4A_3each_interaction as common


EXPECTED_SPECIES = ("H", "C", "O")
EXPECTED_COUNTS = (176, 24, 60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="CONTCAR with the original atom order")
    parser.add_argument("--output", type=Path, required=True, help="New calculation directory")
    parser.add_argument("--potcar", type=Path, help="Author-supplied POTCAR; default: beside --source")
    parser.add_argument("--kpoints", type=Path, help="KPOINTS; default: beside --source")
    parser.add_argument("--source-label", help="Frame label recorded with the source checksum")
    args = parser.parse_args()
    SOURCE, OUT = args.source.resolve(), args.output.resolve()
    ROOT = SOURCE.parent
    kpoints = args.kpoints.resolve() if args.kpoints else ROOT / "KPOINTS"
    if not kpoints.is_file():
        raise FileNotFoundError(kpoints)
    if OUT.exists():
        raise RuntimeError(f"Refusing to overwrite existing directory: {OUT}")
    lattice, species, counts, coords = common.read_poscar(SOURCE)
    if species != EXPECTED_SPECIES or counts != EXPECTED_COUNTS:
        raise RuntimeError(f"Unexpected composition: {species} {counts}")
    ranges = common.species_ranges(species, counts)
    pairs = common.identify_waters(lattice, coords, ranges)
    if len(pairs) != 60:
        raise RuntimeError(f"Expected 60 waters, found {len(pairs)}")

    length_z = math.sqrt(sum(value * value for value in lattice[2]))
    oxygens = list(pairs)
    z_values = [(coords[oxygen][2] % 1.0) * length_z for oxygen in oxygens]
    unwrapped, layer_low, layer_high, middle, largest_gap = common.circular_layer(z_values, length_z)
    selected = sorted(
        zip(oxygens, z_values, unwrapped),
        key=lambda row: (abs(row[2] - middle), row[0]),
    )[:3]
    expected_serials = [228, 238, 220]
    if [row[0] + 1 for row in selected] != expected_serials:
        raise RuntimeError(f"Unexpected selected O atoms: {[row[0] + 1 for row in selected]}")

    potcars = common.split_potcar(args.potcar.resolve() if args.potcar else ROOT / "POTCAR", species)
    incar = """SYSTEM = COMPONENT
ISTART = 0
ICHARG = 2
GGA = PE
IVDW = 12
PREC = Accurate
ENCUT = 450
EDIFF = 1E-6
NELMIN = 6
NELM = 300
ALGO = Normal
ISMEAR = 0
SIGMA = 0.06
ISPIN = 1
ISYM = 0
LREAL = Auto
IBRION = -1
NSW = 0
LCHARG = .FALSE.
LWAVE = .FALSE.
NCORE = 6
"""

    OUT.mkdir(parents=True)
    all_indices = list(range(sum(counts)))
    common.write_calc(
        OUT / "ab_full", "ab_full", species, counts, all_indices,
        lattice, coords, potcars, incar, ROOT, kpoints=kpoints,
    )
    a_counts = (counts[0] - 2, counts[1], counts[2] - 1)
    rows = ["rank\tO_serial\tH1_serial\tH2_serial\tO_z_A\tabs_from_middle_A\tdirectory"]
    for rank, (oxygen, z_value, z_unwrapped) in enumerate(selected, 1):
        h1, h2 = pairs[oxygen]
        water_dir = OUT / f"water{rank}_O{oxygen + 1}_z{z_value:.3f}A"
        removed = {h1, h2, oxygen}
        a_indices = [index for index in all_indices if index not in removed]
        b_indices = [h1, h2, oxygen]
        common.write_calc(
            water_dir / "a_without_water", f"a_without_O{oxygen + 1}", species,
            a_counts, a_indices, lattice, coords, potcars, incar, ROOT, kpoints=kpoints,
        )
        common.write_calc(
            water_dir / "b_single_water", f"b_O{oxygen + 1}", ("H", "O"),
            (2, 1), b_indices, lattice, coords, potcars, incar, ROOT, kpoints=kpoints,
        )
        rows.append("\t".join((
            str(rank), str(oxygen + 1), str(h1 + 1), str(h2 + 1),
            f"{z_value:.6f}", f"{abs(z_unwrapped - middle):.6f}", water_dir.name,
        )))
    (OUT / "selected_waters.tsv").write_text("\n".join(rows) + "\n")
    (OUT / "water_layer_definition.tsv").write_text(
        "source\twater_O_count\tcell_z_A\twater_z_min_unwrapped_A\t"
        "water_z_max_unwrapped_A\twater_layer_thickness_A\twater_layer_middle_unwrapped_A\t"
        "water_layer_middle_mod_cell_A\tlargest_empty_z_gap_A\n"
        f"{args.source_label or SOURCE.name}\t{len(pairs)}\t{length_z:.6f}\t{layer_low:.6f}\t{layer_high:.6f}\t"
        f"{layer_high - layer_low:.6f}\t{middle:.6f}\t{middle % length_z:.6f}\t{largest_gap:.6f}\n"
    )
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    (OUT / "README.md").write_text(
        "# Three waters closest to the water-layer middle\n\n"
        f"Source: `{args.source_label or SOURCE}`, SHA-256 `{digest}`. All 60 water "
        "molecules were identified with minimum-image O-H distances. The periodic water layer "
        "was unwrapped by cutting at its largest empty z gap.\n\n"
        "Only interaction energies are calculated: `E_int = E_ab - E_a - E_b`; charge "
        "densities are not written.\n"
    )
    shutil.copy2(SCRIPT_DIR / "run_local_feiguding_middle_water3_interaction.sh", OUT / "run_all.sh")
    shutil.copy2(
        SCRIPT_DIR.parent / "finalize_local_cafyoushui_middle_water3_interaction.py",
        OUT / "finalize_results.py",
    )
    print("Selected:", [
        (oxygen + 1, tuple(h + 1 for h in pairs[oxygen]), round(z_value, 6),
         round(abs(z_unwrapped - middle), 6))
        for oxygen, z_value, z_unwrapped in selected
    ])


if __name__ == "__main__":
    main()
