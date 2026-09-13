#!/usr/bin/env python3
"""Prepare every-4-A, three-waters-per-plane interaction-energy calculations."""

from pathlib import Path
import argparse
import csv
import hashlib
import math
import shutil


OH_CUTOFF_A = 1.25
PLANE_SPACING_A = 4.0
WATERS_PER_PLANE = 3
COPY_RESULT_FILES = ("POSCAR", "INCAR", "KPOINTS", "POTCAR", "OUTCAR", "vasp.log", "OSZICAR")


def read_poscar(path):
    lines = path.read_text().splitlines()
    scale = float(lines[1].split()[0])
    lattice = [[scale * float(x) for x in lines[i].split()[:3]] for i in range(2, 5)]
    species = tuple(lines[5].split())
    counts = tuple(map(int, lines[6].split()))
    pos = 7
    if lines[pos].strip().lower().startswith("s"):
        pos += 1
    mode = lines[pos].strip().lower()
    pos += 1
    natoms = sum(counts)
    raw = [list(map(float, lines[pos + i].split()[:3])) for i in range(natoms)]
    if mode.startswith("d"):
        coords = raw
    else:
        if any(abs(lattice[i][j]) > 1e-10 for i in range(3) for j in range(3) if i != j):
            raise RuntimeError("Cartesian input with non-orthogonal cell is not supported")
        coords = [[row[i] / lattice[i][i] for i in range(3)] for row in raw]
    return lattice, species, counts, coords


def species_ranges(species, counts):
    ranges, start = {}, 0
    for symbol, count in zip(species, counts):
        ranges[symbol] = list(range(start, start + count))
        start += count
    return ranges


def min_image_distance(i, j, lattice, coords):
    df = [coords[i][axis] - coords[j][axis] for axis in range(3)]
    df = [value - round(value) for value in df]
    cart = [sum(df[a] * lattice[a][b] for a in range(3)) for b in range(3)]
    return math.sqrt(sum(value * value for value in cart))


def identify_waters(lattice, coords, ranges):
    pairs, used = {}, []
    for oxygen in ranges["O"]:
        close = []
        for hydrogen in ranges["H"]:
            distance = min_image_distance(oxygen, hydrogen, lattice, coords)
            if distance <= OH_CUTOFF_A:
                close.append((distance, hydrogen))
        close.sort()
        if len(close) != 2:
            raise RuntimeError(
                f"O{oxygen + 1} has {len(close)} H atoms within {OH_CUTOFF_A:.2f} A"
            )
        pair = tuple(sorted((close[0][1], close[1][1])))
        pairs[oxygen] = pair
        used.extend(pair)
    if len(set(used)) != 2 * len(pairs):
        raise RuntimeError("Water hydrogen assignments are not unique")
    return pairs


def circular_layer(z_values, length):
    ordered = sorted(z_values)
    gaps = [ordered[i + 1] - ordered[i] for i in range(len(ordered) - 1)]
    gaps.append(ordered[0] + length - ordered[-1])
    cut = max(range(len(gaps)), key=gaps.__getitem__)
    start = ordered[(cut + 1) % len(ordered)]
    unwrapped = [z if z >= start else z + length for z in z_values]
    low, high = min(unwrapped), max(unwrapped)
    return unwrapped, low, high, (low + high) / 2.0, gaps[cut]


def split_potcar(path, species):
    datasets, current = [], []
    with path.open(errors="replace") as handle:
        for line in handle:
            current.append(line)
            if line.strip() == "End of Dataset":
                datasets.append("".join(current))
                current = []
    if len(datasets) < len(species):
        raise RuntimeError(f"Expected at least {len(species)} POTCAR datasets, found {len(datasets)}")
    selected = datasets[:len(species)]
    for symbol, dataset in zip(species, selected):
        marker = f"VRHFIN ={symbol}:"
        if marker not in dataset.replace(" ", ""):
            compact = dataset.replace(" ", "")
            if f"VRHFIN={symbol}:" not in compact:
                raise RuntimeError(f"POTCAR dataset order mismatch for {symbol}")
    return dict(zip(species, selected))


def write_poscar(path, title, symbols, counts, indices, lattice, coords):
    lines = [title, "1.0"]
    lines.extend("  " + "  ".join(f"{x:.16f}" for x in row) for row in lattice)
    lines.extend(("  " + "  ".join(symbols), "  " + "  ".join(map(str, counts)), "Direct"))
    for index in indices:
        wrapped = [value % 1.0 for value in coords[index]]
        lines.append("  " + "  ".join(f"{value:.12f}" for value in wrapped))
    path.write_text("\n".join(lines) + "\n")


def write_calc(calc, title, symbols, counts, indices, lattice, coords, potcars, incar, root, *, kpoints=None):
    calc.mkdir(parents=True, exist_ok=False)
    write_poscar(calc / "POSCAR", title, symbols, counts, indices, lattice, coords)
    (calc / "INCAR").write_text(incar.replace("COMPONENT", title))
    shutil.copy2(kpoints if kpoints is not None else root / "KPOINTS", calc / "KPOINTS")
    (calc / "POTCAR").write_text("".join(potcars[symbol] for symbol in symbols))


def contains_completion(path):
    marker = "General timing and accounting informations"
    if not path.is_file():
        return False
    with path.open(errors="replace") as handle:
        return any(marker in line for line in handle)


def copy_small_result(source, target):
    if not contains_completion(source / "OUTCAR"):
        raise RuntimeError(f"Cannot reuse incomplete calculation: {source}")
    target.mkdir(parents=True, exist_ok=False)
    for name in COPY_RESULT_FILES:
        item = source / name
        if item.is_file():
            shutil.copy2(item, target / name)


def load_reuse_map(reuse_roots):
    result = {}
    for root in reuse_roots:
        table = root / "selected_waters.tsv"
        if not table.is_file():
            continue
        with table.open() as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                oxygen = int(row["O_serial"])
                water_dir = root / row["directory"]
                if contains_completion(water_dir / "a_without_water" / "OUTCAR") and contains_completion(
                    water_dir / "b_single_water" / "OUTCAR"
                ):
                    result[oxygen] = water_dir
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system-label", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-species", required=True)
    parser.add_argument("--expected-counts", required=True)
    parser.add_argument("--ab-source", type=Path, required=True)
    parser.add_argument("--reuse-root", type=Path, action="append", default=[])
    parser.add_argument("--potcar", type=Path, help="Author-supplied POTCAR; default: beside --source")
    parser.add_argument("--kpoints", type=Path, help="KPOINTS; default: beside --source")
    parser.add_argument(
        "--finalizer", type=Path,
        default=Path(__file__).resolve().with_name("finalize_remote_every4A_3each_interaction.py"),
        help="Summary script copied into the output (default: bundled every-4-A finalizer)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source = args.source.resolve()
    root = source.parent
    output = args.output.resolve()
    kpoints = args.kpoints.resolve() if args.kpoints else root / "KPOINTS"
    for required in (kpoints, args.finalizer):
        if not required.is_file():
            raise FileNotFoundError(required)
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite existing directory: {output}")
    lattice, species, counts, coords = read_poscar(source)
    expected_species = tuple(args.expected_species.split(","))
    expected_counts = tuple(map(int, args.expected_counts.split(",")))
    if species != expected_species or counts != expected_counts:
        raise RuntimeError(f"Unexpected composition: {species} {counts}")
    ranges = species_ranges(species, counts)
    pairs = identify_waters(lattice, coords, ranges)

    length_z = math.sqrt(sum(value * value for value in lattice[2]))
    oxygens = list(pairs)
    z_values = [(coords[oxygen][2] % 1.0) * length_z for oxygen in oxygens]
    unwrapped, layer_low, layer_high, middle, largest_gap = circular_layer(z_values, length_z)
    n_low = math.ceil((layer_low - middle) / PLANE_SPACING_A - 1e-10)
    n_high = math.floor((layer_high - middle) / PLANE_SPACING_A + 1e-10)
    planes = [(n, middle + PLANE_SPACING_A * n) for n in range(n_low, n_high + 1)]
    selections = []
    for plane_index, target in planes:
        ranked = sorted(
            zip(oxygens, z_values, unwrapped),
            key=lambda row: (abs(row[2] - target), row[0]),
        )[:WATERS_PER_PLANE]
        for water_rank, (oxygen, z_value, z_unwrapped) in enumerate(ranked, 1):
            selections.append((plane_index, target, water_rank, oxygen, z_value, z_unwrapped))
    selected_oxygens = [row[3] for row in selections]
    if len(set(selected_oxygens)) != len(selected_oxygens):
        raise RuntimeError("A water was selected for more than one target plane")

    reuse_map = load_reuse_map([path.resolve() for path in args.reuse_root])
    potcars = split_potcar(args.potcar.resolve() if args.potcar else root / "POTCAR", species)
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

    output.mkdir(parents=True)
    ab_ref = output / "ab_reference"
    copy_small_result(args.ab_source.resolve(), ab_ref)
    (ab_ref / "reuse_source_path.txt").write_text(str(args.ab_source.resolve()) + "\n")
    all_indices = list(range(sum(counts)))
    a_counts = list(counts)
    a_counts[species.index("H")] -= 2
    a_counts[species.index("O")] -= 1
    rows = [
        "system_label\tplane_index\ttarget_z_unwrapped_A\ttarget_z_mod_cell_A\twater_rank\t"
        "O_serial\tH1_serial\tH2_serial\tO_z_A\tabs_from_target_A\tdirectory\treused\treuse_source"
    ]
    pending = []
    reused_count = 0
    for plane_index, target, water_rank, oxygen, z_value, z_unwrapped in selections:
        h1, h2 = pairs[oxygen]
        plane_tag = f"m{abs(plane_index)}" if plane_index < 0 else f"p{plane_index}"
        relative = Path(f"plane_{plane_tag}_target{target % length_z:.3f}A") / (
            f"water{water_rank}_O{oxygen + 1}_z{z_value:.3f}A"
        )
        water_dir = output / relative
        reuse_source = reuse_map.get(oxygen + 1)
        if reuse_source is not None:
            copy_small_result(reuse_source / "a_without_water", water_dir / "a_without_water")
            copy_small_result(reuse_source / "b_single_water", water_dir / "b_single_water")
            (water_dir / "reuse_source_path.txt").write_text(str(reuse_source) + "\n")
            reused = "yes"
            reused_count += 1
        else:
            removed = {h1, h2, oxygen}
            a_indices = [index for index in all_indices if index not in removed]
            b_indices = [h1, h2, oxygen]
            write_calc(
                water_dir / "a_without_water", f"a_without_O{oxygen + 1}", species,
                tuple(a_counts), a_indices, lattice, coords, potcars, incar, root, kpoints=kpoints,
            )
            write_calc(
                water_dir / "b_single_water", f"b_O{oxygen + 1}", ("H", "O"),
                (2, 1), b_indices, lattice, coords, potcars, incar, root, kpoints=kpoints,
            )
            pending.append(str(relative))
            reused = "no"
            reuse_source = ""
        rows.append("\t".join((
            args.system_label, str(plane_index), f"{target:.6f}", f"{target % length_z:.6f}",
            str(water_rank), str(oxygen + 1), str(h1 + 1), str(h2 + 1), f"{z_value:.6f}",
            f"{abs(z_unwrapped - target):.6f}", str(relative), reused, str(reuse_source),
        )))
    (output / "selected_waters.tsv").write_text("\n".join(rows) + "\n")
    (output / "pending_water_dirs.txt").write_text("\n".join(pending) + ("\n" if pending else ""))
    plane_rows = ["plane_index\ttarget_z_unwrapped_A\ttarget_z_mod_cell_A"]
    plane_rows.extend(
        f"{index}\t{target:.6f}\t{target % length_z:.6f}" for index, target in planes
    )
    (output / "planes.tsv").write_text("\n".join(plane_rows) + "\n")
    (output / "water_layer_definition.tsv").write_text(
        "source\twater_O_count\tcell_z_A\twater_z_min_unwrapped_A\twater_z_max_unwrapped_A\t"
        "water_layer_middle_unwrapped_A\twater_layer_middle_mod_cell_A\tlargest_empty_z_gap_A\t"
        "plane_spacing_A\tplane_count\twaters_per_plane\n"
        f"{args.source_label}\t{len(pairs)}\t{length_z:.6f}\t{layer_low:.6f}\t{layer_high:.6f}\t"
        f"{middle:.6f}\t{middle % length_z:.6f}\t{largest_gap:.6f}\t{PLANE_SPACING_A:.6f}\t"
        f"{len(planes)}\t{WATERS_PER_PLANE}\n"
    )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    (output / "README.md").write_text(
        f"# {args.system_label}: three waters at every 4 A plane\n\n"
        f"Source: `{source}`, label `{args.source_label}`, SHA-256 `{digest}`. The periodic water "
        "layer is unwrapped by cutting at its largest empty z gap. Target planes are anchored "
        "at the layer center and separated by 4 A; each plane uses its three nearest unique waters.\n\n"
        "Only interaction energies are calculated: `E_int = E_ab - E_a - E_b`. Previously "
        "completed identical water calculations are reused without copying charge-density files.\n"
    )
    shutil.copy2(args.finalizer.resolve(), output / "finalize_results.py")
    print(
        f"Prepared {args.system_label}: planes={len(planes)} waters={len(selections)} "
        f"reused={reused_count} new={len(pending)}"
    )


if __name__ == "__main__":
    main()
