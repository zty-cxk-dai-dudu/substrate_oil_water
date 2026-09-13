#!/usr/bin/env python3
import argparse
import math
import os
import shutil
import subprocess
from pathlib import Path


def read_poscar(path):
    lines = Path(path).read_text().splitlines()
    scale = float(lines[1].split()[0])
    lattice = [[float(x) * scale for x in lines[i].split()[:3]] for i in range(2, 5)]
    species = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    coord_start = 8
    if lines[7].strip().lower().startswith("s"):
        coord_start = 9
    mode = lines[coord_start - 1].strip().lower()
    coords = []
    labels = []
    idx = 0
    for elem, count in zip(species, counts):
        for j in range(count):
            coords.append([float(x) for x in lines[coord_start + idx].split()[:3]])
            labels.append(f"{elem}{j + 1}")
            idx += 1
    return lattice, species, counts, mode, coords, labels


def frac_delta(a, b):
    return [a[i] - b[i] - round(a[i] - b[i]) for i in range(3)]


def cart_from_frac(frac, lattice):
    return [
        frac[0] * lattice[0][i] + frac[1] * lattice[1][i] + frac[2] * lattice[2][i]
        for i in range(3)
    ]


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def z_cart(frac, lattice):
    return frac[0] * lattice[0][2] + frac[1] * lattice[1][2] + frac[2] * lattice[2][2]


def infer_water_triplets(lattice, species, counts, mode, coords):
    if not mode.startswith("d"):
        raise SystemExit("This helper currently expects Direct POSCAR coordinates.")
    if "O" not in species or "H" not in species:
        raise SystemExit(f"Expected POSCAR to contain O and H, got {species}")
    starts = {}
    atom_id = 1
    for elem, count in zip(species, counts):
        starts[elem] = atom_id
        atom_id += count
    oxygen_ids = list(range(starts["O"], starts["O"] + counts[species.index("O")]))
    hydrogen_ids = list(range(starts["H"], starts["H"] + counts[species.index("H")]))
    triplets = []
    for o_id in oxygen_ids:
        o_frac = coords[o_id - 1]
        distances = []
        for h_id in hydrogen_ids:
            d_cart = cart_from_frac(frac_delta(coords[h_id - 1], o_frac), lattice)
            distances.append((norm(d_cart), h_id))
        nearest = sorted(distances)[:2]
        if nearest[1][0] > 1.35:
            raise SystemExit(
                f"O{o_id}: second-nearest H is {nearest[1][0]:.3f} A; "
                "water mapping is ambiguous."
            )
        triplets.append(
            {
                "o_id": o_id,
                "h1_id": nearest[0][1],
                "h2_id": nearest[1][1],
                "oh1": nearest[0][0],
                "oh2": nearest[1][0],
                "o_z": z_cart(o_frac, lattice),
            }
        )
    return triplets


def build_layers(triplets, width):
    z_min = min(t["o_z"] for t in triplets)
    z_max = max(t["o_z"] for t in triplets)
    layers = []
    lo = z_min
    i = 1
    while lo <= z_max + 1e-9:
        hi = lo + width
        if hi > z_max:
            selected = [t for t in triplets if lo <= t["o_z"] <= hi + 1e-9]
        else:
            selected = [t for t in triplets if lo <= t["o_z"] < hi]
        if selected:
            layers.append((i, lo, hi, selected))
            i += 1
        lo = hi
    return layers


def write_selected_atoms(path, coords, labels, selected):
    with Path(path).open("w") as f:
        f.write("ATOMS_ID ATOM_LABEL   X_POSITION    Y_POSITION    Z_POSITION     SELECTED?\n")
        for atom_id, (label, xyz) in enumerate(zip(labels, coords), start=1):
            flag = "T" if atom_id in selected else "F"
            f.write(
                f"{atom_id:4d} {label:>8s}"
                f" {xyz[0]:13.8f} {xyz[1]:13.8f} {xyz[2]:13.8f}"
                f" {flag:>9s}\n"
            )


def symlink(src, dst):
    src = Path(src)
    dst = Path(dst)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def safe_layer_name(index, lo, hi):
    return f"layer{index:02d}_z{lo:.3f}_{hi:.3f}".replace(".", "p")


def main():
    parser = argparse.ArgumentParser(
        description="Run VASPKIT 728 for upward 3 A water layers, selecting complete waters."
    )
    parser.add_argument("--vaspkit", default="./vaspkit")
    parser.add_argument("--xdatcar", default="XDATCAR")
    parser.add_argument("--width", type=float, default=3.0)
    parser.add_argument("--outdir", default="vdos_water_layers_3A_up_from_interface_vaspkit_728")
    args = parser.parse_args()

    cwd = Path.cwd()
    outdir = cwd / args.outdir
    outdir.mkdir(exist_ok=True)
    lattice, species, counts, mode, coords, labels = read_poscar(cwd / "POSCAR")
    triplets = infer_water_triplets(lattice, species, counts, mode, coords)
    layers = build_layers(triplets, args.width)

    with (outdir / "water_atom_map.csv").open("w") as f:
        f.write("O_serial,O_atom_id,H1_atom_id,H2_atom_id,O_z_A,OH1_A,OH2_A\n")
        for t in triplets:
            f.write(
                f"{t['o_id']},{t['o_id']},{t['h1_id']},{t['h2_id']},"
                f"{t['o_z']:.8f},{t['oh1']:.6f},{t['oh2']:.6f}\n"
            )

    summary = outdir / "layer_summary.csv"
    with summary.open("w") as f:
        f.write(
            "layer,run_dir,z_min_A,z_max_A,n_waters,n_atoms,O_serials,status,output_dat,output_txt\n"
        )

    for index, lo, hi, selected_waters in layers:
        run_name = safe_layer_name(index, lo, hi)
        run_dir = outdir / run_name
        run_dir.mkdir(exist_ok=True)
        selected_atoms = set()
        o_serials = []
        for t in selected_waters:
            selected_atoms.update([t["o_id"], t["h1_id"], t["h2_id"]])
            o_serials.append(t["o_id"])

        for name in ["POSCAR", "INCAR", "KPOINTS"]:
            symlink(cwd / name, run_dir / name)
        symlink(cwd / args.xdatcar, run_dir / "XDATCAR")
        symlink(Path(args.vaspkit), run_dir / "vaspkit")
        write_selected_atoms(run_dir / "SELECTED_ATOMS_LIST", coords, labels, selected_atoms)

        with (run_dir / "selected_waters.csv").open("w") as f:
            f.write("O_serial,O_atom_id,H1_atom_id,H2_atom_id,O_z_A\n")
            for t in selected_waters:
                f.write(f"{t['o_id']},{t['o_id']},{t['h1_id']},{t['h2_id']},{t['o_z']:.8f}\n")

        proc = subprocess.run(
            ["./vaspkit"],
            cwd=run_dir,
            input="728\n4\n4\n0\n1\n",
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        (run_dir / "vaspkit_728.log").write_text(proc.stdout)
        dat_path = run_dir / "VDOS.dat"
        status = "ok" if proc.returncode == 0 and dat_path.exists() else f"failed:{proc.returncode}"
        out_dat = outdir / f"{run_name}.dat"
        out_txt = outdir / f"{run_name}.txt"
        if dat_path.exists():
            shutil.copy2(dat_path, out_dat)
            shutil.copy2(dat_path, out_txt)
        with summary.open("a") as f:
            f.write(
                f"{index},{run_name},{lo:.8f},{hi:.8f},{len(selected_waters)},"
                f"{len(selected_atoms)},\"{' '.join(map(str, o_serials))}\",{status},{out_dat},{out_txt}\n"
            )
        print(f"{run_name}: waters={len(selected_waters)} atoms={len(selected_atoms)} {status}", flush=True)
        if proc.returncode != 0:
            raise SystemExit(f"VASPKIT failed for {run_name}; see {run_dir / 'vaspkit_728.log'}")


if __name__ == "__main__":
    main()
