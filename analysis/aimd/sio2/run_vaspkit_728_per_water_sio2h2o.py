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
    lattice = []
    for i in range(2, 5):
        lattice.append([float(x) * scale for x in lines[i].split()[:3]])
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
            vals = [float(x) for x in lines[coord_start + idx].split()[:3]]
            coords.append(vals)
            labels.append(f"{elem}{j + 1}")
            idx += 1
    return lattice, species, counts, mode, coords, labels


def frac_delta(a, b):
    d = [a[i] - b[i] for i in range(3)]
    return [x - round(x) for x in d]


def cart_from_frac(frac, lattice):
    return [
        frac[0] * lattice[0][i] + frac[1] * lattice[1][i] + frac[2] * lattice[2][i]
        for i in range(3)
    ]


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def frac_from_cart(cart, lattice):
    det = (
        lattice[0][0] * (lattice[1][1] * lattice[2][2] - lattice[1][2] * lattice[2][1])
        - lattice[0][1] * (lattice[1][0] * lattice[2][2] - lattice[1][2] * lattice[2][0])
        + lattice[0][2] * (lattice[1][0] * lattice[2][1] - lattice[1][1] * lattice[2][0])
    )
    if abs(det) < 1.0e-12:
        raise SystemExit("Singular POSCAR lattice.")
    # Coordinates are row vectors; solve frac @ lattice = cart.
    a, b, c = lattice
    m = [
        [a[0], b[0], c[0]],
        [a[1], b[1], c[1]],
        [a[2], b[2], c[2]],
    ]
    x, y, z = cart
    inv_det = 1.0 / det
    return [
        (
            x * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - y * (m[0][1] * m[2][2] - m[0][2] * m[2][1])
            + z * (m[0][1] * m[1][2] - m[0][2] * m[1][1])
        )
        * inv_det,
        (
            -x * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + y * (m[0][0] * m[2][2] - m[0][2] * m[2][0])
            - z * (m[0][0] * m[1][2] - m[0][2] * m[1][0])
        )
        * inv_det,
        (
            x * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
            - y * (m[0][0] * m[2][1] - m[0][1] * m[2][0])
            + z * (m[0][0] * m[1][1] - m[0][1] * m[1][0])
        )
        * inv_det,
    ]


def coords_as_frac(mode, coords, lattice):
    if mode.startswith("d"):
        return coords
    if mode.startswith("c") or mode.startswith("k"):
        return [frac_from_cart(xyz, lattice) for xyz in coords]
    raise SystemExit(f"Unknown POSCAR coordinate mode: {mode}")


def infer_water_triplets(lattice, species, counts, mode, coords, expected_waters):
    if "O" not in species or "H" not in species:
        raise SystemExit(f"Expected POSCAR to contain O and H, got {species}")
    frac_coords = coords_as_frac(mode, coords, lattice)
    starts = {}
    atom_id = 1
    for elem, count in zip(species, counts):
        starts[elem] = atom_id
        atom_id += count
    oxygen_ids = list(range(starts["O"], starts["O"] + counts[species.index("O")]))
    hydrogen_ids = list(range(starts["H"], starts["H"] + counts[species.index("H")]))
    candidates = []
    for o_id in oxygen_ids:
        o_frac = frac_coords[o_id - 1]
        distances = []
        for h_id in hydrogen_ids:
            h_frac = frac_coords[h_id - 1]
            d_frac = frac_delta(h_frac, o_frac)
            d_cart = cart_from_frac(d_frac, lattice)
            distances.append((norm(d_cart), h_id))
        for dist, h_id in sorted(distances)[:8]:
            if dist <= 1.25:
                candidates.append((dist, o_id, h_id))
    candidates.sort()
    assigned = {o_id: [] for o_id in oxygen_ids}
    used_h = set()
    for dist, o_id, h_id in candidates:
        if h_id in used_h or len(assigned[o_id]) >= 2:
            continue
        assigned[o_id].append((h_id, dist))
        used_h.add(h_id)
    triplets = []
    for o_id in oxygen_ids:
        if len(assigned[o_id]) == 2:
            (h1, d1), (h2, d2) = assigned[o_id]
            triplets.append((o_id, h1, h2, d1, d2))
    if len(triplets) != expected_waters:
        raise SystemExit(
            f"Expected {expected_waters} waters from O-H geometry, "
            f"identified {len(triplets)}."
        )
    return triplets


def write_selected_atoms(path, lattice, species, counts, mode, coords, labels, selected):
    with Path(path).open("w") as f:
        f.write("ATOMS_ID ATOM_LABEL   X_POSITION    Y_POSITION    Z_POSITION     SELECTED?\n")
        for atom_id, (label, xyz) in enumerate(zip(labels, coords), start=1):
            flag = "T" if atom_id in selected else "F"
            f.write(
                f"{atom_id:4d} {label:>8s}"
                f" {xyz[0]:13.8f} {xyz[1]:13.8f} {xyz[2]:13.8f}"
                f" {flag:>9s}\n"
            )


def copy_or_link(src, dst):
    src = Path(src)
    dst = Path(dst)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def main():
    parser = argparse.ArgumentParser(
        description="Run VASPKIT task 728 for each water molecule selected by O serial."
    )
    parser.add_argument("--vaspkit", default="./vaspkit")
    parser.add_argument("--outdir", default="vdos_per_water_vaspkit_728")
    parser.add_argument("--limit", type=int, default=0, help="Debug: run first N waters only.")
    parser.add_argument("--expected-waters", type=int, default=100)
    args = parser.parse_args()

    cwd = Path.cwd()
    outdir = cwd / args.outdir
    outdir.mkdir(exist_ok=True)

    lattice, species, counts, mode, coords, labels = read_poscar(cwd / "POSCAR")
    triplets = infer_water_triplets(
        lattice, species, counts, mode, coords, args.expected_waters
    )
    if args.limit:
        triplets = triplets[: args.limit]

    map_path = outdir / "water_atom_map.csv"
    with map_path.open("w") as f:
        f.write("O_serial,O_atom_id,H1_atom_id,H2_atom_id,OH1_A,OH2_A\n")
        for o_id, h1, h2, d1, d2 in triplets:
            f.write(f"{o_id},{o_id},{h1},{h2},{d1:.6f},{d2:.6f}\n")

    summary_path = outdir / "summary.csv"
    with summary_path.open("w") as f:
        f.write("O_serial,run_dir,status,output_dat,output_txt\n")

    for o_id, h1, h2, _, _ in triplets:
        run_dir = outdir / f"O{o_id}"
        run_dir.mkdir(exist_ok=True)
        for name in ["POSCAR", "XDATCAR", "INCAR", "KPOINTS"]:
            copy_or_link(cwd / name, run_dir / name)
        copy_or_link(Path(args.vaspkit), run_dir / "vaspkit")
        write_selected_atoms(
            run_dir / "SELECTED_ATOMS_LIST",
            lattice,
            species,
            counts,
            mode,
            coords,
            labels,
            {o_id, h1, h2},
        )
        proc = subprocess.run(
            ["./vaspkit"],
            cwd=run_dir,
            input="728\n4\n4\n0\n1\n",
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        log_path = run_dir / "vaspkit_728.log"
        log_path.write_text(proc.stdout)
        dat_path = run_dir / "VDOS.dat"
        txt_path = outdir / f"O{o_id}.txt"
        status = "ok" if proc.returncode == 0 and dat_path.exists() else f"failed:{proc.returncode}"
        if dat_path.exists():
            shutil.copy2(dat_path, outdir / f"O{o_id}.dat")
            shutil.copy2(dat_path, txt_path)
        with summary_path.open("a") as f:
            f.write(f"{o_id},{run_dir.name},{status},{outdir / f'O{o_id}.dat'},{txt_path}\n")
        print(f"O{o_id}: {status}", flush=True)
        if proc.returncode != 0:
            raise SystemExit(f"VASPKIT failed for O{o_id}; see {log_path}")


if __name__ == "__main__":
    main()
