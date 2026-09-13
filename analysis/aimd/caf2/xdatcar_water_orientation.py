#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取 VASP AIMD 的 XDATCAR 中给定水分子的相对界面法向角 (tilt, θ) 和取向角 (azimuth, φ) 随时间（默认前 50 ps）的变化。

新增：支持用 **原子绝对序号 (serial)** 选择目标水分子（以 **O 原子的绝对序号** 为准）。
- 以前：`--mol-index` 按 O 的顺序编号（1..#O）。
- 现在也可以：`--serial-index` 直接给出 XDATCAR 中 O 原子的 **绝对序号（1-based，全原子计数）**。

定义：
- 分子偶极方向 μ：以 O 原子为参考，取两个 O–H 归一向量的和并归一化，即
    μ = norm( r_H1 - r_O )/|…| + norm( r_H2 - r_O )/|…|，再整体归一化。
- 界面法向 n：用户给定单位向量（默认 z 方向 (0,0,1)）。
- 倾角 θ（degrees）：μ 与 n 的夹角，θ = arccos( μ·n ).
- 方位角 φ（degrees）：μ 在界面平面上的投影相对参考 x' 轴的极角。x' 轴取为将全局 x 轴投影到界面平面并归一化；若与 n 共线则改用全局 y 轴。

假设：
- XDATCAR 的晶格在整个轨迹中不变（NVT/NVE 常见）。若在变胞（NPT）下，请改用 vasprun.xml/OUTCAR 获取逐步晶格。
- 物种行包含 O 与 H（大小写不敏感，支持如 "Ow"、"H" 等）。
- H 的归属采用每一帧基于近邻的方式：对所选 O，选择两颗到 O 的最近 H（O–H 截断默认 1.2 Å，可调）。
- 周期性边界条件：以分数坐标下的最小镜像法处理 O↔H 位移。

输出：
- CSV：<prefix>_mol<M>.csv 或 <prefix>_serial<S>.csv，包含 [frame,time_ps,theta_deg,phi_deg,z_O(A)]。
- 图像：对应的 _theta.png 与 _phi.png（时间-角度曲线）。

用法示例：
1) 按 O 顺序（1..#O）选择：
    python xdatcar_water_orientation.py \
        --xdatcar XDATCAR \
        --mol-index 5 \
        --timestep-fs 1.0 \
        --max-time-ps 50 \
        --normal 0,0,1 \
        --oh-cutoff 1.2 \
        --out-prefix orient

2) 用绝对序号（例如 XDATCAR 的第 140 号原子是 O）：
    python xdatcar_water_orientation.py \
        --xdatcar XDATCAR \
        --serial-index 140 \
        --timestep-fs 1.0 \
        --max-time-ps 50 \
        --normal 0,0,1 \
        --oh-cutoff 1.2 \
        --out-prefix orient

可同时指定多个目标：`--mol-index 3 7 12` 或 `--serial-index 140 233 318`
如物种标签非常规，可指定：`--oxygen-label O --hydrogen-label H`
"""
from __future__ import annotations
import argparse
import math
import os
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------- 基础几何与工具 -------------------------
def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def frac_to_cart(frac: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """分数 -> 笛卡尔，frac: (...,3), lattice: (3,3) 行向量为晶格基。"""
    return frac @ lattice


def min_image_delta_frac(d_frac: np.ndarray) -> np.ndarray:
    """将分数位移包装到 [-0.5, 0.5) 区间（最小镜像）。"""
    return d_frac - np.round(d_frac)


def angle_deg_between(u: np.ndarray, v: np.ndarray, eps: float = 1e-12) -> float:
    u_n = normalize(u)
    v_n = normalize(v)
    dot = float(np.clip(np.dot(u_n, v_n), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def plane_axes_from_normal(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """由法向 n 生成平面正交基 (x', y', n)。x' 为全局 x(1,0,0) 在平面的投影（或退化时用 y）。"""
    n = normalize(n)
    x_global = np.array([1.0, 0.0, 0.0])
    y_global = np.array([0.0, 1.0, 0.0])
    xprime = x_global - np.dot(x_global, n) * n
    if np.linalg.norm(xprime) < 1e-8:
        xprime = y_global - np.dot(y_global, n) * n
    xprime = normalize(xprime)
    yprime = normalize(np.cross(n, xprime))
    return xprime, yprime, n


def azimuth_deg_in_plane(v: np.ndarray, xprime: np.ndarray, yprime: np.ndarray, n: np.ndarray) -> float:
    # 先投影到平面
    v_par = v - np.dot(v, n) * n
    if np.linalg.norm(v_par) < 1e-12:
        return float('nan')
    vx = np.dot(v_par, xprime)
    vy = np.dot(v_par, yprime)
    phi = math.degrees(math.atan2(vy, vx))
    if phi < 0:
        phi += 360.0
    return phi


# ------------------------- XDATCAR 解析 -------------------------
class Xdatcar:
    def __init__(self, path: str):
        self.path = path
        self.comment = ''
        self.scale = 1.0
        self.lattice = np.eye(3)
        self.species: List[str] = []
        self.counts: List[int] = []
        self.n_atoms = 0
        self.frames_frac: List[np.ndarray] = []  # 每帧 N×3 的分数坐标
        self.coord_mode_per_frame: List[str] = []
        self._parse()

    def _parse(self):
        with open(self.path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [ln.rstrip() for ln in f]
        if len(lines) < 8:
            raise ValueError('XDATCAR 内容过短，无法解析。')

        it = 0
        self.comment = lines[it]; it += 1
        self.scale = float(lines[it].split()[0]); it += 1
        # 晶格 3 行
        lat = []
        for _ in range(3):
            lat.append([float(x) for x in lines[it].split()[:3]])
            it += 1
        self.lattice = self.scale * np.array(lat)  # 3x3

        # 可能存在：元素名行 + 元素数行
        species_line = lines[it].split(); it += 1
        counts_line = lines[it].split(); it += 1
        # 健壮性检查
        def is_configuration_line(s: str) -> bool:
            s = s.strip().lower()
            return s.startswith('direct configuration=') or s.startswith('cartesian configuration=')

        if is_configuration_line(' '.join(counts_line)):
            raise ValueError('未检测到元素计数行。请检查 XDATCAR 是否为标准格式（含元素名与元素数两行）。')

        if all(tok.replace('-', '').replace('.', '').isdigit() for tok in species_line):
            counts = [int(x) for x in species_line]
            species = [f'X{i+1}' for i in range(len(counts))]
            it -= 1
        else:
            species = species_line
            counts = [int(x) for x in counts_line]

        self.species = species
        self.counts = counts
        self.n_atoms = sum(self.counts)

        # 逐帧读取
        idx = it
        while idx < len(lines):
            line = lines[idx].strip()
            if not line:
                idx += 1
                continue
            low = line.lower()
            if low.startswith('direct configuration='):
                coord_mode = 'Direct'; idx += 1
            elif low.startswith('cartesian configuration='):
                coord_mode = 'Cartesian'; idx += 1
            else:
                if low.startswith('direct'):
                    coord_mode = 'Direct'; idx += 1
                elif low.startswith('cartesian'):
                    coord_mode = 'Cartesian'; idx += 1
                else:
                    break
            if idx + self.n_atoms > len(lines):
                break
            coords = []
            for j in range(self.n_atoms):
                parts = lines[idx + j].split()
                if len(parts) < 3:
                    raise ValueError(f'第 {len(self.frames_frac)+1} 帧坐标行格式错误。')
                coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
            idx += self.n_atoms
            coords = np.array(coords)
            if coord_mode.lower().startswith('cart'):
                inv_lat = np.linalg.inv(self.lattice)
                coords = coords @ inv_lat
            coords = coords - np.floor(coords)
            self.frames_frac.append(coords)
            self.coord_mode_per_frame.append(coord_mode)

        if not self.frames_frac:
            raise ValueError('未在 XDATCAR 中解析到任何帧。')


# ------------------------- 物种与索引工具 -------------------------
def species_ranges(species: List[str], counts: List[int]) -> Dict[str, Tuple[int, int]]:
    """返回每个物种在原子列表中的半开区间 [start, end)。"""
    ranges = {}
    start = 0
    for name, cnt in zip(species, counts):
        ranges[name] = (start, start + cnt)
        start += cnt
    return ranges


def pick_label(names: List[str], wanted: str | None, default_key: str) -> str:
    if wanted and any(n.lower() == wanted.lower() for n in names):
        return next(n for n in names if n.lower() == wanted.lower())
    key = default_key.lower()
    cand = [n for n in names if key in n.lower()]
    if cand:
        exact = [n for n in cand if n.lower() == key]
        return exact[0] if exact else cand[0]
    return names[0]


def species_name_of_serial(serial_0based: int, species: List[str], counts: List[int]) -> str:
    start = 0
    for name, cnt in zip(species, counts):
        if start <= serial_0based < start + cnt:
            return name
        start += cnt
    return 'UNKNOWN'


# ------------------------- 水分子构型与角度 -------------------------
def water_mu_for_O(frac_coords: np.ndarray, lattice: np.ndarray, o_index: int,
                   H_indices_pool: np.ndarray, oh_cutoff: float = 1.2) -> Tuple[np.ndarray, int, int]:
    O_frac = frac_coords[o_index]
    H_fracs = frac_coords[H_indices_pool]
    delta_frac = H_fracs - O_frac[None, :]
    delta_frac = min_image_delta_frac(delta_frac)
    delta_cart = delta_frac @ lattice
    dists = np.linalg.norm(delta_cart, axis=1)
    order = np.argsort(dists)
    if len(order) < 2:
        raise RuntimeError('H 池中少于 2 个原子。')
    h1i = H_indices_pool[order[0]]
    h2i = H_indices_pool[order[1]]
    if dists[order[0]] > oh_cutoff or dists[order[1]] > oh_cutoff:
        raise RuntimeError(f'找不到两颗 H 满足 O–H 截断 {oh_cutoff} Å（最短 {dists[order[0]]:.2f}, 次短 {dists[order[1]]:.2f}）。')

    v1 = delta_cart[order[0]]
    v2 = delta_cart[order[1]]
    mu = normalize(v1) + normalize(v2)
    mu = normalize(mu)
    return mu, int(h1i), int(h2i)


# ------------------------- 主逻辑 -------------------------
def process(
    xdatcar_path: str,
    mol_indices: List[int] | None,
    serial_indices: List[int] | None,
    timestep_fs: float,
    max_time_ps: float,
    normal_vec: Tuple[float, float, float],
    oh_cutoff: float,
    out_prefix: str,
    oxygen_label: str | None,
    hydrogen_label: str | None,
):
    xd = Xdatcar(xdatcar_path)
    rng = species_ranges(xd.species, xd.counts)

    o_label = pick_label(xd.species, oxygen_label, default_key='O')
    h_label = pick_label(xd.species, hydrogen_label, default_key='H')
    if o_label not in rng or h_label not in rng:
        raise ValueError(f'无法在物种中找到氧/氢：{xd.species}')

    o_start, o_end = rng[o_label]
    h_start, h_end = rng[h_label]
    nO = o_end - o_start

    # 统一得到目标 O 的绝对索引（0-based）以及用于输出命名的标签
    targets: List[Tuple[int, str]] = []

    if mol_indices:
        for m in mol_indices:
            if m < 1 or m > nO:
                raise ValueError(f'水分子序号 {m} 越界（1..{nO}）。如需按原子绝对序号指定，请改用 --serial-index。')
            o_abs = o_start + (m - 1)
            targets.append((o_abs, f"mol{m}"))

    if serial_indices:
        for s in serial_indices:
            if s < 1 or s > xd.n_atoms:
                raise ValueError(f'绝对序号 {s} 越界（1..{xd.n_atoms}）。')
            s0 = s - 1
            sp = species_name_of_serial(s0, xd.species, xd.counts)
            if not (o_start <= s0 < o_end):
                raise ValueError(
                    f'绝对序号 {s} 对应的物种为 {sp}，不在氧区间 {o_start+1}..{o_end}（1-based）。请提供 **O 原子的**绝对序号。')
            targets.append((s0, f"serial{s}"))

    if not targets:
        raise ValueError('未提供目标序号。请使用 --mol-index 或 --serial-index。')

    H_pool = np.arange(h_start, h_end, dtype=int)

    n_frames_total = len(xd.frames_frac)
    if timestep_fs <= 0:
        raise ValueError('--timestep-fs 必须为正。')
    frames_limit = int(math.floor(max_time_ps * 1000.0 / timestep_fs)) + 1
    n_frames = min(n_frames_total, frames_limit)

    lattice = xd.lattice  # 假设不变
    n_vec = normalize(np.array(normal_vec, dtype=float))
    xprime, yprime, n_unit = plane_axes_from_normal(n_vec)

    os.makedirs(os.path.dirname(out_prefix) or '.', exist_ok=True)

    for o_idx, tag in targets:
        records = []
        last_h_pair: Tuple[int, int] | None = None
        for f in range(n_frames):
            frac = xd.frames_frac[f]
            # 计算 μ
            try:
                mu, h1i, h2i = water_mu_for_O(frac, lattice, o_idx, H_pool, oh_cutoff=oh_cutoff)
                last_h_pair = (h1i, h2i)
            except RuntimeError:
                if last_h_pair is not None:
                    h1i, h2i = last_h_pair
                    O_frac = frac[o_idx]
                    H_fracs = frac[[h1i, h2i]]
                    d_frac = min_image_delta_frac(H_fracs - O_frac)
                    d_cart = d_frac @ lattice
                    mu = normalize(normalize(d_cart[0]) + normalize(d_cart[1]))
                else:
                    mu = np.array([np.nan, np.nan, np.nan])
                    h1i = h2i = -1

            if np.any(np.isnan(mu)):
                theta = float('nan'); phi = float('nan')
            else:
                theta = angle_deg_between(mu, n_unit)
                phi = azimuth_deg_in_plane(mu, xprime, yprime, n_unit)

            O_cart = frac_to_cart(frac[o_idx][None, :], lattice)[0]
            time_ps = f * timestep_fs / 1000.0
            records.append({
                'frame': f,
                'time_ps': time_ps,
                'theta_deg': theta,
                'phi_deg': phi,
                'z_O(A)': O_cart[2],
                'O_index_0based': o_idx,
                'H1_index_0based': h1i,
                'H2_index_0based': h2i,
            })

        df = pd.DataFrame(records)
        csv_path = f"{out_prefix}_{tag}.csv"
        df.to_csv(csv_path, index=False)

        # 绘图：时间-θ
        plt.figure(figsize=(8, 4.2))
        plt.plot(df['time_ps'], df['theta_deg'], lw=1.2)
        plt.xlabel('Time (ps)')
        plt.ylabel('Tilt θ (deg)')
        plt.title(f'Water ({tag}): tilt vs time')
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_{tag}_theta.png", dpi=200)
        plt.close()

        # 绘图：时间-φ
        plt.figure(figsize=(8, 4.2))
        plt.plot(df['time_ps'], df['phi_deg'], lw=1.2)
        plt.xlabel('Time (ps)')
        plt.ylabel('Azimuth φ (deg)')
        plt.title(f'Water ({tag}): azimuth vs time')
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_{tag}_phi.png", dpi=200)
        plt.close()

        print(f"目标 {tag}: 已写入 {csv_path}，以及对应的 PNG 曲线图。")


# ------------------------- CLI -------------------------
def parse_normal(s: str) -> Tuple[float, float, float]:
    parts = [p for p in s.replace(';', ',').split(',') if p.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError('法向需要 3 个逗号分隔的数，例如 0,0,1')
    v = np.array([float(x) for x in parts], dtype=float)
    if np.linalg.norm(v) < 1e-12:
        raise argparse.ArgumentTypeError('法向向量不能为零。')
    v = normalize(v)
    return float(v[0]), float(v[1]), float(v[2])


def main():
    ap = argparse.ArgumentParser(description='从 XDATCAR 提取指定水分子的倾角与方位角随时间的变化。')
    ap.add_argument('--xdatcar', type=str, default='XDATCAR', help='XDATCAR 文件路径（默认当前目录 XDATCAR）')

    idx_group = ap.add_mutually_exclusive_group(required=True)
    idx_group.add_argument('--mol-index', type=int, nargs='+', help='水分子序号（按 O 的顺序，1-based）。可给多个。')
    idx_group.add_argument('--serial-index', type=int, nargs='+', help='**原子绝对序号（1-based）**，应为 O 原子在全原子序列中的序号。可给多个。')

    ap.add_argument('--timestep-fs', type=float, default=1.0, help='相邻帧时间步长（fs），用于构造时间轴（默认 1.0 fs）')
    ap.add_argument('--max-time-ps', type=float, default=50.0, help='输出的时间上限（ps），默认 50 ps')
    ap.add_argument('--normal', type=parse_normal, default=(0.0, 0.0, 1.0), help='界面法向单位向量，形如 0,0,1（默认 z 轴）')
    ap.add_argument('--oh-cutoff', type=float, default=1.2, help='O–H 最近邻判定截断（Å），默认 1.2 Å')
    ap.add_argument('--out-prefix', type=str, default='orientation', help='输出前缀（默认 orientation）')
    ap.add_argument('--oxygen-label', type=str, default=None, help='氧元素在 XDATCAR 物种行中的标签（默认自动识别）')
    ap.add_argument('--hydrogen-label', type=str, default=None, help='氢元素在 XDATCAR 物种行中的标签（默认自动识别）')

    args = ap.parse_args()

    process(
        xdatcar_path=args.xdatcar,
        mol_indices=args.mol_index,
        serial_indices=args.serial_index,
        timestep_fs=args.timestep_fs,
        max_time_ps=args.max_time_ps,
        normal_vec=args.normal,
        oh_cutoff=args.oh_cutoff,
        out_prefix=args.out_prefix,
        oxygen_label=args.oxygen_label,
        hydrogen_label=args.hydrogen_label,
    )


if __name__ == '__main__':
    main()

