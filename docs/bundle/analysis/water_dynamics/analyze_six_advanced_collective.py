#!/usr/bin/env python3
"""Six complementary collective-motion analyses for 500 ps water films."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np


CM_PER_PS = 33.3564095198152
MASS_O = 15.9994
MASS_H = 1.00794
BANDS = ((20.0, 60.0), (60.0, 120.0), (120.0, 300.0))


def minimum_image(delta: np.ndarray, cell: np.ndarray) -> np.ndarray:
    frac = delta @ np.linalg.inv(cell)
    frac -= np.rint(frac)
    return frac @ cell


def water_topology(symbols: np.ndarray, positions: np.ndarray,
                   cell: np.ndarray, cutoff: float = 1.30) -> np.ndarray:
    oxygen = np.flatnonzero(symbols == "O")
    hydrogen = np.flatnonzero(symbols == "H")
    candidates = []
    for o in oxygen:
        distance = np.linalg.norm(
            minimum_image(positions[hydrogen] - positions[o], cell), axis=1
        )
        candidates.extend((float(d), int(o), int(h))
                          for h, d in zip(hydrogen, distance) if d <= cutoff)
    candidates.sort()
    assigned = {int(o): [] for o in oxygen}
    used_h = set()
    for _distance, o, h in candidates:
        if len(assigned[o]) < 2 and h not in used_h:
            assigned[o].append(h)
            used_h.add(h)
    waters = [(int(o), *assigned[int(o)]) for o in oxygen
              if len(assigned[int(o)]) == 2]
    if len(waters) != len(oxygen):
        raise RuntimeError(f"Only {len(waters)}/{len(oxygen)} waters assigned")
    return np.asarray(waters, dtype=np.int64)


def load_candidate_mean_z(path: Path) -> dict[int, float]:
    result = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            result[int(row["O_serial_1based"])] = float(row["mean_z_minus_GDS_A"])
    return result


def write_csv(path: Path, header: list[str], rows) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def source_parts(paths: list[Path], total_frames: int, stride: int,
                 datasets: tuple[str, ...]):
    global_start = 0
    remaining = total_frames
    for path in paths:
        if remaining <= 0:
            break
        with h5py.File(path, "r") as handle:
            if not bool(handle.attrs.get("complete", False)):
                raise RuntimeError(f"Incomplete source {path}")
            available = min(len(handle["step"]), remaining)
            offset = (-global_start) % stride
            indices = np.arange(offset, available, stride, dtype=np.int64)
            yield {name: np.asarray(handle[name][indices]) for name in datasets}
            global_start += available
            remaining -= available
    if remaining:
        raise RuntimeError(f"Missing {remaining} requested source frames")


def directed_hbond(o_xyz: np.ndarray, h_xyz: np.ndarray, cell: np.ndarray,
                   ho_cutoff: float, oo_cutoff: float,
                   angle_cutoff: float) -> np.ndarray:
    h_to_acceptor = minimum_image(
        o_xyz[:, None, None, :, :] - h_xyz[:, :, :, None, :], cell
    )
    ho_distance = np.linalg.norm(h_to_acceptor, axis=-1)
    donor_to_acceptor = minimum_image(
        o_xyz[:, None, :, :] - o_xyz[:, :, None, :], cell
    )
    oo_distance = np.linalg.norm(donor_to_acceptor, axis=-1)
    h_to_donor = minimum_image(o_xyz[:, :, None, :] - h_xyz, cell)
    h_to_donor = h_to_donor[:, :, :, None, :]
    cosine = np.sum(h_to_donor * h_to_acceptor, axis=-1) / np.maximum(
        np.linalg.norm(h_to_donor, axis=-1)
        * np.linalg.norm(h_to_acceptor, axis=-1), 1.0e-30
    )
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    not_self = ~np.eye(o_xyz.shape[1], dtype=bool)[None, :, None, :]
    state = ((ho_distance <= ho_cutoff)
             & (oo_distance[:, :, None, :] <= oo_cutoff)
             & (angle >= angle_cutoff) & not_self)
    return np.any(state, axis=2)


def graph_prepass(args, paths, cell, waters, fixed_layer):
    oxygen, h1, h2 = waters.T
    h_by_water = np.stack((h1, h2), axis=1)
    nwater = len(waters)
    upper_i, upper_j = np.triu_indices(nwater, 1)
    valid_pairs = ((fixed_layer[upper_i] >= 0) & (fixed_layer[upper_j] >= 0))
    left = np.minimum(fixed_layer[upper_i], fixed_layer[upper_j])
    right = np.maximum(fixed_layer[upper_i], fixed_layer[upper_j])
    pair_code = left * args.n_layers + right
    total = np.zeros((args.n_layers, args.n_layers), dtype=float)
    samples = 0
    for part_number, part in enumerate(source_parts(
        paths, args.source_frames, args.hbond_stride, ("positions_A",)
    ), start=1):
        positions = np.asarray(part["positions_A"], dtype=np.float32)
        for start in range(0, len(positions), 10):
            chunk = positions[start:start + 10]
            o_xyz = chunk[:, oxygen]
            h_xyz = chunk[:, h_by_water]
            directed = directed_hbond(
                o_xyz, h_xyz, cell, args.hbond_ho,
                args.hbond_oo, args.hbond_angle,
            )
            adjacency = directed | np.swapaxes(directed, 1, 2)
            for frame in adjacency:
                edge = frame[upper_i, upper_j] & valid_pairs
                counts = np.bincount(pair_code[edge], minlength=args.n_layers ** 2)
                counts = counts.reshape(args.n_layers, args.n_layers)
                total += counts
                samples += 1
        if part_number == 1 or part_number % 10 == 0:
            print(json.dumps({"graph_parts_done": part_number,
                              "graph_samples": samples}), flush=True)
    expected = len(range(0, args.source_frames, args.hbond_stride))
    if samples != expected:
        raise RuntimeError(f"Graph samples {samples}/{expected}")

    mean_edges_upper = total / samples
    mean_edges = mean_edges_upper + mean_edges_upper.T
    np.fill_diagonal(mean_edges, np.diag(mean_edges_upper))
    counts = np.bincount(fixed_layer[fixed_layer >= 0], minlength=args.n_layers)
    weight = np.zeros_like(mean_edges)
    for i in range(args.n_layers):
        for j in range(i + 1, args.n_layers):
            denom = math.sqrt(max(counts[i] * counts[j], 1))
            weight[i, j] = weight[j, i] = mean_edges[i, j] / denom
    degree = np.sum(weight, axis=1)
    invsqrt = np.zeros_like(degree)
    invsqrt[degree > 0] = 1.0 / np.sqrt(degree[degree > 0])
    laplacian = np.eye(args.n_layers) - invsqrt[:, None] * weight * invsqrt[None, :]
    laplacian[degree == 0] = 0.0
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    return mean_edges, weight, counts, eigenvalues, eigenvectors, samples


def cross_spectrum(transformed: np.ndarray, window_norm: float) -> np.ndarray:
    return np.einsum("fla,fma->flm", transformed,
                     np.conjugate(transformed), optimize=True) / window_norm


def coherence_from_spectrum(spectrum: np.ndarray) -> np.ndarray:
    diagonal = np.real(np.diagonal(spectrum, axis1=-2, axis2=-1))
    denominator = np.sqrt(np.maximum(diagonal[..., :, None]
                                     * diagonal[..., None, :], 1.0e-300))
    return spectrum / denominator


def band_name(lo: float, hi: float) -> str:
    return f"{lo:g}-{hi:g}cm-1"


def stream_native(args, paths, cell, waters, fixed_layer,
                  graph_eigenvectors, source_dt_fs):
    oxygen, h1, h2 = waters.T
    nwater = len(waters)
    groups = {layer: np.flatnonzero(fixed_layer == layer)
              for layer in range(args.n_layers)}
    central_members = np.flatnonzero((fixed_layer >= 2) & (fixed_layer <= 9))
    if not len(central_members):
        raise RuntimeError("No waters in central layers 3-10")
    block_frames = int(round(args.block_ps * 1000.0 / source_dt_fs))
    if args.source_frames % block_frames:
        raise RuntimeError("Source frames must divide into spectral blocks")
    nblocks = args.source_frames // block_frames
    window = np.blackman(block_frames)
    window_norm = float(np.sum(window ** 2))
    full_frequency = np.fft.rfftfreq(block_frames, d=source_dt_fs / 1000.0) * CM_PER_PS
    export_mask = full_frequency <= args.max_frequency_cm
    frequency = full_frequency[export_mask]
    band_masks = {(lo, hi): (frequency >= lo) & (frequency < hi)
                  for lo, hi in BANDS}

    layer_signal = np.empty((args.source_frames, args.n_layers, 3), dtype=np.float32)
    global_signal = np.empty((args.source_frames, 3), dtype=np.float32)
    sampled_layer_position = []
    sampled_global_position = []
    previous_o = None
    unwrapped_o = None

    variants = ("raw", "global_COM_corrected")
    cross_accum = {variant: np.zeros((len(frequency), args.n_layers,
                                      args.n_layers), dtype=np.complex128)
                   for variant in variants}
    band_block = {variant: [] for variant in variants}
    graph_spectrum = {variant: np.zeros((args.n_layers, len(frequency)), dtype=float)
                      for variant in variants}

    q_orders = np.asarray([int(value) for value in args.q_orders.split(",")], dtype=int)
    q_actual = 2.0 * np.pi * q_orders / cell[0, 0]
    q_spectrum = np.zeros((len(q_orders), len(frequency)), dtype=float)

    buffers_pos = []
    buffers_vel = []
    buffered = 0
    source_cursor = 0
    block_index = 0

    def process_block(positions: np.ndarray, velocities: np.ndarray) -> None:
        nonlocal block_index, source_cursor, previous_o, unwrapped_o
        vcom = (MASS_O * velocities[:, oxygen]
                + MASS_H * (velocities[:, h1] + velocities[:, h2])) \
            / (MASS_O + 2 * MASS_H)
        layer_v = np.zeros((block_frames, args.n_layers, 3), dtype=np.float64)
        for layer, members in groups.items():
            if len(members):
                layer_v[:, layer] = np.mean(vcom[:, members], axis=1)
        global_v = np.mean(vcom, axis=1)
        layer_signal[source_cursor:source_cursor + block_frames] = layer_v
        global_signal[source_cursor:source_cursor + block_frames] = global_v

        for local_index in range(0, block_frames, args.position_stride):
            wrapped = np.asarray(positions[local_index, oxygen], dtype=float)
            if previous_o is None:
                unwrapped_o = wrapped.copy()
            else:
                unwrapped_o = unwrapped_o + minimum_image(wrapped - previous_o, cell)
            previous_o = wrapped
            layer_pos = np.zeros((args.n_layers, 3), dtype=float)
            for layer, members in groups.items():
                if len(members):
                    layer_pos[layer] = np.mean(unwrapped_o[members], axis=0)
            sampled_layer_position.append(layer_pos)
            sampled_global_position.append(np.mean(unwrapped_o, axis=0))

        signals = {
            "raw": layer_v,
            "global_COM_corrected": layer_v - global_v[:, None, :],
        }
        for variant, signal in signals.items():
            centered = signal - np.mean(signal, axis=0, keepdims=True)
            transformed = np.fft.rfft(centered * window[:, None, None], axis=0)[export_mask]
            spectrum = cross_spectrum(transformed, window_norm)
            cross_accum[variant] += spectrum
            blocks_for_variant = {}
            for band, mask in band_masks.items():
                blocks_for_variant[band] = np.sum(spectrum[mask], axis=0)
            band_block[variant].append(blocks_for_variant)

            projected = np.einsum("tla,lm->tma", signal, graph_eigenvectors,
                                  optimize=True)
            projected -= np.mean(projected, axis=0, keepdims=True)
            mode_fft = np.fft.rfft(projected * window[:, None, None], axis=0)[export_mask]
            graph_spectrum[variant] += np.sum(np.abs(mode_fft) ** 2, axis=2).T / window_norm

        rho = np.empty((block_frames, len(q_actual)), dtype=np.complex128)
        x = positions[:, oxygen[central_members], 0]
        for iq, q in enumerate(q_actual):
            rho[:, iq] = np.sum(np.exp(1.0j * q * x), axis=1)
        rho -= np.mean(rho, axis=0, keepdims=True)
        rho_fft = np.fft.fft(rho * window[:, None], axis=0)[:len(full_frequency)]
        rho_fft = rho_fft[export_mask]
        q_spectrum[:] += (np.abs(rho_fft) ** 2 / (len(central_members) * window_norm)).T

        source_cursor += block_frames
        block_index += 1
        print(json.dumps({"native_blocks_done": block_index,
                          "native_blocks_total": nblocks}), flush=True)

    remaining = args.source_frames
    for path in paths:
        if remaining <= 0:
            break
        with h5py.File(path, "r") as handle:
            available = min(len(handle["step"]), remaining)
            start = 0
            while start < available:
                take = min(block_frames - buffered, available - start)
                buffers_pos.append(np.asarray(
                    handle["positions_A"][start:start + take], dtype=np.float32
                ))
                buffers_vel.append(np.asarray(
                    handle["velocities_A_per_fs"][start:start + take], dtype=np.float32
                ))
                start += take
                buffered += take
                if buffered == block_frames:
                    process_block(np.concatenate(buffers_pos), np.concatenate(buffers_vel))
                    buffers_pos.clear(); buffers_vel.clear(); buffered = 0
            remaining -= available
    if remaining or buffered or block_index != nblocks:
        raise RuntimeError(f"Native stream incomplete: {remaining=} {buffered=} "
                           f"blocks={block_index}/{nblocks}")

    for variant in variants:
        cross_accum[variant] /= nblocks
        graph_spectrum[variant] /= nblocks
    q_spectrum /= nblocks
    return {
        "frequency": frequency,
        "band_masks": band_masks,
        "cross_spectrum": cross_accum,
        "band_block": band_block,
        "graph_spectrum": graph_spectrum,
        "q_orders": q_orders,
        "q_actual": q_actual,
        "q_spectrum": q_spectrum,
        "layer_signal": np.asarray(layer_signal, dtype=float),
        "global_signal": np.asarray(global_signal, dtype=float),
        "sampled_layer_position": np.asarray(sampled_layer_position),
        "sampled_global_position": np.asarray(sampled_global_position),
        "groups": groups,
        "central_members": central_members,
        "nblocks": nblocks,
        "block_frames": block_frames,
    }


def export_complex_coherence(args, native):
    summary_rows = []
    block_rows = []
    for variant, spectrum in native["cross_spectrum"].items():
        for band, mask in native["band_masks"].items():
            lo, hi = band
            band_spectrum = np.sum(spectrum[mask], axis=0)
            coherence = coherence_from_spectrum(band_spectrum)
            for i in range(args.n_layers):
                for j in range(args.n_layers):
                    value = coherence[i, j]
                    summary_rows.append([
                        variant, band_name(lo, hi), lo, hi, i + 1, j + 1,
                        float(np.real(value)), float(np.imag(value)),
                        float(np.abs(value)), float(np.degrees(np.angle(value))),
                    ])
            for block_index, block in enumerate(native["band_block"][variant], start=1):
                block_coherence = coherence_from_spectrum(block[band])
                for i in range(args.n_layers):
                    for j in range(args.n_layers):
                        value = block_coherence[i, j]
                        block_rows.append([
                            variant, block_index, band_name(lo, hi), lo, hi,
                            i + 1, j + 1, float(np.real(value)),
                            float(np.imag(value)), float(np.abs(value)),
                            float(np.degrees(np.angle(value))),
                        ])
    write_csv(
        args.output / "complex_layer_coherence_band_summary.csv",
        ["velocity_variant", "band", "frequency_min_cm-1", "frequency_max_cm-1",
         "layer_i", "layer_j", "coherence_real", "coherence_imag",
         "coherence_magnitude", "coherence_phase_degree"], summary_rows,
    )
    write_csv(
        args.output / "complex_layer_coherence_band_blocks_10ps.csv",
        ["velocity_variant", "block", "band", "frequency_min_cm-1",
         "frequency_max_cm-1", "layer_i", "layer_j", "coherence_real",
         "coherence_imag", "coherence_magnitude", "coherence_phase_degree"],
        block_rows,
    )


def export_pca(args, native):
    raw_position = native["sampled_layer_position"]
    global_position = native["sampled_global_position"]
    position_variants = {
        "raw": raw_position,
        "global_COM_corrected": raw_position - global_position[:, None, :],
    }
    eigen_rows = []
    mode_rows = []
    lag = int(round(1.0 / (args.position_stride * args.source_dt_fs / 1000.0)))
    for variant, position in position_variants.items():
        displacement = position[lag:] - position[:-lag]
        flattened = displacement.reshape(len(displacement), -1)
        flattened -= np.mean(flattened, axis=0, keepdims=True)
        covariance = flattened.T @ flattened / len(flattened)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        eigenvectors = eigenvectors[:, order]
        total = np.sum(eigenvalues)
        cumulative = 0.0
        for mode, value in enumerate(eigenvalues, start=1):
            fraction = value / max(total, 1.0e-300)
            cumulative += fraction
            eigen_rows.append([variant, mode, value, fraction, cumulative])
        for mode in range(min(args.pca_modes, eigenvectors.shape[1])):
            vector = eigenvectors[:, mode].reshape(args.n_layers, 3)
            for layer in range(args.n_layers):
                mode_rows.append([
                    variant, mode + 1, eigenvalues[mode], layer + 1,
                    layer * args.layer_width, (layer + 1) * args.layer_width,
                    vector[layer, 0], vector[layer, 1], vector[layer, 2],
                    np.linalg.norm(vector[layer]),
                ])
    write_csv(
        args.output / "layer_displacement_PCA_eigenvalues.csv",
        ["position_variant", "mode", "eigenvalue_A2",
         "variance_fraction", "cumulative_variance_fraction"], eigen_rows,
    )
    write_csv(
        args.output / "layer_displacement_PCA_modes.csv",
        ["position_variant", "mode", "eigenvalue_A2", "layer",
         "z_min_A", "z_max_A", "eigenvector_x", "eigenvector_y",
         "eigenvector_z", "layer_vector_amplitude"], mode_rows,
    )


def export_correlation_length(args, native):
    length_rows = []
    separation_rows = []
    centers = np.arange(args.n_layers) * args.layer_width + 0.5 * args.layer_width
    for variant, spectrum in native["cross_spectrum"].items():
        coherence = coherence_from_spectrum(spectrum)
        for fi, frequency in enumerate(native["frequency"]):
            distances = []
            values = []
            for separation in range(1, args.n_layers):
                pair_values = [abs(coherence[fi, i, i + separation])
                               for i in range(args.n_layers - separation)]
                mean_value = float(np.mean(pair_values))
                distance = float(centers[separation] - centers[0])
                distances.append(distance)
                values.append(mean_value)
                separation_rows.append([
                    variant, frequency, separation, distance,
                    len(pair_values), mean_value,
                ])
            x = np.asarray(distances)
            y = np.asarray(values)
            valid = np.isfinite(y) & (y > 1.0e-8)
            if np.sum(valid) >= 4:
                slope, intercept = np.polyfit(x[valid], np.log(y[valid]), 1)
                prediction = intercept + slope * x[valid]
                residual = np.sum((np.log(y[valid]) - prediction) ** 2)
                total = np.sum((np.log(y[valid]) - np.mean(np.log(y[valid]))) ** 2)
                r2 = 1.0 - residual / max(total, 1.0e-300)
                xi = -1.0 / slope if slope < 0 else math.nan
            else:
                slope = intercept = r2 = xi = math.nan
            length_rows.append([
                variant, frequency, xi, slope, intercept, r2, int(np.sum(valid)),
                "fit_to_mean_magnitude_coherence_vs_layer_separation",
            ])
    write_csv(
        args.output / "frequency_dependent_spatial_correlation_length.csv",
        ["velocity_variant", "frequency_cm-1", "xi_A", "log_slope_A-1",
         "log_intercept", "fit_R2", "separations_used", "definition"], length_rows,
    )
    write_csv(
        args.output / "frequency_dependent_coherence_by_separation.csv",
        ["velocity_variant", "frequency_cm-1", "layer_separation",
         "distance_A", "layer_pairs", "mean_coherence_magnitude"], separation_rows,
    )


def export_stft(args, native):
    window_frames = int(round(args.stft_window_ps * 1000.0 / args.source_dt_fs))
    step_frames = int(round(args.stft_step_ps * 1000.0 / args.source_dt_fs))
    if window_frames > args.source_frames:
        raise RuntimeError("STFT window exceeds source")
    window = np.blackman(window_frames)
    frequency = np.fft.rfftfreq(window_frames,
                                d=args.source_dt_fs / 1000.0) * CM_PER_PS
    # Use the central two-thirds of the film.  For the 12-layer CaF2/oil
    # systems this is exactly layers 3-10; the rule also extends cleanly to
    # films with a different physical thickness.
    margin = max(1, int(round(args.n_layers / 6.0)))
    central_layers = np.arange(margin, args.n_layers - margin)
    raw = native["layer_signal"]
    corrected = raw - native["global_signal"][:, None, :]
    rows = []
    for variant, signal in (("raw", raw), ("global_COM_corrected", corrected)):
        for start in range(0, args.source_frames - window_frames + 1, step_frames):
            # Copy because the raw branch is a view into the stored native signal;
            # in-place mean removal must not alter later windows or the corrected branch.
            part = np.array(
                signal[start:start + window_frames, central_layers],
                dtype=np.float64,
                copy=True,
            )
            part -= np.mean(part, axis=0, keepdims=True)
            transformed = np.fft.rfft(part * window[:, None, None], axis=0)
            for lo, hi in BANDS:
                mask = (frequency >= lo) & (frequency < hi)
                selected = transformed[mask]
                power = np.sum(np.abs(selected) ** 2, axis=2)
                pair_numerator = 0.0
                pair_denominator = 0.0
                for i in range(len(central_layers)):
                    for j in range(i + 1, len(central_layers)):
                        cross = np.real(np.sum(
                            selected[:, i] * np.conjugate(selected[:, j]), axis=1
                        ))
                        denom = np.sqrt(power[:, i] * power[:, j])
                        pair_numerator += float(np.sum(cross))
                        pair_denominator += float(np.sum(denom))
                signed_alignment = pair_numerator / max(pair_denominator, 1.0e-300)
                self_power = float(np.sum(power))
                coherent_power = float(np.sum(np.abs(np.sum(selected, axis=1)) ** 2))
                rows.append([
                    variant, start * args.source_dt_fs / 1000.0,
                    (start + window_frames) * args.source_dt_fs / 1000.0,
                    (start + 0.5 * window_frames) * args.source_dt_fs / 1000.0,
                    band_name(lo, hi), lo, hi, signed_alignment,
                    coherent_power / max(self_power, 1.0e-300),
                    (coherent_power - self_power) / max(self_power, 1.0e-300),
                ])
    write_csv(
        args.output / "STFT_phase_locking_time_series.csv",
        ["velocity_variant", "time_min_ps", "time_max_ps", "time_center_ps",
         "band", "frequency_min_cm-1", "frequency_max_cm-1",
         "signed_pair_phase_alignment", "coherent_over_self",
         "signed_cross_over_self"], rows,
    )
    summary_rows = []
    for variant in ("raw", "global_COM_corrected"):
        for lo, hi in BANDS:
            selected = [row for row in rows
                        if row[0] == variant and row[4] == band_name(lo, hi)]
            for column, name in ((7, "signed_pair_phase_alignment"),
                                 (8, "coherent_over_self"),
                                 (9, "signed_cross_over_self")):
                values = np.asarray([row[column] for row in selected])
                summary_rows.append([
                    variant, band_name(lo, hi), name, len(values),
                    np.mean(values), np.std(values, ddof=1),
                    np.std(values, ddof=1) / np.sqrt(len(values)),
                    np.quantile(values, 0.05), np.quantile(values, 0.95),
                ])
    write_csv(
        args.output / "STFT_phase_locking_summary.csv",
        ["velocity_variant", "band", "metric", "windows", "mean", "std",
         "naive_window_SEM", "q05", "q95"], summary_rows,
    )


def export_common_q(args, native):
    rows = []
    peaks = []
    frequency = native["frequency"]
    for iq, (order, q) in enumerate(zip(native["q_orders"], native["q_actual"])):
        spectrum = native["q_spectrum"][iq]
        for freq, value in zip(frequency, spectrum):
            rows.append([iq + 1, int(order), q, freq, value])
        band = (frequency >= 5.0) & (frequency <= 300.0)
        peak_index = np.flatnonzero(band)[np.argmax(spectrum[band])]
        peaks.append([iq + 1, int(order), q, frequency[peak_index],
                      spectrum[peak_index]])
    write_csv(
        args.output / "common_q_dynamic_structure_factor.csv",
        ["common_q_index", "reciprocal_order_x", "actual_q_A-1",
         "frequency_cm-1", "S_q_omega"], rows,
    )
    write_csv(
        args.output / "common_q_peak_summary.csv",
        ["common_q_index", "reciprocal_order_x", "actual_q_A-1",
         "peak_frequency_5_300cm-1", "peak_S_q_omega"], peaks,
    )


def export_graph_modes(args, mean_edges, graph_weight, graph_counts,
                       graph_eigenvalues, graph_eigenvectors, native):
    weight_rows = []
    for i in range(args.n_layers):
        for j in range(args.n_layers):
            weight_rows.append([
                i + 1, j + 1, int(graph_counts[i]), int(graph_counts[j]),
                mean_edges[i, j], graph_weight[i, j],
            ])
    write_csv(
        args.output / "hbond_layer_graph_weights.csv",
        ["layer_i", "layer_j", "waters_i", "waters_j",
         "mean_HB_edges_per_frame", "normalized_graph_weight"], weight_rows,
    )
    mode_rows = []
    for mode in range(args.n_layers):
        for layer in range(args.n_layers):
            mode_rows.append([
                mode + 1, graph_eigenvalues[mode], layer + 1,
                layer * args.layer_width, (layer + 1) * args.layer_width,
                graph_eigenvectors[layer, mode],
            ])
    write_csv(
        args.output / "hbond_layer_graph_laplacian_modes.csv",
        ["graph_mode", "normalized_laplacian_eigenvalue", "layer",
         "z_min_A", "z_max_A", "eigenvector_amplitude"], mode_rows,
    )
    spectrum_rows = []
    peak_rows = []
    frequency = native["frequency"]
    for variant, spectrum in native["graph_spectrum"].items():
        for mode in range(args.n_layers):
            for fi, freq in enumerate(frequency):
                spectrum_rows.append([
                    variant, mode + 1, graph_eigenvalues[mode], freq,
                    spectrum[mode, fi],
                ])
            band = (frequency >= 5.0) & (frequency <= 300.0)
            peak_index = np.flatnonzero(band)[np.argmax(spectrum[mode, band])]
            peak_rows.append([
                variant, mode + 1, graph_eigenvalues[mode],
                frequency[peak_index], spectrum[mode, peak_index],
            ])
    write_csv(
        args.output / "hbond_graph_mode_velocity_spectra.csv",
        ["velocity_variant", "graph_mode", "normalized_laplacian_eigenvalue",
         "frequency_cm-1", "projected_velocity_power"], spectrum_rows,
    )
    write_csv(
        args.output / "hbond_graph_mode_peak_summary.csv",
        ["velocity_variant", "graph_mode", "normalized_laplacian_eigenvalue",
         "peak_frequency_5_300cm-1", "peak_power"], peak_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--segment-glob", default="*segment_*.h5")
    parser.add_argument("--candidate-screen", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--system", required=True)
    parser.add_argument("--source-frames", default=500000, type=int)
    parser.add_argument("--layer-width", default=2.0, type=float)
    parser.add_argument("--n-layers", default=12, type=int)
    parser.add_argument("--block-ps", default=10.0, type=float)
    parser.add_argument("--position-stride", default=100, type=int)
    parser.add_argument("--hbond-stride", default=100, type=int)
    parser.add_argument("--hbond-ho", default=2.45, type=float)
    parser.add_argument("--hbond-oo", default=3.50, type=float)
    parser.add_argument("--hbond-angle", default=150.0, type=float)
    parser.add_argument("--max-frequency-cm", default=300.0, type=float)
    parser.add_argument("--q-orders", required=True,
                        help="Comma-separated reciprocal x orders")
    parser.add_argument("--pca-modes", default=6, type=int)
    parser.add_argument("--stft-window-ps", default=10.0, type=float)
    parser.add_argument("--stft-step-ps", default=1.0, type=float)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    paths = sorted(args.input.glob(args.segment_glob))
    if not paths:
        raise RuntimeError(f"No input HDF5 in {args.input}")
    with h5py.File(paths[0], "r") as handle:
        symbols = np.asarray(handle["symbols"]).astype(str)
        cell = np.asarray(handle["cell_A"], dtype=float)
        positions0 = np.asarray(handle["positions_A"][0], dtype=float)
        args.source_dt_fs = float(handle.attrs["output_interval_fs"])
    waters = water_topology(symbols, positions0, cell)
    oxygen = waters[:, 0]
    candidate_mean_z = load_candidate_mean_z(args.candidate_screen)
    mean_z = np.asarray([candidate_mean_z[int(value) + 1] for value in oxygen])
    fixed_layer = np.floor(mean_z / args.layer_width).astype(int)
    fixed_layer[(fixed_layer < 0) | (fixed_layer >= args.n_layers)] = -1

    graph = graph_prepass(args, paths, cell, waters, fixed_layer)
    mean_edges, graph_weight, graph_counts, graph_values, graph_vectors, graph_samples = graph
    native = stream_native(args, paths, cell, waters, fixed_layer,
                           graph_vectors, args.source_dt_fs)
    export_complex_coherence(args, native)
    export_pca(args, native)
    export_correlation_length(args, native)
    export_stft(args, native)
    export_common_q(args, native)
    export_graph_modes(args, mean_edges, graph_weight, graph_counts,
                       graph_values, graph_vectors, native)

    metadata = {
        "status": "complete", "system": args.system,
        "source_frames": args.source_frames,
        "source_sampling_interval_fs": args.source_dt_fs,
        "water_count": len(waters), "fixed_layer_water_counts": graph_counts.tolist(),
        "spectral_blocks": native["nblocks"], "spectral_block_ps": args.block_ps,
        "frequency_resolution_cm-1": float(native["frequency"][1]
                                                - native["frequency"][0]),
        "bands_cm-1": BANDS,
        "velocity_variants": {
            "raw": "molecular COM velocities with collective water translation retained",
            "global_COM_corrected": "instantaneous all-water mean COM velocity subtracted",
        },
        "six_analyses": [
            "band-resolved complex layer coherence",
            "layer-displacement PCA modes",
            "frequency-dependent spatial coherence length",
            "sliding-STFT phase locking",
            "commensurate common-q S(q,omega)",
            "H-bond layer-graph Laplacian mode velocity projection",
        ],
        "common_q_actual_A-1": native["q_actual"].tolist(),
        "common_q_reciprocal_orders": native["q_orders"].tolist(),
        "hbond_graph_samples": graph_samples,
        "method_limits": [
            "Current earlier signed cross/self spectra already retained collective COM velocity; both raw and corrected variants are output here.",
            "Complex coherence is based on static mean-z layer signals and Welch-averaged cross spectra.",
            "Correlation length fits magnitude coherence in a finite 24 A film and may be undefined when no monotonic decay exists.",
            "STFT windows overlap; naive window SEM is descriptive and not an independent-sample uncertainty.",
            "H-bond graph modes use a time-averaged 12-layer graph, not instantaneous molecular graph eigenmodes or Hessian phonons.",
        ],
        "elapsed_s": time.time() - started,
    }
    (args.output / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    (args.output / "COMPLETE").write_text("complete\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
