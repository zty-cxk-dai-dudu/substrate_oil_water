#!/usr/bin/env python3
"""Export complete, regularly sampled frames from XDATCAR or LAMMPS dumps."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from ase import Atoms
from ase.io import read


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def scan_xdatcar(path):
    with Path(path).open('rb') as stream:
        title = stream.readline().decode().strip()
        scale = float(stream.readline())
        cell = np.array([[float(x) for x in stream.readline().split()] for _ in range(3)])
        length_scale = (-scale / abs(np.linalg.det(cell))) ** (1 / 3) if scale < 0 else scale
        cell *= length_scale
        species = stream.readline().decode().split()
        counts = [int(x) for x in stream.readline().split()]
        if len(species) != len(counts) or any(not re.fullmatch(r'[A-Z][a-z]?', s) for s in species):
            raise ValueError('XDATCAR must explicitly list element symbols and counts.')
        symbols = [symbol for symbol, count in zip(species, counts) for _ in range(count)]
        n = len(symbols)
        frames = []
        while True:
            marker = stream.readline()
            if not marker:
                break
            if not marker.strip():
                continue
            match = re.fullmatch(rb'\s*(Direct|Cartesian)\s+configuration\s*=\s*(\d+)\s*', marker)
            if not match:
                raise ValueError('Unexpected XDATCAR frame marker: ' + repr(marker[:120]))
            offset = stream.tell()
            for _ in range(n):
                if not stream.readline():
                    raise ValueError('Incomplete XDATCAR frame.')
            frames.append({'offset': offset, 'configuration': int(match[2]),
                           'direct': match[1] == b'Direct'})
    return {'title': title, 'cell': cell, 'symbols': symbols, 'frames': frames, 'length_scale': length_scale}


def scan_lammps(path):
    frames = []
    expected_ids = None
    with Path(path).open('rb') as stream:
        while True:
            marker = stream.readline()
            if not marker:
                break
            if marker.strip() != b'ITEM: TIMESTEP':
                raise ValueError('Unexpected LAMMPS frame marker.')
            step = int(stream.readline())
            assert stream.readline().strip() == b'ITEM: NUMBER OF ATOMS'
            n = int(stream.readline())
            box_header = stream.readline().decode().split()
            if box_header[1:3] != ['BOX', 'BOUNDS']:
                raise ValueError('Missing LAMMPS box bounds.')
            if any(x in box_header for x in ['xy', 'xz', 'yz']):
                raise ValueError('This reader expects an orthorhombic LAMMPS dump.')
            if box_header[3:] != ['pp', 'pp', 'pp']:
                raise ValueError('This reader expects periodic LAMMPS boundaries in all three directions.')
            bounds = np.array([[float(x) for x in stream.readline().split()] for _ in range(3)])
            assert bounds.shape == (3, 2)
            columns = stream.readline().decode().split()[2:]
            if not all(x in columns for x in ['id', 'element', 'x', 'y', 'z']):
                raise ValueError('Dump must include id, element and Cartesian x/y/z columns.')
            offset = stream.tell()
            for _ in range(n):
                if not stream.readline():
                    raise ValueError('Incomplete LAMMPS frame.')
            if expected_ids is not None and n != expected_ids:
                raise ValueError('Atom count changes between frames.')
            expected_ids = n
            frames.append({'offset': offset, 'configuration': step, 'atoms': n,
                           'cell': np.diag(bounds[:, 1] - bounds[:, 0]),
                           'origin': bounds[:, 0], 'columns': columns})
    return {'frames': frames}


def export(source, output, *, source_format, start_frame=None, end_frame=None,
           stride=100, last_frames=20001, timestep_fs=None):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError('Source and output must be different files.')
    before = source.stat()
    scanned = scan_xdatcar(source) if source_format == 'xdatcar' else scan_lammps(source)
    records = scanned['frames']
    count = len(records)
    end = count if end_frame is None else end_frame
    start = max(1, end - last_frames + 1) if start_frame is None else start_frame
    if not 1 <= start <= end <= count or stride < 1:
        raise ValueError('Requested source frame range or stride is invalid.')
    selected = list(range(start - 1, end, stride))
    identifiers = [records[index]['configuration'] for index in selected]
    differences = np.diff(identifiers)
    if len(differences) and (np.any(differences <= 0) or len(np.unique(differences)) != 1):
        raise ValueError('Selected configurations must be increasing with a regular interval.')
    output.parent.mkdir(parents=True, exist_ok=True)
    mapping = []
    expected_symbols = None
    expected_atom_ids = None
    frames_for_validation = []
    with source.open('rb') as stream, output.open('w') as target:
        for ordinal, index in enumerate(selected):
            record = records[index]
            stream.seek(record['offset'])
            if source_format == 'xdatcar':
                symbols = scanned['symbols']
                rows = [stream.readline().split()[:3] for _ in symbols]
                raw = np.array(rows, dtype=float)
                cell = scanned['cell']
                positions = raw @ cell if record['direct'] else raw * scanned['length_scale']
                atom_ids = np.arange(1, len(symbols) + 1)
            else:
                columns = record['columns']
                data = [stream.readline().decode().split() for _ in range(record['atoms'])]
                data.sort(key=lambda row: int(row[columns.index('id')]))
                atom_ids = np.array([int(row[columns.index('id')]) for row in data])
                symbols = [row[columns.index('element')] for row in data]
                positions = np.array([[float(row[columns.index(axis)]) for axis in ['x', 'y', 'z']] for row in data])
                positions -= record['origin']
                cell = record['cell']
            if len(set(atom_ids.tolist())) != len(symbols):
                raise ValueError('Duplicate atom identifiers.')
            if not np.isfinite(positions).all() or not np.isfinite(cell).all():
                raise ValueError('Non-finite coordinates or cell.')
            if expected_symbols is None:
                expected_symbols = symbols
                expected_atom_ids = atom_ids.copy()
            elif symbols != expected_symbols:
                raise ValueError('Element order changes between selected frames.')
            elif not np.array_equal(atom_ids, expected_atom_ids):
                raise ValueError('Atom identifiers change between selected frames.')
            lattice = ' '.join(f'{value:.12f}' for value in cell.ravel())
            comment = (f'Lattice="{lattice}" Properties=species:S:1:pos:R:3:atom_id:I:1 '
                       f'pbc="T T T" source_frame={index + 1} '
                       f'source_configuration={record["configuration"]}')
            entry = {'output_frame': ordinal + 1, 'source_frame': index + 1,
                     'source_configuration': record['configuration']}
            if timestep_fs is not None:
                time_ps = record['configuration'] * timestep_fs / 1000
                comment += f' time_ps={time_ps:.9f}'
                entry['time_ps'] = time_ps
            target.write(f'{len(symbols)}\n{comment}\n')
            for symbol, position, atom_id in zip(symbols, positions, atom_ids):
                target.write(f'{symbol} {position[0]:.10f} {position[1]:.10f} {position[2]:.10f} {atom_id}\n')
            mapping.append(entry)
            frames_for_validation.append((symbols, positions.copy(), cell.copy(), atom_ids.copy()))
    parsed = read(output, index=':', format='extxyz')
    if len(parsed) != len(selected):
        raise ValueError('Exported frame count differs after parsing.')
    max_error = 0.0
    for frame, (symbols, positions, cell, ids) in zip(parsed, frames_for_validation):
        assert frame.get_chemical_symbols() == symbols
        assert np.array_equal(frame.arrays['atom_id'], ids)
        assert np.allclose(frame.cell.array, cell, rtol=0, atol=1e-10)
        error = float(np.abs(frame.positions - positions).max())
        max_error = max(error, max_error)
        assert error <= 6e-11
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Source changed during export.')
    report = {'source': str(source), 'source_format': source_format,
              'source_bytes': before.st_size, 'source_sha256': file_sha256(source),
              'source_frames': count, 'source_start_frame': start, 'source_end_frame': end,
              'stride_source_frames': stride, 'output_frames': len(selected),
              'atoms': len(expected_symbols), 'formula': Atoms(expected_symbols).get_chemical_formula(),
              'file': str(output), 'bytes': output.stat().st_size, 'sha256': file_sha256(output),
              'maximum_roundtrip_coordinate_error_A': max_error, 'frame_index': mapping}
    if timestep_fs is not None:
        report.update(timestep_fs=timestep_fs, time_start_ps=mapping[0]['time_ps'],
                      time_end_ps=mapping[-1]['time_ps'])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--format', dest='source_format', choices=['xdatcar', 'lammps'], required=True)
    parser.add_argument('--start-frame', type=int)
    parser.add_argument('--end-frame', type=int)
    parser.add_argument('--stride', type=int, default=100)
    parser.add_argument('--last-frames', type=int, default=20001)
    parser.add_argument('--timestep-fs', type=float)
    args = vars(parser.parse_args())
    report = export(**args)
    Path(str(args['output']) + '.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'frame_index'}, indent=2))


if __name__ == '__main__':
    main()
