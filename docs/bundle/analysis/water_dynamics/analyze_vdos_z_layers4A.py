#!/usr/bin/env python3
"""Layer-resolved water VDOS from the 1 fs HDF5 velocity trajectory."""
from __future__ import annotations

import argparse, csv, json
from pathlib import Path
import h5py
import numpy as np

CM_PER_PS = 33.3564095198152


def water_topology(symbols, positions, cell):
    oxygen, hydrogen = np.flatnonzero(symbols == "O"), np.flatnonzero(symbols == "H")
    lengths, pairs = np.diag(cell), []
    for o in oxygen:
        d = positions[hydrogen] - positions[o]
        d -= lengths * np.rint(d / lengths)
        for h, r in zip(hydrogen, np.linalg.norm(d, axis=1)):
            if r <= 1.30: pairs.append((float(r), int(o), int(h)))
    pairs.sort(); assigned = {int(o): [] for o in oxygen}; used = set()
    for _, o, h in pairs:
        if len(assigned[o]) < 2 and h not in used:
            assigned[o].append(h); used.add(h)
    # Interface systems include substrate O atoms.  Only O atoms with two uniquely
    # assigned covalent H atoms are complete H2O and enter the water VDOS.
    waters = [(o, *assigned[o]) for o in oxygen if len(assigned[o]) == 2]
    if not waters: raise RuntimeError("could not build any complete water topology")
    return waters


def psd_blocks(v, dt_fs):
    """One-sided non-negative PSD, averaged over atoms/components for one 10 ps block."""
    v = v - v.mean(axis=0, keepdims=True)
    w = np.blackman(len(v)); nfft = 1 << (len(v) * 4 - 1).bit_length()
    xf = np.fft.rfft(v * w[:, None, None], n=nfft, axis=0)
    return np.fft.rfftfreq(nfft, d=dt_fs * 1e-3) * CM_PER_PS, (np.abs(xf) ** 2).mean(axis=(1, 2)) / np.sum(w ** 2)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--input', required=True, type=Path); ap.add_argument('--output', required=True, type=Path); ap.add_argument('--layer-width', type=float, default=4.0); ap.add_argument('--block-ps', type=float, default=10.0)
    a = ap.parse_args(); segs = sorted((a.input / 'segments').glob('*.h5'))
    if not segs or len(segs) % 2: raise RuntimeError('expected an even, nonzero number of complete 5 ps segments')
    a.output.mkdir(parents=True, exist_ok=True)
    with h5py.File(segs[0], 'r') as f:
        symbols=np.asarray(f['symbols']).astype(str); pos0=np.asarray(f['positions_A'][0]); cell=np.asarray(f['cell_A']); dt_fs=float(f.attrs['output_interval_fs'])
    waters=water_topology(symbols,pos0,cell); oidx=np.array([w[0] for w in waters]); zsum=np.zeros(len(waters)); nframes=0
    for p in segs:
        with h5py.File(p,'r') as f:
            z=f['positions_A'][:,oidx,2]; zsum += z.sum(axis=0); nframes += len(z)
    mean_z=zsum/nframes; edges=np.arange(0.0, np.ceil((mean_z.max()+1e-9)/a.layer_width)*a.layer_width+a.layer_width*0.5, a.layer_width)
    layer_ids=np.digitize(mean_z, edges, right=False)-1
    layers=[]
    for i in range(len(edges)-1):
        wi=np.flatnonzero(layer_ids==i)
        if len(wi): layers.append((i,edges[i],edges[i+1],wi,np.sort(np.array([x for j in wi for x in waters[j]],dtype=int))))
    accum=[None]*len(layers); nblocks=0
    frames_per_block=int(round(a.block_ps*1000/dt_fs))
    for p1,p2 in zip(segs[::2],segs[1::2]):
        with h5py.File(p1,'r') as f1, h5py.File(p2,'r') as f2:
            for k,(_,_,_,_,atoms) in enumerate(layers):
                v=np.concatenate([f1['velocities_A_per_fs'][:,atoms,:],f2['velocities_A_per_fs'][:,atoms,:]],axis=0)
                if len(v)!=frames_per_block: raise RuntimeError('unexpected velocity block length')
                freq,psd=psd_blocks(v,dt_fs); accum[k]=psd if accum[k] is None else accum[k]+psd
        nblocks += 1
    mask=(freq>=1200)&(freq<=4000); labels=[]
    with (a.output/'layer_summary.csv').open('w',newline='') as fh:
        w=csv.writer(fh); w.writerow(['layer_index','z_min_A','z_max_A','z_center_A','water_count','atom_count','O_serials_1based'])
        for k,(i,z0,z1,wi,atoms) in enumerate(layers):
            label=f'{z0:.0f}_{z1:.0f}A'; labels.append(label); psd=accum[k]/nblocks; norm=psd/psd[mask].max()
            w.writerow([i,z0,z1,(z0+z1)/2,len(wi),len(atoms),';'.join(map(str,(oidx[wi]+1).tolist()))])
            with (a.output/f'vdos_layer_{label}.csv').open('w',newline='') as g:
                q=csv.writer(g); q.writerow(['wavenumber_cm-1','VDOS_normalized']); q.writerows(zip(freq[mask],norm[mask]))
            accum[k]=norm
    with (a.output/'vdos_layers_4A_source_data.csv').open('w',newline='') as fh:
        w=csv.writer(fh); w.writerow(['wavenumber_cm-1']+labels)
        for row in zip(freq[mask],*[x[mask] for x in accum]): w.writerow(row)
    with (a.output/'water_mean_z.csv').open('w',newline='') as fh:
        w=csv.writer(fh); w.writerow(['water_index','O_serial_1based','mean_O_z_A','layer_4A'])
        for j,z in enumerate(mean_z): w.writerow([j+1,oidx[j]+1,z,f'{edges[layer_ids[j]]:.0f}_{edges[layer_ids[j]+1]:.0f}A'])
    meta={'frames':nframes,'sampling_interval_fs':dt_fs,'water_count':len(waters),'layer_width_A':a.layer_width,'selection_basis':'mean O Cartesian z over all analyzed trajectory frames; fixed layer membership','spectral_blocks':nblocks,'spectral_block_ps':a.block_ps,'method':'one-sided velocity PSD averaged over atoms, Cartesian components, and Blackman-tapered blocks','range_cm-1':[1200,4000]}
    (a.output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n'); print(json.dumps(meta,indent=2))

if __name__=='__main__': main()
