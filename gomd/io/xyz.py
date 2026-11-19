"""
io/xyz.py — extended XYZ for MLIP systems (no lone pairs, no connectivity).

Frame layout (ASE-readable; one frame per file here):
    N
    Lattice="Lx 0 0 0 Ly 0 0 0 Lz" Properties=species:S:1:pos:R:3:vel:R:3:mass:R:1:site:I:1 pbc="T T T" charge=0 spin=1 ...
    El  x y z  vx vy vz  mass  site
Units: positions A; velocities m/s (as in coords.inp, NOT ASE's units); mass g/mol.
`site` is the atom's index in the source system with lone-pair sites (for tracing
an atom back); `charge` / `spin` are what UMA's omol task needs. Further key=value
pairs in the comment line (e.g. the Nose-Hoover chain state of a restart, time_ps)
are kept in `info`.
"""
from __future__ import annotations
import os
import re
import shlex
import numpy as np

from gomd.mlip.elements import SYMBOLS

NUMBER = {s: z for z, s in SYMBOLS.items()}


def _fmt_info(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        return '"' + ' '.join(repr(float(t)) for t in np.ravel(v)) + '"'
    if isinstance(v, bool):
        return 'T' if v else 'F'
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    s = str(v)
    return f'"{s}"' if (' ' in s or '=' in s) else s


def write_extxyz(path, symbols, x, box, v=None, mass=None, site=None, info=None, pbc=True):
    """One frame. box (3,) orthorhombic. info: extra comment-line key=value pairs.
    pbc=False writes pbc="F F F" (a cluster in vacuum; box is then only a frame)."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    box = np.asarray(box, dtype=np.float64).reshape(3)
    props = ['species:S:1', 'pos:R:3']
    cols = [np.asarray(symbols, dtype=object)[:, None], x]
    if v is not None:
        props.append('vel:R:3'); cols.append(np.asarray(v, dtype=np.float64))
    if mass is not None:
        props.append('mass:R:1'); cols.append(np.asarray(mass, dtype=np.float64)[:, None])
    if site is not None:
        props.append('site:I:1'); cols.append(np.asarray(site, dtype=np.int64)[:, None])
    L = [repr(float(t)) for t in box]                       # plain floats (numpy 2 repr adds np.float64)
    head = {'Lattice': f'"{L[0]} 0.0 0.0 0.0 {L[1]} 0.0 0.0 0.0 {L[2]}"',
            'Properties': ':'.join(props), 'pbc': '"T T T"' if pbc else '"F F F"'}
    for k, val in (info or {}).items():
        head[k] = _fmt_info(val)
    lines = [str(n), ' '.join(f'{k}={v_}' for k, v_ in head.items())]
    for i in range(n):
        parts = [f'{symbols[i]:<2s}']
        parts += [f'{t:20.12f}' for t in x[i]]
        if v is not None:
            parts += [f'{t:20.9f}' for t in np.asarray(v)[i]]
        if mass is not None:
            parts.append(f'{float(np.asarray(mass)[i]):12.6f}')
        if site is not None:
            parts.append(f'{int(np.asarray(site)[i]):7d}')
        lines.append(' '.join(parts))
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    os.replace(tmp, path)


def read_extxyz(path):
    """-> dict: symbols, numbers, x (n,3), box (3,), pbc (bool), v (n,3) or None,
    mass or None, site or None, info (dict of the other comment-line pairs, as str)."""
    with open(path) as f:
        n = int(f.readline())
        comment = f.readline().strip()
        rows = [f.readline().split() for _ in range(n)]
    kv = {}
    for tok in shlex.split(comment):
        if '=' in tok:
            k, val = tok.split('=', 1)
            kv[k] = val
    lat = np.array([float(t) for t in kv.pop('Lattice').split()]).reshape(3, 3)
    if np.any(np.abs(lat - np.diag(np.diag(lat))) > 1e-12):
        raise ValueError(f'{path}: only orthorhombic cells are supported')
    props = kv.pop('Properties').split(':')
    pbc_flags = kv.pop('pbc', 'T T T').split()
    pbc = [t.upper() in ('T', 'TRUE', '1') for t in pbc_flags]
    if len(set(pbc)) != 1:
        raise ValueError(f'{path}: mixed pbc {pbc_flags} is not supported (all T or all F)')
    out = {'box': np.diag(lat).copy(), 'v': None, 'mass': None, 'site': None, 'info': kv,
           'pbc': pbc[0]}
    col = 0
    for name, typ, cnt in zip(props[0::3], props[1::3], props[2::3]):
        cnt = int(cnt)
        vals = [r[col:col + cnt] for r in rows]
        col += cnt
        if typ == 'S':
            arr = np.array([v_[0] for v_ in vals], dtype=object)
        elif typ == 'I':
            arr = np.array(vals, dtype=np.int64).reshape(n, cnt)
        else:
            arr = np.array(vals, dtype=np.float64).reshape(n, cnt)
        if cnt == 1 and typ != 'S':
            arr = arr[:, 0]
        key = {'species': 'symbols', 'pos': 'x', 'vel': 'v', 'mass': 'mass', 'site': 'site'}.get(name, name)
        out[key] = arr
    out['symbols'] = [str(s) for s in out['symbols']]
    out['numbers'] = np.array([NUMBER[s] for s in out['symbols']], dtype=np.int64)
    return out


def info_float(info, key, default=None):
    if key not in info:
        return default
    return float(info[key])


def info_array(info, key):
    if key not in info:
        return None
    return np.array([float(t) for t in re.split(r'[\s,]+', info[key].strip()) if t])
