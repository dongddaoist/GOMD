#!/usr/bin/env python
"""
GOMD MLIP server: evaluates an ML potential for the GOMD driver.

Runs as its own process, started by gomd/mlip/client.py, so the MLIP stack
(PyTorch, fairchem) never shares a Python process with JAX: separate CUDA
contexts, separate GPU memory pools, and it can live in a separate virtual
environment (fairchem pins its own torch and numpy). This file imports
nothing from gomd and needs only numpy (plus fairchem + ase for UMA).

Protocol over stdin/stdout, every message = header + blob:
    uint64 little-endian length, UTF-8 JSON header,
    uint64 little-endian length, raw bytes
  client -> server
    {"cmd": "init", "numbers": [...], "cell": [a, b, c], "pbc": true,
     "charge": 0, "spin": 1}                    blob: positions float64 (n, 3)
    {"cmd": "eval"}                             blob: positions float64 (n, 3)
    {"cmd": "quit"}
  server -> client
    {"ok": true, "info": {...}}                 reply to init
    {"ok": true, "energy": E}                   blob: forces float64 (n, 3)
    {"ok": false, "error": "...", "traceback": "..."}
Units are the model's: Angstrom, eV, eV/Angstrom.

Backends
  uma   fairchem UMA through FAIRChemCalculator (needs fairchem-core, and a
        Hugging Face token with access to facebook/UMA)
  mock  a harmonic elastic network over the bonded pairs of the first
        configuration (numpy only) — a conservative stand-in for testing the
        GOMD <-> MLIP path without fairchem
"""
from __future__ import annotations
import argparse
import json
import os
import struct
import sys
import time
import traceback

import numpy as np


# ---------------------------------------------------------------- protocol
def _read_exact(f, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            raise EOFError
        buf += chunk
    return bytes(buf)


def read_msg(f):
    (lh,) = struct.unpack('<Q', _read_exact(f, 8))
    header = json.loads(_read_exact(f, lh).decode())
    (lb,) = struct.unpack('<Q', _read_exact(f, 8))
    blob = _read_exact(f, lb) if lb else b''
    return header, blob


def write_msg(f, header, blob=b''):
    h = json.dumps(header).encode()
    f.write(struct.pack('<Q', len(h)) + h + struct.pack('<Q', len(blob)))
    if blob:
        f.write(blob)
    f.flush()


# ---------------------------------------------------------------- backends
class UMABackend:
    def __init__(self, args, numbers, cell, pbc, charge, spin, pos):
        import torch
        from ase import Atoms
        from fairchem.core import pretrained_mlip, FAIRChemCalculator
        import fairchem.core as fc
        device = args.device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        t0 = time.time()
        settings = args.settings
        if settings in LOWMEM_SETTINGS:
            # fairchem's presets turn activation checkpointing off ("default" = nitro,
            # "high-precision"); for ~10^4 atoms it is what fits the GPU. MOLE merging
            # is valid here (fixed composition, task, charge, spin over the run).
            # float64 avoids the fp32 resolution of an absolute energy of ~1e8 kcal/mol.
            from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
            kw = dict(tf32=False, activation_checkpointing=True, merge_mole=True, compile=False)
            if settings == 'lowmem-fp64':
                kw['base_precision_dtype'] = torch.float64
            settings = InferenceSettings(**kw)
        predictor = pretrained_mlip.get_predict_unit(
            args.model, inference_settings=settings, device=device)
        self.calc = FAIRChemCalculator(predictor, task_name=args.task)
        self.atoms = Atoms(numbers=numbers, positions=pos,
                           cell=np.diag(cell) if np.ndim(cell) == 1 else cell, pbc=pbc)
        self.atoms.info['charge'] = int(charge)
        self.atoms.info['spin'] = int(spin)
        self.atoms.calc = self.calc
        gpu = torch.cuda.get_device_name(0) if device == 'cuda' else ''
        self.info = {'backend': 'uma', 'model': args.model, 'task': args.task,
                     'settings': args.settings, 'device': device, 'gpu': gpu,
                     'fairchem': getattr(fc, '__version__', ''),
                     'torch': torch.__version__, 'load_s': round(time.time() - t0, 1)}

    def evaluate(self, pos):
        self.atoms.set_positions(pos, apply_constraint=False)
        e = float(self.atoms.get_potential_energy())
        f = np.asarray(self.atoms.get_forces(), dtype=np.float64)
        return e, f


class MockBackend:
    """E = sum_bonds k/2 (r - r0)^2 over pairs closer than rbond in the first frame."""

    def __init__(self, args, numbers, cell, pbc, charge, spin, pos):
        self.box = np.asarray(cell, dtype=float).reshape(-1)[:3] if np.ndim(cell) == 1 \
            else np.diag(np.asarray(cell, dtype=float))
        self.pbc = bool(pbc)                           # no minimum image for a cluster
        self.k = float(args.mock_k)
        pos = np.asarray(pos, dtype=float)
        n = pos.shape[0]
        ii, jj = [], []
        for s in range(0, n, 256):                     # chunked all-pairs, once
            d = pos[s:s + 256, None, :] - pos[None, :, :]
            if self.pbc:
                d -= self.box * np.round(d / self.box)
            r2 = np.einsum('abk,abk->ab', d, d)
            a, b = np.nonzero(r2 < args.mock_rbond ** 2)
            a = a + s
            keep = a < b
            ii.append(a[keep]); jj.append(b[keep])
        self.i = np.concatenate(ii); self.j = np.concatenate(jj)
        self.r0 = self._dist(pos)[1]
        self.info = {'backend': 'mock', 'pairs': int(self.i.size), 'k_eV_A2': self.k,
                     'rbond_A': float(args.mock_rbond), 'pbc': self.pbc}

    def _dist(self, pos):
        d = pos[self.i] - pos[self.j]
        if self.pbc:
            d -= self.box * np.round(d / self.box)
        return d, np.sqrt(np.einsum('ak,ak->a', d, d))

    def evaluate(self, pos):
        d, r = self._dist(np.asarray(pos, dtype=float))
        dr = r - self.r0
        e = 0.5 * self.k * float(np.sum(dr * dr))
        g = (self.k * dr / r)[:, None] * d             # dE/dx_i for each pair
        f = np.zeros_like(pos, dtype=float)
        np.add.at(f, self.i, -g)
        np.add.at(f, self.j, g)
        return e, f


BACKENDS = {'uma': UMABackend, 'mock': MockBackend}
# --settings values built here (fairchem InferenceSettings); anything else is passed to
# fairchem as a preset name ("default" = nitro, "high-precision", "batch", ...)
LOWMEM_SETTINGS = ('lowmem', 'lowmem-fp64')


# ---------------------------------------------------------------- main loop
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--backend', choices=sorted(BACKENDS), required=True)
    ap.add_argument('--model', default='uma-s-1p2p1')
    ap.add_argument('--task', default='omol')
    ap.add_argument('--settings', default='default')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--mock-k', type=float, default=10.0, help='eV/A^2')
    ap.add_argument('--mock-rbond', type=float, default=1.9, help='A')
    args = ap.parse_args()

    # The server never plots. Jupyter / Colab export MPLBACKEND=module://matplotlib_inline...
    # to child processes; the MLIP environment lacks that package, so the first
    # `import matplotlib.pyplot` (fairchem's UMA backbone does one) fails. Use Agg.
    os.environ['MPLBACKEND'] = 'Agg'
    # Large systems run UMA close to the GPU's memory (7,400 atoms on a 40 GB A100:
    # 37 GB in use, a 2.75 GB request failing with 3 GB reserved but fragmented).
    # Expandable segments let PyTorch grow allocations instead of fragmenting.
    # Read when CUDA initialises, i.e. before the backend imports torch.
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

    # Keep the protocol channel clean: anything a library prints to stdout
    # (fd 1) goes to stderr instead.
    proto_in = os.fdopen(os.dup(0), 'rb', buffering=0)
    proto_out = os.fdopen(os.dup(1), 'wb')
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)

    backend = None
    n = None
    while True:
        try:
            header, blob = read_msg(proto_in)
        except EOFError:
            return 0
        cmd = header.get('cmd')
        try:
            if cmd == 'quit':
                write_msg(proto_out, {'ok': True})
                return 0
            if cmd == 'init':
                numbers = np.asarray(header['numbers'], dtype=np.int64)
                n = numbers.size
                pos = np.frombuffer(blob, dtype='<f8').reshape(n, 3)
                backend = BACKENDS[args.backend](
                    args, numbers, np.asarray(header['cell'], dtype=float),
                    bool(header.get('pbc', True)), int(header.get('charge', 0)),
                    int(header.get('spin', 1)), pos)
                write_msg(proto_out, {'ok': True, 'info': backend.info})
            elif cmd == 'eval':
                if backend is None:
                    raise RuntimeError('eval before init')
                pos = np.frombuffer(blob, dtype='<f8').reshape(n, 3)
                e, f = backend.evaluate(pos)
                f = np.ascontiguousarray(f, dtype='<f8')
                if f.shape != (n, 3):
                    raise RuntimeError(f'forces have shape {f.shape}, expected {(n, 3)}')
                write_msg(proto_out, {'ok': True, 'energy': e}, f.tobytes())
            else:
                raise RuntimeError(f'unknown command {cmd!r}')
        except Exception as exc:                         # report, keep serving
            write_msg(proto_out, {'ok': False, 'error': f'{type(exc).__name__}: {exc}',
                                  'traceback': traceback.format_exc()})
            if cmd == 'init':
                return 1


if __name__ == '__main__':
    sys.exit(main())
