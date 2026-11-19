"""
GOMD side of the MLIP: starts gomd/mlip/server.py as a subprocess and turns its
answers into GOMD units.

    provider = MLIPProvider.start(backend='uma', python=..., numbers=..., box=...,
                                  x0=..., model='uma-s-1p2p1', task='omol')
    e_kcal, f_kcal = provider(x)        # x: (nat, 3) incl. lone-pair sites
    provider.close()

Only real atoms (atomic number > 0) are sent; lone-pair sites get zero force.
Energies are returned in kcal/mol and forces in kcal/mol/A, the PFF's units.
"""
from __future__ import annotations
import os
import subprocess
import sys
import time

import numpy as np

from .server import read_msg, write_msg

EV_TO_KCAL = 96.485332123310018 / 4.184          # 1 eV = 23.0605 kcal/mol
SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'server.py')


class MLIPServerError(RuntimeError):
    pass


class MLIPClient:
    def __init__(self, cmd, numbers, cell, pbc, charge, spin, pos):
        self.cmd = cmd
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.n = int(len(numbers))
        hdr = {'cmd': 'init', 'numbers': [int(z) for z in numbers],
               'cell': [float(c) for c in np.asarray(cell).reshape(-1)],
               'pbc': bool(pbc), 'charge': int(charge), 'spin': int(spin)}
        self.info = self._call(hdr, pos)[0]['info']

    def _call(self, header, pos=None):
        blob = b'' if pos is None else np.ascontiguousarray(pos, dtype='<f8').tobytes()
        try:
            write_msg(self.proc.stdin, header, blob)
            reply, rblob = read_msg(self.proc.stdout)
        except (EOFError, BrokenPipeError, OSError):
            rc = self.proc.wait(timeout=30)
            raise MLIPServerError(
                f"MLIP server exited (code {rc}) — see its messages above. "
                f"command: {' '.join(self.cmd)}") from None
        if not reply.get('ok'):
            raise MLIPServerError(f"MLIP server: {reply.get('error')}\n{reply.get('traceback', '')}")
        return reply, rblob

    def evaluate(self, pos):
        reply, blob = self._call({'cmd': 'eval'}, pos)
        return float(reply['energy']), np.frombuffer(blob, dtype='<f8').reshape(self.n, 3)

    def close(self):
        if self.proc.poll() is None:
            try:
                self._call({'cmd': 'quit'})
            except MLIPServerError:
                pass
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class MLIPProvider:
    """Callable (nat, 3) positions -> (E kcal/mol, F (nat, 3) kcal/mol/A)."""

    def __init__(self, client, numbers):
        self.client = client
        self.numbers = np.asarray(numbers)
        self.real = np.flatnonzero(self.numbers > 0)
        self.nat = self.numbers.size
        self.ncall = 0
        self.seconds = 0.0

    @classmethod
    def start(cls, backend, numbers, box, x0, python=None, model='uma-s-1p2p1',
              task='omol', settings='default', device='auto', charge=0, spin=1,
              extra_args=(), pbc=True):
        numbers = np.asarray(numbers)
        real = np.flatnonzero(numbers > 0)
        cmd = [python or sys.executable, '-I', SERVER, '--backend', backend,
               '--model', model, '--task', task, '--settings', settings,
               '--device', device, *extra_args]
        client = MLIPClient(cmd, numbers[real], np.asarray(box), bool(pbc), charge, spin,
                            np.asarray(x0)[real])
        return cls(client, numbers)

    @property
    def info(self):
        return self.client.info

    def __call__(self, x):
        t0 = time.perf_counter()
        x = np.asarray(x, dtype=np.float64)
        e, f_real = self.client.evaluate(x[self.real])
        f = np.zeros((self.nat, 3))
        f[self.real] = f_real * EV_TO_KCAL
        self.ncall += 1
        self.seconds += time.perf_counter() - t0
        return e * EV_TO_KCAL, f

    def close(self):
        self.client.close()
