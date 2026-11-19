"""PFF vs MLIP agreement at one configuration (real atoms only).

Energies of two different models have unrelated zeros, so the energy metric is
the change of the difference since the first comparison:
    dE_shift(t) = [E_mlip(t) - E_pff(t)] - [E_mlip(0) - E_pff(0)]
Forces are compared directly, dF = F_mlip - F_pff, in kcal/mol/A. F_pff is the
physical PFF force after lone-pair redistribution (not the SHAKE-combined
force), so both describe the same real atoms.
"""
from __future__ import annotations
import csv
import numpy as np

from .elements import SYMBOLS


class Comparator:
    def __init__(self, numbers, path, propagate):
        self.numbers = np.asarray(numbers)
        self.real = np.flatnonzero(self.numbers > 0)
        self.zs = [int(z) for z in np.unique(self.numbers[self.real])]
        self.de0 = None
        self.rows = []
        self.propagate = propagate
        self.cols = (['step', 'propagated', 'E_pff', 'E_mlip', 'dE_shift', 'dE_shift_per_atom',
                      'F_rmse', 'F_mae', 'F_max_atom', 'F_rel', 'cos', 'F_rms_pff', 'F_rms_mlip']
                     + [f'F_rmse_{SYMBOLS[z]}' for z in self.zs] + ['mlip_s'])
        self._f = open(path, 'w', newline='')
        self._w = csv.writer(self._f)
        self._w.writerow(self.cols)
        self._f.flush()

    def add(self, step, e_pff, f_pff, e_mlip, f_mlip, mlip_seconds=float('nan')):
        fp = np.asarray(f_pff)[self.real]
        fm = np.asarray(f_mlip)[self.real]
        d = fm - fp
        de = float(e_mlip) - float(e_pff)
        if self.de0 is None:
            self.de0 = de
        nrm = lambda a: float(np.sqrt(np.sum(a * a)))
        row = {
            'step': int(step), 'propagated': self.propagate,
            'E_pff': float(e_pff), 'E_mlip': float(e_mlip),
            'dE_shift': de - self.de0, 'dE_shift_per_atom': (de - self.de0) / self.real.size,
            'F_rmse': float(np.sqrt(np.mean(d * d))),
            'F_mae': float(np.mean(np.abs(d))),
            'F_max_atom': float(np.max(np.sqrt(np.sum(d * d, axis=1)))),
            'F_rel': nrm(d) / max(nrm(fp), 1e-300),
            'cos': float(np.sum(fp * fm)) / max(nrm(fp) * nrm(fm), 1e-300),
            'F_rms_pff': float(np.sqrt(np.mean(fp * fp))),
            'F_rms_mlip': float(np.sqrt(np.mean(fm * fm))),
            'mlip_s': float(mlip_seconds),
        }
        zr = self.numbers[self.real]
        for z in self.zs:
            dz = d[zr == z]
            row[f'F_rmse_{SYMBOLS[z]}'] = float(np.sqrt(np.mean(dz * dz)))
        self.rows.append(row)
        self._w.writerow([row[c] for c in self.cols])
        self._f.flush()
        return row

    @staticmethod
    def line(row):
        return (f"# compare step {row['step']:>6}: dE_shift {row['dE_shift']:+.4f} kcal/mol  "
                f"F_rmse {row['F_rmse']:.4f}  F_max {row['F_max_atom']:.3f} kcal/mol/A  "
                f"F_rel {row['F_rel']:.4f}  cos {row['cos']:.6f}")

    def summary(self):
        if not self.rows:
            return []
        de = np.array([r['dE_shift'] for r in self.rows])
        fr = np.array([r['F_rmse'] for r in self.rows])
        cs = np.array([r['cos'] for r in self.rows])
        out = [f"# compare summary ({len(self.rows)} samples): "
               f"dE_shift std {de.std():.4f} kcal/mol, range [{de.min():+.4f}, {de.max():+.4f}]; "
               f"F_rmse mean {fr.mean():.4f} kcal/mol/A; cos mean {cs.mean():.6f}"]
        zs = ' '.join(f"{SYMBOLS[z]} {np.mean([r[f'F_rmse_{SYMBOLS[z]}'] for r in self.rows]):.3f}"
                      for z in self.zs)
        out.append(f"# compare F_rmse by element (kcal/mol/A): {zs}")
        return out

    def close(self):
        self._f.close()
