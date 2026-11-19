"""
io/restart.py — write a restart input directory (coords.inp format).

The new coords.inp is made from the run's own input coords.inp as a template,
so every site keeps its label and charge type and the reference parser
(pkg/input_parser.read_coords) reads it unchanged:
  * per site: position line "x y z label chargetype", velocity line "vx vy vz"
    (free format; written with 12 / 9 decimals so a restart continues the
    trajectory to ~1e-12 A, not to the 1e-6 A of the template);
  * box, then Xlogs / Vlogs / Glogs (the Nose-Hoover chain state, written when
    the run was NVT; otherwise the template's values are kept), then the three
    barostat lines (kept);
  * comment lines (starting with '*') are kept; "Time = ... picoseconds" is
    advanced by the simulated time.
All sites, lone pairs included, are wrapped into [0, L): the parser shifts
every atom when it meets a negative coordinate.

write_restart_dir(input_dir, out_dir, ...) writes coords.inp and copies the
other input files (mdrun.params, ff.dat, connectivity.dat, temp.ramp and
params.h if present), so the next run is just --input-dir out_dir. params.h
carries the array limits pkg needs for larger systems (md-input3).
"""
from __future__ import annotations
import os
import re
import shutil
import numpy as np

OTHER_INPUTS = ('mdrun.params', 'ff.dat', 'connectivity.dat', 'temp.ramp', 'params.h')


def _is_data(line):
    return not line.strip().startswith('*')


def write_coords(template, path, x, v, box, nhc=None, time_add_ps=0.0):
    """Write coords.inp at `path` from `template` with new positions x (nat,3),
    velocities v (nat,3) in m/s, box (3,), and, if given, the thermostat state
    nhc = (xlogs, vlogs, glogs) (arrays of the chain length)."""
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    box = np.asarray(box, dtype=np.float64).reshape(3)
    nat = x.shape[0]
    xw = x - box * np.floor(x / box)                       # wrap into [0, L)
    xw = np.where(xw >= box, xw - box, xw)                 # guard x == L after rounding
    with open(template) as f:
        lines = f.read().splitlines()
    ndata = sum(1 for l in lines if _is_data(l))
    if ndata < 2 * nat + 7:
        raise ValueError(f"{template}: {ndata} data lines, expected {2 * nat + 7} for {nat} sites")

    def fmt_list(a):
        return ' '.join(f"{float(t): .16e}" for t in np.atleast_1d(np.asarray(a)))

    out, k = [], 0
    for line in lines:
        if not _is_data(line):
            m = re.search(r'Time\s*=\s*([-+0-9.eEdD]+)', line)
            if m and time_add_ps:
                t_new = float(m.group(1).replace('D', 'e').replace('d', 'e')) + time_add_ps
                line = line[:m.start(1)] + f"{t_new:.3f}" + line[m.end(1):]
            out.append(line)
            continue
        if k < 2 * nat:
            iat = k // 2
            if k % 2 == 0:
                tok = line.split()
                out.append(f"{xw[iat, 0]:22.12f}{xw[iat, 1]:22.12f}{xw[iat, 2]:22.12f}   "
                           f"{tok[3]:<4s}{tok[4]:>4s}")
            else:
                out.append(f"{v[iat, 0]:22.9f}{v[iat, 1]:22.9f}{v[iat, 2]:22.9f}")
        elif k == 2 * nat:
            out.append(f"  {fmt_list(box)}")
        elif nhc is not None and k in (2 * nat + 1, 2 * nat + 2, 2 * nat + 3):
            out.append(f"  {fmt_list(nhc[k - 2 * nat - 1])}")
        else:
            out.append(line)
        k += 1
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        f.write('\n'.join(out) + '\n')
    os.replace(tmp, path)                                   # never leave a half-written file


def set_kmax(params_path, klimit):
    """Rewrite the Ewald k-limits in an mdrun.params copy, if they differ.

    The 13th data line (pkg read_mdrun_params, 'Electrostatics') holds a 12-character
    label and then alpha, kmax_x, kmax_y, kmax_z, enable. Only the three integers
    change; the line is left alone when they already match. Returns True if changed.
    The slab layout (gomd/electrodes.py) scales kmax_z with the box height, and a
    restart in the taller box must keep the scaled value: read back with the
    input's kmax_z, the 205 A cell of md-input3 started 9.2 kcal/mol off.
    """
    with open(params_path) as f:
        lines = f.read().split('\n')
    data = [k for k, l in enumerate(lines) if l.strip() and not l.strip().startswith('*')]
    k = data[12]
    line = lines[k]
    rest = line[12:].split()
    new = [str(int(t)) for t in klimit]
    if rest[1:4] == new:
        return False
    rest[1:4] = new
    lines[k] = line[:12] + '   ' + '   '.join(rest)
    with open(params_path, 'w') as f:
        f.write('\n'.join(lines))
    return True


def write_restart_dir(input_dir, out_dir, x, v, box, nhc=None, time_add_ps=0.0, klimit=None):
    """coords.inp (from input_dir/coords.inp as the template) + copies of the
    other inputs into out_dir. klimit: the Ewald k-limits the run used, written
    into the mdrun.params copy when they differ from it (see set_kmax).
    Returns out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    src = os.path.abspath(input_dir); dst = os.path.abspath(out_dir)
    if src == dst:
        raise ValueError('restart directory must differ from the input directory')
    for name in OTHER_INPUTS:
        p = os.path.join(src, name)
        if os.path.exists(p):
            shutil.copyfile(p, os.path.join(dst, name))
    if klimit is not None and os.path.exists(os.path.join(dst, 'mdrun.params')):
        set_kmax(os.path.join(dst, 'mdrun.params'), klimit)
    write_coords(os.path.join(src, 'coords.inp'), os.path.join(dst, 'coords.inp'),
                 x, v, box, nhc=nhc, time_add_ps=time_add_ps)
    return out_dir
