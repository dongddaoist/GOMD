"""
Restart round trip: write a restart after N steps, read it back with the
reference parser, and continue; compare with the uninterrupted run.

    python -m gomd.tests.check_restart --input-dir ./md-input --ref-dir ./pkg --polarizable

  1. N steps (NVT, as main_scan) from --input-dir
  2. write_restart_dir -> --tmp; load it with the same loader as every run
  3. positions, velocities and the Nose-Hoover chain state read back = written
  4. N more steps uninterrupted vs N steps from the restart: PE, KE, H_ext
     (a restarted run starts from the physical force and a fresh neighbor list
     and SCF history, so it tracks the uninterrupted one closely but not bitwise)
"""
from __future__ import annotations
import argparse, os, re, shutil, sys


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', default='./md-input')
    ap.add_argument('--ref-dir', default='./pkg')
    ap.add_argument('--polarizable', action='store_true')
    ap.add_argument('--steps', type=int, default=100)
    ap.add_argument('--tmp', default='md-output/check_restart_tmp')
    args = ap.parse_args(argv)
    from jax import config as jc
    jc.update('jax_enable_x64', True)
    import jax
    import numpy as np
    from gomd.tests.common import build_md
    from gomd.io.restart import write_restart_dir
    from gomd.io.assembly_loader import load_m2_system

    n = args.steps
    A = build_md(args.input_dir, args.ref_dir, polarizable=args.polarizable, ensemble='nvt')
    c1, h1 = A['run'](A['carry0'], nsteps=n)
    S = jax.device_get(c1['S'])
    box = np.asarray(A['D']['box'])
    shutil.rmtree(args.tmp, ignore_errors=True)
    write_restart_dir(args.input_dir, args.tmp, S['x'], S['v'], box,
                      nhc=tuple(np.asarray(a) for a in S['nhc']), time_add_ps=n * A['D']['dt'] / 1000)
    t_old = re.search(r'Time\s*=\s*([0-9.]+)', open(os.path.join(args.input_dir, 'coords.inp')).read())
    t_new = re.search(r'Time\s*=\s*([0-9.]+)', open(os.path.join(args.tmp, 'coords.inp')).read())

    D2 = load_m2_system(args.tmp, args.ref_dir)
    xw = S['x'] - box * np.floor(S['x'] / box)
    dx = np.abs(D2['x'] - xw); dx = np.minimum(dx, box - dx)
    dv = np.abs(D2['v'] - S['v'])
    E2 = D2['ensemble']
    dn = max(float(np.max(np.abs(np.asarray(E2[k]) - np.asarray(S['nhc'][i])) /
                       np.maximum(np.abs(np.asarray(S['nhc'][i])), 1e-300)))
             for i, k in enumerate(('xlogs', 'vlogs', 'glogs')))
    ok_read = dx.max() < 1e-10 and dv.max() < 1e-7 and dn < 1e-14 and D2['nat'] == A['D']['nat']
    print(f"read back: max |dx| {dx.max():.1e} A, max |dv| {dv.max():.1e} m/s, NHC state rel {dn:.1e}; "
          f"Time {t_old.group(1)} -> {t_new.group(1)} ps   {'OK' if ok_read else 'MISMATCH'}", flush=True)

    c2, h2 = A['run'](c1, nsteps=n)                      # uninterrupted continuation
    B = build_md(args.tmp, args.ref_dir, polarizable=args.polarizable, ensemble='nvt')
    c3, h3 = B['run'](B['carry0'], nsteps=n)             # from the restart
    h2 = jax.device_get(h2); h3 = jax.device_get(h3)
    worst = 0.0
    for k in ('pe', 'ke', 'hext'):
        d = np.abs(np.asarray(h3[k]) - np.asarray(h2[k]))
        worst = max(worst, float(d.max()))
        print(f"  {k:<5} max |restart - uninterrupted| over {n} steps: {d.max():.4f} kcal/mol "
              f"(step 1: {d[0]:.4f}, step {n}: {d[-1]:.4f})")
    ok_run = worst < 1.0
    print('ALL OK' if ok_read and ok_run else 'MISMATCH')
    shutil.rmtree(args.tmp, ignore_errors=True)
    return 0 if ok_read and ok_run else 1


if __name__ == '__main__':
    sys.exit(main())
