"""
io/make_lp_free.py — lone-pair-free copy of a GOMD system, for MLIP runs.

    python -m gomd.io.make_lp_free --input-dir ./md-input --ref-dir ./pkg --out-dir ./md-input2

Loads the system the way every GOMD run does (pkg/ setup, so the lone-pair
sites are recognised from the force field), drops every lone-pair site, and
writes out-dir/system.xyz (extended XYZ, see io/xyz.py):
  * element symbols from the force-field masses (Fs -> F, O= -> O, S- -> S, ...)
  * positions (wrapped into the box), velocities (m/s) and masses of the real
    atoms, unchanged and in the same order
  * site: the atom's index in the source system
  * charge: the total charge of the source system (lone-pair charges included,
    so 0 here); spin multiplicity 1
plus out-dir/README.txt (provenance and composition). No connectivity and no
force-field file: an MLIP needs only elements, positions and the cell.
"""
from __future__ import annotations
import argparse, os, re, sys, time


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', default='./md-input')
    ap.add_argument('--ref-dir', default='./pkg')
    ap.add_argument('--out-dir', default='./md-input2')
    ap.add_argument('--spin', type=int, default=1)
    args = ap.parse_args(argv)
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    import numpy as np
    from gomd.io.assembly_loader import load_m2_system
    from gomd.mlip.elements import numbers_from_masses, formula, SYMBOLS
    from gomd.io.xyz import write_extxyz, read_extxyz

    D = load_m2_system(args.input_dir, args.ref_dir)
    real = ~np.asarray(D['ldummy'])
    site = np.flatnonzero(real)
    numbers = numbers_from_masses(D['mass'], D['ldummy'])[site]
    symbols = [SYMBOLS[int(z)] for z in numbers]
    box = np.asarray(D['box'])
    x = np.asarray(D['x'])[site]
    x = x - box * np.floor(x / box)
    v = np.asarray(D['v'])[site]
    mass = np.asarray(D['mass'])[site]
    charge = int(round(float(np.sum(D['q']))))
    m = re.search(r'Time\s*=\s*([0-9.]+)', open(os.path.join(args.input_dir, 'coords.inp')).read())
    t_ps = float(m.group(1)) if m else 0.0
    ndof = 3 * len(site) - 3
    ke = 0.5 * np.sum(mass[:, None] * v * v) / 4184000.0
    T = 2 * ke / (ndof * 1.98709e-3)
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, 'system.xyz')
    write_extxyz(path, symbols, x, box, v=v, mass=mass, site=site,
                 info={'charge': charge, 'spin': args.spin, 'time_ps': t_ps,
                       'velocity_units': 'm/s', 'source': args.input_dir.rstrip('/')})
    back = read_extxyz(path)                                  # round-trip check
    ok = (back['symbols'] == symbols and np.allclose(back['x'], x, atol=1e-11)
          and np.allclose(back['v'], v, atol=1e-8) and np.array_equal(back['site'], site))
    nlp = int(np.sum(~real))
    with open(os.path.join(args.out_dir, 'README.txt'), 'w') as f:
        f.write(f"""Lone-pair-free copy of {args.input_dir.rstrip('/')} for MLIP runs
made {time.strftime('%Y-%m-%d %H:%M')} by: python -m gomd.io.make_lp_free --input-dir {args.input_dir} --out-dir {args.out_dir}

system.xyz  extended XYZ, one frame (positions A, velocities m/s, masses g/mol)
  {len(site)} atoms = {D['nat']} sites - {nlp} lone-pair sites
  composition {formula(numbers)}
  box {box[0]:.6f} x {box[1]:.6f} x {box[2]:.6f} A, periodic
  charge {charge} (total of the source system, lone-pair charges included), spin multiplicity {args.spin}
  state: coords.inp at t = {t_ps} ps; kinetic temperature {T:.2f} K over 3N - 3 = {ndof} dof
         (no constraints here; the PFF counts 15192 dof with its 7008 SHAKE bonds)
  column 'site' = the atom's index in the source system (lone-pair sites included)

Run MD on it with the MLIP driver:  python -m gomd.mlip_md --input {path} --mlip uma|mock ...
""")
    print(f"{path}: {len(site)} atoms ({formula(numbers)}), {nlp} lone-pair sites dropped, charge {charge}, "
          f"spin {args.spin}, T {T:.2f} K over {ndof} dof; read-back {'OK' if ok else 'MISMATCH'}")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
