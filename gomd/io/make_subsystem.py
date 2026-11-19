"""
io/make_subsystem.py — a smaller MLIP test system cut from a GOMD system, made of
complete molecules, real atoms only (no lone-pair sites).

    python -m gomd.io.make_subsystem --input-dir md-input --ref-dir pkg \
        --shape sphere --natoms 2000 --out md-input2/sphere_2000.xyz
    python -m gomd.io.make_subsystem ... --shape cube --natoms 2000 --out md-input2/cube_2000.xyz

1. Molecules are the connected components of the bond graph over ALL sites,
   so each lone-pair site belongs to its molecule and the molecular charge
   (sum of site charges, lone pairs included) is an integer: Li+ +1, FSI- -1,
   solvent 0.
2. Every molecule is made whole across the periodic boundary (minimum image
   from its first atom) and placed at the image of its centre of mass (COM)
   nearest the selection centre (default: the box centre).
3. Selection of whole molecules by COM:
     sphere  all molecules with |COM - centre| <= R, R grown until the real
             atom count reaches --natoms; written as a cluster in vacuum
             (pbc F F F, frame = extent + 2 x --vacuum)
     cube    all molecules with their COM inside a cube of edge a (a from
             --natoms and the density); written in a periodic box of edge a.
             Molecules that stick out of the cube are wrapped to the far face,
             where they may come close to others: the closest contact between
             atoms of different molecules is reported.
4. Neutrality (default): while the net charge is not 0, the nearest unselected
   counter-ion (by COM distance) is added.
5. Output: real atoms of the selected molecules, positions, velocities (m/s),
   masses, 'site' (index in the source system), charge, spin; a section is
   appended to <out dir>/README.txt.
"""
from __future__ import annotations
import argparse, os, re, sys, time


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', default='md-input')
    ap.add_argument('--ref-dir', default='pkg')
    ap.add_argument('--shape', choices=['sphere', 'cube'], default='sphere')
    ap.add_argument('--natoms', type=int, default=2000, help='target number of real atoms')
    ap.add_argument('--center', default=None, help='x,y,z in A (default: box centre)')
    ap.add_argument('--vacuum', type=float, default=15.0, help='sphere: A of vacuum on each side')
    ap.add_argument('--no-neutral', action='store_true')
    ap.add_argument('--spin', type=int, default=1)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    import numpy as np
    from gomd.io.assembly_loader import load_m2_system
    from gomd.mlip.elements import numbers_from_masses, formula, SYMBOLS
    from gomd.io.xyz import write_extxyz, read_extxyz

    D = load_m2_system(args.input_dir, args.ref_dir)
    nat = D['nat']; box = np.asarray(D['box'], dtype=np.float64)
    x = np.asarray(D['x'], dtype=np.float64); v = np.asarray(D['v'], dtype=np.float64)
    mass = np.asarray(D['mass'], dtype=np.float64); q = np.asarray(D['q'], dtype=np.float64)
    real = ~np.asarray(D['ldummy'])
    numbers = numbers_from_masses(mass, D['ldummy'])
    mi = lambda d: d - box * np.round(d / box)

    # ---- 1. molecules over all sites (lone pairs included) ----
    parent = np.arange(nat)
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    b = np.asarray(D['conn']['bonds'])
    for i, j in zip(b[0], b[1]):
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[ri] = rj
    roots = np.array([find(a) for a in range(nat)])
    mols = [np.flatnonzero(roots == r) for r in np.unique(roots)]
    if any(not real[s].any() for s in mols):
        raise SystemExit('a molecule without real atoms (lone pair not bonded to its parent?)')

    c = (np.array([float(t) for t in args.center.split(',')]) if args.center else box / 2.0)
    info_m = []                                          # per molecule
    xs_all = np.empty_like(x)
    for s in mols:
        r0 = s[np.argmax(real[s])]                       # first real atom
        xs = x[r0] + mi(x[s] - x[r0])                    # whole molecule
        sr = s[real[s]]
        com = np.sum(mass[sr, None] * xs[real[s]], axis=0) / mass[sr].sum()
        d = mi(com - c)
        xs = xs + (c + d - com)                          # COM at its image nearest the centre
        xs_all[s] = xs
        qm = float(q[s].sum())
        if abs(qm - round(qm)) > 1e-3:
            raise SystemExit(f'molecule with non-integer charge {qm:.4f}')
        el = set(SYMBOLS[int(z)] for z in numbers[sr])
        kind = 'Li+' if el == {'Li'} else ('FSI-' if 'S' in el else 'solvent')
        info_m.append({'sites': s, 'real': sr, 'd': d, 'r': float(np.linalg.norm(d)),
                       'charge': int(round(qm)), 'kind': kind})

    # ---- 2. selection ----
    order = np.argsort([m['r'] for m in info_m])
    if args.shape == 'sphere':
        sel, count = [], 0
        for k in order:                                  # grow R until the target is reached
            sel.append(k); count += info_m[k]['real'].size
            if count >= args.natoms:
                break
        R = info_m[sel[-1]]['r']
        sel = [k for k in order if info_m[k]['r'] <= R]  # every molecule with COM inside R
        size_note = f'radius {R:.3f} A'
    else:
        dens = real.sum() / np.prod(box)
        a = (args.natoms / dens) ** (1.0 / 3.0)
        if a >= box.min():
            raise SystemExit(f'cube edge {a:.2f} A is not smaller than the box')
        sel = [k for k in order if np.all(np.abs(info_m[k]['d']) < a / 2.0)]
        size_note = f'edge {a:.3f} A'

    added = []
    if not args.no_neutral:
        net = sum(info_m[k]['charge'] for k in sel)
        chosen = set(sel)
        while net != 0:
            want = -1 if net > 0 else 1
            k = next((k for k in order if k not in chosen and info_m[k]['charge'] == want), None)
            if k is None:
                raise SystemExit('cannot neutralise: no counter-ion left')
            sel.append(k); chosen.add(k); added.append(k); net += want

    # ---- 3. output ----
    idx = np.concatenate([info_m[k]['real'] for k in sel])
    mol_of = np.concatenate([np.full(info_m[k]['real'].size, i) for i, k in enumerate(sel)])
    xo = xs_all[idx].copy()
    if args.shape == 'sphere':
        xo -= c
        ext = float(np.max(np.abs(xo)))
        frame = 2.0 * (ext + args.vacuum)
        xo += frame / 2.0
        obox = np.array([frame] * 3); pbc = False
    else:
        xo -= (c - a / 2.0)
        obox = np.array([a] * 3); pbc = True
        xo -= obox * np.floor(xo / obox)
    sym = [SYMBOLS[int(z)] for z in numbers[idx]]
    charge = int(sum(info_m[k]['charge'] for k in sel))

    # closest contact between atoms of different molecules (cube: minimum image)
    dmin = np.inf
    for s0 in range(0, len(idx), 512):
        dd = xo[s0:s0 + 512, None, :] - xo[None, :, :]
        if pbc:
            dd -= obox * np.round(dd / obox)
        r2 = np.einsum('abk,abk->ab', dd, dd)
        same = mol_of[s0:s0 + 512, None] == mol_of[None, :]
        r2[same] = np.inf
        dmin = min(dmin, float(np.sqrt(r2.min())))

    kinds = [info_m[k]['kind'] for k in sel]
    comp = ', '.join(f"{kd} {kinds.count(kd)}" for kd in ('solvent', 'Li+', 'FSI-'))
    ndof = 3 * len(idx) - 3
    T = 2 * (0.5 * np.sum(mass[idx, None] * v[idx] ** 2) / 4184000.0) / (ndof * 1.98709e-3)
    m = re.search(r'Time\s*=\s*([0-9.]+)', open(os.path.join(args.input_dir, 'coords.inp')).read())
    t_ps = float(m.group(1)) if m else 0.0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_extxyz(args.out, sym, xo, obox, v=v[idx], mass=mass[idx], site=idx, pbc=pbc,
                 info={'charge': charge, 'spin': args.spin, 'time_ps': t_ps, 'velocity_units': 'm/s',
                       'source': args.input_dir.rstrip('/'), 'subsystem': args.shape})
    back = read_extxyz(args.out)
    ok = back['symbols'] == sym and back['pbc'] == pbc and np.allclose(back['x'], xo, atol=1e-11)
    summary = (f"{os.path.basename(args.out)}: {args.shape}, {size_note}, {len(idx)} atoms "
               f"({formula(numbers[idx])}) in {len(sel)} complete molecules ({comp})"
               + (f", {len(added)} counter-ion(s) added for neutrality" if added else "")
               + f"; charge {charge}, spin {args.spin}; "
               + (f"cluster in vacuum, frame {obox[0]:.2f} A (pbc F)" if not pbc
                  else f"periodic box {obox[0]:.3f} A")
               + f"; closest contact between molecules {dmin:.3f} A; "
               f"T {T:.1f} K over {ndof} dof; read-back {'OK' if ok else 'MISMATCH'}")
    print(summary)
    with open(os.path.join(os.path.dirname(os.path.abspath(args.out)), 'README.txt'), 'a') as f:
        f.write(f"\n{time.strftime('%Y-%m-%d %H:%M')}  python -m gomd.io.make_subsystem --input-dir "
                f"{args.input_dir} --shape {args.shape} --natoms {args.natoms} --out {args.out}\n  "
                + summary + "\n")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
