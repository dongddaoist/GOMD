"""
tests/check_electrodes.py — P0 checks of gomd.electrodes on the graphite cell (md-input3).

    python -m gomd.tests.check_electrodes --input-dir md-input3 --ref-dir pkg [--slab-height 205]

Loads the system once and prepares it twice (periodic box, then the slab box),
from the same input. Checks:
  1. every electrode layer is flat (one z) and, snapped, is ideal graphene: each
     atom has exactly 3 in-plane neighbours at the bond length b (to 1e-9 A);
  2. the layer spacings are the same in both layouts, and the slab layout puts
     every electrolyte atom between the two electrodes, with the requested box
     height and at least rlist of vacuum;
  3. the slab layout moves the electrolyte rigidly: every electrolyte-electrolyte
     minimum-image distance within 15 A is unchanged;
  4. frozen atoms, dropped terms and ndof are as expected for this input
     (8 sheets x 448 atoms: 672 bonds, 1344 bends, 2688 torsions, 1344 out-of-plane
     terms each; ndof 46632 -> 35880).
Exit code 0 when everything passes.
"""
from __future__ import annotations
import argparse, copy, os, sys, time


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', default='md-input3')
    ap.add_argument('--ref-dir', default='pkg')
    ap.add_argument('--lower', default='first:1792')
    ap.add_argument('--upper', default='last:1792')
    ap.add_argument('--slab-height', type=float, default=205.0)
    args = ap.parse_args(argv)
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    import numpy as np
    from gomd.io.assembly_loader import load_m2_system
    from gomd.electrodes import prepare_electrodes

    t0 = time.time()
    D0 = load_m2_system(args.input_dir, args.ref_dir)
    print(f"loaded {D0['nat']} sites in {time.time() - t0:.0f} s")
    rlist = D0['rcut'] + 1.0
    ok = True

    def check(name, cond, detail=''):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(': ' + detail) if detail else ''}")

    out = {}
    for mode, h in (('periodic', None), ('slab', args.slab_height)):
        D = copy.deepcopy(D0)
        print(f"\n== {mode} layout")
        info = prepare_electrodes(D, args.lower, args.upper, geometry='flat', slab_height=h,
                                  rlist=rlist, emit=lambda s: print('   ', s))
        out[mode] = (D, info)
        x = D['x']; box = D['box']; fr = D['frozen']
        # 1. flat, ideal graphene
        worst_z = worst_b = 0.0; bad_nb = 0
        sel = {'lower': np.flatnonzero(fr)[:info['n_lower']],
               'upper': np.flatnonzero(fr)[info['n_lower']:]}
        for name, idx in sel.items():
            z = x[idx, 2]
            for zl in np.unique(np.round(z, 6)):
                a = idx[np.abs(z - zl) < 1e-6]
                worst_z = max(worst_z, float(np.ptp(x[a, 2])))
                d = x[a, None, :2] - x[None, a, :2]
                d -= box[:2] * np.round(d / box[:2])
                r = np.sqrt((d ** 2).sum(-1))
                np.fill_diagonal(r, np.inf)
                b = info['layers'][name][0]['b']
                nb = (r < 1.2 * b)
                bad_nb += int((nb.sum(1) != 3).sum())
                worst_b = max(worst_b, float(np.abs(r[nb] - b).max()))
        check('layers flat', worst_z < 1e-9, f"max z spread {worst_z:.2e} A")
        check('ideal graphene', bad_nb == 0 and worst_b < 1e-9,
              f"atoms without exactly 3 neighbours: {bad_nb}; max |r - b| {worst_b:.2e} A")
        nlayer = sum(len(info['layers'][k]) for k in ('lower', 'upper'))
        check('8 layers of 448', nlayer == 8 and all(r['n'] == 448 for k in ('lower', 'upper')
                                                    for r in info['layers'][k]))
        # 4. dropped terms, ndof
        dr = info['dropped']
        check('dropped terms', dr == {'bonds': 8 * 672, 'bends': 8 * 1344, 'torts': 8 * 2688,
                                      'out-of-plane': 8 * 1344, 'constraints': 0}, str(dr))
        check('ndof', info['ndof'] == (46632, 35880), str(info['ndof']))
        check('frozen velocities zero', float(np.abs(D['v'][fr]).max()) == 0.0)

    # 2. same spacings; electrolyte inside; box
    Dp, ip = out['periodic']; Ds, is_ = out['slab']
    zs = sorted(np.unique(np.round(Ds['x'][Ds['frozen'], 2], 6)))
    sp = np.diff(sorted(ip['layers']['lower'][i]['z'] for i in range(4)))
    ss = np.diff(zs[:4])
    check('lower-electrode spacings equal in both layouts', np.allclose(sp, ss, atol=1e-9),
          f"{np.round(sp, 4)} vs {np.round(ss, 4)}")
    check('slab box height', abs(Ds['box'][2] - args.slab_height) < 1e-12, f"{Ds['box'][2]}")
    check('vacuum >= rlist', is_['vacuum'] >= rlist, f"{is_['vacuum']:.1f} A")
    mob = ~Ds['frozen'] & ~Ds['ldummy']
    zlo_top = is_['lower_z'][1]; zup_bot = is_['upper_z'][0]
    zm = Ds['x'][mob, 2]
    check('electrolyte between the electrodes', (zm > zlo_top).all() and (zm < zup_bot).all(),
          f"electrolyte z {zm.min():.2f}..{zm.max():.2f}, faces {zlo_top:.2f} / {zup_bot:.2f}")
    check('inner gap unchanged', abs(is_['inner_gap'] - ip['inner_gap']) < 1e-9,
          f"{is_['inner_gap']:.4f} A")
    # 3. rigid electrolyte: pair distances (min image in the old box vs the new one)
    m = np.flatnonzero(~Ds['frozen'])
    rng = np.random.default_rng(0)
    pick = rng.choice(m, 400, replace=False)
    def dists(D, box):
        d = D['x'][pick, None, :] - D['x'][None, m, :]
        d -= box * np.round(d / box)
        r = np.sqrt((d ** 2).sum(-1))
        return np.where(r < 15.0, r, 0.0)
    r0 = dists(D0, D0['box']); r1 = dists(Ds, Ds['box'])
    check('electrolyte moved rigidly', np.abs(r0 - r1).max() < 1e-9,
          f"max change {np.abs(r0 - r1).max():.2e} A over {(r0 > 0).sum()} pairs")
    print('\nALL PASS' if ok else '\nSOME CHECKS FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
