"""
Induced (polarization) force = -dE_pol/dx, by central finite differences.

    python -m gomd.tests.check_polforce_fd --input-dir ./md-input --ref-dir ./pkg [--rtaper 10]

At a tightly converged SCF the PCG energy is variational, so the total
induced force (pair term, exclusion and 1-4 corrections, k-space Q-mu force)
must be minus the gradient of E_pol(x) with the dipoles re-solved at every
displaced configuration. Checks --natoms random real atoms x 3 components.
With --rtaper < rcut the dipole-dipole taper is on, which also checks tau and
its derivative. The field uses the tabulated erfc (piecewise linear), so the
two sides agree to ~1e-5 relative, not to machine precision.
"""
from __future__ import annotations
import argparse, sys


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', default='./md-input')
    ap.add_argument('--ref-dir', default='./pkg')
    ap.add_argument('--rtaper', type=float, default=None)
    ap.add_argument('--natoms', type=int, default=8)
    ap.add_argument('--h', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args(argv)
    from jax import config as jc
    jc.update('jax_enable_x64', True)
    import jax, jax.numpy as jnp
    import numpy as np
    from gomd.main_scan import build_Dj
    from gomd.io.assembly_loader import load_m2_system
    from gomd.forces.neighbor import build_neighbor_pairs
    from gomd.forces.ewald import build_kvectors
    from gomd.forces.dummy import update_dummy_coords
    from gomd.forces.polarization_full import make_polarization_parts

    D = load_m2_system(args.input_dir, args.ref_dir)
    if args.rtaper is not None:
        D['pol_data']['rtaper'] = float(args.rtaper)
    D['pol_data']['tolpol'] = 1e-22
    nat = D['nat']; Dj = build_Dj(D); box = Dj['box']; dm = Dj['dummy']
    x = update_dummy_coords(Dj['x'], dm['jik'], dm['adummy'], dm['bdummy'], dm['cdummy'], box, nat)
    ii, jj = build_neighbor_pairs(np.asarray(x), D['box'], D['rcut'], skin=1.0, exclusions=D['excl_set'])
    ii, jj = jnp.asarray(ii), jnp.asarray(jj)
    kx, ky, kz = [jnp.asarray(a) for a in build_kvectors(D['box'], *D['klimit'])]
    pp = make_polarization_parts(D, kx, ky, kz, scf_max_iter=400, solver='pcg')
    comp = jax.jit(pp['compute'])
    e0, f0, mu0, st = comp(x, ii, jj)
    print(f"device {jax.devices()[0]}  rtaper {D['pol_data']['rtaper']} (rcut {D['rcut']}; taper "
          f"{'ON' if D['pol_data']['rtaper'] < D['rcut'] else 'off'})  E_pol {float(e0):.6f} kcal/mol  "
          f"SCF {int(st['niter'])} products", flush=True)
    rng = np.random.default_rng(args.seed)
    real = np.flatnonzero(~np.asarray(D['ldummy']) & (np.asarray(D['pol_data']['alpha_atom']) > 0))
    worst = 0.0
    xn = np.asarray(x); bx = np.asarray(D['box']); el = bx / 2; rc = float(D['rcut'])

    def crosses(a, k):
        """Does moving atom a by +-h along k carry a pair across rcut? (energy jump)"""
        out = []
        for s in (args.h, -args.h):
            xa = xn[a].copy(); xa[k] += s
            r = []
            for p in (xn[a], xa):
                d = xn - p
                d -= np.where(np.abs(d) > el, np.sign(d) * bx, 0.0)
                r.append(np.sqrt((d * d).sum(1)) <= rc)
            out.append(np.any(r[0] != r[1]))
        return any(out)

    nskip = 0
    for a in rng.choice(real, args.natoms, replace=False):
        for k in range(3):
            xp = x.at[a, k].add(args.h); xm = x.at[a, k].add(-args.h)
            ep = float(comp(xp, ii, jj)[0]); em = float(comp(xm, ii, jj)[0])
            fd = -(ep - em) / (2 * args.h)
            an = float(f0[a, k])
            rel = abs(fd - an) / max(abs(an), 1e-3)
            if crosses(a, k):
                nskip += 1
                note = '  (a pair crosses rcut within +-h: energy jump, not counted)'
            else:
                worst = max(worst, rel); note = ''
            print(f"  atom {a:5d} {'xyz'[k]}  analytic {an:+.6f}  -dE/dx {fd:+.6f}  rel diff {rel:.1e}{note}",
                  flush=True)
    if nskip:
        print(f"  {nskip} component(s) skipped: the hard cutoff at rcut makes E_pol jump there")
    print(f"max rel diff {worst:.1e}  ({'OK' if worst < 1e-3 else 'MISMATCH'})")
    return 0 if worst < 1e-3 else 1


if __name__ == '__main__':
    sys.exit(main())
