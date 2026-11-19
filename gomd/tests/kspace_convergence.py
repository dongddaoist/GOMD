"""
tests/kspace_convergence.py — how converged is the k-space sum on the slab cell?

    python -m gomd.tests.kspace_convergence --input-dir md-input3 --ref-dir pkg \
        --slab-height 205 [--gpu-timing] [--out md-output/kspace_convergence.txt]

The cell is prepared as in a run (fixed flat electrodes, slab box, planar term).
For each k-limit set the k-space part (forces/ewald2d_ref.recip3d, numpy, chunked)
is compared with a tightly converged reference (kmax 11/11/66 by default) at the
same alpha:
    dE      reciprocal energy difference, kcal/mol
    F       force error on the mobile atoms, rms / max, kcal/mol/A
    E-field field error at the polarizable sites (drives the SCF), rms / max, V/A
    phi_el  potential error at the electrode sites (what a constant-potential
            electrode sees), rms / max, V; compared up to the mean, since only
            differences enter
--gpu-timing also times GOMD's shared k-space pass (tables + energy + force +
qmu_grad, jitted) on the device for each set, or reports that it does not fit.
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np

KCAL_PER_EV = 23.0605


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', default='md-input3')
    ap.add_argument('--ref-dir', default='pkg')
    ap.add_argument('--lower', default='first:1792')
    ap.add_argument('--upper', default='last:1792')
    ap.add_argument('--slab-height', type=float, default=205.0)
    ap.add_argument('--sets', default='7,7,25;8,8,35;9,9,45;9,9,55;10,10,60')
    ap.add_argument('--reference', default='11,11,66')
    ap.add_argument('--alpha', type=float, default=None, help='default: mdrun.params')
    ap.add_argument('--gpu-timing', action='store_true')
    ap.add_argument('--out', default=None)
    args = ap.parse_args(argv)
    if not args.gpu_timing:
        os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    import jax
    jax.config.update('jax_enable_x64', True)
    from gomd.io.assembly_loader import load_m2_system
    from gomd.electrodes import prepare_electrodes
    from gomd.forces.ewald import build_kvectors
    from gomd.forces.ewald2d_ref import recip3d, COULOMB

    lines = []
    def emit(s=''):
        print(s, flush=True); lines.append(s)

    D = load_m2_system(args.input_dir, args.ref_dir)
    prepare_electrodes(D, args.lower, args.upper, geometry='flat', slab_height=args.slab_height,
                       rlist=D['rcut'] + 1.0, emit=lambda s: None)
    x = np.asarray(D['x']); q = np.asarray(D['q']); box = np.asarray(D['box'])
    alpha = float(args.alpha if args.alpha is not None else D['alpha'])
    frozen = D['frozen']; ld = np.asarray(D['ldummy'])
    mob = ~frozen & ~ld
    pol = np.asarray(D['pol_data']['alpha_atom']) > 0 if D.get('pol_data') else mob
    pts = np.full(len(q), np.inf)
    V_PER = COULOMB / KCAL_PER_EV                # e/A -> V
    emit(f"# k-space convergence, {args.input_dir}: {D['nat']} sites, box "
         + ' x '.join(f'{b:.2f}' for b in box) + f" A (slab, planar term), alpha {alpha}")
    emit(f"# real-space cutoff {D['rcut']} A: erfc(alpha rcut) = "
         f"{__import__('scipy.special', fromlist=['erfc']).erfc(alpha * D['rcut']):.1e} (unchanged by kmax)")

    def run(kset):
        kx, ky, kz = build_kvectors(box, *kset)
        t0 = time.time()
        U, phi, F, E = recip3d(x, q, pts, box, alpha, np.stack([kx, ky, kz], 1), weight=2.0,
                               planar=True, chunk=192)
        return kx.size, COULOMB * U, phi, COULOMB * F, E, time.time() - t0

    kref = tuple(int(t) for t in args.reference.split(','))
    nref, Uref, pref, Fref, Eref, tref = run(kref)
    emit(f"# reference kmax {'/'.join(map(str, kref))}: {nref} k-vectors ({tref:.0f} s on the CPU), "
         f"E_recip {Uref:.4f} kcal/mol; force rms {np.sqrt((Fref[mob] ** 2).mean()):.3f} kcal/mol/A")
    emit()
    emit(f"{'kmax':>10} {'nk':>6} {'exp(-kc^2/4a^2)':>15} {'dE kcal/mol':>12} "
         f"{'F rms':>8} {'F max':>8} {'Efld rms':>9} {'Efld max':>9} {'phi_el rms':>10} {'phi_el max':>10}"
         + (f" {'GPU ms':>7}" if args.gpu_timing else ''))
    emit(f"{'':>10} {'':>6} {'':>15} {'':>12} {'kcal/mol/A':>17} {'V/A':>19} {'V':>21}")
    for spec in args.sets.split(';'):
        kset = tuple(int(t) for t in spec.split(','))
        n, U, phi, F, E, _ = run(kset)
        kc = min(2 * np.pi * kset[a] / box[a] for a in range(3))
        dF = np.sqrt(((F - Fref)[mob] ** 2).sum(1))
        dE = np.sqrt(((E - Eref)[pol] ** 2).sum(1)) * V_PER
        dp = (phi - pref)[frozen]; dp = (dp - dp.mean()) * V_PER
        row = (f"{'/'.join(map(str, kset)):>10} {n:6d} {np.exp(-kc * kc / (4 * alpha * alpha)):15.1e} "
               f"{U - Uref:12.4f} {np.sqrt((dF ** 2).mean()):8.4f} {dF.max():8.4f} "
               f"{np.sqrt((dE ** 2).mean()):9.2e} {dE.max():9.2e} {np.sqrt((dp ** 2).mean()):10.2e} "
               f"{np.abs(dp).max():10.2e}")
        if args.gpu_timing:
            row += f" {gpu_ms(box, q, kset, alpha, x):>7}"
        emit(row)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w') as f:
            f.write('\n'.join(lines) + '\n')
    return 0


def gpu_ms(box, q, kset, alpha, x):
    import jax, jax.numpy as jnp
    from gomd.forces.ewald import build_kvectors
    from gomd.forces.recip_shared import make_recip
    try:
        kx, ky, kz = build_kvectors(box, *kset)
        rec = make_recip(box, q, kx, ky, kz, alpha, slab=True)
        mu = jnp.zeros((len(q), 3))
        @jax.jit
        def step(y):
            rt = rec['tables'](y)
            return rec['energy'](rt), rec['force'](rt), rec['qmu_grad'](rt, mu)
        xj = jnp.asarray(x)
        jax.block_until_ready(step(xj))
        t0 = time.perf_counter()
        for _ in range(5):
            jax.block_until_ready(step(xj))
        return f"{(time.perf_counter() - t0) / 5 * 1e3:.1f}"
    except Exception as exc:                       # out of device memory
        return 'OOM' if 'RESOURCE_EXHAUSTED' in str(exc) or 'memory' in str(exc).lower() else 'err'


if __name__ == '__main__':
    sys.exit(main())
