"""
main_scan.py — GOMD driver: on-device neighbor build + lax.scan loop + segsum.

Combines the three M4 fixes:
  Fix 1  on-device neighbor build (neighbor_ondevice)   — kills the ~4.5s host build
  Fix 2  whole trajectory in one lax.scan (integrator_scan) — no Python/host between steps
  Fix 3  fp32 + segment_sum pair kernel (pair_splined_segsum) — fixes the GPU scatter

Runs the trajectory in chunks of --print-every steps so it can log a (PE, KE)
sample between chunks; each chunk is one on-device scan. The neighbor list is
rebuilt INSIDE the scan every --rebuild-every steps via lax.cond.

Usage:
    python -m gomd.main_scan --input-dir ./md-input --ref-dir ./pkg \
        --steps 200 --print-every 50 --rebuild-every 100 --out md-output/md_scan.out
"""
from __future__ import annotations
import sys, os, time, argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', default='.')
    ap.add_argument('--ref-dir', required=True)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--print-every', type=int, default=50)
    ap.add_argument('--rebuild-every', type=int, default=100)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--polarizable', action='store_true',
                    help='Enable the full polarizable force field (M3): '
                         'induced-dipole field+SCF+energy+forces each step.')
    ap.add_argument('--out', default='md_scan.out')
    args = ap.parse_args()

    if args.cpu:
        os.environ['JAX_PLATFORMS'] = 'cpu'
    from jax import config as jc
    jc.update('jax_enable_x64', True)
    import jax, jax.numpy as jnp
    import numpy as np

    args.out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    _logf = open(args.out, 'w')
    def _emit(line):
        print(line); _logf.write(line + '\n'); _logf.flush()
    _emit(f'# GOMD scan log  devices={jax.devices()}')

    from gomd.io.assembly_loader import load_m2_system
    from gomd.forces.neighbor import build_neighbor_pairs
    from gomd.forces.neighbor_ondevice import (
        build_neighbor_ondevice, exclusions_to_flat)
    from gomd.forces.ewald import build_kvectors, self_energy
    from gomd.integrator_scan import make_scan_step, make_runner

    print("Loading system (reference setup)...")
    t0 = time.time()
    D = load_m2_system(args.input_dir, args.ref_dir)
    print(f"  loaded in {time.time()-t0:.1f}s  nat={D['nat']} ndummy={D['ndummy']}")

    kx, ky, kz = build_kvectors(D['box'], *D['klimit'])
    kx = jnp.asarray(kx); ky = jnp.asarray(ky); kz = jnp.asarray(kz)
    ewself = self_energy(D['q'], D['alpha'])

    def J(a): return jnp.asarray(a)
    def build_Dj(D):
        Dj = dict(D)
        for k in ['x','v','box','q','atomtype','mass','massinv','ldummy',
                  'typee','coefft','coeffft','coeffee','coefft_m','coeffft_m','zs',
                  'excl_qq','excl_i','excl_j']:
            Dj[k] = J(D[k])
        Dj['bonds']=J(D['conn']['bonds']); Dj['bends']=J(D['conn']['bends']); Dj['torts']=J(D['conn']['torts'])
        Dj['stretch']=J(D['ffb']['stretch']); Dj['bend']=J(D['ffb']['bend'])
        Dj['twist']=J(D['ffb']['twist']); Dj['nprms']=J(D['ffb']['nprms'])
        Dj['con'] = {'con_i':J(D['con']['con_i']),'con_j':J(D['con']['con_j']),
                     'd2':J(D['con']['d2']),'massred':J(D['con']['massred']),
                     'tol':D['con']['tol'],'nbcon':D['con']['nbcon'],
                     'constrain':D['con']['constrain']}
        Dj['dummy'] = {'jik':J(D['dummy']['jik']),'adummy':J(D['dummy']['adummy']),
                       'bdummy':J(D['dummy']['bdummy']),'cdummy':J(D['dummy']['cdummy']),
                       'ndummy':D['dummy']['ndummy']}
        # Fix 3: fp32 + segment_sum pair kernel
        Dj['use_segsum_pair'] = True
        Dj['coefft_m32'] = J(D['coefft_m']).astype(jnp.float32)
        Dj['coeffft_m32'] = J(D['coeffft_m']).astype(jnp.float32)
        Dj['zs32'] = J(D['zs']).astype(jnp.float32)
        return Dj
    Dj = build_Dj(D)

    # Size capacity from one host build (host build used ONCE, at setup only).
    xnp = np.asarray(D['x'])
    ii0, jj0 = build_neighbor_pairs(xnp, D['box'], D['rcut'], skin=1.0,
                                    exclusions=D['excl_set'])
    capacity = int(ii0.size * 1.25)
    _emit(f"neighbor pairs: {ii0.size}, padded capacity: {capacity}")

    rlist2 = float((D['rcut'] + 1.0) ** 2)
    excl_flat = exclusions_to_flat(D['excl_set'], D['nat'])

    # initial pairs (on device, no host loop hereafter)
    idx_i0, idx_j0 = build_neighbor_ondevice(
        Dj['x'], Dj['box'], rlist2, excl_flat,
        nat=D['nat'], capacity=capacity)

    polarization_fn = None
    if args.polarizable:
        if D.get('pol_data') is None:
            raise SystemExit("--polarizable set but this system is not "
                             "polarizable (no pol_data from loader).")
        from gomd.forces.polarization_full import make_polarization_fn
        polarization_fn = make_polarization_fn(D, kx, ky, kz)
        _emit("polarization: ENABLED (full induced-dipole SCF each step)")
    else:
        _emit("polarization: disabled (classical force field)")

    scan_body, force_fn = make_scan_step(
        Dj, kx, ky, kz, ewself, excl_flat, capacity,
        args.rebuild_every, rlist2, polarization_fn=polarization_fn)
    run = make_runner(scan_body)

    # initial forces
    x0 = Dj['x']
    x0, pe0, f0, _comps0, mu0 = force_fn(x0, idx_i0, idx_j0)
    f0.block_until_ready()
    label = "polarizable" if args.polarizable else "classical"
    _emit(f"Initial {label} PE: {float(pe0):.4f} kcal/mol")

    # dipole history for the ASPC predictor (zeros => cold start, as in the ref)
    if args.polarizable:
        mu1 = mu0; mu2 = jnp.zeros_like(mu0); mu3 = jnp.zeros_like(mu0)
    else:
        z = jnp.zeros((D['nat'], 3))
        mu1 = z; mu2 = z; mu3 = z

    _emit(f"{'step':>6} {'PE':>14} {'KE':>12} {'TE':>14} {'ms/step':>10}")
    x, v, f = x0, Dj['v'], f0
    idx_i, idx_j = idx_i0, idx_j0
    step = jnp.array(1)

    # warm-compile one chunk (excluded from timing)
    chunk = args.print_every
    (carry, hist) = run(x, v, f, idx_i, idx_j, step, mu1, mu2, mu3, nsteps=chunk)
    carry[0].block_until_ready()
    # NOTE: discard warm-up result; restart timing from the true initial state
    x, v, f = x0, Dj['v'], f0
    idx_i, idx_j = idx_i0, idx_j0
    step = jnp.array(1)
    mh1, mh2, mh3 = mu1, mu2, mu3

    t_start = time.time(); last = t_start
    done = 0
    while done < args.steps:
        n = min(chunk, args.steps - done)
        carry, hist = run(x, v, f, idx_i, idx_j, step, mh1, mh2, mh3, nsteps=n)
        x, v, f, idx_i, idx_j, step, mh1, mh2, mh3 = carry
        x.block_until_ready()
        now = time.time(); ms = (now - last) / n * 1000; last = now
        done += n
        pe = float(hist[0][-1]); ke = float(hist[1][-1]); te = pe + ke
        _emit(f"{done:>6} {pe:>14.4f} {ke:>12.4f} {te:>14.4f} {ms:>10.1f}")

    tot = time.time() - t_start
    _emit(f"=== done: {tot:.1f}s total, {tot/args.steps*1000:.1f} ms/step avg ===")
    _logf.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
