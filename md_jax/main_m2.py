"""
main_m2.py — Full M2 MD driver (classical APPLE&P, no polarization).

Runs real, stable MD on the 8716-atom system using the assembled JAX engine.
Unlike M1 (Coulomb-only, blows up), this has the full classical force field:
vdW + real-space elec (splines) + bonded + Ewald, with SHAKE + dummy atoms.

Usage:
    python -m wmi_md_jax.main_m2 --input-dir <dir> --ref-dir <ref_pkg> \
        --steps 50 --print-every 10 --rebuild-every 10
"""
from __future__ import annotations
import sys, os, time, argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', default='.')
    ap.add_argument('--ref-dir', required=True,
                    help="Path to the reference Python package (for setup + parsers).")
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--print-every', type=int, default=10)
    ap.add_argument('--rebuild-every', type=int, default=10,
                    help="Rebuild neighbor list every N steps (host).")
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--fp32', action='store_true',
                    help='Use mixed precision (fp32 pair force).')
    ap.add_argument('--out', default='md_jax.out',
                    help='File to write the per-step MD log to.')
    args = ap.parse_args()

    if args.cpu:
        os.environ['JAX_PLATFORMS'] = 'cpu'
    from jax import config as jc
    jc.update('jax_enable_x64', True)
    import jax, jax.numpy as jnp
    import numpy as np
    print(f"JAX devices: {jax.devices()}  backend: {jax.default_backend()}")

    import os as _os
    args.out = _os.path.abspath(args.out)
    _os.makedirs(_os.path.dirname(args.out) or '.', exist_ok=True)
    _logf = open(args.out, 'w')
    def _emit(line):
        print(line)
        _logf.write(line + '\n'); _logf.flush()
    _emit(f'# wmi_md_jax M2 MD log  devices={jax.devices()}')

    from wmi_md_jax.io.assembly_loader import load_m2_system
    from wmi_md_jax.forces.neighbor import build_neighbor_pairs
    from wmi_md_jax.forces.ewald import build_kvectors, self_energy
    from wmi_md_jax.integrator_m2 import make_force_fn, vv_step

    print("Loading system (reference setup)...")
    t0 = time.time()
    D = load_m2_system(args.input_dir, args.ref_dir)
    print(f"  loaded in {time.time()-t0:.1f}s  nat={D['nat']} ndummy={D['ndummy']}")

    # k-vectors + self-energy (static for NVE)
    kx, ky, kz = build_kvectors(D['box'], *D['klimit'])
    ewself = self_energy(D['q'], D['alpha'])

    def J(a): return jnp.asarray(a)
    def build_Dj(D):
        Dj = dict(D)
        for k in ['x','v','box','q','atomtype','mass','massinv','ldummy',
                  'typee','coefft','coeffft','coeffee','coefft_m','coeffft_m','zs','excl_qq','excl_i','excl_j']:
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
        if args.fp32:
            Dj['use_fp32_pair'] = True
            Dj['coefft_m32'] = J(D['coefft_m']).astype(jnp.float32)
            Dj['coeffft_m32'] = J(D['coeffft_m']).astype(jnp.float32)
            Dj['zs32'] = J(D['zs']).astype(jnp.float32)
        return Dj
    Dj = build_Dj(D)

    # Fixed-capacity neighbor list: pad pairs to a constant length so the
    # jit'd force fn compiles ONCE (not on every rebuild). Capacity is sized
    # from the first build with headroom for fluctuations.
    from wmi_md_jax.forces.neighbor import build_neighbor_pairs_padded
    xnp = np.asarray(D['x'])
    ii0, jj0 = build_neighbor_pairs(xnp, D['box'], D['rcut'], skin=1.0,
                                    exclusions=D['excl_set'])
    capacity = int(ii0.size * 1.25)        # 25% headroom
    _emit(f"neighbor pairs: {ii0.size}, padded capacity: {capacity}")

    def rebuild(xnp):
        ii, jj, nreal = build_neighbor_pairs_padded(
            xnp, D['box'], D['rcut'], skin=1.0,
            exclusions=D['excl_set'], capacity=capacity)
        return J(ii), J(jj)

    idx_i, idx_j = rebuild(xnp)
    # Build the force fn ONCE; on rebuild we only swap the pair arrays (same
    # shape -> no recompile).
    force_fn = make_force_fn(Dj, kx, ky, kz, ewself)

    # initial forces
    x0 = Dj['x']
    x0, pe0, f0, comps0 = force_fn(x0, idx_i, idx_j)
    f0.block_until_ready()
    _emit(f"Initial classical PE: {float(pe0):.4f} kcal/mol")

    state = {'x': x0, 'v': Dj['v'], 'f': f0, 'pe': pe0, 'ke': 0.0}

    _emit(f"{'step':>6} {'PE':>14} {'KE':>12} {'TE':>14} {'ms/step':>10}")
    t_start = time.time(); last = t_start
    for i in range(1, args.steps + 1):
        # periodic neighbor rebuild (host) — same padded shape, no recompile
        if i % args.rebuild_every == 0:
            xnp = np.asarray(state['x'])
            idx_i, idx_j = rebuild(xnp)   # same padded shape -> no recompile
        state = vv_step(state, Dj, force_fn, idx_i, idx_j)
        if i % args.print_every == 0:
            state['x'].block_until_ready()
            now = time.time(); ms = (now-last)/args.print_every*1000; last = now
            pe = float(state['pe']); ke = float(state['ke']); te = pe+ke
            _emit(f"{i:>6} {pe:>14.4f} {ke:>12.4f} {te:>14.4f} {ms:>10.1f}")

    state['x'].block_until_ready()
    tot = time.time() - t_start
    _emit(f"=== done: {tot:.1f}s total, {tot/args.steps*1000:.1f} ms/step avg ===")
    _logf.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
