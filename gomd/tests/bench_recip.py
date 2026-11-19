"""
Shared reciprocal-space pass vs the three separate ones: agreement + timing.

    python -m gomd.tests.bench_recip --input-dir ./md-input --ref-dir ./pkg

  energy   recip_shared.energy        vs ewald.reciprocal_energy_force (energy)
  force    recip_shared.force         vs ewald.reciprocal_energy_force (force)
  field    tables()['field']          vs pol_field._recip_field
  Q-mu     recip_shared.qmu_grad      vs jax.grad of sum(mu * _recip_field) (autodiff)
"""
from __future__ import annotations
import argparse, os, statistics, sys, time


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', default='./md-input')
    ap.add_argument('--ref-dir', default='./pkg')
    ap.add_argument('--reps', type=int, default=10)
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args(argv)
    if args.cpu:
        os.environ['JAX_PLATFORMS'] = 'cpu'
    from jax import config as jc
    jc.update('jax_enable_x64', True)
    import jax, jax.numpy as jnp
    import numpy as np
    from gomd.io.assembly_loader import load_m2_system
    from gomd.forces.ewald import build_kvectors, reciprocal_energy_force
    from gomd.forces.pol_field import _recip_field
    from gomd.forces.recip_shared import make_recip

    def block(o):
        for leaf in jax.tree_util.tree_leaves(o):
            leaf.block_until_ready()
        return o

    def bench(fn, *a):
        out = block(fn(*a)); ts = []
        for _ in range(args.reps):
            t0 = time.perf_counter(); block(fn(*a)); ts.append(time.perf_counter() - t0)
        return out, statistics.median(ts) * 1e3

    D = load_m2_system(args.input_dir, args.ref_dir)
    x = jnp.asarray(D['x']); q = jnp.asarray(D['q']); box = jnp.asarray(D['box'])
    kxn, kyn, kzn = build_kvectors(D['box'], *D['klimit'])
    kx, ky, kz = jnp.asarray(kxn), jnp.asarray(kyn), jnp.asarray(kzn)
    alpha = float(D['alpha'])
    al = jnp.asarray(D['pol_data']['alpha_atom'])
    print(f"device {jax.devices()[0]}  nat {D['nat']}  nk {kxn.size}", flush=True)

    R = make_recip(D['box'], D['q'], kxn, kyn, kzn, alpha)
    (e_ref, f_ref), t_ew = bench(jax.jit(lambda y: reciprocal_energy_force(y, q, box, kx, ky, kz, alpha)), x)
    E_ref, t_fd = bench(jax.jit(lambda y: _recip_field(y, q, box, kx, ky, kz, alpha)), x)
    mu = al[:, None] * E_ref * 30.0                        # a dipole field of realistic size
    g_ref, t_ad = bench(jax.jit(lambda y, m: jax.grad(
        lambda z: jnp.sum(jax.lax.stop_gradient(m) * _recip_field(z, q, box, kx, ky, kz, alpha)))(y)), x, mu)

    rt, t_tab = bench(jax.jit(R['tables']), x)
    e_new, _ = bench(jax.jit(R['energy']), rt)
    f_new, _ = bench(jax.jit(R['force']), rt)
    g_new, t_qm = bench(jax.jit(R['qmu_grad']), rt, mu)

    def rel(a, b):
        return float(jnp.max(jnp.abs(a - b)) / jnp.maximum(jnp.max(jnp.abs(b)), 1e-300))
    checks = {
        'energy': abs(float(e_new) - float(e_ref)) / abs(float(e_ref)),
        'force': rel(f_new, f_ref),
        'field': rel(rt['field'], E_ref),
        'Q-mu force': rel(g_new, g_ref),
    }
    ok = True
    for k, v in checks.items():
        good = v < 1e-10
        ok &= good
        print(f"  {k:<11} max rel diff {v:.2e}  {'ok' if good else 'MISMATCH'}")
    sep = t_ew + t_fd + t_ad
    shared = t_tab + t_qm
    print(f"  separate: Ewald {t_ew:.2f} + field {t_fd:.2f} + Q-mu autodiff {t_ad:.2f} = {sep:.2f} ms")
    print(f"  shared:   tables(+field) {t_tab:.2f} + Q-mu {t_qm:.2f} = {shared:.2f} ms  "
          f"({sep / shared:.1f}x)  [energy/force from the tables: negligible]")
    print('ALL OK' if ok else 'MISMATCH')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
