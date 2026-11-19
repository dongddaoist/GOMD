"""
tests/check_ewald2d.py — P1 checks of the slab electrostatics.

    python -m gomd.tests.check_ewald2d

A small neutral slab: two "electrodes" of two Gaussian layers each (6 x 6 sites,
eta 1.979 1/A, small random charges) around 40 random point charges, in a
12 x 13 A cell. All energies in e^2/A (Gaussian units).

  1. exact 2D Ewald (forces/ewald2d_ref.ewald2d_exact) is independent of alpha:
     energy, site potentials dU/dq and forces at alpha 0.3 / 0.4 / 0.5, each with
     real and k-space parts converged far below the tolerance;
  2. analytic limits: an isolated Gaussian pair (+1/-1, 3 A apart in z) and a
     point-Gaussian pair in a 300 A cell match q^2 [eta/sqrt(2 pi) ...] - erf(...)/d
     to the lateral-image dipole energy (~1e-6);
  3. the trapezoid form (3D sum in a box with vacuum + planar dipole term,
     ewald3dc) converges to the exact 2D sum as the vacuum grows, and without
     the planar term it does not;
  4. GOMD's shared k-space pass (forces/recip_shared, slab=True, GPU or CPU)
     equals the numpy k-space reference on the same k-vectors: energy, force,
     field; points only (the pass has no Gaussian sites yet);
  5. the pass's force is -dE/dx (jax.grad) and its qmu_grad is
     d/dx sum_i mu_i . E_i at fixed mu (jax.grad), planar term included.
Exit code 0 when all pass.
"""
from __future__ import annotations
import sys, time
import numpy as np


def make_system(seed=0):
    rng = np.random.default_rng(seed)
    Lx, Ly = 12.0, 13.0
    gx, gy = np.meshgrid(np.arange(6) * Lx / 6, np.arange(6) * Ly / 6, indexing='ij')
    lay = np.stack([gx.ravel(), gy.ravel()], 1)
    xs, eta, q = [], [], []
    for z0, shift in ((0.0, 0.0), (3.4, 1.0), (24.0, 0.5), (27.4, 1.5)):
        p = np.column_stack([lay + shift, np.full(len(lay), z0)])
        xs.append(p); eta.append(np.full(len(lay), 1.979)); q.append(rng.normal(0, 0.05, len(lay)))
    npnt = 40
    p = np.column_stack([rng.uniform(0, Lx, npnt), rng.uniform(0, Ly, npnt), rng.uniform(6.5, 21.0, npnt)])
    xs.append(p); eta.append(np.full(npnt, np.inf)); q.append(rng.choice([-1.0, 1.0], npnt) * rng.uniform(0.3, 1.0, npnt))
    x = np.concatenate(xs); eta = np.concatenate(eta); q = np.concatenate(q)
    x[:, 0] %= Lx; x[:, 1] %= Ly
    q[-1] -= q.sum()                                    # neutral
    return x, q, eta, (Lx, Ly)


def main():
    from gomd.forces import ewald2d_ref as R
    ok = True

    def check(name, cond, detail=''):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(': ' + detail) if detail else ''}")

    x, q, eta, Lxy = make_system()
    print(f"system: {len(q)} sites ({int(np.isinf(eta).sum())} point, {int((~np.isinf(eta)).sum())} "
          f"Gaussian), cell {Lxy[0]} x {Lxy[1]} A, z {x[:, 2].min():.1f}..{x[:, 2].max():.1f} A")

    # ---- 1. alpha independence of the exact 2D sum ----
    t0 = time.time()
    res = {}
    for a in (0.3, 0.4, 0.5):
        kmax = (int(np.ceil(11 * a * Lxy[0] / (2 * np.pi))), int(np.ceil(11 * a * Lxy[1] / (2 * np.pi))))
        res[a] = R.ewald2d_exact(x, q, eta, Lxy, a, rcut=7.0 / a, kmax=kmax)
    U0, p0, F0 = res[0.4]
    dU = max(abs(res[a][0] - U0) for a in res)
    dp = max(np.abs(res[a][1] - p0).max() for a in res)
    dF = max(np.abs(res[a][2] - F0).max() for a in res)
    print(f"\n1. exact 2D Ewald at alpha 0.3/0.4/0.5 ({time.time() - t0:.0f} s): U = {U0:.10f} e^2/A")
    check('energy independent of alpha', dU < 1e-9, f"max diff {dU:.1e} e^2/A")
    check('site potentials independent of alpha', dp < 1e-9, f"max diff {dp:.1e} e/A")
    check('forces independent of alpha', dF < 1e-9, f"max diff {dF:.1e} e^2/A^2 (max |F| {np.abs(F0).max():.3f})")
    check('net force zero', np.abs(F0.sum(0)).max() < 1e-9, f"{np.abs(F0.sum(0)).max():.1e}")

    # ---- 2. analytic limits in a wide cell ----
    eta0 = 1.979; d = 3.0; L = 300.0; a = 0.3
    kmax = (int(np.ceil(11 * a * L / (2 * np.pi))),) * 2
    xp = np.array([[10.0, 10.0, 0.0], [10.0, 10.0, d]])
    Ugg = R.ewald2d_exact(xp, np.array([1.0, -1.0]), np.array([eta0, eta0]), (L, L), a, 7 / a, kmax)[0]
    Upg = R.ewald2d_exact(xp, np.array([1.0, -1.0]), np.array([np.inf, eta0]), (L, L), a, 7 / a, kmax)[0]
    from scipy.special import erf
    ref_gg = 2 * eta0 / np.sqrt(2 * np.pi) - erf(eta0 * d / np.sqrt(2)) / d
    ref_pg = eta0 / np.sqrt(2 * np.pi) - erf(eta0 * d) / d
    img = 9.03 * d * d / (2 * L ** 3)                  # lateral images of the z dipole (~1.5e-6)
    print(f"\n2. isolated pairs, {L:g} A cell (lateral-image dipole energy ~{img:.1e})")
    check('Gaussian-Gaussian pair', abs(Ugg - ref_gg - img) < 3e-7,
          f"U {Ugg:.9f}, analytic {ref_gg:.9f}, diff {Ugg - ref_gg:+.2e}")
    check('point-Gaussian pair', abs(Upg - ref_pg - img) < 3e-7,
          f"U {Upg:.9f}, analytic {ref_pg:.9f}, diff {Upg - ref_pg:+.2e}")

    # ---- 3. trapezoid (EW3DC) vs exact ----
    a = 0.4; rc = 7.0 / a
    thick = x[:, 2].max() - x[:, 2].min()
    print(f"\n3. 3D sum + planar term vs exact 2D (alpha {a}; slab {thick:.1f} A, Lx {Lxy[0]} A)")
    errs = []
    for gap in (18.0, 24.0, 36.0, 48.0):
        Lz = thick + gap
        xz = x.copy(); xz[:, 2] += gap / 2 - x[:, 2].min()
        kmax = tuple(int(np.ceil(11 * a * L / (2 * np.pi))) for L in (Lxy[0], Lxy[1], Lz))
        U, p, F = R.ewald3dc(xz, q, eta, (Lxy[0], Lxy[1], Lz), a, rc, kmax)
        e = (abs(U - U0), np.abs(p - p0 - (p - p0).mean()).max(), np.abs(F - F0).max())
        errs.append(e)
        print(f"     gap {gap:4.0f} A: |dU| {e[0]:.1e}  |dphi| {e[1]:.1e}  |dF| {e[2]:.1e}"
              f"   (exp(-2 pi gap/Lx) = {np.exp(-2 * np.pi * gap / Lxy[0]):.0e})")
    check('converges to the exact 2D sum', errs[-1][0] < 1e-8 and errs[-1][2] < 1e-8
          and errs[0][2] > errs[-1][2], f"gap 48 A: dU {errs[-1][0]:.1e}, dF {errs[-1][2]:.1e}")
    Unp, _, Fnp = R.ewald3dc(xz, q, eta, (Lxy[0], Lxy[1], Lz), a, rc, kmax, planar=False)
    check('the planar term is needed', abs(Unp - U0) > 1e3 * errs[-1][0] + 1e-6,
          f"without it at gap 48 A: dU {abs(Unp - U0):.1e}, dF {np.abs(Fnp - F0).max():.1e}")
    print("     (the potential is compared up to a constant: a 3D sum fixes its zero differently)")

    # ---- 4./5. GOMD shared k-space pass (points only) ----
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from gomd.forces.ewald import build_kvectors
    from gomd.forces.recip_shared import make_recip, COULOMB_CONST
    pts = np.isinf(eta)
    xpq = xz[pts]; qp = q[pts] - q[pts].mean()          # neutral point set
    box = np.array([Lxy[0], Lxy[1], Lz])
    kx, ky, kz = build_kvectors(box, 9, 9, 40)
    rec = make_recip(box, qp, kx, ky, kz, a, slab=True)
    rt = rec['tables'](jnp.asarray(xpq))
    Ug = float(rec['energy'](rt)); Fg = np.asarray(rec['force'](rt)); Eg = np.asarray(rt['field'])
    Ur, _, Fr, Er = R.recip3d(xpq, qp, np.full(len(qp), np.inf), box, a,
                             np.stack([kx, ky, kz], 1), weight=2.0, planar=True)
    print(f"\n4. GOMD recip_shared (slab=True, {rec['nk']} k-vectors, {jax.devices()[0].platform}) "
          f"vs numpy, {len(qp)} point charges")
    check('energy', abs(Ug - COULOMB_CONST * Ur) < 1e-9 * max(1.0, abs(Ug)),
          f"{Ug:.10f} vs {COULOMB_CONST * Ur:.10f} kcal/mol")
    check('force', np.abs(Fg - COULOMB_CONST * Fr).max() < 1e-9,
          f"max diff {np.abs(Fg - COULOMB_CONST * Fr).max():.1e} kcal/mol/A")
    check('field', np.abs(Eg - Er).max() < 1e-12, f"max diff {np.abs(Eg - Er).max():.1e} e/A^2")

    xj = jnp.asarray(xpq)
    gE = -np.asarray(jax.grad(lambda y: rec['energy'](rec['tables'](y)))(xj))
    rng = np.random.default_rng(1)
    mu = jnp.asarray(rng.normal(0, 0.3, (len(qp), 3)))
    gW = np.asarray(jax.grad(lambda y: jnp.sum(mu * rec['tables'](y)['field']))(xj))
    gQ = np.asarray(rec['qmu_grad'](rt, mu))
    print("\n5. gradients (planar term included)")
    check('force = -dE/dx', np.abs(gE - Fg).max() < 1e-9 * max(1.0, np.abs(Fg).max()),
          f"max diff {np.abs(gE - Fg).max():.1e}")
    check('qmu_grad = d/dx sum mu.E', np.abs(gW - gQ).max() < 1e-10,
          f"max diff {np.abs(gW - gQ).max():.1e}")
    print('\nALL PASS' if ok else '\nSOME CHECKS FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
