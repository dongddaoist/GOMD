"""
selftest.py — GOMD self-test, building its reference on the fly from pkg/.

    python -m gomd.selftest --input-dir ./md-input --ref-dir ./pkg

Checks (each prints PASS / FAIL):
  1. every classical force term is the exact negative gradient of its energy
     (analytic F vs JAX autodiff, fp64) — matching the reference alone did not
     catch the torsion sign bug, this does
  2. lone-pair force redistribution = exact chain rule of the site placement
  3. torsion energy and force: GOMD vs the reference pkg
  4. Nose-Hoover chain half step: GOMD vs the reference nhc_integrate
  5. MLIP plumbing: mock MLIP server round trip, forces vs finite differences
     of its energy, eV -> kcal/mol conversion, zero force on lone-pair sites
  6. SCF solvers (polarizable force fields): PCG and Jacobi reach the same
     dipoles at a tight tolerance, and PCG warm-started from a 1%-biased guess
     returns to the converged energy (the case that defeats the Jacobi
     step-change test)
  7. on-device neighbor build = host cell-list build (pairs within fp32
     rounding of rlist excepted), (i,j)-ordered, padded with (0,0)
  8. the shared k-space pass (recip_shared) = the separate ones: classical
     reciprocal energy and force, SCF field, and the k-space Q-mu force in
     closed form vs autodiff

Runs on the CPU by default (it is a correctness test, not a benchmark);
--gpu runs it on the default JAX backend instead.
Exit code 0 only if every check passes.
"""
from __future__ import annotations
import argparse, copy, os, sys, time


def main(argv=None):
    ap = argparse.ArgumentParser(description='GOMD self-test')
    ap.add_argument('--input-dir', default='./md-input')
    ap.add_argument('--ref-dir', default='./pkg')
    ap.add_argument('--gpu', action='store_true', help='use the default JAX backend, not the CPU')
    ap.add_argument('--skip-mlip', action='store_true')
    ap.add_argument('--skip-pol', action='store_true', help='skip the SCF solver check (6)')
    args = ap.parse_args(argv)
    args.input_dir = os.path.abspath(args.input_dir)
    args.ref_dir = os.path.abspath(args.ref_dir)
    if not args.gpu:
        os.environ['JAX_PLATFORMS'] = 'cpu'
    from jax import config
    config.update('jax_enable_x64', True)
    import jax, jax.numpy as jnp
    import numpy as np

    from gomd.io.assembly_loader import load_m2_system
    from gomd.forces.neighbor import build_neighbor_pairs
    from gomd.forces.pair_splined import splined_pair_energy_force
    from gomd.forces.bonded import bond_energy_force, bend_energy_force, torsion_energy_force
    from gomd.forces.ewald import build_kvectors, reciprocal_energy_force, exclusion_correction
    from gomd.forces.dummy import update_dummy_coords, dummy_force_redistribution
    from gomd import thermostat as T

    results = []
    def report(name, ok, detail):
        results.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    t0 = time.time()
    D = load_m2_system(args.input_dir, args.ref_dir)
    print(f"system loaded in {time.time()-t0:.0f}s: nat={D['nat']} ndummy={D['ndummy']}  "
          f"jax {jax.__version__} on {jax.devices()[0].platform}", flush=True)
    J = jnp.asarray
    nat = D['nat']; box = J(D['box'])
    dm = D['dummy']; jik, ad, bd, cd = J(dm['jik']), J(dm['adummy']), J(dm['bdummy']), J(dm['cdummy'])
    x = update_dummy_coords(J(D['x']), jik, ad, bd, cd, box, nat)
    real = ~np.asarray(D['ldummy'])
    ii, jj = build_neighbor_pairs(np.asarray(x), D['box'], D['rcut'], skin=1.0, exclusions=D['excl_set'])
    ii, jj = J(ii), J(jj)
    kx, ky, kz = [J(a) for a in build_kvectors(D['box'], *D['klimit'])]
    rc2 = D['rcut'] ** 2

    # ---- 1. analytic force vs autodiff, per term ----
    terms = {
        'pair (vdW + real-space elec)': lambda y: splined_pair_energy_force(
            y, ii, jj, J(D['atomtype']), J(D['typee']), J(D['coefft_m']), J(D['coeffft_m']),
            J(D['zs']), D['deltainv'], rc2, box, nat),
        'bond': lambda y: bond_energy_force(y, J(D['conn']['bonds']), J(D['ffb']['stretch']), box, nat),
        'bend': lambda y: bend_energy_force(y, J(D['conn']['bends']), J(D['ffb']['bend']), box, nat),
        'torsion': lambda y: torsion_energy_force(y, J(D['conn']['torts']), J(D['ffb']['twist']),
                                                  J(D['ffb']['nprms']), box, nat, D['max_torsion_terms']),
        'Ewald reciprocal': lambda y: reciprocal_energy_force(y, J(D['q']), box, kx, ky, kz, D['alpha']),
        'Ewald exclusion correction': lambda y: exclusion_correction(
            y, J(D['excl_qq']), J(D['excl_i']), J(D['excl_j']), box, D['alpha'], nat),
    }
    # The exclusion-correction energy uses the reference's Abramowitz-Stegun erfc
    # polynomial (|error| ~1e-7) while its force uses the exact derivative, so
    # it agrees with -grad E to ~1e-5 only; every other term is exact.
    limit = {'Ewald exclusion correction': 1e-4}
    f_sites = jnp.zeros((nat, 3))
    for name, fn in terms.items():
        e, f = fn(x)
        g = jax.grad(lambda y: fn(y)[0])(x)
        rel = float(jnp.max(jnp.abs(f + g)) / jnp.maximum(jnp.max(jnp.abs(g)), 1e-300))
        f_sites = f_sites + f
        tol = limit.get(name, 1e-8)
        report(f"force = -dE/dx, {name}", rel < tol, f"max rel diff {rel:.1e} (limit {tol:g})")

    # ---- 2. lone-pair redistribution vs chain rule ----
    place = lambda y: update_dummy_coords(y, jik, ad, bd, cd, box, nat)
    exact = -jax.grad(lambda y: -jnp.sum(jax.lax.stop_gradient(f_sites) * place(y)))(x)
    red = dummy_force_redistribution(f_sites, x, jik, ad, bd, cd, box, nat)
    rel = float(jnp.max(jnp.abs((red - exact)[real])) / jnp.max(jnp.abs(exact[real])))
    report("lone-pair force redistribution = chain rule", rel < 1e-10, f"max rel diff {rel:.1e}")

    # ---- 3. torsion vs reference pkg ----
    import forces as ref_forces                      # pkg/ (on sys.path via the loader)
    torts = np.asarray(D['conn']['torts'])
    e_g, f_g = terms['torsion'](x)
    e_r, f_r = ref_forces.compute_torsion_forces(
        np.ascontiguousarray(np.asarray(x).T), torts, int(torts.shape[1]),
        np.asarray(D['ffb']['twist']), np.asarray(D['ffb']['nprms']), np.asarray(D['box']),
        np.asarray(D['box']) * 0.5, np.zeros((3, nat)))
    rel = float(np.max(np.abs(f_r.T - np.asarray(f_g))) / np.max(np.abs(np.asarray(f_g))))
    de = abs(float(e_g) - float(e_r))
    report("torsion GOMD vs reference", rel < 1e-10 and de < 1e-8,
           f"force max rel diff {rel:.1e}, energy diff {de:.1e} kcal/mol")

    # ---- 4. NHC half step vs reference ----
    import input_parser, integrator as RI
    import main as refmain
    cwd = os.getcwd()
    try:
        os.chdir(args.input_dir)
        st0 = input_parser.load_simulation(args.input_dir)
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            refmain.setup_system(st0)
    finally:
        os.chdir(cwd)
    worst = 0.0
    for nnos in (1, 3):
        st = copy.deepcopy(st0)
        st.params.nnos = nnos
        for a in (st.thermostat.xlogs, st.thermostat.vlogs, st.thermostat.glogs):
            a[1:] = 0.0
        if nnos == 3:
            st.thermostat.vlogs[1] = 3e-3; st.thermostat.vlogs[2] = -2e-3
        RI.init_thermostat(st); RI.init_integration(st)
        P = T.make_nhc_params(st.Tset, st.ndof, nnos, st.params.qfreq, st.params.delt)
        mass = J(st.atoms.mass); nd = J(~st.atoms.ldummy)[:, None]
        v = J(st.atoms.v.T.copy())
        s = tuple(J(a[:nnos].copy()) for a in (st.thermostat.xlogs, st.thermostat.vlogs, st.thermostat.glogs))
        half = jax.jit(lambda v, s: T.nhc_half(v, mass, nd, s, P))
        m = ~st.atoms.ldummy
        for _ in range(3):
            RI.nhc_integrate(st)
            v, s = half(v, s)
            vr = st.atoms.v.T
            worst = max(worst, float(np.max(np.abs(np.asarray(v)[m] - vr[m])) / np.max(np.abs(vr[m]))))
            for a, b in zip(s, (st.thermostat.xlogs, st.thermostat.vlogs)):
                b = b[:nnos]
                worst = max(worst, float(np.max(np.abs(np.asarray(a) - b) / np.maximum(np.abs(b), 1e-300))))
    report("Nose-Hoover chain half step vs reference (chain 1 and 3)", worst < 1e-12,
           f"max rel diff {worst:.1e}")

    # ---- 5. MLIP plumbing (mock server) ----
    if not args.skip_mlip:
        from gomd.mlip.client import MLIPProvider, EV_TO_KCAL
        from gomd.mlip.elements import numbers_from_masses, formula
        numbers = numbers_from_masses(D['mass'], D['ldummy'])
        xs = np.asarray(x)
        p = MLIPProvider.start('mock', numbers, D['box'], xs)
        rng = np.random.default_rng(1)
        xt = xs.copy(); xt[real] += rng.normal(0, 0.02, (int(real.sum()), 3))
        e, f = p(xt)
        h = 1e-5; worst = 0.0
        for i in rng.choice(np.flatnonzero(real), 6, replace=False):
            for k in range(3):
                xp = xt.copy(); xp[i, k] += h; xm = xt.copy(); xm[i, k] -= h
                fd = -(p(xp)[0] - p(xm)[0]) / (2 * h)
                worst = max(worst, abs(fd - f[i, k]) / max(abs(f[i, k]), 1e-6))
        lp_zero = bool(np.all(f[~real] == 0))
        p.close()
        report("MLIP server round trip (mock): force vs finite difference",
               worst < 1e-5 and lp_zero,
               f"max rel diff {worst:.1e}; lone-pair sites zero force: {lp_zero}; "
               f"elements {formula(numbers)}")

    # ---- 7. on-device neighbor build vs the host cell-list build ----
    from gomd.forces.neighbor_ondevice import exclusions_to_flat, build_neighbor_ondevice_count
    rl = D['rcut'] + 1.0
    cap = int(ii.shape[0] * 1.25)
    di, dj, dn = build_neighbor_ondevice_count(x, box, rl * rl, exclusions_to_flat(D['excl_set'], nat),
                                               nat=nat, capacity=cap)
    dn = int(dn)
    kd = np.asarray(di[:dn]).astype(np.int64) * nat + np.asarray(dj[:dn])
    kh = np.asarray(ii).astype(np.int64) * nat + np.asarray(jj)
    diff = np.setxor1d(kd, kh)
    xs = np.asarray(x)
    dv = xs[diff % nat] - xs[diff // nat]
    dv -= D['box'] * np.round(dv / D['box'])
    edge = np.abs(np.sqrt(np.sum(dv * dv, axis=1)) - rl)
    pad = bool(np.all(np.asarray(di[dn:]) == 0) and np.all(np.asarray(dj[dn:]) == 0))
    report("on-device neighbor build = host build", bool(np.all(edge < 1e-4)) and pad and
           bool(np.all(np.diff(kd) > 0)),
           f"{dn} vs {kh.size} pairs; {diff.size} differ, all within {edge.max() if diff.size else 0:.1e} A "
           f"of rlist (fp32 test); (i,j)-ordered, padding (0,0): {pad}")

    # ---- 8. shared k-space pass vs the separate ones ----
    from gomd.forces.recip_shared import make_recip
    from gomd.forces.pol_field import _recip_field
    R = make_recip(D['box'], D['q'], kx, ky, kz, D['alpha'])
    qj = J(D['q'])
    rt = R['tables'](x)
    e_ref, f_ref = reciprocal_energy_force(x, qj, box, kx, ky, kz, D['alpha'])
    E_ref = _recip_field(x, qj, box, kx, ky, kz, D['alpha'])
    mu_t = J(np.random.default_rng(3).normal(0.0, 0.05, (nat, 3)))
    g_ref = jax.grad(lambda y: jnp.sum(mu_t * _recip_field(y, qj, box, kx, ky, kz, D['alpha'])))(x)
    relm = lambda a, b: float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(b)))
    d = {'energy': abs(float(R['energy'](rt)) - float(e_ref)) / abs(float(e_ref)),
         'force': relm(R['force'](rt), f_ref), 'field': relm(rt['field'], E_ref),
         'Q-mu force': relm(R['qmu_grad'](rt, mu_t), g_ref)}
    report("shared k-space pass = separate passes", max(d.values()) < 1e-10,
           ', '.join(f"{k} {v:.1e}" for k, v in d.items()))

    # ---- 6. SCF solvers: PCG vs Jacobi ----
    if not args.skip_pol and D.get('pol_data') is not None:
        from gomd.forces.polarization_full import make_polarization_parts
        from gomd.forces.polarization import solve_scf, solve_scf_pcg, COULOMB_CONST
        pp = make_polarization_parts(D, kx, ky, kz, scf_max_iter=200, solver='pcg')
        G = jax.jit(pp['geometry'])(x, ii, jj)
        Ef = jax.jit(pp['field'])(x, ii, jj)
        al, arf = pp['alpha_atom'], pp['arf']
        fac = (G['rx'], G['ry'], G['rz'], G['tfact'], G['three_zf_l5'])
        e_std = lambda m: float(-0.5 * COULOMB_CONST * jnp.sum(Ef * m))
        mu_j, n_j = solve_scf(Ef, al, ii, jj, *fac, jnp.zeros((nat, 3)), arf, 1e-24, nat, 200, 0)
        mu_p, n_p, _ = solve_scf_pcg(Ef, al, ii, jj, *fac, None, arf, 1e-24, nat, 200, 0)
        scale = float(jnp.max(jnp.abs(mu_j)))
        dmu = float(jnp.max(jnp.abs(mu_p - mu_j))) / scale
        e_ref = e_std(mu_j)
        de = abs(e_std(mu_p) - e_ref)
        report("SCF: PCG = Jacobi at tight tolerance", dmu < 1e-8 and de < 1e-6,
               f"max rel dipole diff {dmu:.1e}, E_pol diff {de:.1e} kcal/mol "
               f"(E_pol {e_ref:.4f}; {int(n_p)} vs {int(n_j)} T.mu products)")
        mu_b = 1.01 * mu_j
        mw, nw, rw = solve_scf_pcg(Ef, al, ii, jj, *fac, mu_b, arf, 1e-8, nat, 200, 0)
        e_w = e_std(mw) - 0.5 * COULOMB_CONST * float(jnp.sum(mw * rw))
        mjw, njw = solve_scf(Ef, al, ii, jj, *fac, mu_b, arf, 1e-8, nat, 200, 0)
        report("SCF: PCG warm start from a 1%-biased guess, tolpol 1e-8", abs(e_w - e_ref) < 0.1,
               f"energy error {e_w - e_ref:+.5f} kcal/mol after {int(nw)} products "
               f"(Jacobi from the same guess: {e_std(mjw) - e_ref:+.4f} after {int(njw)})")

    n_ok = sum(results)
    print(f"\n{n_ok}/{len(results)} checks passed ({time.time()-t0:.0f}s)")
    return 0 if n_ok == len(results) else 1


if __name__ == '__main__':
    sys.exit(main())
