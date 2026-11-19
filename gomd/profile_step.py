"""
profile_step.py — where the time of one GOMD MD step goes, on the real system.

    python -m gomd.profile_step --input-dir ./md-input --ref-dir ./pkg \
        --polarizable --tolpol 1e-8 --out md-output/profile.txt

Sections (each prints a table; everything is also written to --out):
  1. device and fp64/fp32 throughput (elementwise math and pair scatters) —
     decides whether fp32 or segment_sum would pay off on this GPU
  2. neighbor list: build time, pairs in the list vs inside rcut vs capacity
  3. every force phase on its own, jit'd and timed (median of --reps calls,
     compile excluded): classical terms, lone pairs, polarization field parts,
     one SCF matrix-vector product, the SCF, the induced-force parts, SHAKE,
     the integrator halves
  4. the MD step itself in lax.scan (ms/step), WITHOUT neighbor rebuilds, and,
     at each checkpoint, the largest site drift since the build and the pairs
     inside rcut that the stale list misses — i.e. how long a list with this
     skin really stays valid
  5. (polarizable) SCF solvers: Jacobi vs PCG at tolpol 1e-4 ... 1e-12 —
     T.mu products, time, dipole and energy error against a tightly converged
     reference, and the two tight references against each other (validates PCG)

Timings are medians of --reps calls of a jit'd function with block_until_ready;
the first call (compile) is excluded. Phase times do not add up exactly to the
step time: XLA fuses across phase boundaries inside the full step.
"""
from __future__ import annotations
import argparse, math, os, statistics, sys, time


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description='GOMD per-phase profiler')
    ap.add_argument('--input-dir', default='./md-input')
    ap.add_argument('--ref-dir', default='./pkg')
    ap.add_argument('--polarizable', action='store_true')
    ap.add_argument('--tolpol', type=float, default=1e-8,
                    help='SCF tolerance for the phase timings and the MD step (default 1e-8)')
    ap.add_argument('--scf-solver', choices=['jacobi', 'pcg'], default='pcg',
                    help='solver timed in section 3 and used in the MD step of section 4 '
                         '(pcg there starts from the predicted dipoles, as in main_scan)')
    ap.add_argument('--scf-max-iter', type=int, default=300)
    ap.add_argument('--ensemble', choices=['auto', 'nve', 'nvt'], default='auto')
    ap.add_argument('--skin', type=float, default=1.0)
    ap.add_argument('--reps', type=int, default=10, help='timed calls per phase (median)')
    ap.add_argument('--md-steps', type=int, default=200,
                    help='MD steps run without a neighbor rebuild (section 4)')
    ap.add_argument('--chunk', type=int, default=25, help='checkpoint interval of section 4')
    ap.add_argument('--skip-solvers', action='store_true', help='skip section 5')
    ap.add_argument('--cpu', action='store_true', help='force the CPU backend')
    ap.add_argument('--out', default=None, help='also write the report here')
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.md_steps < 1 or args.chunk < 1:
        raise SystemExit('--md-steps and --chunk must be >= 1')
    if args.cpu:
        os.environ['JAX_PLATFORMS'] = 'cpu'
    from jax import config as jc
    jc.update('jax_enable_x64', True)
    import jax, jax.numpy as jnp
    import numpy as np

    from gomd.main_scan import build_Dj
    from gomd.io.assembly_loader import load_m2_system
    from gomd.forces.neighbor import build_neighbor_pairs
    from gomd.forces.neighbor_ondevice import exclusions_to_flat, count_pairs_within
    from gomd.forces.ewald import (build_kvectors, self_energy, reciprocal_energy_force,
                                   exclusion_correction)
    from gomd.forces.pair_splined_segsum import splined_pair_energy_force_segsum
    from gomd.forces.bonded import bond_energy_force, bend_energy_force, torsion_energy_force
    from gomd.forces.all_forces import compute_all_forces
    from gomd.forces.dummy import dummy_force_redistribution
    from gomd.constraints.shake import shake_constrain
    from gomd.integrator_scan import make_md_step, make_runner, UNIT, VSCALE
    from gomd import thermostat as nhc_mod

    out_f = None
    if args.out:
        args.out = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        out_f = open(args.out, 'w')

    def emit(s=''):
        print(s, flush=True)
        if out_f:
            out_f.write(s + '\n'); out_f.flush()

    def block(out):
        for leaf in jax.tree_util.tree_leaves(out):
            if hasattr(leaf, 'block_until_ready'):
                leaf.block_until_ready()
        return out

    timings = {}

    def bench(name, fn, *a, reps=None, note=''):
        """Median wall time (ms) of fn(*a) over reps calls; first call = compile."""
        reps = args.reps if reps is None else reps
        t0 = time.perf_counter()
        out = block(fn(*a))
        t_first = time.perf_counter() - t0
        ts = []
        for _ in range(max(reps, 1)):
            t0 = time.perf_counter()
            out = block(fn(*a))
            ts.append(time.perf_counter() - t0)
        ms = statistics.median(ts) * 1e3
        timings[name] = ms
        emit(f"  {name:<52} {ms:10.3f} ms   (first call {t_first:6.2f} s){'  ' + note if note else ''}")
        return out, ms

    dev = jax.devices()[0]
    emit(f"# GOMD profile  {time.strftime('%Y-%m-%d %H:%M:%S')}  jax {jax.__version__}  "
         f"device {dev.platform}: {getattr(dev, 'device_kind', '?')}")
    emit(f"# args: {' '.join(sys.argv[1:]) if argv is None else ' '.join(argv)}")

    # ------------------------------------------------------------------ setup
    t0 = time.time()
    D = load_m2_system(args.input_dir, args.ref_dir)
    nat = D['nat']
    emit(f"# system: {nat} sites ({D['ndummy']} lone-pair), loaded in {time.time() - t0:.1f} s")
    E = D['ensemble']
    nvt = args.ensemble == 'nvt' or (args.ensemble == 'auto' and E['nvt'])
    nhc_params = (nhc_mod.make_nhc_params(E['Tset'], E['ndof'], E['nnos'], E['qfreq'], D['dt'])
                  if nvt else None)
    emit(f"# ensemble for the integrator timings: {'NVT' if nvt else 'NVE'}"
         + (" (mdrun.params asks for NPT, which GOMD does not run)" if E['npt'] and not nvt else ""))
    kx, ky, kz = [jnp.asarray(a) for a in build_kvectors(D['box'], *D['klimit'])]
    from gomd.forces.recip_shared import make_recip
    recip = make_recip(D['box'], D['q'], kx, ky, kz, D['alpha'])     # as main_scan
    ewself = self_energy(D['q'], D['alpha'])
    Dj = build_Dj(D)
    Dj['con']['tol'] = 1e-12
    box = Dj['box']
    rcut2 = float(D['rcut']) ** 2
    rlist2 = float((D['rcut'] + args.skin) ** 2)
    ii_h, jj_h = build_neighbor_pairs(np.asarray(D['x']), D['box'], D['rcut'], skin=args.skin,
                                      exclusions=D['excl_set'])
    capacity = int(ii_h.size * 1.25)
    excl_flat = exclusions_to_flat(D['excl_set'], nat)
    ncut0 = int(count_pairs_within(jnp.asarray(D['x']), jnp.asarray(ii_h), jnp.asarray(jj_h),
                                   box, float((D['rcut'] + 0.05) ** 2)))
    cut_capacity = int(ncut0 * 1.10)              # as main_scan --cut-capacity 1.10

    pp = None
    pol_fn = None
    if args.polarizable:
        if D.get('pol_data') is None:
            raise SystemExit('--polarizable: this system has no polarization data')
        from gomd.forces.polarization_full import make_polarization_parts
        D['pol_data']['tolpol'] = float(args.tolpol)
        pp = make_polarization_parts(D, kx, ky, kz, scf_max_iter=args.scf_max_iter,
                                     solver=args.scf_solver, recip=recip)
        pol_fn = jax.jit(pp['compute'])

    # rebuild_every=0 + no drift trigger: section 4 runs on one stale list
    parts = make_md_step(Dj, kx, ky, kz, ewself, excl_flat, capacity, 0, rlist2,
                         polarization_fn=pol_fn, use_aspc=(args.scf_solver == 'pcg'),
                         nhc_params=nhc_params,
                         natreal=E['natreal'], shake_max_iter=100, skin=args.skin,
                         drift_rebuild=False, cut_capacity=cut_capacity, recip=recip)
    J = jax.jit

    # ------------------------------------------------------------------ 1. device
    emit("\n## 1. device throughput")
    n_el = 4_000_000

    def burn(a):
        return jax.lax.fori_loop(0, 50, lambda i, y: jnp.sqrt(y * y + 1.0) * 0.5 + jnp.exp(-y) * 0.25, a)
    a32 = jnp.linspace(0.0, 1.0, n_el, dtype=jnp.float32)
    _, t32 = bench('elementwise fp32 (4M x 50 sqrt+exp)', J(burn), a32)
    _, t64 = bench('elementwise fp64 (4M x 50 sqrt+exp)', J(burn), a32.astype(jnp.float64))
    emit(f"  -> fp64 / fp32 = {t64 / max(t32, 1e-9):.1f}x")

    # ------------------------------------------------------------------ 2. neighbor list
    emit("\n## 2. neighbor list")
    x = J(parts['place_dummies'])(Dj['x']); block(x)
    (ii, jj, cnt), t_nb = bench('neighbor build (all pairs on device)', J(parts['rebuild']), x)
    n_list = int(cnt)
    n_cut = int(count_pairs_within(x, ii, jj, box, rcut2))
    (pi, pj, nc), t_cmp = bench('pair compaction to rcut + 0.05 A (every step)',
                                J(parts['compact']), x, ii, jj)
    emit(f"  pairs in rlist {n_list:,}  inside rcut {n_cut:,} ({n_cut / n_list:.0%})  "
         f"list capacity {capacity:,}; compacted {int(nc):,} into {cut_capacity:,} slots -> the "
         f"pair kernels process {cut_capacity / max(n_cut, 1):.2f}x the pairs that interact "
         f"({capacity / max(n_cut, 1):.2f}x without compaction)")

    v64 = jnp.ones((2 * capacity, 3)) * 1e-3
    vd = jnp.concatenate([jnp.where(ii != jj, ii, nat), jnp.where(ii != jj, jj, nat)])
    bench('scatter fp64, 2 x capacity rows, padding into atom 0',
          J(lambda v, a, b: jnp.zeros((nat, 3)).at[a].add(v[:capacity]).at[b].add(v[capacity:])),
          v64, ii, jj)
    bench('scatter fp64, same rows, padding dropped',
          J(lambda v, s: jnp.zeros((nat, 3)).at[s].add(v, mode='drop')), v64, vd)
    bench('scatter fp32 segment_sum, padding dropped',
          J(lambda v, s: jax.ops.segment_sum(v, s, num_segments=nat)), v64.astype(jnp.float32), vd)
    bench('gather fp64 x[i], x[j] (capacity pairs)', J(lambda y, a, b: y[b] - y[a]), x, ii, jj)
    emit("  (the force phases below run on the compacted list, as the MD step does)")

    # ------------------------------------------------------------------ 3. phases
    emit("\n## 3. force phases (one evaluation each)")
    pair = J(lambda y, a, b: splined_pair_energy_force_segsum(
        y, a, b, Dj['atomtype'], Dj['typee'], Dj['coefft_m32'], Dj['coeffft_m32'],
        Dj['zs32'], Dj['deltainv'], rcut2, box, nat))
    bench('pair vdW + real-space elec (fp32 segsum)', pair, x, pi, pj)
    bench('bonds', J(lambda y: bond_energy_force(y, Dj['bonds'], Dj['stretch'], box, nat)), x)
    bench('bends', J(lambda y: bend_energy_force(y, Dj['bends'], Dj['bend'], box, nat)), x)
    bench('torsions', J(lambda y: torsion_energy_force(y, Dj['torts'], Dj['twist'], Dj['nprms'],
                                                       box, nat, Dj['max_torsion_terms'])), x)
    rt, _ = bench(f'k-space tables, shared ({int(kx.shape[0])} k: cos/sin, S(k), field)',
                  J(recip['tables']), x)
    bench('Ewald exclusion correction',
          J(lambda y: exclusion_correction(y, Dj['excl_qq'], Dj['excl_i'], Dj['excl_j'],
                                           box, Dj['alpha'], nat)), x)
    (pe_c, f_c, _), _ = bench('= classical total (compute_all_forces)',
                              J(lambda y, a, b, t: compute_all_forces(
                                  y, Dj, a, b, kx, ky, kz, ewself,
                                  recip_ef=(recip['energy'](t), recip['force'](t)))),
                              x, pi, pj, rt)
    if D['ndummy'] > 0:
        dm = Dj['dummy']
        bench('lone-pair site placement', J(parts['place_dummies']), x)
        bench('lone-pair force redistribution',
              J(lambda f, y: dummy_force_redistribution(f, y, dm['jik'], dm['adummy'],
                                                        dm['bdummy'], dm['cdummy'], box, nat)),
              f_c, x)

    mu = None
    if pp is not None:
        fp = pp['field_parts']
        G, _ = bench('pol: pair geometry + Thole factors', J(pp['geometry']), x, pi, pj)
        bench('pol field: real-space pairs (tabulated erfc)', J(fp['real']), x, pi, pj)
        if pp['has14']:
            bench('pol field: 1-4 fold-in', J(fp['real14']), x)
        bench('pol field: exclusion correction', J(fp['excl']), x)
        if pp['has14']:
            bench('pol field: 1-4 correction', J(fp['corr14']), x)
        Ef, _ = bench('= pol field total (reciprocal part from the shared tables)',
                      J(pp['field']), x, pi, pj, rt)
        mu_test = pp['alpha_atom'][:, None] * Ef
        _, t_mv = bench('pol SCF: one T.mu product', J(pp['town']), mu_test, G, pi, pj)
        (mu, niter, _r), t_scf = bench(f"pol SCF: {args.scf_solver}, tolpol {args.tolpol:g}",
                                       J(pp['scf']), Ef, G, pi, pj)
        emit(f"  -> {int(niter)} T.mu products; {t_scf / max(int(niter), 1):.3f} ms each "
             f"(one bare product {t_mv:.3f} ms)")
        fpp = pp['force_parts']
        bench('pol forces: main pair term (phase 5)', J(fpp['main']), x, mu, G, pi, pj)
        bench('pol forces: exclusion correction (6)', J(fpp['excl']), x, mu)
        if pp['has14']:
            bench('pol forces: 1-4 correction (7)', J(fpp['c14']), x, mu)
        bench('pol forces: k-space charge-dipole (closed form)', J(fpp['recip_qmu']), x, mu, rt)
        bench('= pol forces total', J(pp['forces']), x, mu, G, pi, pj, rt)
        bench('= polarization total (geometry+field+SCF+E+F)', pol_fn, x, pi, pj, None, rt)

    (x1, pe1, f1, comps1, mu1), t_ff = bench('== force_fn: everything above + lone pairs',
                                             J(parts['force_fn']), x, ii, jj)

    # integrator halves
    v0 = Dj['v']
    nhc0 = ((jnp.asarray(E['xlogs']), jnp.asarray(E['vlogs']), jnp.asarray(E['glogs']))
            if nvt else (jnp.zeros(1), jnp.zeros(1), jnp.zeros(1)))
    S0 = {'x': x1, 'v': v0, 'f': f1, 'nhc': nhc0, 'step': jnp.array(1)}
    S1, _ = bench('pre_force: [NHC] kick drift SHAKE', J(parts['pre_force']), S0)
    x2, pe2, f2, _, _ = J(parts['force_fn'])(S1['x'], ii, jj)
    bench('post_force: SHAKE pred.-corr. kick [NHC]', J(parts['post_force']), S1, x2, pe2, f2)
    con = Dj['con']
    if con['constrain'] and con['nbcon'] > 0:
        dt = Dj['dt']
        rmass = UNIT * Dj['massinv']
        nd = (~Dj['ldummy'])[:, None]
        xp = jnp.where(nd, x2 + VSCALE * dt * S1['v'] + VSCALE * dt * dt * f2 * rmass[:, None], x2)
        xp = xp - box * jnp.floor(xp / box)
        shk = J(lambda a, b: shake_constrain(a, b, con['con_i'], con['con_j'], con['d2'],
                                             con['massred'], Dj['massinv'], box, nat,
                                             con['tol'], 100))
        (_xc, _sig, nit), _ = bench(f"SHAKE alone, tol {con['tol']:g}", shk, xp, x2)
        emit(f"  -> {int(nit)} Jacobi sweeps over {con['nbcon']} constraints")

    # ------------------------------------------------------------------ 4. MD step + drift
    emit(f"\n## 4. MD step in lax.scan ({'NVT' if nvt else 'NVE'}), no neighbor rebuild, "
         f"{args.md_steps} steps from coords.inp")
    if pp is not None:
        mu_h = (mu1, mu1, mu1)
    else:
        z = jnp.zeros((nat, 3)); mu_h = (z, z, z)
    S0 = {'x': x1, 'v': v0, 'f': f1, 'nhc': nhc0, 'step': jnp.array(1)}
    carry0 = parts['init_carry'](S0, x, ii, jj, cnt, mu_h, pe1, f1)
    run = make_runner(parts['scan_body'])
    chunk = max(1, min(args.chunk, args.md_steps))
    t0 = time.perf_counter()
    block(run(carry0, nsteps=chunk))
    emit(f"  compile + first {chunk} steps: {time.perf_counter() - t0:.1f} s")
    emit(f"  {'step':>6} {'t (fs)':>8} {'ms/step':>9} {'max drift A':>12} {'missed pairs':>13}"
         + (f" {'scf_it':>7}" if pp is not None else ""))
    carry = carry0
    done = 0
    ms_all = []
    while done < args.md_steps:
        n = min(chunk, args.md_steps - done)
        t0 = time.perf_counter()
        carry, hist = run(carry, nsteps=n)
        block(carry)
        ms = (time.perf_counter() - t0) / n * 1e3
        ms_all.append(ms)
        done += n
        xs = carry['S']['x']
        dv = xs - x
        dv = dv - box * jnp.round(dv / box)
        drift = float(jnp.sqrt(jnp.max(jnp.sum(dv * dv, axis=1))))
        ii2, jj2, c2 = parts['rebuild'](xs)
        missed = int(count_pairs_within(xs, ii2, jj2, box, rcut2)
                     - count_pairs_within(xs, ii, jj, box, rcut2))
        line = f"  {done:>6} {done * D['dt']:>8.1f} {ms:>9.2f} {drift:>12.3f} {missed:>13d}"
        if pp is not None:
            line += f" {float(np.mean(np.asarray(hist['scf_it']))):>7.2f}"
        emit(line)
    ms_step = statistics.median(ms_all)
    emit(f"  -> {ms_step:.2f} ms/step (median over chunks); a rebuild costs about {t_nb:.2f} ms "
         f"(build only; the MD loop adds two cheap pair counts) = {t_nb / ms_step:.2f} steps")
    emit(f"  -> a list is valid while max drift < skin/2 = {args.skin / 2:g} A; the MD driver "
         f"rebuilds at {parts['nb_trigger'] + 0.0:.2f} A (or every --rebuild-every steps)")

    # ------------------------------------------------------------------ budget
    emit("\n## budget (ms, one evaluation each; the fused MD step is the real number)")
    # leaf phases only: totals ('= ...'), the bare T.mu product (inside the SCF),
    # SHAKE alone (inside the integrator halves) and the micro-benchmarks are left out
    keys = [k for k in timings
            if not k.startswith(('elementwise', 'scatter', 'gather', 'neighbor', '=',
                                 'SHAKE alone', 'pol SCF: one'))]
    tot = sum(timings[k] for k in keys)
    for k in sorted(keys, key=lambda k: -timings[k]):
        emit(f"  {timings[k]:9.3f}  {timings[k] / max(tot, 1e-9):6.1%}  {k}")
    emit(f"  {tot:9.3f}  sum of the phases (+ neighbor build / interval)")
    emit(f"  {ms_step:9.3f}  measured MD step in lax.scan")

    # ------------------------------------------------------------------ 5. SCF solvers
    if pp is not None and not args.skip_solvers:
        from gomd.forces.polarization import solve_scf, solve_scf_pcg, COULOMB_CONST
        emit("\n## 5. SCF solvers at the initial configuration (cold start)")
        alpha_atom = pp['alpha_atom']; arf = pp['arf']
        pol = alpha_atom > 0
        diag = jnp.where(pol, 1.0 / jnp.where(pol, alpha_atom, 1.0) - arf, 0.0)
        maxit = args.scf_max_iter
        G = J(pp['geometry'])(x, pi, pj)
        Ef = J(pp['field'])(x, pi, pj)

        jac = J(lambda Ee, Gg, a, b, tol: solve_scf(
            Ee, alpha_atom, a, b, Gg['rx'], Gg['ry'], Gg['rz'], Gg['tfact'], Gg['three_zf_l5'],
            jnp.zeros((nat, 3)), arf, tol, nat, maxit, 0))
        pcg = J(lambda Ee, Gg, a, b, tol: solve_scf_pcg(
            Ee, alpha_atom, a, b, Gg['rx'], Gg['ry'], Gg['rz'], Gg['tfact'], Gg['three_zf_l5'],
            None, arf, tol, nat, maxit, 0))
        resid = J(lambda Ee, m, Gg, a, b: jnp.where(
            pol[:, None], Ee - (diag[:, None] * m - pp['town'](m, Gg, a, b)), 0.0))

        def energies(m):
            r = resid(Ef, m, G, pi, pj)
            e_std = float(-0.5 * COULOMB_CONST * jnp.sum(Ef * m))
            e_var = e_std - float(0.5 * COULOMB_CONST * jnp.sum(m * r))
            rj = float(jnp.max(jnp.sum((alpha_atom[:, None] * r) ** 2, axis=1)))
            return e_std, e_var, rj

        mu_rp, n_rp, _ = block(pcg(Ef, G, pi, pj, 1e-26))
        mu_rj, n_rj = block(jac(Ef, G, pi, pj, 1e-26))
        scale = float(jnp.max(jnp.abs(mu_rp)))
        d_ref = float(jnp.max(jnp.abs(mu_rp - mu_rj)))
        e_rp = energies(mu_rp); e_rj = energies(mu_rj)
        emit(f"  references: PCG {int(n_rp)} products (residual {e_rp[2]:.1e}), Jacobi {int(n_rj)} "
             f"products (residual {e_rj[2]:.1e}); max|mu_pcg - mu_jac| = {d_ref:.2e} "
             f"(max|mu| {scale:.3f}); E_pol {e_rp[0]:.6f} vs {e_rj[0]:.6f} kcal/mol")
        if int(n_rp) >= maxit or int(n_rj) >= maxit:
            emit(f"  note: a reference hit --scf-max-iter {maxit}; it is as converged as fp64 allows")
        e_ref = e_rp[0]
        emit(f"  {'solver':<7} {'tolpol':>8} {'T.mu':>5} {'ms':>9} {'max|dmu|':>10} {'rms|dmu|':>10} "
             f"{'E_pol err':>11} {'E_var err':>11}   (errors vs the PCG reference, kcal/mol)")
        for tol in (1e-4, 1e-6, 1e-8, 1e-10, 1e-12):
            for name, fn in (('jacobi', jac), ('pcg', pcg)):
                res, ms = bench(f'scf {name} {tol:g}', fn, Ef, G, pi, pj, tol, reps=3)
                m, nit = res[0], res[1]
                dm = m - mu_rp
                dmax = float(jnp.max(jnp.sqrt(jnp.sum(dm * dm, axis=1))))
                drms = float(jnp.sqrt(jnp.sum(dm * dm) / max(int(jnp.sum(pol)), 1)))
                e_std, e_var, _ = energies(m)
                emit(f"  {name:<7} {tol:>8.0e} {int(nit):>5d} {ms:>9.2f} {dmax:>10.2e} {drms:>10.2e} "
                     f"{e_std - e_ref:>+11.4f} {e_var - e_ref:>+11.4f}")
        emit("  (E_pol err: the -1/2 E.mu energy the Jacobi path logs; E_var err: the variational "
             "energy the PCG path logs)")

    emit("\n# done")
    if out_f:
        out_f.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
