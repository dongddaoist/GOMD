"""
main_scan.py — GOMD driver.

Forces: the APPLE&P polarizable force field (PFF, on device) and/or an ML
interatomic potential (MLIP: UMA through fairchem, in its own process).
  --propagate pff   (default) the PFF drives the dynamics; the whole trajectory
                    runs on device in lax.scan chunks.
  --propagate mlip  the MLIP drives the dynamics (one host round trip per step).
  --mlip uma|mock   also evaluate the other engine every --compare-every steps
                    at the same configuration and log the agreement to
                    <out>.compare.csv (energy-difference drift, force RMSE /
                    MAE / max / relative error / cosine, per-element RMSE).

Ensembles: NVE, or NVT with a Nose-Hoover chain (as the reference). --ensemble
auto takes it from mdrun.params. NPT is refused: the reference barostat is fed
an invalid virial and GOMD has no virial yet.

Log columns: NVE  step PE KE TE [scf_it] ms/step
             NVT  step PE KE TE T H_ext [scf_it] ms/step   (H_ext = conserved energy)
scf_it (polarizable PFF runs) = mean SCF matrix-vector products per step over
the print interval. ms/step is the MD time; MLIP comparison time is reported
separately. Neighbor list: rebuilt every --rebuild-every steps and whenever a
site has drifted more than skin/2 - 0.01 A; overflow stops the run, and pairs
a stale list missed are reported (# nb lines and the final summary).

Usage:
    python -m gomd.main_scan --input-dir ./md-input --ref-dir ./pkg \
        --steps 200 --print-every 50 --rebuild-every 100 --polarizable \
        --out md-output/md_scan.out
    ... --mlip uma --compare-every 50                       # PFF + UMA alongside
    ... --mlip uma --propagate mlip --compare-every 50      # UMA dynamics
"""
from __future__ import annotations
import sys, os, time, argparse, math


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description='GOMD: GPU-accelerated Open MD')
    ap.add_argument('--input-dir', default='.')
    ap.add_argument('--ref-dir', required=True)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--print-every', type=int, default=50)
    ap.add_argument('--rebuild-every', type=int, default=100,
                    help='scheduled neighbor rebuild interval in steps (0 = only on drift)')
    ap.add_argument('--cpu', action='store_true', help='force the CPU backend (debugging)')
    ap.add_argument('--polarizable', action='store_true',
                    help='full polarizable force field: induced-dipole SCF each step')
    g = ap.add_argument_group('neighbor list')
    g.add_argument('--skin', type=float, default=1.0, help='rlist - rcut, A')
    g.add_argument('--no-drift-rebuild', action='store_true',
                   help='rebuild only every --rebuild-every steps, not when a site has '
                        'drifted more than skin/2 (diagnostic: the log then reports the pairs '
                        'the stale list missed at each rebuild)')
    g.add_argument('--nb-capacity', type=float, default=1.25,
                   help='pair-list capacity as a multiple of the first build')
    g.add_argument('--cut-capacity', type=float, default=1.10,
                   help='capacity of the per-step list of pairs inside rcut + 0.05 A, as a '
                        'multiple of the first count (0 = run the pair kernels on the whole list)')
    g = ap.add_argument_group('accuracy')
    g.add_argument('--tolpol', type=float, default=None,
                   help='SCF tolerance on max|dmu|^2 (default: the mdrun.params value)')
    g.add_argument('--scf-max-iter', type=int, default=50)
    g.add_argument('--scf-solver', choices=['jacobi', 'pcg'], default=None,
                   help='pcg (default) = preconditioned conjugate gradient (residual test, '
                        'variational energy, safe to warm-start); jacobi = the reference iteration '
                        '(default with --reference-parity)')
    g.add_argument('--scf-guess', choices=['cold', 'predict'], default=None,
                   help='SCF start: predict (default with pcg: extrapolate the last 3 steps) or '
                        'cold (mu = 0; default with jacobi)')
    g.add_argument('--shake-tol', type=float, default=None,
                   help='SHAKE tolerance on mean (d0^2 - d^2)^2, A^4 (default 1e-12; '
                        'the reference uses 1e-6)')
    g.add_argument('--shake-max-iter', type=int, default=100)
    g.add_argument('--reference-pol-forces', action='store_true',
                   help='omit the k-space charge-dipole polarization force, as the reference '
                        'pkg does (non-conservative)')
    g.add_argument('--reference-parity', action='store_true',
                   help='reproduce the reference pkg: SHAKE tol 1e-6 (unless --shake-tol) and '
                        '--reference-pol-forces')
    ap.add_argument('--out', default='md_scan.out')
    g = ap.add_argument_group('restart')
    g.add_argument('--restart-dir', default=None,
                   help='write a restart input directory here (coords.inp with the final positions, '
                        'velocities and thermostat state + copies of the other inputs); continue with '
                        '--input-dir DIR. Default <out>_restart; "none" disables')
    g.add_argument('--restart-every', type=int, default=0,
                   help='also rewrite the restart every N steps (0 = only at the end)')
    g = ap.add_argument_group('ensemble')
    g.add_argument('--dt', type=float, default=None, help='time step, fs (default: mdrun.params)')
    g.add_argument('--ensemble', choices=['auto', 'nve', 'nvt'], default='auto',
                   help='auto = from mdrun.params (default)')
    g = ap.add_argument_group('MLIP')
    g.add_argument('--mlip', choices=['none', 'uma', 'mock'], default='none',
                   help='ML potential to run alongside (or to propagate with)')
    g.add_argument('--propagate', choices=['pff', 'mlip'], default='pff',
                   help='which engine drives the dynamics')
    g.add_argument('--compare-every', type=int, default=0,
                   help='evaluate the other engine every N steps (default: --print-every)')
    g.add_argument('--compare-out', default=None, help='default: <out>.compare.csv')
    g.add_argument('--mlip-model', default='uma-s-1p2p1',
                   help='fairchem model name or checkpoint path (uma-s-1p2p1, uma-s-1p1, uma-m-1p1, ...)')
    g.add_argument('--mlip-task', default='omol', help='UMA task head: omol, omat, omc, odac, oc20')
    g.add_argument('--mlip-settings', default='default',
                   help='fairchem inference settings: default, high-precision, ...')
    g.add_argument('--mlip-device', default='auto', help='auto, cuda or cpu')
    g.add_argument('--mlip-python', default=None,
                   help='Python that runs the MLIP server (e.g. a venv with fairchem); '
                        'default: this Python')
    g.add_argument('--charge', type=int, default=None,
                   help='total charge for UMA omol (default: sum of PFF charges, rounded)')
    g.add_argument('--spin', type=int, default=1, help='spin multiplicity for UMA omol')
    g = ap.add_argument_group('electrodes (constant-potential task, P0: fixed electrodes normal to z)')
    g.add_argument('--electrode-lower', default=None,
                   help="atoms of the lower electrode: 'first:N', 'last:N', 'a-b' (1-based, "
                        "inclusive) or single indices, comma-separated")
    g.add_argument('--electrode-upper', default=None, help='atoms of the upper electrode (same syntax)')
    g.add_argument('--electrode-geometry', choices=['flat', 'snapshot'], default='flat',
                   help='flat: each layer at its mean z, graphene layers snapped onto the ideal '
                        'lattice; snapshot: input positions')
    g.add_argument('--slab-height', type=float, default=None,
                   help='make the box this tall in z (A), cell in the middle, vacuum outside the '
                        'electrodes (default: keep the periodic input box). The k-space sum is '
                        'then the 2D Ewald sum: 3D sum in this box + the planar dipole term')
    g.add_argument('--kmax', default=None,
                   help='Ewald k-limits kx,ky,kz (default: mdrun.params, kz scaled with '
                        '--slab-height). On md-input3 in the 205 A slab, 8,8,35 cuts the k-space '
                        'force error from 0.035 to 0.002 kcal/mol/A (tests/kspace_convergence.py)')
    g.add_argument('--no-planar-term', action='store_true',
                   help='slab box without the planar dipole term (3D Ewald with tinfoil '
                        'boundaries; for comparisons only)')
    return ap.parse_args(argv)


def build_Dj(D):
    """Device copies of the loader arrays used by the force kernels."""
    import jax.numpy as jnp
    J = jnp.asarray
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
    # fp32 + segment_sum pair kernel
    Dj['use_segsum_pair'] = True
    Dj['coefft_m32'] = J(D['coefft_m']).astype(jnp.float32)
    Dj['coeffft_m32'] = J(D['coeffft_m']).astype(jnp.float32)
    Dj['zs32'] = J(D['zs']).astype(jnp.float32)
    return Dj


def main(argv=None):
    args = parse_args(argv)
    if args.propagate == 'mlip' and args.mlip == 'none':
        raise SystemExit('--propagate mlip needs --mlip uma (or mock)')
    use_mlip = args.mlip != 'none'
    if use_mlip and args.compare_every <= 0:
        args.compare_every = args.print_every
    if not use_mlip:
        args.compare_every = 0
    if args.no_drift_rebuild and args.rebuild_every <= 0:
        raise SystemExit('--no-drift-rebuild with --rebuild-every 0 would never rebuild the '
                         'neighbor list')
    if args.skin <= 0.1:
        raise SystemExit(f'--skin {args.skin} A is too small (the drift trigger is skin/2 - 0.01 A)')
    # SCF defaults: PCG + predicted start (2,000-step NVE, 2026-11-17: same energy
    # conservation as Jacobi at tolpol 1e-8 with 2.3 instead of 10 T.mu products per
    # step); --reference-parity falls back to the reference Jacobi cold start.
    if args.scf_solver is None:
        args.scf_solver = 'jacobi' if args.reference_parity else 'pcg'
    if args.scf_guess is None:
        args.scf_guess = 'predict' if args.scf_solver == 'pcg' else 'cold'

    # Environment must be set before JAX initializes.
    if args.cpu:
        os.environ['JAX_PLATFORMS'] = 'cpu'
    if use_mlip:
        # the MLIP server process shares the GPU: do not let JAX grab 75% of it
        os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    from jax import config as jc
    jc.update('jax_enable_x64', True)
    import jax, jax.numpy as jnp
    import numpy as np

    args.out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    input_dir_abs = os.path.abspath(args.input_dir)
    restart_dir = None
    if (args.restart_dir or '').lower() != 'none':
        restart_dir = (os.path.abspath(args.restart_dir) if args.restart_dir
                       else os.path.splitext(args.out)[0] + '_restart')
        if restart_dir == input_dir_abs:
            raise SystemExit('--restart-dir must differ from --input-dir')
    _logf = open(args.out, 'w')
    def _emit(line):
        print(line, flush=True); _logf.write(line + '\n'); _logf.flush()
    _emit(f'# GOMD scan log  devices={jax.devices()}  jax {jax.__version__}')

    from gomd.io.assembly_loader import load_m2_system
    from gomd.forces.neighbor import build_neighbor_pairs
    from gomd.forces.neighbor_ondevice import (
        build_neighbor_ondevice, exclusions_to_flat)
    from gomd.forces.ewald import build_kvectors, self_energy
    from gomd.integrator_scan import make_md_step, make_runner
    from gomd import thermostat as nhc_mod

    print("Loading system (reference setup)...")
    t0 = time.time()
    D = load_m2_system(args.input_dir, args.ref_dir)
    print(f"  loaded in {time.time()-t0:.1f}s  nat={D['nat']} ndummy={D['ndummy']}")
    if args.dt is not None:
        _emit(f"time step: {args.dt:g} fs (mdrun.params: {D['dt']:g} fs)")
        D['dt'] = float(args.dt)

    # ---------------- fixed electrodes (constant-potential task, P0) ----------------
    if (args.electrode_lower is None) != (args.electrode_upper is None):
        raise SystemExit('give both --electrode-lower and --electrode-upper')
    frozen = None
    if args.electrode_lower is not None:
        if use_mlip:
            raise SystemExit('fixed electrodes are not supported together with an MLIP')
        from gomd.electrodes import prepare_electrodes
        prepare_electrodes(D, args.electrode_lower, args.electrode_upper,
                           geometry=args.electrode_geometry, slab_height=args.slab_height,
                           rlist=D['rcut'] + args.skin, emit=_emit)
        frozen = D['frozen']
    elif args.slab_height is not None:
        raise SystemExit('--slab-height needs the electrodes (--electrode-lower / --electrode-upper)')
    if args.kmax is not None:
        kset = tuple(int(t) for t in args.kmax.split(','))
        if len(kset) != 3 or min(kset) < 1:
            raise SystemExit(f'--kmax {args.kmax}: give three positive integers kx,ky,kz')
        _emit(f"Ewald k-limits: {kset} (input: {tuple(D['klimit'])})")
        D['klimit'] = kset
    ndeforms = int(D['conn'].get('deforms', np.zeros((8, 0))).shape[1])
    if ndeforms:
        raise SystemExit(f"the connectivity has {ndeforms} out-of-plane (deformation) terms on "
                         f"mobile atoms; GOMD does not evaluate them (they are only dropped "
                         f"on frozen electrodes)")

    # ---------------- ensemble ----------------
    E = D['ensemble']
    ens = args.ensemble
    if ens == 'auto':
        if E['npt']:
            raise SystemExit(
                "mdrun.params asks for NPT, which GOMD does not support: the reference "
                "barostat is fed an invalid virial (pair virial = pair energy; bonded, "
                "SHAKE and lone-pair virials missing) and GOMD has no virial yet. "
                "Run with --ensemble nvt or --ensemble nve.")
        ens = 'nvt' if E['nvt'] else 'nve'
    if E['multibig'] > 1:
        _emit(f"# WARNING: mdrun.params sets Large Steps = {E['multibig']} (RESPA); GOMD "
              f"does not port RESPA and evaluates every force at dt = {D['dt']} fs")
    nhc_params = None
    if ens == 'nvt':
        nhc_params = nhc_mod.make_nhc_params(E['Tset'], E['ndof'], E['nnos'], E['qfreq'], D['dt'])
        _emit(f"ensemble: NVT, Nose-Hoover chain (MTK, 5x5 Yoshida-Suzuki)  T = {E['Tset']} K  "
              f"chain = {E['nnos']}  OmegaQ = {E['qfreq']} 1/fs  ndof = {E['ndof']}")
    else:
        _emit(f"ensemble: NVE  ndof = {E['ndof']}")

    kx, ky, kz = build_kvectors(D['box'], *D['klimit'])
    # one k-space pass per step for the classical reciprocal term, the SCF field
    # and the k-space Q-mu force (forces/recip_shared.py)
    from gomd.forces.recip_shared import make_recip
    slab = args.slab_height is not None and not args.no_planar_term
    recip = make_recip(D['box'], D['q'], kx, ky, kz, D['alpha'], slab=slab)
    _emit(f"k-space: alpha {D['alpha']:g}, kmax {'/'.join(str(k) for k in D['klimit'])}, "
          f"{recip['nk']} k-vectors; "
          + ("2D Ewald (3D sum in the slab box + planar dipole term)" if slab else
             "3D Ewald, tinfoil boundaries"
             + (" (slab box WITHOUT the planar term)" if args.slab_height is not None else "")))
    kx = jnp.asarray(kx); ky = jnp.asarray(ky); kz = jnp.asarray(kz)
    ewself = self_energy(D['q'], D['alpha'])

    J = jnp.asarray
    Dj = build_Dj(D)
    # SHAKE tolerance: the reference's 1e-6 stops the Jacobi iteration after ~2
    # sweeps (bond errors up to 3e-4 A); the residual makes the constraint force
    # ring with period 2 steps (+-4 kcal/mol in TE). 1e-12 takes ~11 sweeps.
    ref_shake_tol = float(Dj['con']['tol'])
    if args.shake_tol is not None:
        Dj['con']['tol'] = float(args.shake_tol)
    elif not args.reference_parity:
        Dj['con']['tol'] = 1e-12
    if Dj['con']['constrain'] and Dj['con']['nbcon'] > 0:
        _emit(f"SHAKE: {Dj['con']['nbcon']} constraints, tol {Dj['con']['tol']:g} "
              f"(reference {ref_shake_tol:g}), max {args.shake_max_iter} iterations")

    # Size capacity from one host build (host build used ONCE, at setup only).
    xnp = np.asarray(D['x'])
    ii0, jj0 = build_neighbor_pairs(xnp, D['box'], D['rcut'], skin=args.skin,
                                    exclusions=D['excl_set'])
    if frozen is not None:                    # as the device build: no frozen-frozen pairs
        keep = ~(frozen[ii0] & frozen[jj0])
        _emit(f"neighbor pairs between frozen atoms left out: {int((~keep).sum())}")
        ii0, jj0 = ii0[keep], jj0[keep]
    capacity = int(ii0.size * args.nb_capacity)
    _emit(f"neighbor pairs: {ii0.size}, padded capacity: {capacity}")

    rlist2 = float((D['rcut'] + args.skin) ** 2)
    excl_flat = exclusions_to_flat(D['excl_set'], D['nat'])
    cut_capacity = None
    if args.cut_capacity > 0:
        from gomd.forces.neighbor_ondevice import count_pairs_within
        ncut0 = int(count_pairs_within(jnp.asarray(D['x']), jnp.asarray(ii0), jnp.asarray(jj0),
                                       jnp.asarray(D['box']), float((D['rcut'] + 0.05) ** 2)))
        cut_capacity = int(ncut0 * args.cut_capacity)
        _emit(f"pairs inside rcut + 0.05 A: {ncut0}; per-step compacted list capacity {cut_capacity}")

    polarization_fn = None
    if args.polarizable:
        if D.get('pol_data') is None:
            raise SystemExit("--polarizable set but this system is not "
                             "polarizable (no pol_data from loader).")
        from gomd.forces.polarization_full import make_polarization_fn
        if args.tolpol is not None:
            D['pol_data']['tolpol'] = float(args.tolpol)
        if args.reference_parity:
            args.reference_pol_forces = True
        if D['pol_data']['tolpol'] > 1e-6:
            _emit(f"# WARNING: tolpol {D['pol_data']['tolpol']:g} is loose — the SCF stops before "
                  f"E_pol converges (on the 8,716-site PEO/LiFSI system Jacobi at 1e-4 left E_pol "
                  f"19 kcal/mol off). Use --tolpol 1e-8: with PCG + predicted start it costs about "
                  f"2.3 T.mu products per step.")
        polarization_fn = make_polarization_fn(D, kx, ky, kz, scf_max_iter=args.scf_max_iter,
                                               recip_qmu_force=not args.reference_pol_forces,
                                               solver=args.scf_solver, recip=recip)
        _emit(f"polarization: ENABLED (full induced-dipole SCF each step, "
              f"{args.scf_solver} solver, {args.scf_guess} start, "
              f"tolpol {D['pol_data']['tolpol']:g}, max {args.scf_max_iter} iterations"
              + ("; k-space charge-dipole force OMITTED (reference parity)" if args.reference_pol_forces
                 else "; with k-space charge-dipole force") + ")")
        if args.scf_guess == 'predict' and args.scf_solver == 'jacobi':
            _emit("# WARNING: --scf-guess predict with the Jacobi solver can stop at a wrong fixed "
                  "point (E_pol ~24 kcal/mol off from a 1% biased start); use --scf-solver pcg")
    else:
        _emit("polarization: disabled (classical force field)")
    pff_label = "polarizable" if args.polarizable else "classical"

    parts = make_md_step(
        Dj, kx, ky, kz, ewself, excl_flat, capacity,
        args.rebuild_every, rlist2, polarization_fn=polarization_fn,
        use_aspc=(args.scf_guess == 'predict'),
        nhc_params=nhc_params, natreal=E['natreal'], shake_max_iter=args.shake_max_iter,
        skin=args.skin,
        drift_rebuild=(not args.no_drift_rebuild) and args.propagate == 'pff',
        cut_capacity=cut_capacity, recip=recip, frozen=frozen)
    force_fn = parts['force_fn']
    if args.propagate == 'pff':
        rules = ([f"every {args.rebuild_every} steps"] if args.rebuild_every > 0 else []) + \
                ([] if args.no_drift_rebuild else
                 [f"whenever a site drifts > {parts['nb_trigger']:.2f} A"])
        _emit(f"neighbor list: rlist {D['rcut'] + args.skin:g} A (skin {args.skin:g}), rebuilt "
              + " and ".join(rules))

    @jax.jit
    def pff_fresh(x):
        """PFF at an arbitrary configuration: fresh pairs, SCF cold start."""
        ii, jj, cnt = parts['rebuild'](x)
        xo, pe, f, comps, _ = force_fn(x, ii, jj)
        return xo, pe, f, jnp.maximum(cnt - capacity, comps.get('ncut', 0) - (cut_capacity or 0))

    # ---------------- MLIP ----------------
    provider = comparator = None
    if use_mlip:
        from gomd.mlip.elements import numbers_from_masses, formula
        from gomd.mlip.client import MLIPProvider
        from gomd.mlip.compare import Comparator
        numbers = numbers_from_masses(D['mass'], D['ldummy'])
        charge = args.charge if args.charge is not None else int(round(float(np.sum(D['q']))))
        x_sites = np.asarray(jax.jit(parts['place_dummies'])(Dj['x']))
        _emit(f"MLIP: {args.mlip}  ({formula(numbers)}; {int((numbers > 0).sum())} atoms sent, "
              f"{D['ndummy']} lone-pair sites excluded)  charge {charge}  spin {args.spin}")
        if args.mlip == 'uma':
            _emit(f"MLIP: model {args.mlip_model}  task {args.mlip_task}  settings "
                  f"{args.mlip_settings}  periodic cell " + ' x '.join(f'{b:.4f}' for b in D['box']) + " A")
        t0 = time.time()
        provider = MLIPProvider.start(
            args.mlip, numbers, D['box'], x_sites, python=args.mlip_python,
            model=args.mlip_model, task=args.mlip_task, settings=args.mlip_settings,
            device=args.mlip_device, charge=charge, spin=args.spin)
        _emit(f"MLIP server ready in {time.time()-t0:.1f}s: {provider.info}")
        cmp_path = args.compare_out or (os.path.splitext(args.out)[0] + '.compare.csv')
        comparator = Comparator(numbers, cmp_path, args.propagate)
        _emit(f"compare: {('MLIP' if args.propagate == 'pff' else 'PFF (' + pff_label + ')')} "
              f"evaluated every {args.compare_every} steps -> {cmp_path}")

    def compare(step, x, e_prop, f_prop):
        """Evaluate the non-propagating engine at x and log the agreement."""
        t0 = time.perf_counter()
        if args.propagate == 'pff':
            e_m, f_m = provider(np.asarray(x))
            e_p, f_p = float(e_prop), np.asarray(f_prop)
        else:
            _, e_p, f_p, over = pff_fresh(jnp.asarray(x))
            if int(over) > 0:
                _emit(f"# WARNING: step {step}: a PFF pair list overflows by {int(over)} pairs; this "
                      f"PFF comparison is wrong. Raise --nb-capacity / --cut-capacity.")
            e_p, f_p = float(e_p), np.asarray(f_p)
            e_m, f_m = float(e_prop), np.asarray(f_prop)
        row = comparator.add(step, e_p, f_p, e_m, f_m, time.perf_counter() - t0)
        _emit(comparator.line(row))

    # ---------------- initial state ----------------
    x0 = Dj['x']
    if E['nnos'] and ens == 'nvt':
        nhc0 = (J(E['xlogs']), J(E['vlogs']), J(E['glogs']))
    else:
        nhc0 = (jnp.zeros(1), jnp.zeros(1), jnp.zeros(1))

    x0 = jax.jit(parts['place_dummies'])(x0)          # build at the placed sites
    x_built = x0
    idx_i0, idx_j0, cnt0 = parts['rebuild'](x0)
    if int(cnt0) > capacity:
        raise SystemExit(f"initial neighbor list overflows: {int(cnt0)} pairs > capacity "
                         f"{capacity}; raise --nb-capacity")
    if args.propagate == 'pff':
        x0, pe0, f0, comps0, mu0 = force_fn(x0, idx_i0, idx_j0)
        f0.block_until_ready()
        if cut_capacity and int(comps0['ncut']) > cut_capacity:
            raise SystemExit(f"initial pairs inside rcut ({int(comps0['ncut'])}) exceed the "
                             f"compacted capacity {cut_capacity}; raise --cut-capacity")
        _emit(f"Initial {pff_label} PE: {float(pe0):.4f} kcal/mol"
              + (f"  (SCF: {int(comps0['scf_it'])} T.mu products, E_pol "
                 f"{float(comps0['epol']):.4f})" if args.polarizable else ""))
    else:
        x0 = jax.jit(parts['place_dummies'])(x0)
        e0, f0n = provider(np.asarray(x0))
        pe0, f0 = jnp.asarray(e0), jnp.asarray(f0n)
        mu0 = None
        _emit(f"Initial MLIP PE: {e0:.4f} kcal/mol  ({e0/23.060547830619026:.6f} eV)")

    # dipole history for the SCF predictor (unused with --scf-guess cold). All
    # three slots start at mu0, so the first prediction 3(mu1-mu2)+mu3 = mu0.
    if args.polarizable and mu0 is not None:
        mu = (mu0, mu0, mu0)
    else:
        z = jnp.zeros((D['nat'], 3)); mu = (z, z, z)

    S0 = {'x': x0, 'v': Dj['v'], 'f': f0, 'nhc': nhc0, 'step': jnp.array(1)}
    ke0 = float(nhc_mod.kinetic_energy(Dj['v'], Dj['mass'], (~Dj['ldummy'])[:, None]))
    _emit(f"Initial KE: {ke0:.4f} kcal/mol  T = {nhc_mod.temperature(ke0, E['ndof']):.2f} K")

    nvt = ens == 'nvt'
    show_scf = args.polarizable and args.propagate == 'pff'

    from gomd.io.restart import write_restart_dir

    def save_restart(S, nsteps_done):
        """coords.inp (positions, velocities, NHC state) + the other inputs -> restart_dir."""
        nhc_state = tuple(np.asarray(a) for a in S['nhc']) if nvt else None
        write_restart_dir(input_dir_abs, restart_dir, np.asarray(S['x']), np.asarray(S['v']),
                          D['box'], nhc=nhc_state, time_add_ps=nsteps_done * D['dt'] / 1000.0,
                          klimit=D['klimit'])
    scf_hdr = f" {'scf_it':>7}" if show_scf else ""
    if nvt:
        _emit(f"{'step':>6} {'PE':>14} {'KE':>12} {'TE':>14} {'T':>9} {'H_ext':>14}{scf_hdr} {'ms/step':>10}")
    else:
        _emit(f"{'step':>6} {'PE':>14} {'KE':>12} {'TE':>14}{scf_hdr} {'ms/step':>10}")

    def row(step, pe, ke, hext, ms, scf=float('nan')):
        te = pe + ke
        s = f" {scf:>7.2f}" if show_scf else ""
        if nvt:
            T = nhc_mod.temperature(ke, E['ndof'])
            return f"{step:>6} {pe:>14.4f} {ke:>12.4f} {te:>14.4f} {T:>9.2f} {hext:>14.4f}{s} {ms:>10.1f}"
        return f"{step:>6} {pe:>14.4f} {ke:>12.4f} {te:>14.4f}{s} {ms:>10.1f}"

    if comparator is not None:
        compare(0, x0, pe0, f0)

    t_md = 0.0          # MD time only (excludes MLIP comparisons)
    t_md_since = 0.0; steps_since = 0
    exit_code = 0
    nb_summary = None
    if args.propagate == 'pff':
        # ---------- PFF: lax.scan chunks ----------
        run = make_runner(parts['scan_body'])
        chunk = args.print_every
        if args.compare_every:
            chunk = math.gcd(args.print_every, args.compare_every)
        carry0 = parts['init_carry'](S0, x_built, idx_i0, idx_j0, cnt0, mu, pe0, f0)
        # warm-compile one chunk (excluded from timing); result discarded
        c_w, _ = run(carry0, nsteps=min(chunk, args.steps))
        c_w['S']['x'].block_until_ready(); del c_w
        carry = carry0
        done = 0
        scf_sum = 0.0; scf_n = 0; scf_tot = 0.0; scf_capped = False; ncut_max = 0
        nb_last = jax.device_get(carry0['nb'])
        nb_printed = nb_last
        while done < args.steps:
            n = min(chunk, args.steps - done)
            t0 = time.time()
            carry, hist = run(carry, nsteps=n)
            carry['S']['x'].block_until_ready()
            dt_ = time.time() - t0
            t_md += dt_; t_md_since += dt_; steps_since += n
            done += n
            if show_scf:
                it = np.asarray(hist['scf_it'])
                scf_sum += float(it.sum()); scf_n += n; scf_tot += float(it.sum())
                if not scf_capped and int(it.max()) >= args.scf_max_iter:
                    scf_capped = True
                    _emit(f"# WARNING: the SCF hit --scf-max-iter {args.scf_max_iter} before "
                          f"tolpol (first seen by step {done}); E_pol and the forces are unconverged")
            nb = jax.device_get(carry['nb'])
            if cut_capacity:
                nc = int(np.max(np.asarray(hist['ncut'])))
                ncut_max = max(ncut_max, nc)
                if nc > cut_capacity:
                    _emit(f"# ERROR: the per-step list of pairs inside rcut overflowed in steps "
                          f"{done - n + 1}-{done}: {nc} pairs > capacity {cut_capacity}. Pairs were "
                          f"dropped, so the state at step {done} is not valid; the run is stopped. "
                          f"Rerun with --cut-capacity {args.cut_capacity * nc / cut_capacity * 1.05:.2f}")
                    exit_code = 2
                    nb_last = nb
                    break
            if int(nb['overflow']) > 0:
                _emit(f"# ERROR: neighbor list overflow in steps {done - n + 1}-{done}: "
                      f"{int(nb['count_max'])} pairs > capacity {capacity}. Pairs were dropped from "
                      f"the overflow on, so the state at step {done} is not valid; the run is "
                      f"stopped. Rerun with --nb-capacity "
                      f"{args.nb_capacity * int(nb['count_max']) / capacity * 1.1:.2f}")
                exit_code = 2
                nb_last = nb
                break
            if int(nb['missed']) > int(nb_last['missed']):
                _emit(f"# WARNING: by step {done} a stale neighbor list missed up to "
                      f"{int(nb['missed'])} pairs inside rcut (peak drift so far "
                      f"{math.sqrt(float(nb['disp2_peak'])):.3f} A; skin/2 = {args.skin/2:g} A)")
            nb_last = nb
            if (restart_dir and args.restart_every and done < args.steps
                    and done // args.restart_every > (done - n) // args.restart_every):
                save_restart(carry['S'], done)
            if args.compare_every and (done % args.compare_every == 0 or done == args.steps):
                compare(done, carry['S']['x'], carry['pe'], carry['fphys'])
            if done % args.print_every == 0 or done == args.steps:
                _emit(row(done, float(hist['pe'][-1]), float(hist['ke'][-1]),
                          float(hist['hext'][-1]), t_md_since / steps_since * 1000,
                          scf_sum / max(scf_n, 1)))
                d_extra = int(nb['nextra']) - int(nb_printed['nextra'])
                if d_extra > 0:
                    _emit(f"# nb: {int(nb['nbuild']) - int(nb_printed['nbuild'])} rebuilds since the "
                          f"last line, {d_extra} of them on drift > {parts['nb_trigger']:.2f} A")
                nb_printed = nb
                t_md_since = 0.0; steps_since = 0; scf_sum = 0.0; scf_n = 0
        nb = nb_last
        nb_summary = (f"# neighbor list: {int(nb['nbuild'])} rebuilds in {done} steps "
                      f"({int(nb['nextra'])} on drift alone), peak drift before a rebuild "
                      f"{math.sqrt(float(nb['disp2_peak'])):.3f} A (trigger "
                      + ("off" if args.no_drift_rebuild else f"{parts['nb_trigger']:.2f} A")
                      + f"), max pairs {int(nb['count_max'])} / capacity {capacity} "
                      f"({int(nb['count_max']) / capacity:.0%}), max pairs missed by a stale list "
                      f"{int(nb['missed'])}")
        if cut_capacity:
            nb_summary += (f"\n# pairs inside rcut + 0.05 A per step: max {ncut_max} / capacity "
                           f"{cut_capacity} ({ncut_max / cut_capacity:.0%})")
        if show_scf and done:
            nb_summary += f"\n# SCF: {scf_tot / done:.2f} T.mu products per step on average"
    else:
        # ---------- MLIP: host loop, MLIP between the two jit'd halves ----------
        pre = jax.jit(parts['pre_force'])
        place = jax.jit(parts['place_dummies'])
        post = jax.jit(parts['post_force'])
        Sw = pre(S0); xw = place(Sw['x'])                 # warm-compile (discarded)
        post(Sw, xw, pe0, f0)[0]['x'].block_until_ready(); del Sw, xw
        S = S0
        for step in range(1, args.steps + 1):
            t0 = time.time()
            S = pre(S)
            x = place(S['x'])
            e, fm = provider(np.asarray(x))
            S, out = post(S, x, jnp.asarray(e), jnp.asarray(fm))
            S['x'].block_until_ready()
            dt_ = time.time() - t0
            t_md += dt_; t_md_since += dt_; steps_since += 1
            if args.compare_every and (step % args.compare_every == 0 or step == args.steps):
                compare(step, x, e, fm)
            if step % args.print_every == 0 or step == args.steps:
                _emit(row(step, float(out['pe']), float(out['ke']), float(out['hext']),
                          t_md_since / steps_since * 1000))
                t_md_since = 0.0; steps_since = 0
            if restart_dir and args.restart_every and step % args.restart_every == 0 \
                    and step < args.steps:
                save_restart(S, step)

    n_run = done if args.propagate == 'pff' else args.steps
    _emit(f"=== done: {t_md:.1f}s MD, {t_md/max(n_run, 1)*1000:.1f} ms/step avg ===")
    if restart_dir:
        if exit_code == 0:
            S_end = carry['S'] if args.propagate == 'pff' else S
            save_restart(S_end, n_run)
            _emit(f"restart: {restart_dir}  (step {n_run}; continue with --input-dir {restart_dir})")
        else:
            _emit("restart: not written (the final state is not valid)")
    if nb_summary:
        _emit(nb_summary)
    if frozen is not None and args.propagate == 'pff':
        xe = np.asarray(carry['S']['x'])[frozen]; x0e = np.asarray(Dj['x'])[frozen]
        dxe = xe - x0e
        dxe -= np.asarray(D['box']) * np.round(dxe / np.asarray(D['box']))
        _emit(f"# electrodes: max displacement of the {int(frozen.sum())} frozen atoms over the run "
              f"{float(np.abs(dxe).max()):.3e} A, max |v| "
              f"{float(np.abs(np.asarray(carry['S']['v'])[frozen]).max()):.3e} m/s")
    if provider is not None:
        _emit(f"MLIP: {provider.ncall} evaluations, {provider.seconds/max(provider.ncall,1)*1000:.0f} ms each")
        for line in comparator.summary():
            _emit(line)
        comparator.close()
        provider.close()
    _logf.close()
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
