"""
mlip_md.py — molecular dynamics driven by an ML potential, on a lone-pair-free
system read from extended XYZ (no force field, no connectivity, no constraints).

    python -m gomd.mlip_md --input md-input2/system.xyz --mlip mock --steps 2000
    python -m gomd.mlip_md --input md-input2/system.xyz --mlip uma --ensemble nvt \
        --mlip-python /path/to/uma_env/bin/python --steps 2000 --fd-check 4

The MLIP runs in its own process (gomd/mlip/server.py: UMA via fairchem, or the
numpy mock), exactly as for main_scan; only real atoms exist here.

Integrator: velocity Verlet in the PFF integrator's units (x A, v m/s, dt fs,
F kcal/mol/A, m g/mol: v += dt/2 * 41.84 F/m, x += 1e-5 dt v), every bond
flexible, so dt <= 0.5 fs. The centre-of-mass momentum is removed at the
start (and after each NVT step, as a uniform shift). ndof = 3N - 3.
NVT: the same Nose-Hoover chain as main_scan (gomd/thermostat.py, MTK,
5 x 5 Yoshida-Suzuki), state carried in the restart file.

Log: NVE  step PE KE TE ms/step mlip_ms;  NVT  step PE KE TE T H_ext ms/step mlip_ms.
PE is the MLIP energy (kcal/mol, absolute: its zero is the model's).
--fd-check N: before the run, compare the forces with central finite
differences of the energy on N random atoms x 3 components.
Restart: <out>_restart.xyz (positions, velocities, chain state), at the end
and every --restart-every steps; continue with --input <that file>.
"""
from __future__ import annotations
import argparse, math, os, sys, time

UNIT = 41.84            # (kcal/mol/A) / (g/mol) -> m/s per fs
VSCALE = 1.0e-5         # m/s * fs -> A
KCAL = 4184.0 * 1000.0  # (g/mol)(m/s)^2 -> kcal/mol
KB = 1.98709e-3         # kcal/(mol K)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description='GOMD MLIP-driven MD on an extended-XYZ system')
    ap.add_argument('--input', required=True, help='extended XYZ (md-input2/system.xyz or a restart)')
    ap.add_argument('--mlip', choices=['uma', 'mock'], required=True)
    ap.add_argument('--ensemble', choices=['nve', 'nvt'], default='nve')
    ap.add_argument('--dt', type=float, default=0.5, help='fs')
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--print-every', type=int, default=10)
    ap.add_argument('--temp', type=float, default=296.15, help='NVT set point, K')
    ap.add_argument('--nhc-chain', type=int, default=3,
                    help='chain length; 1 (mdrun.params) rings at the thermostat period on a '
                         'flexible cluster: sd(T) 24 K vs 5.9 K with 3 (canonical 5.4 K), mock, 1 ps')
    ap.add_argument('--omega-q', type=float, default=0.01, help='NHC frequency, 1/fs')
    ap.add_argument('--fd-check', type=int, default=0, help='finite-difference force check on N atoms')
    ap.add_argument('--fd-h', type=float, default=1e-2,
                    help='A; 4-point stencil, error O(h^4) plus ~eps/h for an energy rough at eps '
                         '(UMA fp32: eps ~1e-3 kcal/mol)')
    ap.add_argument('--out', default='md-output/mlip_md.out')
    ap.add_argument('--restart-every', type=int, default=0)
    ap.add_argument('--traj-every', type=int, default=0, help='append positions to <out>.traj.xyz')
    g = ap.add_argument_group('MLIP')
    g.add_argument('--mlip-model', default='uma-s-1p2p1')
    g.add_argument('--mlip-task', default='omol')
    g.add_argument('--mlip-settings', default='default')
    g.add_argument('--mlip-device', default='auto')
    g.add_argument('--mlip-python', default=None)
    g.add_argument('--charge', type=int, default=None, help='default: from the XYZ (charge=)')
    g.add_argument('--spin', type=int, default=None, help='default: from the XYZ (spin=)')
    g.add_argument('--mock-k', type=float, default=10.0)
    g.add_argument('--mock-rbond', type=float, default=1.9)
    g.add_argument('--seed', type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')        # the thermostat math only; the GPU is the MLIP's
    # fp64, as in main_scan. Without it nhc_half rounds v and the chain state to fp32
    # every half step: H_ext then drifts ~ -6e-4 kcal/mol per step whatever dt is
    # (2,002 atoms; found 2026-11-18 from a UMA NVT run and the mock).
    import jax
    jax.config.update('jax_enable_x64', True)
    import numpy as np
    from gomd.io.xyz import read_extxyz, write_extxyz, info_float, info_array
    from gomd.mlip.client import MLIPProvider, EV_TO_KCAL
    from gomd.mlip.elements import formula
    from gomd import thermostat as nhc_mod

    args.out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    logf = open(args.out, 'w')
    def emit(s=''):
        print(s, flush=True); logf.write(s + '\n'); logf.flush()

    S = read_extxyz(args.input)
    x = np.asarray(S['x'], dtype=np.float64).copy()
    n = x.shape[0]
    box = S['box']
    v = (np.asarray(S['v']) if S['v'] is not None else np.zeros((n, 3))).astype(np.float64)
    if S['mass'] is not None:
        m = np.asarray(S['mass'], dtype=np.float64)
    else:
        from gomd.mlip.elements import _WEIGHTS
        m = np.array([_WEIGHTS[int(z)][1] for z in S['numbers']])
    info = S['info']
    charge = args.charge if args.charge is not None else int(info_float(info, 'charge', 0))
    spin = args.spin if args.spin is not None else int(info_float(info, 'spin', 1))
    t0_ps = info_float(info, 'time_ps', 0.0)
    ndof = 3 * n - 3
    rmass = (UNIT / m)[:, None]
    M = m.sum()
    v -= (m[:, None] * v).sum(0) / M                     # zero total momentum (uniform shift)
    pbc = bool(S.get('pbc', True))                       # False: a cluster in vacuum
    if pbc:
        x = x - box * np.floor(x / box)
    ke = lambda vv: 0.5 * float(np.sum(m[:, None] * vv * vv)) / KCAL

    emit(f"# GOMD MLIP MD  {time.strftime('%Y-%m-%d %H:%M:%S')}  input {args.input}")
    emit(f"# {n} atoms ({formula(S['numbers'])}), "
         + ("periodic box " if pbc else "cluster, no PBC (frame ")
         + ' x '.join(f'{b:.4f}' for b in box) + ("" if pbc else ")")
         + f" A, charge {charge}, spin {spin}, ndof {ndof} (no constraints), dt {args.dt} fs, "
         + (f"NVT {args.temp} K (NHC chain {args.nhc_chain}, OmegaQ {args.omega_q} 1/fs)"
            if args.ensemble == 'nvt' else "NVE"))
    extra = (['--mock-k', str(args.mock_k), '--mock-rbond', str(args.mock_rbond)]
             if args.mlip == 'mock' else [])
    t = time.time()
    prov = MLIPProvider.start(args.mlip, S['numbers'], box, x, python=args.mlip_python,
                              model=args.mlip_model, task=args.mlip_task, settings=args.mlip_settings,
                              device=args.mlip_device, charge=charge, spin=spin, extra_args=extra,
                              pbc=pbc)
    emit(f"# MLIP {args.mlip} ready in {time.time() - t:.1f} s: {prov.info}")
    e, f = prov(x)
    emit(f"# initial: PE {e:.4f} kcal/mol ({e / EV_TO_KCAL / n:.6f} eV/atom), KE {ke(v):.4f} kcal/mol, "
         f"T {2 * ke(v) / (ndof * KB):.2f} K, |F| rms {math.sqrt(float(np.mean(f * f))):.3f} kcal/mol/A, "
         f"sum F {np.abs(f.sum(0)).max():.2e}")

    def fd_check(x, f, label):
        # 4-point central stencil. Its truncation error is O(h^4), so h can be large
        # enough that the energy's own roughness eps stays small once divided by h:
        # UMA in fp32 is rough at ~1e-3 kcal/mol, which a 2-point stencil with
        # h = 1e-3 A turns into ~0.5 kcal/mol/A of noise (sphere_2000, 2026-11-18).
        rng = np.random.default_rng(args.seed)
        worst = worst_abs = 0.0; h = args.fd_h
        for a in rng.choice(n, args.fd_check, replace=False):
            for k in range(3):
                def e_at(s):
                    xs = x.copy(); xs[a, k] += s * h
                    return prov(xs)[0]
                fd = -(8.0 * (e_at(1) - e_at(-1)) - (e_at(2) - e_at(-2))) / (12.0 * h)
                dabs = abs(fd - f[a, k])
                rel = dabs / max(abs(f[a, k]), 1e-2)
                worst = max(worst, rel); worst_abs = max(worst_abs, dabs)
                emit(f"#   fd-check atom {a:5d} {S['symbols'][a]:<2s} {'xyz'[k]}  F {f[a, k]:+10.5f}  "
                     f"-dE/dx {fd:+10.5f}  diff {fd - f[a, k]:+.2e}  rel diff {rel:.1e}")
        emit(f"# fd-check at the {label} configuration (4-point stencil, h = {h} A, |F| floor 0.01): "
             f"max rel diff {worst:.1e}, max abs diff {worst_abs:.2e} kcal/mol/A")

    def energy_probe(x, e, f):
        """How finely does the MLIP resolve energy? An absolute (e.g. DFT-referenced)
        total energy summed in fp32 moves in steps of ~1e-7 |E|, which would make
        finite differences and NVE energy checks meaningless."""
        e_again = prov(x)[0]
        a = int(np.argmax(np.sum(f * f, axis=1)))
        fa = float(np.sqrt(np.sum(f[a] ** 2)))
        u = f[a] / fa
        emit(f"# energy probe: |E| = {abs(e):.1f} kcal/mol (fp32 step there ~ "
             f"{abs(e) * 2 ** -23:.3g}); same configuration again: dE {e_again - e:+.3e}")
        for s in (1e-4, 1e-3, 1e-2):
            xp = x.copy(); xp[a] += s * u
            de = prov(xp)[0] - e
            emit(f"#   atom {a} ({S['symbols'][a]}) moved {s:g} A along F: dE {de:+.6f} kcal/mol, "
                 f"expected -|F| s = {-fa * s:+.6f}")

    if args.fd_check > 0 and float(np.abs(f).max()) > 1e-6:
        energy_probe(x, e, f)
        fd_check(x, f, 'initial')

    nhc = P = None
    if args.ensemble == 'nvt':
        P = nhc_mod.make_nhc_params(args.temp, ndof, args.nhc_chain, args.omega_q, args.dt)
        st = [info_array(info, k) for k in ('nhc_xlogs', 'nhc_vlogs', 'nhc_glogs')]
        if all(s is not None and s.size == args.nhc_chain for s in st):
            nhc = tuple(st); emit("# NHC chain state from the input file")
        else:
            if st[0] is not None:
                emit(f"# NHC chain state in the input has length {st[0].size}, not {args.nhc_chain}: "
                     f"starting a fresh chain")
            nhc = tuple(np.zeros(args.nhc_chain) for _ in range(3))
    nd = np.ones((n, 1), dtype=bool)
    if P is not None:
        # Compiled once. Run eagerly, the chain's unrolled 5 x 5 scalar updates dispatch
        # one op at a time: 6 / 19 ms per step for chains of 1 / 3 locally, ~57 ms per
        # step on Colab's CPU (2026-11-18).
        nhc_half = jax.jit(lambda v_, st_: nhc_mod.nhc_half(v_, m, nd, st_, P))
        nhc_energy = jax.jit(lambda st_: nhc_mod.nhc_energy(st_, P))
        nhc = tuple(np.asarray(s, dtype=np.float64) for s in nhc)

    def save_restart(step):
        path = os.path.splitext(args.out)[0] + '_restart.xyz'
        inf = {'charge': charge, 'spin': spin, 'time_ps': t0_ps + step * args.dt / 1000.0,
               'velocity_units': 'm/s', 'source': args.input}
        if nhc is not None:
            inf.update(nhc_xlogs=np.asarray(nhc[0]), nhc_vlogs=np.asarray(nhc[1]),
                       nhc_glogs=np.asarray(nhc[2]))
        write_extxyz(path, S['symbols'], x, box, v=v, mass=m, site=S['site'], info=inf, pbc=pbc)
        return path

    traj = open(os.path.splitext(args.out)[0] + '.traj.xyz', 'w') if args.traj_every else None
    nvt = args.ensemble == 'nvt'
    emit(f"{'step':>6} {'PE':>16} {'KE':>12} {'TE':>16}"
         + (f" {'T':>8} {'H_ext':>16}" if nvt else '') + f" {'ms/step':>9} {'mlip_ms':>8}")
    tm = 0.0; tml = 0.0; cnt = 0; tml_tot = 0.0; t_all = time.time()
    cons = []                                            # TE (NVE) or H_ext (NVT) per step
    temps = []
    for step in range(1, args.steps + 1):
        t = time.perf_counter()
        if nvt:
            v, nhc = nhc_half(v, nhc); v = np.array(v)                     # writable copy
        v = v + 0.5 * args.dt * rmass * f
        x = x + args.dt * VSCALE * v
        if pbc:
            x = x - box * np.floor(x / box)
        tq = time.perf_counter()
        e, f = prov(x)
        dq = time.perf_counter() - tq
        v = v + 0.5 * args.dt * rmass * f
        if nvt:
            v, nhc = nhc_half(v, nhc); v = np.array(v)                     # writable copy
            v -= (m[:, None] * v).sum(0) / M
        tm += time.perf_counter() - t; tml += dq; tml_tot += dq; cnt += 1
        k_ = ke(v)
        h = e + k_ + (float(nhc_energy(nhc)) if nvt else 0.0)
        cons.append(h); temps.append(2 * k_ / (ndof * KB))
        if step % args.print_every == 0 or step == args.steps:
            line = f"{step:>6} {e:>16.4f} {k_:>12.4f} {e + k_:>16.4f}"
            if nvt:
                line += f" {2 * k_ / (ndof * KB):>8.2f} {h:>16.4f}"
            emit(line + f" {tm / cnt * 1e3:>9.1f} {tml / cnt * 1e3:>8.1f}")
            tm = tml = 0.0; cnt = 0
        if traj is not None and step % args.traj_every == 0:
            traj.write(f"{n}\nLattice=\"{box[0]} 0 0 0 {box[1]} 0 0 0 {box[2]}\" Properties=species:S:1:pos:R:3 pbc=\"{'T T T' if pbc else 'F F F'}\" "
                       f"step={step} time_ps={t0_ps + step * args.dt / 1000:.4f}\n")
            traj.write('\n'.join(f"{S['symbols'][i]} {x[i, 0]:.6f} {x[i, 1]:.6f} {x[i, 2]:.6f}"
                                 for i in range(n)) + '\n')
        if args.restart_every and step % args.restart_every == 0 and step < args.steps:
            save_restart(step)
    wall = time.time() - t_all
    c = np.asarray(cons); tt = np.arange(1, len(c) + 1) * args.dt
    use = tt >= 0.1 * tt[-1]                             # skip the first 10% (start-up)
    if use.sum() > 2:
        p = np.polyfit(tt[use], c[use], 1)
        emit(f"# {'H_ext' if nvt else 'TE'}: drift {p[0]:+.5f} kcal/mol/fs, sd about the fit "
             f"{np.std(c[use] - np.polyval(p, tt[use])):.4f} kcal/mol, mean minus initial "
             f"{c[use].mean() - c[0]:+.4f}; T mean {np.mean(np.asarray(temps)[use]):.2f} K "
             f"(sd {np.std(np.asarray(temps)[use]):.2f})")
    if args.fd_check > 0:
        energy_probe(x, e, f)
        fd_check(x, f, 'final')
    path = save_restart(args.steps)
    emit(f"=== done: {args.steps} steps in {wall:.1f} s ({wall / max(args.steps, 1) * 1e3:.1f} ms/step; "
         f"MLIP {tml_tot / max(args.steps, 1) * 1e3:.1f} ms/step, {prov.ncall} evaluations) ===")
    emit(f"restart: {path}")
    if traj is not None:
        traj.close()
    prov.close()
    logf.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
