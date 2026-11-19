"""
Ensemble validation of the GOMD integrator (NVE timestep scaling, NVT
canonical statistics, equipartition, velocities along constrained bonds).

    python -m gomd.tests.ensemble_check nve --input-dir ./md-input --ref-dir ./pkg \
        --polarizable --dts 0.25,0.5,1.0 --time 1000
    python -m gomd.tests.ensemble_check nvt --input-dir ./md-input --ref-dir ./pkg \
        --polarizable --equil 30000 --steps 40000

Observables are computed on the GPU every --sample-every steps:
  * KE, PE, TE (NVE) or H_ext (NVT), every step, from the scan outputs
  * translational (centre-of-mass) temperature of each group of molecules:
    Li+, FSI-, solvent. Internal constraints do not touch these 3 dof per
    molecule, so in equilibrium each must equal T (equipartition).
  * velocity along the constrained bonds: u = (v_i - v_j) . r_ij/|r_ij|.
    SHAKE fixes the bond lengths, but this integrator has no RATTLE velocity
    step, so u need not be 0; KE_c = sum 1/2 mu_ij u^2 is kinetic energy in
    degrees of freedom the temperature (ndof = 3 N - N_con - 3) does not count.
NVE: TE drift and its fluctuation about the fit vs dt (velocity Verlet:
fluctuation ~ dt^2). NVT (after --equil steps): mean T vs Tset, the KE
fluctuation vs the canonical sqrt(ndof/2) k Tset, group temperatures.
Errors are standard errors from 10 block averages.
"""
from __future__ import annotations
import argparse, math, os, sys, time

KCAL = 4184.0 * 1000.0          # (g/mol)(m/s)^2 -> kcal/mol
KB = 1.98709e-3                 # kcal/(mol K)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['nve', 'nvt'])
    ap.add_argument('--input-dir', default='./md-input')
    ap.add_argument('--ref-dir', default='./pkg')
    ap.add_argument('--polarizable', action='store_true')
    ap.add_argument('--tolpol', type=float, default=1e-8)
    ap.add_argument('--dts', default='0.25,0.5,1.0', help='NVE: timesteps (fs)')
    ap.add_argument('--time', type=float, default=1000.0, help='NVE: simulated time per dt (fs)')
    ap.add_argument('--equil', type=int, default=30000, help='NVT: equilibration steps (discarded)')
    ap.add_argument('--steps', type=int, default=40000, help='NVT: production steps')
    ap.add_argument('--sample-every', type=int, default=10)
    ap.add_argument('--then-nve', type=int, default=0,
                    help='NVT: continue N steps in NVE from the equilibrated state')
    ap.add_argument('--rtaper', type=float, default=None,
                    help='override the dipole-dipole taper start (A); < rcut turns the taper on')
    ap.add_argument('--out', default=None, help='report file (default md-output/ensemble_<mode>.txt)')
    args = ap.parse_args(argv)
    from jax import config as jc
    jc.update('jax_enable_x64', True)
    import jax, jax.numpy as jnp
    import numpy as np
    from gomd.tests.common import build_md, molecules

    out = args.out or os.path.join('md-output', f'ensemble_{args.mode}.txt')
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    fo = open(out, 'w')
    def emit(s=''):
        print(s, flush=True); fo.write(s + '\n'); fo.flush()

    def make_obs(B, mol, nmol, gid, ngroup):
        D, Dj = B['D'], B['Dj']
        mass = Dj['mass']; nd = (~Dj['ldummy'])[:, None]; box = Dj['box']
        con = Dj['con']; ci, cj = con['con_i'], con['con_j']
        mi, mj = mass[ci], mass[cj]
        mred = mi * mj / (mi + mj)
        molj = jnp.asarray(mol); gidj = jnp.asarray(gid)

        @jax.jit
        def obs(S):
            v = jnp.where(nd, S['v'], 0.0); x = S['x']
            ke = 0.5 * jnp.sum(mass[:, None] * v * v) / KCAL
            P = jax.ops.segment_sum(mass[:, None] * v, molj, num_segments=nmol)
            M = jax.ops.segment_sum(mass, molj, num_segments=nmol)
            ke_mol = 0.5 * jnp.sum(P * P, axis=1) / jnp.where(M > 0, M, 1.0) / KCAL
            ke_grp = jax.ops.segment_sum(ke_mol, gidj, num_segments=ngroup)
            d = x[ci] - x[cj]
            d = d - box * jnp.round(d / box)
            rhat = d / jnp.sqrt(jnp.sum(d * d, axis=1))[:, None]
            u = jnp.sum((v[ci] - v[cj]) * rhat, axis=1)
            ke_con = 0.5 * jnp.sum(mred * u * u) / KCAL
            return jnp.concatenate([jnp.stack([ke, ke_con, jnp.sqrt(jnp.mean(u * u))]), ke_grp])
        return obs

    def run_chunks(B, nsteps, chunk, obs=None, label=''):
        run = B['run']; carry = B['carry0'] if '_carry' not in B else B['_carry']
        pe, ke, h, o = [], [], [], []
        done = 0; t0 = time.perf_counter(); tlast = t0
        while done < nsteps:
            n = min(chunk, nsteps - done)
            carry, hist = run(carry, nsteps=n)
            hh = jax.device_get(hist)
            pe.append(hh['pe']); ke.append(hh['ke']); h.append(hh['hext'])
            if int(np.max(hh['ncut'])) > B['cut_capacity'] or int(jax.device_get(carry['nb']['overflow'])):
                raise SystemExit('pair list overflow — results invalid')
            if obs is not None:
                o.append(np.asarray(obs(carry['S'])))
            done += n
            if time.perf_counter() - tlast > 60:
                tlast = time.perf_counter()
                T = 2 * float(np.mean(hh['ke'])) / (B['E']['ndof'] * KB)
                print(f"    {label} {done}/{nsteps} steps, T {T:.1f} K, "
                      f"{(tlast - t0) / done * 1e3:.1f} ms/step", flush=True)
        B['_carry'] = carry
        return (np.concatenate(pe), np.concatenate(ke), np.concatenate(h),
                (np.array(o) if o else None), (time.perf_counter() - t0) / nsteps * 1e3)

    def block_err(a, nb=10):
        a = np.asarray(a); m = len(a) // nb
        if m < 1:
            return float('nan')
        bm = a[:m * nb].reshape(nb, m).mean(axis=1)
        return float(bm.std(ddof=1) / math.sqrt(nb))

    def block_sd_err(a, nb=10):
        a = np.asarray(a); m = len(a) // nb
        bs = a[:m * nb].reshape(nb, m).std(axis=1)
        return float(bs.std(ddof=1) / math.sqrt(nb))

    emit(f"# GOMD ensemble check ({args.mode})  {time.strftime('%Y-%m-%d %H:%M:%S')}  "
         f"jax {jax.__version__}  {jax.devices()[0].device_kind}")
    emit(f"# {'polarizable, tolpol %g (PCG + predict)' % args.tolpol if args.polarizable else 'classical'}"
         f"; SHAKE 1e-12, drift rebuilds, rcut compaction, shared k-space")

    if args.mode == 'nve':
        rows = []
        for dt in [float(s) for s in args.dts.split(',')]:
            B = build_md(args.input_dir, args.ref_dir, polarizable=args.polarizable,
                         tolpol=args.tolpol, ensemble='nve', dt=dt, rtaper=args.rtaper)
            mol, nmol, groups = molecules(B['D'])
            names = sorted(set(groups)); gid = np.array([names.index(g) for g in groups])
            obs = make_obs(B, mol, nmol, gid, len(names))
            nsteps = int(round(args.time / dt))
            pe, ke, te, o, ms = run_chunks(B, nsteps, args.sample_every, obs, f"dt {dt}")
            t = np.arange(1, nsteps + 1) * dt
            use = t >= 10.0
            p = np.polyfit(t[use], te[use], 1)
            sd_te = float(np.std(te[use] - np.polyval(p, t[use])))
            sd_ke = float(np.std(ke[use]))
            kec = o[:, 1] / o[:, 0]
            rows.append((dt, nsteps, p[0], sd_te, sd_ke, float(kec.mean()), float(o[:, 2].mean()), ms))
            emit(f"  dt {dt:5.2f} fs  {nsteps:5d} steps  TE drift {p[0]:+.5f} kcal/mol/fs  "
                 f"sd(TE) {sd_te:.4f}  sd(KE) {sd_ke:.2f}  sd(TE)/sd(KE) {sd_te / sd_ke:.1e}  "
                 f"KE along constrained bonds {100 * kec.mean():.3f}% of KE "
                 f"(rms u {o[:, 2].mean():.1f} m/s)  {ms:.1f} ms/step")
        emit("\n## fluctuation scaling (velocity Verlet: sd(TE) ~ dt^2)")
        d0, s0 = rows[0][0], rows[0][3]
        for dt, n, dr, sd, sk, kc, ur, ms in rows:
            emit(f"  dt {dt:5.2f}: sd(TE)/sd(TE at {d0:g}) = {sd / s0:6.2f}   (dt/{d0:g})^2 = "
                 f"{(dt / d0) ** 2:6.2f}")
    else:
        B = build_md(args.input_dir, args.ref_dir, polarizable=args.polarizable,
                     tolpol=args.tolpol, ensemble='nvt', rtaper=args.rtaper)
        E = B['E']; ndof = E['ndof']; Tset = E['Tset']
        mol, nmol, groups = molecules(B['D'])
        names = sorted(set(groups)); gid = np.array([names.index(g) for g in groups])
        nmol_g = np.array([groups.count(g) for g in names])
        obs = make_obs(B, mol, nmol, gid, len(names))
        emit(f"# NVT: Tset {Tset} K, Nose-Hoover chain {E['nnos']}, OmegaQ {E['qfreq']} 1/fs, "
             f"ndof {ndof}, dt {B['D']['dt']} fs; molecules: "
             + ', '.join(f"{g} {c}" for g, c in zip(names, nmol_g)))
        emit(f"## equilibration: {args.equil} steps ({args.equil * B['D']['dt'] / 1000:.1f} ps)")
        pe, ke, h, _, ms = run_chunks(B, args.equil, 1000, None, 'equil')
        T = 2 * ke / (ndof * KB)
        per_ps = int(round(1000 / B['D']['dt']))
        emit("  T per ps: " + ' '.join(f"{T[k:k + per_ps].mean():.0f}" for k in range(0, len(T), per_ps))
             + f"   ({ms:.1f} ms/step)")
        emit(f"## production: {args.steps} steps ({args.steps * B['D']['dt'] / 1000:.1f} ps), "
             f"observables every {args.sample_every} steps")
        pe, ke, h, o, ms = run_chunks(B, args.steps, args.sample_every, obs, 'prod')
        T = 2 * ke / (ndof * KB)
        sig_can = math.sqrt(ndof / 2.0) * KB * Tset
        t = np.arange(len(h)) * B['D']['dt']
        hd = np.polyfit(t, h, 1)[0]
        emit(f"  T = {T.mean():.2f} +- {block_err(T):.2f} K   (Tset {Tset})")
        emit(f"  sd(KE) = {ke.std():.2f} +- {block_sd_err(ke):.2f} kcal/mol   canonical "
             f"sqrt(ndof/2) k Tset = {sig_can:.2f}   ratio {ke.std() / sig_can:.3f}")
        emit(f"  H_ext drift {hd:+.5f} kcal/mol/fs, sd {np.std(h - np.polyval(np.polyfit(t, h, 1), t)):.3f}"
             f"   ({ms:.1f} ms/step)")
        kc = o[:, 1] / o[:, 0]
        emit(f"  KE along constrained bonds: {100 * kc.mean():.3f}% of KE (rms u {o[:, 2].mean():.1f} m/s);"
             f" in temperature units {2 * o[:, 1].mean() / (ndof * KB):.2f} K")

        def equipartition(o, label):
            emit(f"  {label}: translational (COM) temperature per group, 3 dof per molecule:")
            ke_com = np.zeros(len(o))
            for k, g in enumerate(names):
                Tg = 2 * o[:, 3 + k] / (3 * nmol_g[k] * KB)
                ke_com += o[:, 3 + k]
                emit(f"    {g:<8} {nmol_g[k]:4d} molecules  T_com = {Tg.mean():7.2f} +- "
                     f"{block_err(Tg):5.2f} K")
            dof_int = ndof - 3 * int(nmol_g.sum())
            Ti = 2 * (o[:, 0] - ke_com) / (dof_int * KB)
            emit(f"    internal ({dof_int} dof = ndof - 3 x molecules): T = {Ti.mean():7.2f} +- "
                 f"{block_err(Ti):5.2f} K;  all ndof: T = {(2 * o[:, 0] / (ndof * KB)).mean():.2f} K")
        equipartition(o, 'production')
        o0 = np.asarray(obs(B['carry0']['S']))[None, :]
        emit("  for reference, the coords.inp velocities (t = 0, before any dynamics):")
        for k, g in enumerate(names):
            emit(f"    {g:<8} T_com = {2 * o0[0, 3 + k] / (3 * nmol_g[k] * KB):7.2f} K")
        ke_com0 = o0[0, 3:].sum()
        emit(f"    internal T = {2 * (o0[0, 0] - ke_com0) / ((ndof - 3 * int(nmol_g.sum())) * KB):.2f} K;"
             f"  all ndof T = {2 * o0[0, 0] / (ndof * KB):.2f} K")
        np.savez(os.path.splitext(out)[0] + '.npz', pe=pe, ke=ke, hext=h, obs=o, groups=names)

        if args.then_nve > 0:
            B2 = build_md(args.input_dir, args.ref_dir, polarizable=args.polarizable,
                          tolpol=args.tolpol, ensemble='nve', rtaper=args.rtaper)
            B2['_carry'] = B['_carry']           # same pytree (chain length 1)
            emit(f"## NVE continuation from the equilibrated NVT state: {args.then_nve} steps "
                 f"(no thermostat)")
            pe2, ke2, te2, o2, ms2 = run_chunks(B2, args.then_nve, args.sample_every, obs, 'nve')
            t2 = np.arange(len(te2)) * B2['D']['dt']
            emit(f"  TE drift {np.polyfit(t2, te2, 1)[0]:+.5f} kcal/mol/fs; T = "
                 f"{(2 * ke2 / (ndof * KB)).mean():.2f} K   ({ms2:.1f} ms/step)")
            equipartition(o2, 'NVE')
    fo.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
