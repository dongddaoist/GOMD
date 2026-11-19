"""
integrator_scan.py — the GOMD MD step, on device.

One MD step is split into three pieces so that the same arithmetic drives both
ways of running it:

    pre_force(S)                 thermostat half step (NVT), first half kick,
                                 drift + PBC wrap, SHAKE of the drifted positions
    <force evaluation at S['x']> PFF on device, or an MLIP on the host
    post_force(S, x, pe, f)      SHAKE predictor-corrector (combined force),
                                 second half kick, thermostat half step and
                                 COM-velocity removal (NVT), KE / H_ext

S = {'x', 'v', 'f', 'nhc', 'step'} is the small dynamic state; 'f' is the
SHAKE-combined force used by the next kick, 'nhc' the thermostat chain state
(xlogs, vlogs, glogs).

  * PFF propagation: scan_body = neighbor rebuild (lax.cond) + pre_force +
    PFF force + post_force, and the whole trajectory runs inside one lax.scan
    (no Python, no host transfers between steps). The scan carry adds the
    neighbor pairs, the positions they were built at, neighbor-list
    statistics, the dipole history, and the last physical PE and force (so a
    parallel MLIP evaluation can be compared with exactly the forces the
    integrator used). Build it with init_carry().
  * Neighbor rebuild: every rebuild_every steps (0 = never on a schedule), at
    the start of the step as before, and, with drift_rebuild, after the drift
    whenever any site of the force-evaluation configuration is more than
    skin/2 - drift_margin from where it was at the last build (the list then
    holds every pair inside rcut for this step's force, for any dt). Each
    rebuild also counts the pairs inside rcut that the stale list was missing,
    and whether the fresh list overflowed its fixed capacity (carry['nb']).
  * MLIP propagation: the driver calls jit(pre_force), the MLIP on the host,
    then jit(post_force) once per step (see gomd/mlip).

The step follows the reference pkg/integrator.py _vanilla_vv_step and
run_simulation:
  [NHC half step] -> kick -> drift -> SHAKE(x, bond vectors from the pre-drift
  x) -> forces -> SHAKE predictor-corrector -> kick -> [NHC half step ->
  COM-velocity removal]            ([...] = NVT only)
SHAKE after the drift is applied in NVE as well, as in the reference. Once the
positions are on the constraints the predictor-corrector already lands the
drift there and this SHAKE exits at once; it matters on the first step (the
input velocities have components along the constrained bonds) and whenever
velocities are rescaled (NVT). Without it (the Nov 9 engine) the first NVE
step showed a one-off ~8 kcal/mol total-energy dip.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from functools import partial

from .forces.all_forces import compute_all_forces
from .forces.dummy import update_dummy_coords, dummy_force_redistribution
from .constraints.shake import shake_constrain
from .forces.neighbor_ondevice import (build_neighbor_ondevice_count, count_pairs_within,
                                       compact_pairs)
from . import thermostat as nhc_mod

UNIT = 41.84
VSCALE = 1.0e-5


def _mi(d, box):
    return d - box * jnp.round(d / box)


def make_md_step(Dj, kx, ky, kz, ewself, excl_flat, capacity,
                 rebuild_every, rlist2, polarization_fn=None, use_aspc=False,
                 nhc_params=None, natreal=None, shake_max_iter=100,
                 skin=1.0, drift_rebuild=True, drift_margin=0.01,
                 cut_capacity=None, cut_margin=0.05, recip=None, frozen=None):
    """Build the step pieces.

    frozen: optional (nat,) bool, atoms held fixed (the electrodes,
        gomd/electrodes.py). They are not kicked, moved or thermostatted (their
        velocities must be zero), the neighbor build skips frozen-frozen pairs,
        and the NVT COM-velocity removal is off: the fixed atoms take momentum
        from the mobile ones, so the mobile COM momentum is not conserved and
        ndof does not count it.

    nhc_params: None for NVE, or thermostat.make_nhc_params(...) for NVT.
    natreal: number of real (non-lone-pair) atoms, needed for NVT COM removal.
    shake_max_iter: cap on SHAKE iterations (tolerance: Dj['con']['tol']).

    rebuild_every: scheduled neighbor rebuild interval in steps (0 = none).
    skin: rlist - rcut (A); rlist2 must be (rcut + skin)^2.
    drift_rebuild: also rebuild when any site has moved more than
        skin/2 - drift_margin (A) since the last build. The test runs after
        the drift, on the positions the force is evaluated at (lone-pair
        sites placed), so it holds for any dt; drift_margin only absorbs the
        fp32 rounding of the build.

    cut_capacity: if set, force_fn first compacts the rlist pairs to those
        inside rcut + cut_margin (fp32 test), into cut_capacity slots, and every
        pair kernel runs on that shorter list (about 2.9M instead of 4.19M
        slots here). The margin keeps every pair either kernel's own rcut test
        (fp32 classical, fp64 polarization) could accept. comps['ncut'] is the
        count; ncut > cut_capacity means pairs were dropped (the driver stops).

    recip: the shared k-space pass (forces.recip_shared.make_recip). If set,
        force_fn builds its cos/sin(k.r) tables once per step and takes the
        classical reciprocal energy and force, the SCF's reciprocal field and
        the k-space Q-mu force from them (pass the same bundle to the
        polarization function). None: each term does its own pass.

    use_aspc: warm-start the SCF from mu_pred = 3*(mu1 - mu2) + mu3. Default
        False: with the Jacobi SCF and the reference max|dmu|^2 <= tolpol test a
        warm start can stop at a wrong fixed point (a 1% predictor error gave
        E_pol ~24 kcal/mol off). Cold start converges to the reference dipoles.
        The PCG solver (polarization_full solver='pcg') is the one meant to
        be warm-started.

    Returns a dict with: force_fn, place_dummies, rebuild, count_within,
    pre_force, post_force, scan_body, init_carry, nb_trigger.
    rebuild(x) -> (idx_i, idx_j, count); count > capacity means overflow.
    """
    nat = Dj['nat']
    box = Dj['box']
    dt = Dj['dt']
    mass = Dj['mass']
    massinv = Dj['massinv']
    ldummy = Dj['ldummy']
    con = Dj['con']
    rcut2 = float(Dj['rcut']) ** 2
    rebuild_every = int(rebuild_every)
    nb_trigger = 0.5 * float(skin) - float(drift_margin)
    if drift_rebuild and nb_trigger <= 0.0:
        raise ValueError(f"skin {skin} A is too small for a drift trigger "
                         f"(skin/2 - margin = {nb_trigger:.3f} A)")
    trig2 = nb_trigger * nb_trigger

    jik = Dj['dummy']['jik']
    adummy = Dj['dummy']['adummy']
    bdummy = Dj['dummy']['bdummy']
    cdummy = Dj['dummy']['cdummy']
    has_dummy = Dj['dummy']['ndummy'] > 0
    do_shake = con['constrain'] and con['nbcon'] > 0
    has_pol = polarization_fn is not None
    nvt = nhc_params is not None
    if nvt and natreal is None:
        raise ValueError("NVT needs natreal for COM-velocity removal")

    rmass = UNIT * massinv
    has_frozen = frozen is not None and bool(jnp.any(jnp.asarray(frozen)))
    frozen = jnp.asarray(frozen, dtype=bool) if has_frozen else None
    # nd: the sites the integrator moves, kicks and thermostats (real, not frozen)
    nd = ((~ldummy) & (~frozen) if has_frozen else ~ldummy)[:, None]

    def place_dummies(x):
        if has_dummy:
            x = update_dummy_coords(x, jik, adummy, bdummy, cdummy, box, nat)
        return x

    rcm2 = (float(Dj['rcut']) + float(cut_margin)) ** 2

    def compact(x, idx_i, idx_j):
        return compact_pairs(x, idx_i, idx_j, box, rcm2, int(cut_capacity))

    # jax.named_scope labels (gomd_*) cost nothing at run time; they name the
    # step's blocks in profiler traces (gomd/tests/trace_step.py)
    def force_fn(x, idx_i, idx_j, mu_init=None):
        with jax.named_scope('gomd_lp_place'):
            x = place_dummies(x)
        ncut = None
        if cut_capacity:
            with jax.named_scope('gomd_compact'):
                idx_i, idx_j, ncut = compact(x, idx_i, idx_j)
        rt = recip_ef = None
        if recip is not None:
            with jax.named_scope('gomd_kspace'):
                rt = recip['tables'](x)
                recip_ef = (recip['energy'](rt), recip['force'](rt))
        with jax.named_scope('gomd_classical'):
            pe, f, comps = compute_all_forces(x, Dj, idx_i, idx_j, kx, ky, kz, ewself,
                                              recip_ef=recip_ef)
        if ncut is not None:
            comps = dict(comps); comps['ncut'] = ncut
        mu_out = None
        if has_pol:
            e_pol, f_pol, mu_out, pstats = polarization_fn(x, idx_i, idx_j, mu_init, rt)
            pe = pe + e_pol
            f = f + f_pol
            comps = dict(comps); comps['epol'] = e_pol; comps['scf_it'] = pstats['niter']
        if has_dummy:
            with jax.named_scope('gomd_lp_forces'):
                f = dummy_force_redistribution(f, x, jik, adummy, bdummy, cdummy, box, nat)
        return x, pe, f, comps, mu_out

    def rebuild(x):
        return build_neighbor_ondevice_count(x, box, rlist2, excl_flat,
                                             nat=nat, capacity=capacity, skip=frozen)

    def count_within(x, idx_i, idx_j):
        return count_pairs_within(x, idx_i, idx_j, box, rcut2)

    def init_carry(S, x_built, idx_i, idx_j, count, mu, pe, f):
        """Scan carry. x_built: positions the pairs (idx_i, idx_j) were built at;
        count: the pair count that build returned; mu: dipole history
        (mu1, mu2, mu3), zeros for a classical force field."""
        count = jnp.asarray(count, dtype=jnp.int64)
        nb = {'nbuild': jnp.zeros((), jnp.int32),       # rebuilds inside the scan
              'nextra': jnp.zeros((), jnp.int32),       # ... triggered by drift alone
              'overflow': jnp.zeros((), jnp.int32),     # rebuilds with count > capacity
              'missed': jnp.zeros((), jnp.int64),       # max pairs in rcut missing from a stale list
              'count_last': count,
              'count_max': count,
              'disp2_peak': jnp.zeros((), x_built.dtype)}   # max squared drift before a rebuild
        return {'S': S, 'idx_i': idx_i, 'idx_j': idx_j, 'x_nb': x_built, 'nb': nb,
                'mu': mu, 'pe': pe, 'fphys': f}

    def _half_kick(v, f):
        return jnp.where(nd, v + (0.5 * dt * rmass)[:, None] * f, v)

    def _drift(x, v):
        x_new = jnp.where(nd, x + dt * VSCALE * v, x)
        return x_new - box * jnp.floor(x_new / box)

    def _wrap(x):
        return x - box * jnp.floor(x / box)

    def pre_force(S):
        x, v, f, nhc = S['x'], S['v'], S['f'], S['nhc']
        if nvt:
            v, nhc = nhc_mod.nhc_half(v, mass, nd, nhc, nhc_params)
        v = _half_kick(v, f)
        x_old = x
        x = _drift(x, v)
        if do_shake:
            # reference velocity_verlet_step: SHAKE(new x, bond vectors from the
            # pre-drift x); a no-op once the trajectory is on the constraints.
            x, _, _ = shake_constrain(
                x, x_old, con['con_i'], con['con_j'], con['d2'], con['massred'],
                massinv, box, nat, con['tol'], shake_max_iter)
            x = jnp.where(nd, _wrap(x), x)
        return dict(S, x=x, v=v, nhc=nhc)

    def post_force(S, x, pe, f):
        """x: positions the force was evaluated at (lone-pair sites placed)."""
        v, nhc = S['v'], S['nhc']
        if do_shake:
            # SHAKE reference position is the post-drift x, NOT the pre-drift one.
            x_ref = x
            x2 = jnp.where(nd, x + VSCALE*dt*v + VSCALE*dt*dt*f*rmass[:, None], x)
            x2 = x2 - box * jnp.floor(x2 / box)
            x2c, sigma, niter = shake_constrain(
                x2, x_ref, con['con_i'], con['con_j'], con['d2'], con['massred'],
                massinv, box, nat, con['tol'], shake_max_iter)
            diff = _mi(x2c - x_ref, box)
            denom = UNIT * VSCALE * dt * dt
            f = jnp.where(nd, (diff - VSCALE*dt*v) * mass[:, None] / denom, 0.0)

        v = _half_kick(v, f)

        if nvt:
            v, nhc = nhc_mod.nhc_half(v, mass, nd, nhc, nhc_params)
            if not has_frozen:
                v = nhc_mod.remove_com_velocity(v, mass, nd, natreal)

        vr = jnp.where(nd, v, 0.0)
        ke = 0.5 * jnp.sum(mass[:, None] * vr * vr) / (4184.0 * 1000.0)
        hext = pe + ke
        if nvt:
            hext = hext + nhc_mod.nhc_energy(nhc, nhc_params)
        out = {'pe': pe, 'ke': ke, 'hext': hext}
        return dict(S, x=x, v=v, f=f, nhc=nhc, step=S['step'] + 1), out

    def _maybe_rebuild(need, x_now, ii_old, jj_old, x_old, nb, on_drift):
        """lax.cond rebuild at x_now; returns (idx_i, idx_j, x_nb, nb)."""
        def _build(_):
            ii, jj, cnt = rebuild(x_now)
            missed = count_within(x_now, ii, jj) - count_within(x_now, ii_old, jj_old)
            # an overflowed fresh list is truncated: its count says nothing
            missed = jnp.where(cnt > capacity, jnp.zeros((), jnp.int64), missed)
            return ii, jj, cnt, missed, x_now

        def _keep(_):
            return ii_old, jj_old, nb['count_last'], jnp.zeros((), jnp.int64), x_old

        ii, jj, cnt, missed, x_nb = jax.lax.cond(need, _build, _keep, None)
        n = need.astype(jnp.int32)
        nb = dict(nb,
                  nbuild=nb['nbuild'] + n,
                  nextra=nb['nextra'] + (n if on_drift else jnp.zeros((), jnp.int32)),
                  overflow=nb['overflow'] + (need & (cnt > capacity)).astype(jnp.int32),
                  missed=jnp.maximum(nb['missed'], missed),
                  count_last=cnt,
                  count_max=jnp.maximum(nb['count_max'], cnt))
        return ii, jj, x_nb, nb

    def scan_body(carry, _):
        S = carry['S']
        nb = carry['nb']
        idx_i, idx_j, x_nb = carry['idx_i'], carry['idx_j'], carry['x_nb']
        mu1, mu2, mu3 = carry['mu']

        # --- (a) scheduled rebuild, at the start of the step (as before) ---
        if rebuild_every > 0:
            with jax.named_scope('gomd_nb_rebuild'):
                need = (S['step'] % rebuild_every) == 0
                idx_i, idx_j, x_nb, nb = _maybe_rebuild(need, S['x'], idx_i, idx_j, x_nb, nb, False)

        with jax.named_scope('gomd_pre_force'):
            S = pre_force(S)

        # --- (b) drift rebuild, at exactly the positions the force sees ---
        with jax.named_scope('gomd_nb_rebuild'):
            x_f = place_dummies(S['x'])
            dvec = _mi(x_f - x_nb, box)
            disp2 = jnp.max(jnp.sum(dvec * dvec, axis=1))
            nb = dict(nb, disp2_peak=jnp.maximum(nb['disp2_peak'], disp2))
            if drift_rebuild:
                idx_i, idx_j, x_nb, nb = _maybe_rebuild(disp2 > trig2, x_f, idx_i, idx_j, x_nb,
                                                        nb, True)
        need = nb['nbuild'] > carry['nb']['nbuild']

        if has_pol:
            mu_pred = 3.0 * (mu1 - mu2) + mu3 if use_aspc else None
            x, pe, f, comps, mu_new = force_fn(S['x'], idx_i, idx_j, mu_pred)
            mu3 = mu2; mu2 = mu1; mu1 = mu_new
            scf_it = comps['scf_it']
        else:
            x, pe, f, comps, _ = force_fn(S['x'], idx_i, idx_j)
            scf_it = jnp.zeros((), jnp.int32)
        with jax.named_scope('gomd_post_force'):
            S, out = post_force(S, x, pe, f)
        out = dict(out, scf_it=scf_it, rebuilt=need,
                   ncut=comps.get('ncut', jnp.zeros((), jnp.int64)))
        carry = {'S': S, 'idx_i': idx_i, 'idx_j': idx_j, 'x_nb': x_nb, 'nb': nb,
                 'mu': (mu1, mu2, mu3), 'pe': pe, 'fphys': f}
        return carry, out

    return {'force_fn': force_fn, 'place_dummies': place_dummies,
            'rebuild': rebuild, 'count_within': count_within,
            'compact': compact if cut_capacity else None,
            'cut_capacity': int(cut_capacity) if cut_capacity else None,
            'pre_force': pre_force, 'post_force': post_force,
            'scan_body': scan_body, 'init_carry': init_carry,
            'nb_trigger': nb_trigger}


def make_runner(scan_body):
    """jit'd fixed-length runner: run(carry, nsteps) -> (carry, per-step outputs).

    nsteps is static, so each distinct chunk length compiles once.
    """
    @partial(jax.jit, static_argnames=('nsteps',))
    def run(carry, nsteps):
        return jax.lax.scan(scan_body, carry, None, length=nsteps)
    return run
