"""
forces/pol_field.py — complete charge electric-field builder for the SCF (M3).

This is the piece whose absence kept polarization out of the loop. The SCF,
energy, and forces were already validated against the reference TOTAL field;
what was missing was a self-contained field that reproduces that total. The
reference assembles it in four parts (compute_polarization phases 1, 1b, 1c,
1d):

    E_total = E_real            (real-space charge field, erfc-damped)
            + E_real_14_foldin  (1-4 Q-mu term folded into the main loop)
            + E_recip           (reciprocal-space charge field)
            + E_excl_corr        (erf correction for excluded/bonded pairs)
            + E_14_corr          (separate 1-4 Q-mu correction)

The bug behind the old `charge_field_realspace` discrepancy: it omitted the
1-4 fold-in (and the rcut mask, and used exact instead of tabulated erfc). With
all three fixed, E_real matches the reference to 1.4e-15. Each part below is
validated against the reference field dump.

Conventions:
    - positions x are (nat,3); reference min-image is a single conditional box
      shift `d -= sign(d)*box where |d|>el` (el = box/2), reproduced exactly
      (NOT round(d/box) — they agree inside rcut<box/2 but we match the form).
    - real-space + 1-4 fold-in use the TABULATED erfc/exp (16384 knots over
      [0,6], linear interp) the reference spline tables were built with.
    - exclusion + 1-4 corrections use the A&S erfc polynomial (erfc_approx),
      exactly as the reference correction routines do.
    - charges are per-atom (q[atomtype], already how the loader stores D['q']).

Pair inputs:
    - pair_i/pair_j: the non-bonded neighbor pairs (same as the spline kernel),
      i<j, exclusions removed, possibly padded. The rcut mask + i!=j handle
      skin and padding.
    - l14_i/l14_j, ex_i/ex_j: directed 1-4 and exclusion pair lists (host-built
      once from the reference listm14/listmex).
"""
from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

PI = np.pi


# ---------------------------------------------------------------------------
# Tabulated Ewald functions (match ewald_tables.py exactly)
# ---------------------------------------------------------------------------
def build_ewald_tables():
    from scipy.special import erfc as _erfc
    TABLE_N = 16384
    TABLE_X_MAX = 6.0
    xs = np.linspace(0.0, TABLE_X_MAX, TABLE_N + 1)
    erfc_tab = _erfc(xs).astype(np.float64)
    exp_tab = np.exp(-xs * xs).astype(np.float64)
    dx_inv = TABLE_N / TABLE_X_MAX
    return jnp.asarray(erfc_tab), jnp.asarray(exp_tab), float(dx_inv)


def _erfc_approx_jax(x):
    """A&S 7.1.26 polynomial — matches the reference erfc_approx()/correction."""
    a1, a2, a3 = 0.254829592, -0.284496736, 1.421413741
    a4, a5, p = -1.453152027, 1.061405429, 0.3275911
    ax = jnp.abs(x)
    t = 1.0 / (1.0 + p * ax)
    val = (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*jnp.exp(-ax*ax)
    return jnp.where(x < 0, 2.0 - val, val)


def _min_image_ref(d, box, el):
    """Reference single-shift min image: shift by box where |d|>el (el=box/2)."""
    return d - jnp.where(jnp.abs(d) > el, jnp.sign(d) * box, 0.0)


# ---------------------------------------------------------------------------
# Real-space charge field (+ 1-4 Q-mu fold-in) — tabulated erfc
# ---------------------------------------------------------------------------
def _tab_lookup(alphar, tab, dx_inv):
    fpos = alphar * dx_inv
    ii = jnp.clip(jnp.floor(fpos).astype(jnp.int32), 0, tab.shape[0] - 2)
    frac = fpos - ii
    val = tab[ii] + frac * (tab[ii + 1] - tab[ii])
    return jnp.where(alphar < 6.0, val, 0.0)


@partial(jax.jit, static_argnames=('nat', 'apply_cut'))
def _field_pairs_tab(x, q, idx_i, idx_j, box, el, alpha, rcut2,
                     erfc_tab, exp_tab, dx_inv, scale, apply_cut, nat):
    """Field from a directed pair list, tabulated erfc. scale multiplies fact
    (1.0 for the main loop, -redQmufactor for the 1-4 fold-in)."""
    xi = x[idx_i]; xj = x[idx_j]
    d = _min_image_ref(xj - xi, box, el)
    z = jnp.sum(d * d, axis=1)
    good = z > 1e-10
    if apply_cut:
        good = good & (z <= rcut2)
    good = good & (idx_i != idx_j)
    r = jnp.sqrt(jnp.where(good, z, 1.0))
    rinv = 1.0 / r
    zinv = rinv * rinv
    alphar = alpha * r
    erfc_v = _tab_lookup(alphar, erfc_tab, dx_inv)
    exp_v = _tab_lookup(alphar, exp_tab, dx_inv)
    twoalphapi = 2.0 * alpha / jnp.sqrt(PI)
    fact = (erfc_v * rinv + twoalphapi * exp_v) * zinv * scale
    fact = jnp.where(good, fact, 0.0)
    ci = q[idx_i]; cj = q[idx_j]
    E = jnp.zeros((nat, 3))
    E = E.at[idx_i].add(-(fact * cj)[:, None] * d)
    E = E.at[idx_j].add((fact * ci)[:, None] * d)
    return E


# ---------------------------------------------------------------------------
# erf-based corrections (exclusion + 1-4) — A&S polynomial erfc
# ---------------------------------------------------------------------------
@partial(jax.jit, static_argnames=('nat', 'half'))
def _field_correction(x, q, ci_i, cj_i, box, el, alpha, scale, half, nat):
    """erf field correction over a directed pair list (ci_i, cj_i are the pair
    index arrays). scale = -1/2 style handled by (scale, half).

    Reference exclusion: ffcorrect = (erf/r^3 - 2a/sqrt(pi) exp/r^2)/2, adds
        +ffcorrect*cj on i, -ffcorrect*ci on j.
    Reference 1-4:       ffcorrect = redQmufactor*(...), no /2.
    """
    i = ci_i; j = cj_i
    xi = x[i]; xj = x[j]
    d = _min_image_ref(xj - xi, box, el)
    z = jnp.sum(d * d, axis=1)
    good = z > 1e-10
    r = jnp.sqrt(jnp.where(good, z, 1.0))
    rinv = 1.0 / r
    zinv = rinv * rinv
    r3inv = zinv * rinv
    alphar = alpha * r
    twoalphapi = 2.0 * alpha / jnp.sqrt(PI)
    erf_val = 1.0 - _erfc_approx_jax(alphar)
    ff = (erf_val * r3inv - twoalphapi * jnp.exp(-alphar*alphar) * zinv) * scale
    if half:
        ff = ff * 0.5
    ff = jnp.where(good, ff, 0.0)
    qi = q[i]; qj = q[j]
    E = jnp.zeros((nat, 3))
    E = E.at[i].add((ff * qj)[:, None] * d)
    E = E.at[j].add(-(ff * qi)[:, None] * d)
    return E


# ---------------------------------------------------------------------------
# Reciprocal-space charge field
# ---------------------------------------------------------------------------
@jax.jit
def _recip_field(x, q, box, kx, ky, kz, alpha):
    """Reciprocal-space charge field (NO COULOMB factor; matches the reference
    compute_recip_efield: elfact = 2*twopifact)."""
    vol = box[0] * box[1] * box[2]
    twopifact = 2.0 * PI / vol
    fouralpha2inv = 1.0 / (4.0 * alpha * alpha)
    elfact = 2.0 * twopifact
    front = 2.0 * PI / box
    hx = kx * front[0]; hy = ky * front[1]; hz = kz * front[2]
    rk2 = hx*hx + hy*hy + hz*hz
    akvec = 2.0 * jnp.exp(-rk2 * fouralpha2inv) / rk2
    kr = (x[:, 0:1]*hx[None, :] + x[:, 1:2]*hy[None, :] + x[:, 2:3]*hz[None, :])
    cos_kr = jnp.cos(kr); sin_kr = jnp.sin(kr)
    qcol = q[:, None]
    cossum = jnp.sum(qcol * cos_kr, axis=0)
    sinsum = jnp.sum(qcol * sin_kr, axis=0)
    prefact = akvec * elfact
    # E_kk,i = sum_k prefact_k*(cossum_k*sin_kr - sinsum_k*cos_kr)*h_kk
    a = prefact * cossum
    b = prefact * sinsum
    cax = jnp.stack([a*hx, a*hy, a*hz], axis=1)   # (nk,3)
    cbx = jnp.stack([b*hx, b*hy, b*hz], axis=1)
    E = sin_kr @ cax - cos_kr @ cbx               # (nat,3)
    return E


# ---------------------------------------------------------------------------
# Public: assemble the total field
# ---------------------------------------------------------------------------
def make_field_builder(Dp, kx, ky, kz):
    """Returns a jit'd build_field(x) -> (nat,3) total charge field for the SCF.

    Dp keys (all JAX arrays unless noted, built once at setup):
        q, box, el, alpha (scalar), rcut2 (scalar), nat (static),
        pair_i, pair_j      (non-bonded neighbor pairs, i<j, padded ok),
        l14_i, l14_j        (directed 1-4 pairs),
        ex_i, ex_j          (directed exclusion pairs),
        redQmufactor (scalar), lredQ_mu14 (bool),
        erfc_tab, exp_tab, dx_inv (Ewald tables).
    kx,ky,kz: Ewald k-vectors (JAX arrays).
    """
    nat = Dp['nat']
    q = Dp['q']; box = Dp['box']; el = Dp['el']
    alpha = Dp['alpha']; rcut2 = Dp['rcut2']
    erfc_tab = Dp['erfc_tab']; exp_tab = Dp['exp_tab']; dx_inv = Dp['dx_inv']
    redQ = Dp['redQmufactor']; has14 = bool(Dp['lredQ_mu14'])
    pair_i = Dp['pair_i']; pair_j = Dp['pair_j']
    l14_i = Dp['l14_i']; l14_j = Dp['l14_j']
    ex_i = Dp['ex_i']; ex_j = Dp['ex_j']

    @jax.jit
    def build_field(x):
        # 1. real-space charge field (main neighbor loop), cut at rcut
        E = _field_pairs_tab(x, q, pair_i, pair_j, box, el, alpha, rcut2,
                             erfc_tab, exp_tab, dx_inv, 1.0, True, nat)
        # 1 (fold-in). 1-4 Q-mu term folded into the main loop (scale -redQ)
        if has14:
            E = E + _field_pairs_tab(x, q, l14_i, l14_j, box, el, alpha, rcut2,
                                     erfc_tab, exp_tab, dx_inv, -redQ, False, nat)
        # 1b. reciprocal-space charge field
        E = E + _recip_field(x, q, box, kx, ky, kz, alpha)
        # 1c. exclusion correction (erf, /2)
        E = E + _field_correction(x, q, ex_i, ex_j, box, el, alpha, 1.0, True, nat)
        # 1d. separate 1-4 Q-mu correction (erf, redQ, no /2)
        if has14:
            E = E + _field_correction(x, q, l14_i, l14_j, box, el, alpha, redQ, False, nat)
        return E

    return build_field
