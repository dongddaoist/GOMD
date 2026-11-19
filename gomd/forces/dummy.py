"""
forces/dummy.py — Lone-pair virtual sites / dummy atoms (M2.6).

Two operations, both vectorized over dummies (this system: 1316 dummies, all
LpGeomType <= 2, the cross-product geometry):

1. update_dummy_coords: place each dummy at
       x_dummy = x_iat + rid
   where (with central parent iat, parents jat, kat):
       rij = x_jat - x_iat ; rjk = x_kat - x_jat ; rik = x_kat - x_iat
       rij_a_rjk = rij + aa*rjk
       rid = -bb * rij_a_rjk / |rij_a_rjk|  +  cc * (rij x rik) / |rij x rik|

2. dummy_force_redistribution: a force F_d on the dummy is redistributed onto
   the three parents (iat, jat, kat) via the chain rule of the placement
   formula, then the dummy's own force is zeroed. Ported verbatim from the
   reference _do_dummy_forces_inner.

jikdummy layout (5, ndummy): [jat, iat, kat, idat, itype]
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from functools import partial


def _mi(d, box):
    return d - box * jnp.round(d / box)


@partial(jax.jit, static_argnames=('nat',))
def update_dummy_coords(x, jik, adummy, bdummy, cdummy, box, nat):
    """Reconstruct dummy positions from parents (LpGeomType<=2 path).

    Args:
        x: (nat,3) positions (dummy rows will be overwritten).
        jik: (5, ndummy) int [jat, iat, kat, idat, itype].
        adummy,bdummy,cdummy: (ntype,) dummy geometry params.
        box: (3,), nat: static int.
    Returns:
        x with dummy positions updated.
    """
    jat = jik[0]; iat = jik[1]; kat = jik[2]; idat = jik[3]; itype = jik[4]
    aa = adummy[itype]; bb = bdummy[itype]; cc = cdummy[itype]   # (nd,)

    rij = _mi(x[jat] - x[iat], box)            # (nd,3)
    rjk = _mi(x[kat] - x[jat], box)
    rik = _mi(x[kat] - x[iat], box)

    rij_a_rjk = rij + aa[:, None] * rjk
    norm2 = jnp.sum(rij_a_rjk * rij_a_rjk, axis=1)
    good = norm2 > 1e-20
    inv = jnp.where(good, 1.0 / jnp.sqrt(jnp.where(good, norm2, 1.0)), 0.0)
    factor = -bb * inv                          # (nd,)
    rid = factor[:, None] * rij_a_rjk           # (nd,3)

    # cross = rij x rik
    cross = jnp.cross(rij, rik)
    crossabs = jnp.sqrt(jnp.maximum(jnp.sum(cross * cross, axis=1), 1e-24))
    rid = rid + (cc / crossabs)[:, None] * cross

    x_dummy = x[iat] + rid                      # (nd,3)
    x_dummy = x_dummy - box * jnp.floor(x_dummy / box)

    x = x.at[idat].set(x_dummy)
    return x


@partial(jax.jit, static_argnames=('nat',))
def dummy_force_redistribution(f, x, jik, adummy, bdummy, cdummy, box, nat):
    """Redistribute forces on dummies onto parents; zero dummy forces.

    Ported from reference _do_dummy_forces_inner (LpGeomType<=2 path).
    Args:
        f: (nat,3) forces (will be modified: parents get force, dummy zeroed).
        x: (nat,3) positions (dummies already placed).
        jik,adummy,bdummy,cdummy,box,nat: as above.
    Returns:
        f updated.
    """
    zero = 1e-12
    jat = jik[0]; iat = jik[1]; kat = jik[2]; idat = jik[3]; itype = jik[4]
    aa = adummy[itype]; bb = bdummy[itype]; cc = cdummy[itype]

    Fd = f[idat]                                # (nd,3) force on dummy

    rij = _mi(x[jat] - x[iat], box)
    rjk = _mi(x[kat] - x[jat], box)
    rik = _mi(x[kat] - x[iat], box)
    rijabs2 = jnp.sum(rij * rij, axis=1)
    rikabs2 = jnp.sum(rik * rik, axis=1)
    rijdotrik = jnp.sum(rij * rik, axis=1)

    rij_a_rjk = rij + aa[:, None] * rjk
    norm2 = jnp.sum(rij_a_rjk * rij_a_rjk, axis=1)
    good = norm2 > 1e-20
    inv = jnp.where(good, 1.0 / jnp.sqrt(jnp.where(good, norm2, 1.0)), 0.0)
    factor = -bb * inv                          # (nd,)

    rid = factor[:, None] * rij_a_rjk
    Fddotrid = jnp.sum(Fd * rid, axis=1)        # (nd,)

    bb_ok = jnp.abs(bb) > zero
    inv_bb2 = jnp.where(bb_ok, 1.0 / jnp.where(bb_ok, bb * bb, 1.0), 0.0)
    f1 = (Fddotrid * inv_bb2)[:, None] * rid    # (nd,3)

    cross = jnp.cross(rij, rik)                  # (nd,3)
    crossdotFd = jnp.sum(cross * Fd, axis=1)
    crossabs2 = jnp.maximum(rijabs2 * rikabs2 - rijdotrik * rijdotrik, 0.0)
    crossabs = jnp.sqrt(crossabs2)
    cok = crossabs >= 1e-10
    t1 = jnp.where(cok, 1.0 / jnp.where(cok, crossabs, 1.0), 0.0)   # tmp1_inv

    # fiout/fjout/fkout (first-derivative part), per the reference
    def comp(a, b, c):  # helper to build a 3-vector from scalar comps
        return jnp.stack([a, b, c], axis=1)

    rij0, rij1, rij2 = rij[:, 0], rij[:, 1], rij[:, 2]
    rik0, rik1, rik2 = rik[:, 0], rik[:, 1], rik[:, 2]
    Fd0, Fd1, Fd2 = Fd[:, 0], Fd[:, 1], Fd[:, 2]

    fiout0 = (Fd1 * (rik2 - rij2) + Fd2 * (rij1 - rik1)) * t1
    fiout1 = (Fd0 * (rij2 - rik2) + Fd2 * (rik0 - rij0)) * t1
    fiout2 = (Fd0 * (rik1 - rij1) + Fd1 * (rij0 - rik0)) * t1

    fjout0 = (Fd1 * (-rik2) + Fd2 * rik1) * t1
    fjout1 = (Fd2 * (-rik0) + Fd0 * rik2) * t1
    fjout2 = (Fd0 * (-rik1) + Fd1 * rik0) * t1

    fkout0 = (Fd1 * rij2 - Fd2 * rij1) * t1
    fkout1 = (Fd2 * rij0 - Fd0 * rij2) * t1
    fkout2 = (Fd0 * rij1 - Fd1 * rij0) * t1

    # second-derivative term
    t3 = t1 * t1 * t1
    cdf = crossdotFd * t3
    fiout0 = fiout0 + cdf * (rikabs2 * rij0 + rijabs2 * rik0 - rijdotrik * (rik0 + rij0))
    fjout0 = fjout0 - cdf * (rikabs2 * rij0 - rijdotrik * rik0)
    fkout0 = fkout0 - cdf * (rijabs2 * rik0 - rijdotrik * rij0)
    fiout1 = fiout1 + cdf * (rikabs2 * rij1 + rijabs2 * rik1 - rijdotrik * (rik1 + rij1))
    fjout1 = fjout1 - cdf * (rikabs2 * rij1 - rijdotrik * rik1)
    fkout1 = fkout1 - cdf * (rijabs2 * rik1 - rijdotrik * rij1)
    fiout2 = fiout2 + cdf * (rikabs2 * rij2 + rijabs2 * rik2 - rijdotrik * (rik2 + rij2))
    fjout2 = fjout2 - cdf * (rikabs2 * rij2 - rijdotrik * rik2)
    fkout2 = fkout2 - cdf * (rijabs2 * rik2 - rijdotrik * rij2)

    # Combine into per-parent forces (component-wise, matching reference)
    ftmp0 = factor * (Fd0 - f1[:, 0])
    ftmp1 = factor * (Fd1 - f1[:, 1])
    ftmp2 = factor * (Fd2 - f1[:, 2])

    tmp1_0 = Fd0 - ftmp0 + fiout0 * cc
    tmp2_0 = (1.0 - aa) * ftmp0 + fjout0 * cc
    tmp3_0 = aa * ftmp0 + fkout0 * cc
    tmp1_1 = Fd1 - ftmp1 + fiout1 * cc
    tmp2_1 = (1.0 - aa) * ftmp1 + fjout1 * cc
    tmp3_1 = aa * ftmp1 + fkout1 * cc
    tmp1_2 = Fd2 - ftmp2 + fiout2 * cc
    tmp2_2 = (1.0 - aa) * ftmp2 + fjout2 * cc
    tmp3_2 = aa * ftmp2 + fkout2 * cc

    # Only apply where geometry valid
    valid = good & cok
    f_iat = jnp.where(valid[:, None], comp(tmp1_0, tmp1_1, tmp1_2), 0.0)
    f_jat = jnp.where(valid[:, None], comp(tmp2_0, tmp2_1, tmp2_2), 0.0)
    f_kat = jnp.where(valid[:, None], comp(tmp3_0, tmp3_1, tmp3_2), 0.0)

    f = f.at[iat].add(f_iat)
    f = f.at[jat].add(f_jat)
    f = f.at[kat].add(f_kat)
    # zero dummy forces
    f = f.at[idat].set(0.0)
    return f
