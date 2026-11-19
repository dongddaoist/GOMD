"""
thermostat.py — Nose-Hoover chain (NVT), ported from the reference pkg/integrator.py.

Same algorithm as the reference (init_thermostat / init_integration /
nhc_integrate): Martyna-Tuckerman-Klein chains with a 5 x 5 Yoshida-Suzuki
factorization, applied as a half step before and after the velocity-Verlet
step. Units follow the reference: time in fs, energy in kcal/mol, velocities in
m/s, thermostat masses in kcal/mol fs^2.

Differences from the reference, on purpose:
  * no clamping of exp() arguments to +-700, no reset of the chain velocities on
    NaN, no clipping of the velocity scale to [0.1, 10]. Those guards never
    trigger in a healthy run and silently corrupt the dynamics when they do;
    here a blow-up shows up in the log (H_ext) instead.
  * the conserved (extended) energy is returned so it can be logged:
        H_ext = KE + PE + sum_i Q_i v_i^2 / 2 + gNkT xi_1 + kT sum_{i>1} xi_i

The thermostat state is the tuple (xlogs, vlogs, glogs), each of length nnos
(the chain length is static, so the chain loops unroll at trace time).
"""
from __future__ import annotations
import numpy as np
import jax.numpy as jnp

GASK_KCAL = 1.98709e-3          # kcal/(mol K), as pkg/constants.py
NHCSTEP1 = 5
NHCSTEP2 = 5


def make_nhc_params(Tset, ndof, nnos, qfreq, dt, multibig=1):
    """Thermostat constants, with the reference's arithmetic order (fp64)."""
    nnos = int(nnos)
    if nnos < 1:
        raise ValueError(f"Num_NHC must be >= 1 (got {nnos})")
    gkt = GASK_KCAL * float(Tset)
    gnkt = int(ndof) * gkt
    qtmass = np.empty(nnos)
    qtmass[0] = gnkt / (qfreq * qfreq)
    if nnos > 1:
        qtmass[1:] = gkt / (qfreq * qfreq)
    stepout = float(dt) * int(multibig)
    w0 = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
    wys = [w0, w0, -4.0 ** (1.0 / 3.0) * w0, w0, w0]
    wdt1 = [wys[i] * stepout / NHCSTEP1 for i in range(NHCSTEP2)]
    return {
        'nnos': nnos, 'gkt': gkt, 'gnkt': gnkt, 'qtmass': qtmass,
        'wdt2': [w / 2.0 for w in wdt1],
        'wdt4': [w / 4.0 for w in wdt1],
        'wdt8': [w / 8.0 for w in wdt1],
        'Tset': float(Tset), 'ndof': int(ndof), 'qfreq': float(qfreq),
    }


def kinetic_energy(v, mass, nd):
    """KE of the real atoms in kcal/mol (v in m/s, mass in g/mol)."""
    vr = jnp.where(nd, v, 0.0)
    return 0.5 * jnp.sum(mass[:, None] * vr * vr) / (4184.0 * 1000.0)


def nhc_half(v, mass, nd, nhc_state, P):
    """One thermostat half step (reference nhc_integrate). Returns (v, state)."""
    xlogs, vlogs, glogs = nhc_state
    n = P['nnos']
    qt = P['qtmass']; gnkt = P['gnkt']; gkt = P['gkt']
    ekin2 = 2.0 * kinetic_energy(v, mass, nd)
    x = [xlogs[i] for i in range(n)]
    vl = [vlogs[i] for i in range(n)]
    g = [glogs[i] for i in range(n)]
    scale = 1.0
    g[0] = (ekin2 - gnkt) / qt[0]
    for _ in range(NHCSTEP1):
        for j in range(NHCSTEP2):
            w2 = P['wdt2'][j]; w4 = P['wdt4'][j]; w8 = P['wdt8'][j]
            vl[n - 1] = vl[n - 1] + g[n - 1] * w4
            for k in range(n - 2, -1, -1):
                aa = jnp.exp(-w8 * vl[k + 1])
                vl[k] = vl[k] * aa * aa + w4 * g[k] * aa
            aa = jnp.exp(-w2 * vl[0])
            scale = scale * aa
            g[0] = (scale * scale * ekin2 - gnkt) / qt[0]
            for k in range(n):
                x[k] = x[k] + vl[k] * w2
            for k in range(n - 1):
                aa = jnp.exp(-w8 * vl[k + 1])
                vl[k] = vl[k] * aa * aa + w4 * g[k] * aa
                g[k + 1] = (qt[k] * vl[k] ** 2 - gkt) / qt[k + 1]
            vl[n - 1] = vl[n - 1] + g[n - 1] * w4
    v = jnp.where(nd, v * scale, v)
    return v, (jnp.stack(x), jnp.stack(vl), jnp.stack(g))


def nhc_energy(nhc_state, P):
    """Thermostat part of the conserved energy (kcal/mol)."""
    xlogs, vlogs, _ = nhc_state
    e = 0.5 * jnp.sum(jnp.asarray(P['qtmass']) * vlogs * vlogs) + P['gnkt'] * xlogs[0]
    if P['nnos'] > 1:
        e = e + P['gkt'] * jnp.sum(xlogs[1:])
    return e


def remove_com_velocity(v, mass, nd, natreal):
    """Reference remove_com_velocity: v_i -= P_total / (natreal * m_i), real atoms."""
    m = jnp.where(nd[:, 0], mass, 1.0)
    vcum = jnp.sum(jnp.where(nd, mass[:, None] * v, 0.0), axis=0)
    return jnp.where(nd, v - vcum[None, :] * (1.0 / (natreal * m))[:, None], v)


def temperature(ke, ndof):
    return 2.0 * ke / (ndof * GASK_KCAL)
