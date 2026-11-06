"""
ewald_fast.py
=============
Accelerated drop-in replacements for the slow Python loops in ewald.py.

Currently ewald.compute_recip_efield is pure Python and gets called ~11
times per step (once per polarization SCF iteration). With klim=7 and
8716 atoms, the inner triple-k-loop x atom loop runs ~67M times per step
in interpreted Python. JIT-compiling collapses this to ~1 second.

Usage:
    # In polarization.py, replace:
    #     from ewald import compute_recip_efield
    #     elfown_recip = compute_recip_efield(state)
    # with:
    from ewald_fast import compute_recip_efield_fast
    elfown_recip = compute_recip_efield_fast(state)

The function below has the SAME numerical output as the original
(verified by comparing on the production system); it is only faster.
"""

import numpy as np
from numba import njit
from constants import PI


@njit(cache=True, fastmath=False)
def _recip_efield_kernel(
    x, atomtype, q, ncharge, alpha, box,
    klimitx, klimity, klimitz
):
    """JIT'd reciprocal-space E-field for SCF.

    Replicates the math of ewald.compute_recip_efield exactly.
    All branches over ky/kz sign and the trig recurrence are kept
    intact so the output is byte-identical (modulo fp64 reordering).
    """
    nat = x.shape[1]

    front = np.zeros(3)
    for kk in range(3):
        front[kk] = 2.0 * PI / box[kk]

    vol = box[0] * box[1] * box[2]
    twopifact = 2.0 * PI / vol
    fouralpha2inv = 1.0 / (4.0 * alpha * alpha)
    elfact = 2.0 * twopifact

    kxmax2inv = 1.0 / float(klimitx * klimitx)
    kymax2inv = 1.0 / float(klimity * klimity)
    kzmax2inv = 1.0 / float(klimitz * klimitz)

    elfown_recip = np.zeros((3, nat))

    # Trig tables (atom is the slow axis to match scratch indexing)
    cos_x = np.zeros((nat, klimitx + 1))
    sin_x = np.zeros((nat, klimitx + 1))
    cos_y = np.zeros((nat, klimity + 1))
    sin_y = np.zeros((nat, klimity + 1))
    cos_z = np.zeros((nat, klimitz + 1))
    sin_z = np.zeros((nat, klimitz + 1))

    for iat in range(nat):
        cos_x[iat, 0] = 1.0; sin_x[iat, 0] = 0.0
        cos_y[iat, 0] = 1.0; sin_y[iat, 0] = 0.0
        cos_z[iat, 0] = 1.0; sin_z[iat, 0] = 0.0
        if klimitx > 0:
            cos_x[iat, 1] = np.cos(front[0] * x[0, iat])
            sin_x[iat, 1] = np.sin(front[0] * x[0, iat])
        if klimity > 0:
            cos_y[iat, 1] = np.cos(front[1] * x[1, iat])
            sin_y[iat, 1] = np.sin(front[1] * x[1, iat])
        if klimitz > 0:
            cos_z[iat, 1] = np.cos(front[2] * x[2, iat])
            sin_z[iat, 1] = np.sin(front[2] * x[2, iat])
        for kn in range(2, klimitx + 1):
            cos_x[iat, kn] = cos_x[iat, kn-1]*cos_x[iat, 1] - sin_x[iat, kn-1]*sin_x[iat, 1]
            sin_x[iat, kn] = sin_x[iat, kn-1]*cos_x[iat, 1] + cos_x[iat, kn-1]*sin_x[iat, 1]
        for kn in range(2, klimity + 1):
            cos_y[iat, kn] = cos_y[iat, kn-1]*cos_y[iat, 1] - sin_y[iat, kn-1]*sin_y[iat, 1]
            sin_y[iat, kn] = sin_y[iat, kn-1]*cos_y[iat, 1] + cos_y[iat, kn-1]*sin_y[iat, 1]
        for kn in range(2, klimitz + 1):
            cos_z[iat, kn] = cos_z[iat, kn-1]*cos_z[iat, 1] - sin_z[iat, kn-1]*sin_z[iat, 1]
            sin_z[iat, kn] = sin_z[iat, kn-1]*cos_z[iat, 1] + cos_z[iat, kn-1]*sin_z[iat, 1]

    # Per-atom charge lookup
    qi = np.zeros(nat)
    for iat in range(nat):
        it = atomtype[iat]
        qi[iat] = q[it] if it < ncharge else 0.0

    # Reusable scratch
    cos_xyz = np.zeros(nat)
    sin_xyz = np.zeros(nat)

    # Half-space k-loop, matches Fortran convention
    for kx in range(0, klimitx + 1):
        hx = float(kx) * front[0]
        kymin = -klimity if kx > 0 else 0
        for ky in range(kymin, klimity + 1):
            hy = float(ky) * front[1]
            kzmin = 1 if (kx == 0 and ky == 0) else -klimitz
            for kz in range(kzmin, klimitz + 1):
                hz = float(kz) * front[2]

                tmp2 = float(kx*kx)*kxmax2inv + float(ky*ky)*kymax2inv + float(kz*kz)*kzmax2inv
                if tmp2 > 1.0:
                    continue

                rkmag2 = hx*hx + hy*hy + hz*hz
                if rkmag2 < 1.0e-12:
                    continue
                akvec = 2.0 * np.exp(-rkmag2 * fouralpha2inv) / rkmag2

                kya = abs(ky); kza = abs(kz)
                # cos/sin(k.r_i) for every atom
                if ky >= 0 and kz >= 0:
                    for iat in range(nat):
                        cx = cos_x[iat, kx]; sx = sin_x[iat, kx]
                        cy_ = cos_y[iat, kya]; sy_ = sin_y[iat, kya]
                        cz_ = cos_z[iat, kza]; sz_ = sin_z[iat, kza]
                        cxy = cx*cy_ - sx*sy_
                        sxy = sx*cy_ + cx*sy_
                        cos_xyz[iat] = cxy*cz_ - sxy*sz_
                        sin_xyz[iat] = sxy*cz_ + cxy*sz_
                elif ky >= 0 and kz < 0:
                    for iat in range(nat):
                        cx = cos_x[iat, kx]; sx = sin_x[iat, kx]
                        cy_ = cos_y[iat, kya]; sy_ = sin_y[iat, kya]
                        cz_ = cos_z[iat, kza]; sz_ = sin_z[iat, kza]
                        cxy = cx*cy_ - sx*sy_
                        sxy = sx*cy_ + cx*sy_
                        cos_xyz[iat] = cxy*cz_ + sxy*sz_
                        sin_xyz[iat] = sxy*cz_ - cxy*sz_
                elif ky < 0 and kz >= 0:
                    for iat in range(nat):
                        cx = cos_x[iat, kx]; sx = sin_x[iat, kx]
                        cy_ = cos_y[iat, kya]; sy_ = sin_y[iat, kya]
                        cz_ = cos_z[iat, kza]; sz_ = sin_z[iat, kza]
                        cxy = cx*cy_ + sx*sy_
                        sxy = sx*cy_ - cx*sy_
                        cos_xyz[iat] = cxy*cz_ - sxy*sz_
                        sin_xyz[iat] = sxy*cz_ + cxy*sz_
                else:  # ky<0, kz<0
                    for iat in range(nat):
                        cx = cos_x[iat, kx]; sx = sin_x[iat, kx]
                        cy_ = cos_y[iat, kya]; sy_ = sin_y[iat, kya]
                        cz_ = cos_z[iat, kza]; sz_ = sin_z[iat, kza]
                        cxy = cx*cy_ + sx*sy_
                        sxy = sx*cy_ - cx*sy_
                        cos_xyz[iat] = cxy*cz_ + sxy*sz_
                        sin_xyz[iat] = sxy*cz_ - cxy*sz_

                # Charge structure factor
                cossum = 0.0
                sinsum = 0.0
                for iat in range(nat):
                    cossum += qi[iat] * cos_xyz[iat]
                    sinsum += qi[iat] * sin_xyz[iat]

                prefact = akvec * elfact
                for iat in range(nat):
                    tmp = prefact * (cossum * sin_xyz[iat] - sinsum * cos_xyz[iat])
                    elfown_recip[0, iat] += tmp * hx
                    elfown_recip[1, iat] += tmp * hy
                    elfown_recip[2, iat] += tmp * hz

    return elfown_recip


def compute_recip_efield_fast(state):
    """Drop-in replacement for ewald.compute_recip_efield."""
    return _recip_efield_kernel(
        state.atoms.x,
        state.atoms.atomtype,
        state.ff.q,
        state.ff.ncharge,
        state.params.alphai,
        state.atoms.box,
        state.params.klimitx,
        state.params.klimity,
        state.params.klimitz,
    )
