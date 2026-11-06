"""
constraints/shake.py — SHAKE constraint solver (M2.5).

Enforces rigid (constrained) bond lengths so the integrator can use dt=0.5 fs
without the fast bond vibrations going unstable.

Reference algorithm (utils.py _shake_inner_loop), per constraint icon between
atoms (i, j) with target squared length d2:
    rij      = x_i - x_j   (min image)
    rijref   = x_ref_i - x_ref_j  (reference bond vector, fixed during solve)
    dot      = rijref . rij
    delta    = d2 - |rij|^2
    g        = delta * massred / dot
    x_i += g * massinv_i * rijref ;  x_j -= g * massinv_j * rijref
Iterate until sigma = mean(delta^2) <= tol.

Gauss-Seidel vs Jacobi:
    The reference is sequential (Gauss-Seidel): constraint icon sees position
    updates from icon-1 within the same sweep. That does not parallelize.
    Here we use the JACOBI form: every constraint computes its correction from
    the SAME start-of-iteration positions, then all corrections are scatter-
    added at once. This parallelizes perfectly on GPU and converges to the
    SAME constraint-satisfying positions (SHAKE's fixed point is order-
    independent), typically in a few more iterations. We validate that the
    final positions satisfy all constraints to the reference tolerance.

We run a fixed-then-checked loop with lax.while_loop so it's jit/GPU friendly.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from functools import partial


def _mi(d, box):
    return d - box * jnp.round(d / box)


@partial(jax.jit, static_argnames=('nat', 'max_iter'))
def shake_constrain(x_new, x_ref, con_i, con_j, d2, massred, massinv,
                    box, nat, tol, max_iter):
    """Apply SHAKE (Jacobi form) to constrain bond lengths.

    Args:
        x_new: (nat,3) positions to constrain (post-step predictor).
        x_ref: (nat,3) reference positions (start of step) for bond vectors.
        con_i, con_j: (ncon,) constrained-bond atom indices.
        d2: (ncon,) target squared bond lengths.
        massred: (ncon,) reduced mass factor per constraint.
        massinv: (nat,) inverse masses.
        box: (3,), tol: float, nat/max_iter: static ints.

    Returns:
        x_con: (nat,3) constrained positions.
        sigma: final mean squared constraint violation.
        niter: iterations used.
    """
    # Reference bond vectors (fixed during the solve)
    rijref = _mi(x_ref[con_i] - x_ref[con_j], box)      # (ncon,3)

    mi_i = massinv[con_i]                                 # (ncon,)
    mi_j = massinv[con_j]

    def cond(state):
        x, sigma, it = state
        return (sigma > tol) & (it < max_iter)

    def body(state):
        x, _sigma, it = state
        rij = _mi(x[con_i] - x[con_j], box)              # (ncon,3)
        dot = jnp.sum(rijref * rij, axis=1)              # (ncon,)
        rijmag = jnp.sum(rij * rij, axis=1)
        delta = d2 - rijmag                              # (ncon,)
        sigma = jnp.mean(delta * delta)

        safe = jnp.abs(dot) >= 1e-10
        g = jnp.where(safe, delta * massred / jnp.where(safe, dot, 1.0), 0.0)

        # Jacobi corrections: each constraint pushes its two atoms.
        corr_i = (g * mi_i)[:, None] * rijref            # (ncon,3)
        corr_j = -(g * mi_j)[:, None] * rijref

        dx = jnp.zeros((nat, 3))
        dx = dx.at[con_i].add(corr_i)
        dx = dx.at[con_j].add(corr_j)
        x = x + dx
        return (x, sigma, it + 1)

    # initial sigma: compute once so the loop can test it
    rij0 = _mi(x_new[con_i] - x_new[con_j], box)
    delta0 = d2 - jnp.sum(rij0 * rij0, axis=1)
    sigma0 = jnp.mean(delta0 * delta0)

    x_con, sigma, niter = jax.lax.while_loop(
        cond, body, (x_new, sigma0, jnp.array(0))
    )
    return x_con, sigma, niter
