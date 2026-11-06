import os
os.environ['JAX_PLATFORMS'] = 'cpu'
from jax import config; config.update('jax_enable_x64', True)
import sys; sys.path.insert(0, '.')
import numpy as np
import jax.numpy as jnp
from wmi_md_jax.constraints.shake import shake_constrain

d = np.load('/tmp/shake_ref.npz')
x_ref = d['x_ref']; x_pert = d['x_pert']; x_con_ref = d['x_con']
con = d['bondcon']; d2 = d['d2']; massred = d['massred']
massinv = d['massinv']; box = d['box']
nbcon = int(d['nbcon']); tol = float(d['tol'])
nat = x_ref.shape[0]
con_i = con[0].astype(np.int32); con_j = con[1].astype(np.int32)

x_con, sigma, niter = shake_constrain(
    jnp.asarray(x_pert), jnp.asarray(x_ref),
    jnp.asarray(con_i), jnp.asarray(con_j),
    jnp.asarray(d2), jnp.asarray(massred), jnp.asarray(massinv),
    jnp.asarray(box), nat, tol, 500)
x_con = np.asarray(x_con); sigma = float(sigma); niter = int(niter)
print(f"JAX SHAKE: {niter} iters, final sigma={sigma:.2e} (tol={tol:.0e})")

# Check constraint satisfaction (the physical requirement)
def max_violation(x):
    mv = 0.0
    for icon in range(nbcon):
        i,j = con_i[icon], con_j[icon]
        dd = x[i]-x[j]; dd -= box*np.round(dd/box)
        mv = max(mv, abs((dd*dd).sum() - d2[icon]))
    return mv

mv_jax = max_violation(x_con)
mv_ref = max_violation(x_con_ref)
print(f"max |r^2 - d^2|:  JAX={mv_jax:.2e}  ref={mv_ref:.2e}")

# Compare final positions JAX vs reference (constrained atoms)
pos_diff = np.abs(x_con - x_con_ref).max()
print(f"position diff JAX vs ref: {pos_diff:.2e}")

# Pass: JAX satisfies constraints to similar tolerance as reference
ok = mv_jax < 1e-2 and sigma <= tol
print(f"\n{'[PASS]' if ok else '[FAIL]'} SHAKE constrains bonds")
