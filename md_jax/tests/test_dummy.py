import os
os.environ['JAX_PLATFORMS'] = 'cpu'
from jax import config; config.update('jax_enable_x64', True)
import sys; sys.path.insert(0, '.')
import numpy as np
import jax.numpy as jnp
from wmi_md_jax.forces.dummy import update_dummy_coords, dummy_force_redistribution

d = np.load('/tmp/dummy_ref.npz')
x_in = d['x_recon_input']; jik = d['jik']; box = d['box']
adummy=d['adummy']; bdummy=d['bdummy']; cdummy=d['cdummy']
nat=int(d['nat']); ndummy=int(d['ndummy'])
x_recon_ref = d['x_recon_ref']
idat = jik[3]   # dummy indices

# --- Test 1: position reconstruction ---
# zero out dummy rows then reconstruct
x_test = x_in.copy()
x_test[idat] = 0.0
x_jax = update_dummy_coords(jnp.asarray(x_test), jnp.asarray(jik),
    jnp.asarray(adummy), jnp.asarray(bdummy), jnp.asarray(cdummy),
    jnp.asarray(box), nat)
x_jax = np.asarray(x_jax)
# compare dummy positions
pos_diff = np.abs(x_jax[idat] - x_recon_ref[idat]).max()
print(f"[pos]  max dummy position diff: {pos_diff:.2e}")

# --- Test 2: force redistribution ---
f_before = d['f_before']; f_after_ref = d['f_after']
# positions: use the reconstructed reference (parents + dummies placed)
f_jax = dummy_force_redistribution(jnp.asarray(f_before), jnp.asarray(x_recon_ref),
    jnp.asarray(jik), jnp.asarray(adummy), jnp.asarray(bdummy), jnp.asarray(cdummy),
    jnp.asarray(box), nat)
f_jax = np.asarray(f_jax)
fdiff = np.abs(f_jax - f_after_ref).max()
print(f"[force] max force diff: {fdiff:.2e}")
print(f"[force] dummy forces zeroed: jax={np.abs(f_jax[idat]).max():.2e} ref={np.abs(f_after_ref[idat]).max():.2e}")

ok = pos_diff < 1e-9 and fdiff < 1e-6
print(f"\n{'[PASS]' if ok else '[FAIL]'} dummy atoms match reference")
