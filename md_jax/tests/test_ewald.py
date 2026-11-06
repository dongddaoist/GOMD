import os
os.environ['JAX_PLATFORMS'] = 'cpu'
from jax import config; config.update('jax_enable_x64', True)
import sys; sys.path.insert(0, '.')
import numpy as np
import jax.numpy as jnp
from wmi_md_jax.forces.ewald import (
    build_kvectors, reciprocal_energy_force, self_energy, exclusion_correction
)

d = np.load('/tmp/ewald_ref2.npz')
x = d['x']; box = d['box']; q = d['q']; alpha = float(d['alpha'])
nat = x.shape[0]

# --- Reciprocal ---
kx, ky, kz = build_kvectors(box, int(d['klimitx']), int(d['klimity']), int(d['klimitz']))
print(f"k-vectors: {kx.size}")
erecip_raw, f_recip = reciprocal_energy_force(
    jnp.asarray(x), jnp.asarray(q), jnp.asarray(box),
    jnp.asarray(kx), jnp.asarray(ky), jnp.asarray(kz), alpha)
erecip_raw = float(erecip_raw); f_recip = np.asarray(f_recip)
print(f"RECIP: E jax={erecip_raw:.6f} ref={float(d['erecip_raw']):.6f} "
      f"diff={abs(erecip_raw-float(d['erecip_raw'])):.2e} Fmax={np.abs(f_recip-d['f_recip']).max():.2e}")

# --- Self energy ---
ews = self_energy(q, alpha)
print(f"SELF:  jax={ews:.6f} ref={float(d['ewself']):.6f} diff={abs(ews-float(d['ewself'])):.2e}")

# --- Exclusion correction ---
ecorr, f_corr = exclusion_correction(
    jnp.asarray(x), jnp.asarray(d['qq']), jnp.asarray(d['excl_i']), jnp.asarray(d['excl_j']),
    jnp.asarray(box), alpha, nat)
ecorr = float(ecorr); f_corr = np.asarray(f_corr)
print(f"CORR:  E jax={ecorr:.6f} ref={float(d['ecorr']):.6f} "
      f"diff={abs(ecorr-float(d['ecorr'])):.2e} Fmax={np.abs(f_corr-d['f_corr']).max():.2e}")

ok = (abs(erecip_raw-float(d['erecip_raw']))<1e-2 and np.abs(f_recip-d['f_recip']).max()<1e-3
      and abs(ews-float(d['ewself']))<1e-3
      and abs(ecorr-float(d['ecorr']))<1e-2 and np.abs(f_corr-d['f_corr']).max()<1e-3)
print(f"\n{'[PASS]' if ok else '[FAIL]'} Ewald matches reference")
