import os
os.environ['JAX_PLATFORMS'] = 'cpu'
from jax import config; config.update('jax_enable_x64', True)
import sys; sys.path.insert(0, '.')
import numpy as np
import jax.numpy as jnp
from wmi_md_jax.forces.bonded import bond_energy_force, bend_energy_force, torsion_energy_force

d = np.load('/tmp/bonded_ref.npz')
x = jnp.asarray(d['x']); box = jnp.asarray(d['box'])
nat = d['x'].shape[0]

# Bonds
eb, fb = bond_energy_force(x, jnp.asarray(d['bonds']), jnp.asarray(d['stretch']), box, nat)
eb = float(eb); fb = np.asarray(fb)
print(f"BOND:    E jax={eb:.6f} ref={float(d['eb']):.6f} diff={abs(eb-float(d['eb'])):.2e}  Fmax={np.abs(fb-d['fb']).max():.2e}")

# Bends
ebe, fbe = bend_energy_force(x, jnp.asarray(d['bends']), jnp.asarray(d['bend']), box, nat)
ebe = float(ebe); fbe = np.asarray(fbe)
print(f"BEND:    E jax={ebe:.6f} ref={float(d['ebe']):.6f} diff={abs(ebe-float(d['ebe'])):.2e}  Fmax={np.abs(fbe-d['fbe']).max():.2e}")

# Torsions
max_terms = int(d['nprms'].max())
print(f"  (max torsion terms = {max_terms})")
et, ft = torsion_energy_force(x, jnp.asarray(d['torts']), jnp.asarray(d['twist']),
                              jnp.asarray(d['nprms']), box, nat, max_terms)
et = float(et); ft = np.asarray(ft)
print(f"TORSION: E jax={et:.6f} ref={float(d['et']):.6f} diff={abs(et-float(d['et'])):.2e}  Fmax={np.abs(ft-d['ft']).max():.2e}")

ok = (abs(eb-float(d['eb']))<1e-4 and np.abs(fb-d['fb']).max()<1e-4 and
      abs(ebe-float(d['ebe']))<1e-3 and np.abs(fbe-d['fbe']).max()<1e-3 and
      abs(et-float(d['et']))<1e-3 and np.abs(ft-d['ft']).max()<1e-3)
print(f"\n{'[PASS]' if ok else '[FAIL]'} bonded forces match reference")
