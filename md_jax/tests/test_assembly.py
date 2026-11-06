import os
os.environ['JAX_PLATFORMS'] = 'cpu'
from jax import config; config.update('jax_enable_x64', True)
import sys; sys.path.insert(0, '.')
import numpy as np
import jax.numpy as jnp
import time

from wmi_md_jax.io.assembly_loader import load_m2_system
from wmi_md_jax.forces.neighbor import build_neighbor_pairs
from wmi_md_jax.forces.ewald import build_kvectors, self_energy
from wmi_md_jax.forces.all_forces import compute_all_forces

REF = '/home/claude/pkg_inspect/pkg'
print("Loading system via assembly loader...")
t0=time.time()
D = load_m2_system(REF, REF)   # input_dir == ref pkg dir (has the input files)
print(f"  loaded in {time.time()-t0:.1f}s, nat={D['nat']}, ndummy={D['ndummy']}")

# Build neighbor pairs (host) with exclusions
t0=time.time()
ii, jj = build_neighbor_pairs(D['x'], D['box'], D['rcut'], skin=0.0, exclusions=D['excl_set'])
print(f"  neighbor build {time.time()-t0:.1f}s, pairs={ii.size}")

# k-vectors + self energy
kx, ky, kz = build_kvectors(D['box'], *D['klimit'])
ewself = self_energy(D['q'], D['alpha'])

# Convert needed arrays to jax and pack a jax-D
def J(a): return jnp.asarray(a)
Dj = dict(D)
for k in ['x','box','q','atomtype','typee','coefft','coeffft','coeffee','zs',
          'excl_qq','excl_i','excl_j']:
    Dj[k] = J(D[k])
Dj['bonds']=J(D['conn']['bonds']); Dj['bends']=J(D['conn']['bends']); Dj['torts']=J(D['conn']['torts'])
Dj['stretch']=J(D['ffb']['stretch']); Dj['bend']=J(D['ffb']['bend'])
Dj['twist']=J(D['ffb']['twist']); Dj['nprms']=J(D['ffb']['nprms'])

x = J(D['x'])
pe, f, comps = compute_all_forces(x, Dj, J(ii), J(jj), J(kx), J(ky), J(kz), ewself)
pe = float(pe); f = np.asarray(f)

ref = np.load('/tmp/classical_ref.npz')
print("\n=== Assembly vs reference (classical, no polarization) ===")
print(f"  PE:  jax={pe:.6f}  ref={float(ref['poten']):.6f}  diff={abs(pe-float(ref['poten'])):.2e}")
print(f"  ebond: jax={float(comps['ebond']):.6f} ref={float(ref['ebond']):.6f}")
print(f"  ebend: jax={float(comps['ebend']):.6f} ref={float(ref['ebend']):.6f}")
print(f"  etort: jax={float(comps['etort']):.6f} ref={float(ref['etort']):.6f}")
fdiff = np.abs(f - ref['f']).max()
print(f"  force max diff: {fdiff:.2e}  (ref RMS {np.sqrt((ref['f']**2).mean()):.4f})")

ok = abs(pe-float(ref['poten']))<1e-2 and fdiff<1e-3
print(f"\n{'[PASS]' if ok else '[FAIL]'} full assembled force matches reference classical")
