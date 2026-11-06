import os
os.environ['JAX_PLATFORMS'] = 'cpu'
from jax import config; config.update('jax_enable_x64', True)
import sys; sys.path.insert(0, '.')
import numpy as np
import jax.numpy as jnp
from wmi_md_jax.forces.polarization import pack_pairs, solve_scf, pol_energy

# Reference converged dipoles + total field
ref = np.load('/tmp/pol_ref.npz')
px,py,pz = ref['px'],ref['py'],ref['pz']
mu_ref = np.stack([px,py,pz],axis=1)        # (nat,3)
elfown_total = ref['elfown']                 # (nat,3) total field used in SCF
epol_ref = float(ref['epol'])
atomtype = ref['atomtype']; c1 = ref['c1']; pol = ref['pol']
arf = float(ref['arf']); tauden=float(ref['tauden'])
rtaper=float(ref['rtaper']); ro=float(ref['ro']); tolpol=float(ref['tolpol'])
ncharge=int(ref['ncharge'])
nat = mu_ref.shape[0]

# Build OrderedSparse pairs from reference listpol (dense)
pp = np.load('/tmp/pol_pairs.npz')
listpol = pp['listpol']; listmpol = pp['listmpol']
ii=[]; jj=[]
for iat in range(nat):
    for k in range(int(listmpol[iat])):
        jat = int(listpol[k,iat])
        ii.append(iat); jj.append(jat)
ii=np.asarray(ii,dtype=np.int32); jj=np.asarray(jj,dtype=np.int32)
print(f"pol pairs: {ii.size}")

# Need positions to pack pairs. Load from spline ref (same x)
d = np.load('/tmp/spline_ref.npz')
x = d['x']; box = d['box']

# Pack pair factors
P = pack_pairs(ii, jj, x, box, atomtype, c1, ncharge, arf, rtaper, ro, tauden)

# alpha per atom
alpha_atom = np.where(atomtype < ncharge, pol[np.minimum(atomtype,ncharge-1)], 0.0)

# Solve SCF from zero initial guess
mu0 = np.zeros((nat,3))
mu, niter = solve_scf(
    jnp.asarray(elfown_total), jnp.asarray(alpha_atom),
    jnp.asarray(P['idx_i']), jnp.asarray(P['idx_j']),
    jnp.asarray(P['rx']), jnp.asarray(P['ry']), jnp.asarray(P['rz']),
    jnp.asarray(P['tfact']), jnp.asarray(P['three_zf_l5']),
    jnp.asarray(mu0), arf, tolpol, nat, 200)
mu = np.asarray(mu); niter=int(niter)
print(f"JAX SCF: {niter} iters")
print(f"mu RMS: jax={np.sqrt((mu**2).mean()):.6e} ref={np.sqrt((mu_ref**2).mean()):.6e}")
mudiff = np.abs(mu - mu_ref).max()
print(f"max dipole diff: {mudiff:.2e}")

# Energy
epol_jax = float(pol_energy(jnp.asarray(elfown_total), jnp.asarray(mu)))
print(f"epol: jax={epol_jax:.6f} ref={epol_ref:.6f} diff={abs(epol_jax-epol_ref):.2e}")

ok = mudiff < 1e-4 and abs(epol_jax-epol_ref) < 1e-2
print(f"\n{'[PASS]' if ok else '[FAIL]'} polarization SCF matches reference")
