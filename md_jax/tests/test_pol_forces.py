import os
os.environ['JAX_PLATFORMS']='cpu'
from jax import config; config.update('jax_enable_x64', True)
import sys; sys.path.insert(0,'.')
import numpy as np, jax.numpy as jnp
from wmi_md_jax.forces.polarization import pol_forces, tau_func_np

ref=np.load('/tmp/pol_ref.npz')
px,py,pz=ref['px'],ref['py'],ref['pz']; mu=np.stack([px,py,pz],axis=1)
atomtype=ref['atomtype']; c1=ref['c1']; ncharge=int(ref['ncharge'])
rtaper=float(ref['rtaper']); ro=float(ref['ro']); tauden=float(ref['tauden'])

d=np.load('/tmp/spline_ref.npz'); x=d['x']; box=d['box']
# need q per atom and alpha_ew
import json
# reload q from assembly? use ewald ref
e=np.load('/tmp/ewald_ref2.npz'); q_atom=e['q']; alpha_ew=float(e['alpha'])

# Build pairs from listpol
pp=np.load('/tmp/pol_pairs.npz'); listpol=pp['listpol']; listmpol=pp['listmpol']
nat=mu.shape[0]
ii=[];jj=[]
for iat in range(nat):
    for k in range(int(listmpol[iat])):
        ii.append(iat); jj.append(int(listpol[k,iat]))
ii=np.asarray(ii,np.int32); jj=np.asarray(jj,np.int32)

# per-pair factors
xi=x[ii]; xj=x[jj]; dr=xj-xi; dr-=box*np.round(dr/box)
rx,ry,rz=dr[:,0],dr[:,1],dr[:,2]
r=np.sqrt(rx*rx+ry*ry+rz*rz); rinv=1.0/r
ti=atomtype[ii]; tj=atomtype[jj]; both=(ti<ncharge)&(tj<ncharge)
c1ij=np.where(both,c1[np.minimum(ti,ncharge-1),np.minimum(tj,ncharge-1)],1.0e6)
expc1r3=np.where(both,np.exp(-c1ij*r**3),0.0)
tau=tau_func_np(r,rtaper,ro,tauden)
# dtau
dtau=np.where((r>rtaper)&(r<ro), ((r-ro)*(r-rtaper)**2)/tauden, 0.0)

f,vir=pol_forces(jnp.asarray(mu),jnp.asarray(q_atom),alpha_ew,
    jnp.asarray(ii),jnp.asarray(jj),jnp.asarray(rx),jnp.asarray(ry),jnp.asarray(rz),
    jnp.asarray(r),jnp.asarray(rinv),jnp.asarray(c1ij),jnp.asarray(expc1r3),
    jnp.asarray(tau),jnp.asarray(dtau),nat)
f=np.asarray(f)
ref2=np.load('/tmp/polf_ref.npz'); findown_ref=ref2['findown']
fdiff=np.abs(f-findown_ref).max()
print(f"findown max diff: {fdiff:.2e}")
print(f"RMS jax={np.sqrt((f**2).mean()):.6e} ref={np.sqrt((findown_ref**2).mean()):.6e}")
ok = fdiff < 1e-6
print(f"{'[PASS]' if ok else '[FAIL]'} pol forces match reference")
