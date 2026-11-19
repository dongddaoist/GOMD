"""Shared setup for the gomd.tests diagnostics: the MD step exactly as main_scan
builds it (PCG + predicted start, drift rebuilds, per-step rcut compaction,
shared k-space pass, SHAKE 1e-12)."""
from __future__ import annotations


def build_md(input_dir, ref_dir, polarizable=True, tolpol=1e-8, ensemble='nve', dt=None,
             skin=1.0, rebuild_every=100, scf_max_iter=300, rtaper=None):
    """Returns a dict: D, Dj, E (ensemble data), parts, run (lax.scan runner),
    carry0, nhc (params or None), recip, capacity, cut_capacity."""
    import jax, jax.numpy as jnp
    import numpy as np
    from gomd.main_scan import build_Dj
    from gomd.io.assembly_loader import load_m2_system
    from gomd.forces.neighbor import build_neighbor_pairs
    from gomd.forces.neighbor_ondevice import exclusions_to_flat, count_pairs_within
    from gomd.forces.ewald import build_kvectors, self_energy
    from gomd.forces.recip_shared import make_recip
    from gomd.integrator_scan import make_md_step, make_runner
    from gomd import thermostat as nhc_mod

    D = load_m2_system(input_dir, ref_dir)
    if dt is not None:
        D['dt'] = float(dt)
    nat = D['nat']; E = D['ensemble']
    Dj = build_Dj(D); Dj['con']['tol'] = 1e-12
    kxn, kyn, kzn = build_kvectors(D['box'], *D['klimit'])
    recip = make_recip(D['box'], D['q'], kxn, kyn, kzn, D['alpha'])
    kx, ky, kz = jnp.asarray(kxn), jnp.asarray(kyn), jnp.asarray(kzn)
    ewself = self_energy(D['q'], D['alpha'])
    ii_h, jj_h = build_neighbor_pairs(np.asarray(D['x']), D['box'], D['rcut'], skin=skin,
                                      exclusions=D['excl_set'])
    capacity = int(ii_h.size * 1.25)
    ncut0 = int(count_pairs_within(jnp.asarray(D['x']), jnp.asarray(ii_h), jnp.asarray(jj_h),
                                   jnp.asarray(D['box']), float((D['rcut'] + 0.05) ** 2)))
    cut_capacity = int(ncut0 * 1.10)
    pol_fn = None
    if polarizable:
        from gomd.forces.polarization_full import make_polarization_fn
        D['pol_data']['tolpol'] = float(tolpol)
        if rtaper is not None:
            D['pol_data']['rtaper'] = float(rtaper)     # dipole-dipole taper on if < rcut
        pol_fn = make_polarization_fn(D, kx, ky, kz, scf_max_iter=scf_max_iter, solver='pcg',
                                      recip=recip)
    nhc = (nhc_mod.make_nhc_params(E['Tset'], E['ndof'], E['nnos'], E['qfreq'], D['dt'])
           if ensemble == 'nvt' else None)
    parts = make_md_step(Dj, kx, ky, kz, ewself, exclusions_to_flat(D['excl_set'], nat), capacity,
                         rebuild_every, float((D['rcut'] + skin) ** 2), polarization_fn=pol_fn,
                         use_aspc=polarizable, nhc_params=nhc, natreal=E['natreal'],
                         skin=skin, drift_rebuild=True, cut_capacity=cut_capacity, recip=recip)
    x0 = jax.jit(parts['place_dummies'])(Dj['x'])
    ii, jj, cnt = parts['rebuild'](x0)
    x0, pe0, f0, comps0, mu0 = jax.jit(parts['force_fn'])(x0, ii, jj)
    mu = (mu0, mu0, mu0) if mu0 is not None else (jnp.zeros((nat, 3)),) * 3
    nhc0 = ((jnp.asarray(E['xlogs']), jnp.asarray(E['vlogs']), jnp.asarray(E['glogs']))
            if nhc else (jnp.zeros(1), jnp.zeros(1), jnp.zeros(1)))
    S0 = {'x': x0, 'v': Dj['v'], 'f': f0, 'nhc': nhc0, 'step': jnp.array(1)}
    carry0 = parts['init_carry'](S0, x0, ii, jj, cnt, mu, pe0, f0)
    return {'D': D, 'Dj': Dj, 'E': E, 'parts': parts, 'run': make_runner(parts['scan_body']),
            'carry0': carry0, 'nhc': nhc, 'recip': recip, 'capacity': capacity,
            'cut_capacity': cut_capacity}


def molecules(D):
    """Molecule id per site (bond-graph components; lone-pair sites -> nmol, i.e.
    dropped by segment sums) and a group name per molecule: 'Li+' (contains Li),
    'FSI-' (contains S), 'solvent' (the rest)."""
    import numpy as np
    nat = D['nat']
    real = ~np.asarray(D['ldummy'])
    parent = np.arange(nat)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    b = np.asarray(D['conn']['bonds'])
    for i, j in zip(b[0], b[1]):
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[ri] = rj
    roots = np.array([find(a) for a in range(nat)])
    uniq = {r: k for k, r in enumerate(sorted(set(roots[real])))}
    nmol = len(uniq)
    mol = np.array([uniq.get(r, nmol) if real[a] else nmol for a, r in enumerate(roots)], np.int32)
    mass = np.asarray(D['mass'])
    groups = []
    for m in range(nmol):
        ms = mass[(mol == m)]
        if np.any(np.abs(ms - 6.94) < 0.05):
            groups.append('Li+')
        elif np.any(np.abs(ms - 32.06) < 0.05):
            groups.append('FSI-')
        else:
            groups.append('solvent')
    return mol, nmol, groups
