"""
io/assembly_loader.py — Extract everything the JAX M2 engine needs.

The reference setup_system (numpy, one-time) builds spline tables, connectivity,
exclusions, constraints, dummy connectivity, and force-field tables. Rather than
re-port all of that setup, we run it once and extract a flat dict of JAX-ready
arrays. This is the clean numpy->JAX boundary for M2.

Returns a dict `D` with everything compute_all_forces + the integrator need.
All positions are stored (nat,3) row-major (transposed from reference (3,nat)).
"""
from __future__ import annotations
import sys
import os
from pathlib import Path
import numpy as np


def load_m2_system(input_dir, ref_pkg_dir):
    """Run the reference setup and extract JAX-ready arrays.

    Args:
        input_dir: directory with mdrun.params, ff.dat, connectivity.dat, coords.inp.
        ref_pkg_dir: path to the reference Python package (so we can import it).

    Returns:
        dict of numpy arrays (caller converts to jax).
    """
    ref_pkg_dir = os.path.abspath(str(ref_pkg_dir))
    input_dir = os.path.abspath(str(input_dir))
    if ref_pkg_dir not in sys.path:
        sys.path.insert(0, ref_pkg_dir)

    from input_parser import load_simulation
    import main as refmain

    cwd = os.getcwd()
    try:
        os.chdir(input_dir)
        state = load_simulation(input_dir)
        refmain.setup_system(state)
    finally:
        os.chdir(cwd)

    a = state.atoms; ff = state.ff; sp = state.splines
    c = state.constraints; ns = state.neighbors; p = state.params
    nat = a.nat; ndummy = a.ndummy
    atomtype = np.asarray(a.atomtype, dtype=np.int32)

    # ---- positions / velocities / masses ----
    x = np.asarray(a.x).T.copy()                  # (nat,3)
    v = np.asarray(a.v).T.copy() if a.v is not None else np.zeros((nat, 3))
    mass = np.asarray(a.mass).copy()
    massinv = np.asarray(a.massinv).copy()
    ldummy = np.asarray(a.ldummy, dtype=bool).copy()
    box = np.asarray(a.box).copy()

    # ---- charges per atom ----
    q_per_atom = np.asarray(ff.q)[atomtype].astype(np.float64)

    # ---- spline tables (M2.2) ----
    coefft = np.asarray(sp.coefft)
    coeffft = np.asarray(sp.coeffft)
    coeffee = np.asarray(sp.coeffee)
    # Moved-axis copies for the optimized splined-pair kernel: (npiece,ninter,6)
    # so one gather yields all 6 coeffs per pair (was 6 separate gathers).
    coefft_m = np.moveaxis(coefft, 0, -1).copy()
    coeffft_m = np.moveaxis(coeffft, 0, -1).copy()
    zs = np.asarray(sp.zs)
    deltainv = float(sp.deltainv)
    typee = np.asarray(ff.typee, dtype=np.int32)
    rcut = float(p.rcut)

    # ---- bonded connectivity (M2.3) ----
    conn = {
        'bonds': np.asarray(state.connectivity.bonds)[:, :state.connectivity.nbonds].astype(np.int32),
        'bends': np.asarray(state.connectivity.bends)[:, :state.connectivity.nbends].astype(np.int32),
        'torts': np.asarray(state.connectivity.torts)[:, :state.connectivity.ntorts].astype(np.int32),
    }
    ffb = {
        'stretch': np.asarray(ff.stretch),
        'bend': np.asarray(ff.bend),
        'twist': np.asarray(ff.twist),
        'nprms': np.asarray(ff.nprms, dtype=np.int32),
    }
    max_torsion_terms = int(np.asarray(ff.nprms).max())

    # ---- exclusions (M2.1 + M2.4 correction) ----
    listex = np.asarray(ns.listex); listmex = np.asarray(ns.listmex)
    elec = np.asarray(ff.electrostatic)
    # exclusion set (i<j) for neighbor build
    excl_set = set()
    excl_i = []; excl_j = []; excl_qq = []
    for i in range(nat):
        for k in range(int(listmex[i])):
            j = int(listex[k, i])
            if j == i:
                continue
            lo, hi = (i, j) if i < j else (j, i)
            excl_set.add((lo, hi))
            # directed pairs with qq for the Ewald correction (match reference loop)
            itypee = typee[atomtype[i], atomtype[j]]
            excl_i.append(i); excl_j.append(j); excl_qq.append(elec[itypee, 0])
    excl_i = np.asarray(excl_i, dtype=np.int32)
    excl_j = np.asarray(excl_j, dtype=np.int32)
    excl_qq = np.asarray(excl_qq)

    # ---- Ewald params (M2.4) ----
    alpha = float(p.alphai)
    klimit = (int(p.klimitx), int(p.klimity), int(p.klimitz))

    # ---- constraints (M2.5) ----
    con = {
        'con_i': np.asarray(c.bondcon)[0, :c.nbcon].astype(np.int32),
        'con_j': np.asarray(c.bondcon)[1, :c.nbcon].astype(np.int32),
        'd2': np.asarray(c.d2)[:c.nbcon].copy(),
        'massred': np.asarray(c.massred)[:c.nbcon].copy(),
        'tol': float(c.tol),
        'nbcon': int(c.nbcon),
        'constrain': bool(p.constrain),
    }

    # ---- dummy connectivity (M2.6) ----
    if ndummy > 0 and a.jikdummy is not None:
        jik = np.asarray(a.jikdummy)[:, :ndummy].astype(np.int32)
    else:
        jik = np.zeros((5, 0), dtype=np.int32)
    dummy = {
        'jik': jik,
        'adummy': np.asarray(ff.adummy),
        'bdummy': np.asarray(ff.bdummy),
        'cdummy': np.asarray(ff.cdummy),
        'ndummy': int(ndummy),
    }

    # ---- polarization (M3) ----
    pol_data = None
    if bool(getattr(ff, 'lpolarizable', False)):
        import polarization as _P
        ps = state.pol_state
        if not ps.initialized:
            _P.init_polarization(state)
        el = np.asarray(a.el).copy()
        # directed 1-4 and exclusion pair lists (host-built once)
        ns = state.neighbors
        list14 = getattr(ns, 'list14', None)
        listm14 = getattr(ns, 'listm14', None)
        l14_i, l14_j = [], []
        if list14 is not None and listm14 is not None:
            for iat in range(nat):
                for k in range(int(listm14[iat])):
                    l14_i.append(iat); l14_j.append(int(list14[k, iat]))
        ex_i14, ex_j14 = [], []
        for iat in range(nat):
            for k in range(int(ns.listmex[iat])):
                ex_i14.append(iat); ex_j14.append(int(ns.listex[k, iat]))
        pol_data = {
            'pol': np.asarray(ff.pol).copy(),          # per-type polarizability
            'alpha_atom': np.asarray(ff.pol)[atomtype].astype(np.float64),
            'c1': np.asarray(ps.c1).copy(),
            'ncharge': int(ff.ncharge),
            'arf': float(ps.arf),
            'rtaper': float(p.rtaper), 'ro': float(p.rcut), 'tauden': float(ps.tauden),
            'tolpol': float(p.tolpol),
            'el': el,
            'l14_i': np.asarray(l14_i, dtype=np.int32),
            'l14_j': np.asarray(l14_j, dtype=np.int32),
            'ex_i': np.asarray(ex_i14, dtype=np.int32),
            'ex_j': np.asarray(ex_j14, dtype=np.int32),
            'lredQ_mu14': bool(p.lredQ_mu14),
            'redQmufactor': float(p.redQmufactor),
            'lredonefour': bool(p.lredonefour),
            'redfactor': float(p.redfactor),
            'lpolarizable': True,
        }

    return {
        'nat': nat, 'ndummy': ndummy,
        'x': x, 'v': v, 'mass': mass, 'massinv': massinv,
        'ldummy': ldummy, 'box': box, 'q': q_per_atom, 'atomtype': atomtype,
        # splines
        'coefft': coefft, 'coeffft': coeffft, 'coeffee': coeffee,
        'coefft_m': coefft_m, 'coeffft_m': coeffft_m,
        'zs': zs, 'deltainv': deltainv, 'typee': typee, 'rcut': rcut,
        # bonded
        'conn': conn, 'ffb': ffb, 'max_torsion_terms': max_torsion_terms,
        # exclusions / ewald
        'excl_set': excl_set, 'excl_i': excl_i, 'excl_j': excl_j, 'excl_qq': excl_qq,
        'alpha': alpha, 'klimit': klimit,
        # constraints / dummy
        'con': con, 'dummy': dummy,
        # polarization (M3); None if classical force field
        'pol_data': pol_data,
        # control
        'dt': float(p.delt),
    }
