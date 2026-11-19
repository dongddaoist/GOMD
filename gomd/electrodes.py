"""
electrodes.py — fixed electrodes for an electrode | electrolyte | electrode cell.

Constant-potential task, phase P0 (2026-11-18). This module only sets the cell
up; the electrode charges are still the force field's (0 for graphite Cg) until
the constant-potential solver (P2) makes them variable.

    prepare_electrodes(D, lower, upper, geometry='flat', slab_height=None,
                       rlist=13.0, emit=print) -> info

  1. Selection. lower / upper are atom specs: comma-separated 'first:N',
     'last:N', 'a-b' (1-based, inclusive) or single 1-based indices. The
     electrodes are normal to z, lower below the electrolyte, upper above it.
     Electrode atoms must be real atoms and bonded only to each other.
  2. Layers. Each electrode is unwrapped in z about its first atom and split
     into layers wherever consecutive z values are more than 1 A apart.
  3. Geometry ('flat', the default). Each layer is put at its mean z. A layer
     that is a graphene sheet commensurate with the box (4 nx ny atoms, a
     rectangular cell 3b x sqrt(3)b in either orientation) is also snapped
     in-plane onto the ideal lattice: every atom goes to its nearest ideal site
     (one-to-one), with the lattice offset fitted to the layer. Other layers
     keep their x, y. 'snapshot' keeps the input positions (unwrapped).
  4. Slab layout (slab_height set). The box is made slab_height tall in z and
     the stack lower electrode | electrolyte | upper electrode is placed in its
     middle with vacuum above and below, so nothing interacts across z through
     the real-space cutoff. Every mobile site keeps its height above the lower
     electrode's bottom layer (taken going up the old periodic box), so a stack
     that the input wrapped across z (here: the lower electrode's bottom sheet
     at z = 131.3 in a 131.6 A box) comes out whole. kmax_z is scaled with the
     height. The driver then adds the planar dipole term to the k-space sum
     (recip_shared slab=True), which makes it the 2D Ewald sum (P1).
  5. Freezing. Electrode velocities are zeroed and D['frozen'] is set; the MD
     step neither kicks, moves nor thermostats frozen atoms
     (integrator_scan.make_md_step(frozen=...)), and the neighbor build skips
     frozen-frozen pairs (constant energy). Bonds, bends and torsions among
     frozen atoms only are dropped; so are out-of-plane terms, which GOMD does
     not evaluate at all and which are therefore refused if they touch a mobile
     atom. SHAKE constraints between two frozen atoms are dropped, one between a
     frozen and a mobile atom is refused. ndof loses the frozen atoms' 3N, and
     gets back the constraints that were dropped.

The potential energy then excludes the frozen-frozen terms (real-space pairs
and bonded), which are constant while the electrodes are frozen and their
charges fixed.
"""
from __future__ import annotations
import numpy as np

SQ3 = np.sqrt(3.0)


def parse_atom_spec(spec, nat):
    """'first:N', 'last:N', 'a-b' (1-based inclusive), 'i', comma-separated -> sorted 0-based."""
    idx = []
    for part in str(spec).split(','):
        part = part.strip()
        if not part:
            continue
        if part.startswith('first:'):
            idx.extend(range(0, int(part[6:])))
        elif part.startswith('last:'):
            idx.extend(range(nat - int(part[5:]), nat))
        elif '-' in part:
            a, b = part.split('-', 1)
            idx.extend(range(int(a) - 1, int(b)))
        else:
            idx.append(int(part) - 1)
    idx = np.unique(np.asarray(idx, dtype=np.int64))
    if idx.size == 0 or idx[0] < 0 or idx[-1] >= nat:
        raise SystemExit(f"electrode spec {spec!r} selects no atoms or atoms outside 1..{nat}")
    return idx


def _mi(d, L):
    return d - L * np.round(d / L)


def _layers(z, Lz, gap=1.0):
    """Unwrap z about the first atom; split at gaps > `gap` A. Returns (z_unwrapped,
    [index arrays into z], bottom to top)."""
    zu = z[0] + _mi(z - z[0], Lz)
    order = np.argsort(zu, kind='stable')
    cuts = np.flatnonzero(np.diff(zu[order]) > gap) + 1
    return zu, np.split(order, cuts)


def _nearest_neighbor_distance(xy, Lxy):
    d = _mi(xy[:, None, :] - xy[None, :, :], Lxy)
    r2 = np.einsum('ijk,ijk->ij', d, d)
    np.fill_diagonal(r2, np.inf)
    return float(np.median(np.sqrt(r2.min(1))))


def snap_graphene(xy, Lxy, max_rms=0.3):
    """Fit a graphene layer (n atoms, in-plane coordinates xy) to the ideal lattice
    commensurate with the in-plane box Lxy. Returns (new_xy or None, info)."""
    n = len(xy)
    b0 = _nearest_neighbor_distance(xy, Lxy)
    best = None
    for orient in (0, 1):                       # 0: cell 3b along x; 1: 3b along y
        cell0 = np.array([3 * b0, SQ3 * b0]) if orient == 0 else np.array([SQ3 * b0, 3 * b0])
        nxy = np.rint(Lxy / cell0).astype(int)
        if (nxy < 1).any() or 4 * nxy[0] * nxy[1] != n:
            continue
        cell = Lxy / nxy
        bx, by = (cell[0] / 3, cell[1] / SQ3) if orient == 0 else (cell[0] / SQ3, cell[1] / 3)
        if abs(bx - by) > 0.01 * b0:            # box not commensurate with one bond length
            continue
        b = 0.5 * (bx + by)
        basis = np.array([[0, 0], [b / 2, SQ3 * b / 2], [1.5 * b, SQ3 * b / 2], [2 * b, 0]])
        if orient == 1:
            basis = basis[:, ::-1]
        I, J = np.meshgrid(np.arange(nxy[0]), np.arange(nxy[1]), indexing='ij')
        cells = np.stack([I.ravel() * cell[0], J.ravel() * cell[1]], axis=1)
        sites = (cells[:, None, :] + basis[None, :, :]).reshape(-1, 2)
        for k in range(4):                      # which basis site the first atom sits on
            off = xy[0] - basis[k]
            for _ in range(3):                  # assign to nearest sites, re-centre the offset
                d = _mi(xy[:, None, :] - (sites[None, :, :] + off), Lxy)
                a = np.einsum('ijk,ijk->ij', d, d).argmin(1)
                res = d[np.arange(n), a]
                off = off + res.mean(0)
            d = _mi(xy[:, None, :] - (sites[None, :, :] + off), Lxy)
            a = np.einsum('ijk,ijk->ij', d, d).argmin(1)
            res = d[np.arange(n), a]
            rms = float(np.sqrt((res ** 2).sum(1).mean()))
            if np.unique(a).size == n and (best is None or rms < best[0]):
                best = (rms, sites[a] + off, b, orient, nxy, float(np.sqrt((res ** 2).sum(1)).max()))
    if best is None or best[0] > max_rms:
        return None, {'snapped': False, 'b': b0,
                      'reason': 'not a commensurate graphene sheet' if best is None
                      else f'rms {best[0]:.3f} A > {max_rms} A'}
    rms, new, b, orient, nxy, dmax = best
    new = new - Lxy * np.floor(new / Lxy)
    return new, {'snapped': True, 'b': b, 'rms': rms, 'max': dmax,
                 'cells': f"{nxy[0]}x{nxy[1]}", 'armchair': 'xy'[orient]}


def prepare_electrodes(D, lower, upper, geometry='flat', slab_height=None, rlist=13.0,
                       emit=print):
    """Select, shape, place and freeze the two electrodes (see the module docstring).
    Mutates D (x, v, box, klimit, conn, con, ensemble ndof; adds 'frozen' and
    'electrodes'). Returns the info dict also stored as D['electrodes']."""
    if geometry not in ('flat', 'snapshot'):
        raise SystemExit(f"electrode geometry {geometry!r}: use flat or snapshot")
    nat = int(D['nat'])
    x = np.array(D['x'], dtype=np.float64)
    v = np.array(D['v'], dtype=np.float64)
    box = np.array(D['box'], dtype=np.float64)
    ldummy = np.asarray(D['ldummy'], dtype=bool)
    sel = {'lower': parse_atom_spec(lower, nat), 'upper': parse_atom_spec(upper, nat)}
    if np.intersect1d(sel['lower'], sel['upper']).size:
        raise SystemExit("the lower and upper electrode specs overlap")
    frozen = np.zeros(nat, dtype=bool)
    for idx in sel.values():
        frozen[idx] = True
    if (frozen & ldummy).any():
        raise SystemExit("an electrode spec includes lone-pair sites")
    b = np.asarray(D['conn']['bonds'])
    cross = frozen[b[0]] != frozen[b[1]]
    if cross.any():
        k = int(np.flatnonzero(cross)[0])
        raise SystemExit(f"electrode atoms are bonded to electrolyte atoms (bond {b[0, k] + 1}-"
                         f"{b[1, k] + 1}): an electrode must be whole molecules")

    # ---- layers and geometry ----
    info = {'geometry': geometry, 'layers': {}}
    zr = {}
    for name, idx in sel.items():
        zu, groups = _layers(x[idx, 2], box[2])
        rows = []
        for g in groups:
            a = idx[g]
            zl = zu[g]
            row = {'n': int(a.size), 'z': float(zl.mean()), 'z_sd': float(zl.std()), 'atoms': a}
            if geometry == 'flat':
                new, fit = snap_graphene(x[a, :2], box[:2])
                row.update(fit)
                if new is not None:
                    x[a, :2] = new
                x[a, 2] = zl.mean()
            else:
                x[a, 2] = zl
            rows.append(row)
        info['layers'][name] = rows
        zr[name] = (min(r['z'] for r in rows), max(r['z'] for r in rows))
        # store the electrode's unwrapped z (flat: layer means) for the slab layout
    zlo_bot, zlo_top = zr['lower']
    zup_bot, zup_top = zr['upper']
    Lz = box[2]

    # ---- heights above the lower electrode's bottom layer, going up the box ----
    s = np.mod(x[:, 2] - zlo_bot, Lz)
    s[sel['lower']] = x[sel['lower'], 2] - zlo_bot                     # 0 .. lower thickness
    s_up_bot = float(np.mod(zup_bot - zlo_bot, Lz))
    s[sel['upper']] = x[sel['upper'], 2] - zup_bot + s_up_bot
    t_lo = zlo_top - zlo_bot
    H = s_up_bot + (zup_top - zup_bot)                                # stack height
    mob = ~frozen & ~ldummy
    outside = mob & ((s <= t_lo) | (s >= s_up_bot))
    if outside.any():
        k = np.flatnonzero(outside)[:5] + 1
        raise SystemExit(f"{int(outside.sum())} electrolyte atoms are not between the lower "
                         f"electrode's top layer and the upper electrode's bottom layer "
                         f"(e.g. atoms {list(k)}); are lower and upper swapped?")
    gap_in = s_up_bot - t_lo
    info.update({'n_lower': int(sel['lower'].size), 'n_upper': int(sel['upper'].size),
                 'inner_gap': float(gap_in), 'stack_height': float(H)})

    if slab_height is not None:
        L = float(slab_height)
        vac = L - H
        if vac < rlist:
            raise SystemExit(f"slab height {L:g} A leaves {vac:.1f} A of vacuum for a "
                             f"{H:.1f} A stack; it must be at least rlist = {rlist:g} A "
                             f"(for the 2D Ewald of P1, much more)")
        base = 0.5 * vac
        x[:, 2] = base + s
        kz_old = int(D['klimit'][2])
        kz_new = int(np.ceil(kz_old * L / Lz))
        D['klimit'] = (int(D['klimit'][0]), int(D['klimit'][1]), kz_new)
        box[2] = L
        info.update({'slab_height': L, 'vacuum': float(vac), 'kmax_z': (kz_old, kz_new),
                     'lower_z': (base, base + t_lo),
                     'upper_z': (base + s_up_bot, base + H)})
    else:
        x = x - box * np.floor(x / box)
        info.update({'slab_height': None})

    for name in ('lower', 'upper'):               # layer z in the final layout
        for r in info['layers'][name]:
            r['z_final'] = float(np.mean(x[r['atoms'], 2]))
    v[frozen] = 0.0
    D['x'] = x; D['v'] = v; D['box'] = box
    D['frozen'] = frozen

    # ---- bonded terms and constraints among frozen atoms ----
    conn = dict(D['conn'])
    dropped = {}
    for key, nrow in (('bonds', 2), ('bends', 3), ('torts', 4)):
        arr = np.asarray(conn[key])
        allf = frozen[arr[:nrow]].all(axis=0) if arr.shape[1] else np.zeros(0, bool)
        conn[key] = arr[:, ~allf]
        dropped[key] = int(allf.sum())
    dfm = np.asarray(conn.get('deforms', np.zeros((8, 0), np.int32)))
    if dfm.shape[1]:
        allf = frozen[dfm[:4]].all(axis=0)
        if not allf.all():
            raise SystemExit(f"{int((~allf).sum())} out-of-plane (deformation) terms involve "
                             f"mobile atoms; GOMD does not evaluate out-of-plane terms")
        conn['deforms'] = dfm[:, ~allf]
        dropped['out-of-plane'] = int(allf.sum())
    D['conn'] = conn
    con = dict(D['con'])
    ci, cj = np.asarray(con['con_i']), np.asarray(con['con_j'])
    fi, fj = frozen[ci], frozen[cj]
    if (fi != fj).any():
        raise SystemExit("a SHAKE constraint joins a frozen and a mobile atom")
    keep = ~(fi & fj)
    n_con_drop = int((~keep).sum())
    for k in ('con_i', 'con_j', 'd2', 'massred'):
        con[k] = np.asarray(con[k])[keep]
    con['nbcon'] = int(keep.sum())
    D['con'] = con
    dropped['constraints'] = n_con_drop

    E = D['ensemble']
    nfz = int(frozen.sum())
    ndof_old = int(E['ndof'])
    E['ndof'] = ndof_old - 3 * nfz + (n_con_drop if con['constrain'] else 0)
    info.update({'dropped': dropped, 'ndof': (ndof_old, int(E['ndof'])), 'n_frozen': nfz})
    D['electrodes'] = info

    # ---- report ----
    emit(f"electrodes: lower {info['n_lower']} atoms, upper {info['n_upper']} atoms, normal "
         f"to z, frozen; geometry {geometry}")
    for name in ('lower', 'upper'):
        for i, r in enumerate(info['layers'][name]):
            fit = ''
            if geometry == 'flat':
                fit = (f"; snapped to graphene ({r['cells']} cells, b {r['b']:.4f} A, armchair "
                       f"along {r['armchair']}): in-plane rms {r['rms']:.3f} A, max {r['max']:.3f} A"
                       if r.get('snapped') else f"; z flattened only ({r.get('reason')})")
            emit(f"  {name} layer {i}: {r['n']} atoms, z {r['z_final']:.3f} (input sd "
                 f"{r['z_sd']:.3f}){fit}")
    emit(f"  inner gap (layer centres) {gap_in:.3f} A; stack height {H:.3f} A")
    if slab_height is not None:
        emit(f"  slab box: z {Lz:.3f} -> {box[2]:.3f} A, vacuum {info['vacuum']:.1f} A; lower "
             f"electrode at z {info['lower_z'][0]:.2f}..{info['lower_z'][1]:.2f}, upper "
             f"{info['upper_z'][0]:.2f}..{info['upper_z'][1]:.2f}; kmax_z {info['kmax_z'][0]} -> "
             f"{info['kmax_z'][1]}")
    emit(f"  frozen: {nfz} atoms; dropped "
         + ", ".join(f"{v} {k}" for k, v in dropped.items())
         + f"; ndof {ndof_old} -> {E['ndof']}; COM-velocity removal off (the electrodes "
           "take momentum)")
    return info
