"""
neighbor_list_fast.py
=====================
JIT-compiled inner kernel for the cell-list neighbor-pair scan.

The original neighbor_list.build_cell_list is ~200 lines of pure Python
including a triple-nested loop over neighbor cells, with `subbox[ibox]`
as a Python list of lists. For nat=8716, this is several seconds per
rebuild — and rebuilds happen on every step where atoms drift more
than driftmax.

This module provides _scan_neighbor_pairs_jit, a Numba-compiled
function that does the pair-scanning work using only NumPy arrays.
The cell-assignment remains in Python (it's O(nat) and not the
bottleneck).

Usage: imported and called by neighbor_list.build_cell_list (patched).
"""

import numpy as np
from numba import njit


@njit(cache=True)
def _scan_neighbor_pairs_jit(
    x, nat, chain, boxofatom,
    cell_count, cell_atoms,           # flat representation of subbox
    idim, idim2, ncheck,
    box, el,
    rcut2, rsbox2,
    listex, listmex,
    inter,
    list_neighbors, lists,
    maxnay, maxnay2
):
    """Scan all neighboring cells for pair candidates.

    Mirrors the body of the second big Python loop in the original
    build_cell_list, but with `subbox[ibox]` replaced by
    `cell_atoms[k, ibox] for k in range(cell_count[ibox])` so it can
    be JIT-compiled.

    Inputs (all arrays):
      cell_count: (ncells,) number of atoms in each cell
      cell_atoms: (max_per_cell, ncells) atom index in slot k of cell c

    Outputs:
      list_neighbors: (maxnay, nat) — modified in place
      listm:          (nat,)        — returned
      lists:          (maxnay2, nat)— modified in place
      listms:         (nat,)        — returned
    """
    listm = np.zeros(nat, dtype=np.int64)
    listms = np.zeros(nat, dtype=np.int64)

    for iat in range(nat):
        ich = chain[iat]
        nayo = 0
        nayos = 0
        ix = boxofatom[0, iat]
        iy = boxofatom[1, iat]
        iz = boxofatom[2, iat]

        for isubz in range(iz - ncheck[2], iz + ncheck[2] + 1):
            izz = isubz
            if izz >= idim[2]:
                izz -= idim[2]
            if izz < 0:
                izz += idim[2]
            izbox = izz * idim2

            for isuby in range(iy - ncheck[1], iy + ncheck[1] + 1):
                iyy = isuby
                if iyy >= idim[1]:
                    iyy -= idim[1]
                if iyy < 0:
                    iyy += idim[1]
                iybox = iyy * idim[0]

                for isubx in range(ix - ncheck[0], ix + ncheck[0] + 1):
                    ixx = isubx
                    if ixx >= idim[0]:
                        ixx -= idim[0]
                    if ixx < 0:
                        ixx += idim[0]
                    ibox = izbox + iybox + ixx

                    n_in_cell = cell_count[ibox]
                    for k in range(n_in_cell):
                        jat = cell_atoms[k, ibox]
                        jch = chain[jat]

                        # Even/odd skip rule for unordered pairs
                        if iat >= jat:
                            if (jat + iat) % 2 == 0:
                                continue
                        else:
                            if (jat + iat) % 2 != 0:
                                continue

                        # Intermolecular flag
                        if not inter:
                            if ich != jch:
                                continue

                        # Distance with PBC (manual since np.sign is tricky in JIT)
                        dx0 = x[0, jat] - x[0, iat]
                        dx1 = x[1, jat] - x[1, iat]
                        dx2 = x[2, jat] - x[2, iat]
                        if abs(dx0) > el[0]:
                            if dx0 > 0:
                                dx0 -= box[0]
                            else:
                                dx0 += box[0]
                        if abs(dx1) > el[1]:
                            if dx1 > 0:
                                dx1 -= box[1]
                            else:
                                dx1 += box[1]
                        if abs(dx2) > el[2]:
                            if dx2 > 0:
                                dx2 -= box[2]
                            else:
                                dx2 += box[2]

                        r2 = dx0*dx0 + dx1*dx1 + dx2*dx2

                        if r2 <= rcut2:
                            # Exclusion list scan (linear search; nex_max ~12 typical)
                            is_excluded = False
                            if ich == jch:
                                nex_iat = listmex[iat]
                                for iex in range(nex_iat):
                                    if listex[iex, iat] == jat:
                                        is_excluded = True
                                        break

                            if not is_excluded:
                                if r2 < rsbox2:
                                    if nayos < maxnay2:
                                        lists[nayos, iat] = jat
                                        nayos += 1
                                if nayo < maxnay:
                                    list_neighbors[nayo, iat] = jat
                                    nayo += 1

        listm[iat] = nayo
        listms[iat] = nayos

    return listm, listms


def build_cell_list_fast(state):
    """Drop-in replacement for neighbor_list.build_cell_list.

    Identical algorithm; the heavy pair-scan loop is JIT-compiled.
    """
    from neighbor_list import DefaultParams

    nat = state.atoms.nat
    box = state.atoms.box
    x = state.atoms.x
    rcut = state.params.rcut
    driftmax = state.params.driftmax
    chain = state.atoms.chain
    inter = state.params.inter

    el = box * 0.5
    state.atoms.el = el

    ros = state.params.ros
    rshort = ros + driftmax
    rsbox = rshort + driftmax
    rsbox2 = rsbox * rsbox
    rcut2 = rcut * rcut

    # Determine box subdivisions (same as original)
    rbig = rcut
    rdimmax = rbig / 2.0
    idim = np.zeros(3, dtype=np.int64)
    ncheck = np.zeros(3, dtype=np.int64)
    size = np.zeros(3)

    for kk in range(3):
        rdimmin = box[kk] / nat**(1.0/3.0)
        if rdimmax < rdimmin:
            rdim = rdimmax
        else:
            icheck = 2
            while True:
                icheck += 1
                rcheck = rbig / icheck
                if rcheck <= rdimmin:
                    break
            rdim = rbig / (icheck - 1)
        idim[kk] = int(box[kk] / rdim)
        if idim[kk] > DefaultParams.maxdim:
            idim[kk] = DefaultParams.maxdim
        rdim = box[kk] / idim[kk]
        ncheck[kk] = int(rbig / rdim) + 1
        if idim[kk] < 2 * ncheck[kk] + 1:
            if idim[kk] % 2 == 0:
                idim[kk] += 1
                rdim = box[kk] / idim[kk]
            ncheck[kk] = (idim[kk] - 1) // 2
        size[kk] = box[kk] / idim[kk]

    if state.params.printout:
        print(f"Boxes: idim = {idim}")

    idim2 = int(idim[0] * idim[1])
    total_cells = int(idim[0] * idim[1] * idim[2])

    # ----- Cell assignment (Python; O(nat), fast) -----
    boxofatom = np.zeros((3, nat), dtype=np.int64)
    # First pass: count per cell
    cell_count_tmp = np.zeros(total_cells, dtype=np.int64)
    box_index_per_atom = np.zeros(nat, dtype=np.int64)
    for iat in range(nat):
        for kk in range(3):
            coord = x[kk, iat]
            while coord < 0:
                coord += box[kk]
            while coord >= box[kk]:
                coord -= box[kk]
            isub_kk = int(coord / size[kk])
            if isub_kk >= idim[kk]:
                isub_kk = idim[kk] - 1
            elif isub_kk < 0:
                isub_kk = 0
            boxofatom[kk, iat] = isub_kk
        ibox = boxofatom[2, iat]*idim2 + boxofatom[1, iat]*idim[0] + boxofatom[0, iat]
        box_index_per_atom[iat] = ibox
        cell_count_tmp[ibox] += 1

    max_per_cell = int(cell_count_tmp.max()) + 1  # safety margin
    cell_atoms = np.zeros((max_per_cell, total_cells), dtype=np.int64)
    cell_count = np.zeros(total_cells, dtype=np.int64)
    for iat in range(nat):
        ibox = box_index_per_atom[iat]
        slot = cell_count[ibox]
        cell_atoms[slot, ibox] = iat
        cell_count[ibox] = slot + 1

    # ----- Output buffers -----
    maxnay = DefaultParams.maxnay
    maxnay2 = maxnay // 2
    list_neighbors = np.zeros((maxnay, nat), dtype=np.int64)
    lists = np.zeros((maxnay2, nat), dtype=np.int64)

    # Exclusions (already built)
    listex = (state.neighbors.listex if state.neighbors.listex is not None
              else np.zeros((40, nat), dtype=np.int64))
    listmex = (state.neighbors.listmex if state.neighbors.listmex is not None
               else np.zeros(nat, dtype=np.int64))
    listex = np.asarray(listex, dtype=np.int64)
    listmex = np.asarray(listmex, dtype=np.int64)
    chain_arr = np.asarray(chain, dtype=np.int64)

    # ----- JIT'd pair scan -----
    listm, listms = _scan_neighbor_pairs_jit(
        x, nat, chain_arr, boxofatom,
        cell_count, cell_atoms,
        idim, idim2, ncheck,
        box, el,
        rcut2, rsbox2,
        listex, listmex,
        bool(inter),
        list_neighbors, lists,
        maxnay, maxnay2
    )

    # Store in state
    state.neighbors.list_neighbors = list_neighbors
    state.neighbors.listm = listm
    state.neighbors.lists = lists
    state.neighbors.listms = listms
    state.neighbors.idim = idim
    state.neighbors.listsubbox = cell_count
