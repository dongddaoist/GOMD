"""
forces/neighbor.py — Neighbor list construction (M2.1).

Produces the set of non-bonded interaction pairs (i<j, within cutoff,
exclusions removed) that every pairwise force kernel consumes.

Design:
    - OrderedSparse format: pairs returned as two int arrays (idx_i, idx_j)
      with idx_i < idx_j. ~2.64M pairs for the 8716-atom system at rcut=12.
    - The BUILD runs in numpy (host). Neighbor lists rebuild infrequently
      (every ~10-50 steps when atoms have drifted > skin/2), so the build is
      not the per-step hot path. The per-step hot path is the force kernels
      that consume these pairs — those are jit'd JAX.
    - Exclusions (1-2, 1-3, 1-4 bonded pairs) are removed during the build,
      matching the reference's listex/listmex semantics.

Why a cell list:
    Brute O(N²) is 38M candidate pairs for 8716 atoms. A cell list reduces
    candidate generation to O(N) by only checking atoms in neighboring cells.
    We still verify against brute force on small systems for correctness.

This matches the reference build_cell_list (neighbor_list.py) in WHICH pairs
it produces, but uses a flat OrderedSparse output instead of the reference's
per-atom (maxnay, nat) dense layout.
"""
from __future__ import annotations
import numpy as np


def _minimum_image(dx, box):
    """Minimum image displacement (numpy), orthorhombic box."""
    return dx - box * np.round(dx / box)


def build_neighbor_pairs(
    x: np.ndarray,
    box: np.ndarray,
    rcut: float,
    skin: float = 1.0,
    exclusions: "set | None" = None,
):
    """Build OrderedSparse (i<j) neighbor pairs within rcut+skin.

    Args:
        x: (nat, 3) positions (numpy).
        box: (3,) orthorhombic box lengths.
        rcut: cutoff radius (Å). Pairs within rcut+skin are included so the
              list stays valid for a few steps before rebuild.
        skin: buffer added to rcut (Å). Larger skin = less frequent rebuild
              but more pairs.
        exclusions: optional set of (i, j) tuples with i<j to exclude
              (bonded 1-2/1-3/1-4). If None, no exclusions applied.

    Returns:
        idx_i, idx_j: (npairs,) int32 arrays, idx_i < idx_j, within cutoff,
              exclusions removed.
    """
    nat = x.shape[0]
    rlist = rcut + skin
    rlist2 = rlist * rlist

    # ----- Cell list construction -----
    # Number of cells per dimension (at least 3 so neighbor search is local).
    ncell = np.maximum(np.floor(box / rlist).astype(np.int64), 1)
    # If the box is too small for a 3x3x3 stencil (ncell < 3 in any dim),
    # fall back to brute force (correctness over speed for tiny systems).
    if np.any(ncell < 3):
        return _brute_pairs(x, box, rlist2, exclusions)

    cell_size = box / ncell
    # Assign each atom to a cell (wrap into [0, ncell) per dim).
    xw = x - box * np.floor(x / box)          # wrap into box
    cell_idx = np.floor(xw / cell_size).astype(np.int64)
    cell_idx = np.clip(cell_idx, 0, ncell - 1)

    ncx, ncy, ncz = int(ncell[0]), int(ncell[1]), int(ncell[2])
    ncells = ncx * ncy * ncz

    def cell_id(cx, cy, cz):
        return (cx % ncx) * ncy * ncz + (cy % ncy) * ncz + (cz % ncz)

    flat_cell = (cell_idx[:, 0] * ncy * ncz
                 + cell_idx[:, 1] * ncz
                 + cell_idx[:, 2])

    # Bucket atoms into cells (list of arrays).
    order = np.argsort(flat_cell, kind='stable')
    sorted_cells = flat_cell[order]
    # boundaries of each cell in the sorted order
    cell_starts = np.searchsorted(sorted_cells, np.arange(ncells))
    cell_ends = np.searchsorted(sorted_cells, np.arange(ncells), side='right')

    idx_i_list = []
    idx_j_list = []

    # 13 half-stencil neighbor offsets + self, to enumerate each pair once.
    # Using the standard "half" neighbor cell scheme avoids double counting.
    half_offsets = [
        (0, 0, 0),
        (1, 0, 0), (-1, 1, 0), (0, 1, 0), (1, 1, 0),
        (-1, -1, 1), (0, -1, 1), (1, -1, 1),
        (-1, 0, 1), (0, 0, 1), (1, 0, 1),
        (-1, 1, 1), (0, 1, 1), (1, 1, 1),
    ]

    for cx in range(ncx):
        for cy in range(ncy):
            for cz in range(ncz):
                c = cell_id(cx, cy, cz)
                ai = order[cell_starts[c]:cell_ends[c]]
                if ai.size == 0:
                    continue
                for (ox, oy, oz) in half_offsets:
                    c2 = cell_id(cx + ox, cy + oy, cz + oz)
                    aj = order[cell_starts[c2]:cell_ends[c2]]
                    if aj.size == 0:
                        continue
                    # Candidate pairs between ai and aj
                    if ox == 0 and oy == 0 and oz == 0:
                        # same cell: only i<j within the cell
                        for a in range(ai.size):
                            i = ai[a]
                            js = ai[a+1:]
                            if js.size == 0:
                                continue
                            dx = _minimum_image(x[i] - x[js], box)
                            r2 = np.einsum('ij,ij->i', dx, dx)
                            m = r2 < rlist2
                            for j in js[m]:
                                lo, hi = (i, j) if i < j else (j, i)
                                idx_i_list.append(lo)
                                idx_j_list.append(hi)
                    else:
                        # different cell: all ai x aj
                        for i in ai:
                            dx = _minimum_image(x[i] - x[aj], box)
                            r2 = np.einsum('ij,ij->i', dx, dx)
                            m = r2 < rlist2
                            for j in aj[m]:
                                lo, hi = (i, j) if i < j else (j, i)
                                idx_i_list.append(lo)
                                idx_j_list.append(hi)

    if len(idx_i_list) == 0:
        return (np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32))

    idx_i = np.asarray(idx_i_list, dtype=np.int64)
    idx_j = np.asarray(idx_j_list, dtype=np.int64)

    # Remove exclusions
    if exclusions:
        keep = np.ones(idx_i.size, dtype=bool)
        for k in range(idx_i.size):
            if (int(idx_i[k]), int(idx_j[k])) in exclusions:
                keep[k] = False
        idx_i = idx_i[keep]
        idx_j = idx_j[keep]

    return idx_i.astype(np.int32), idx_j.astype(np.int32)


def _brute_pairs(x, box, rlist2, exclusions):
    """O(N²) reference pair build for small systems / fallback."""
    nat = x.shape[0]
    idx_i_list = []
    idx_j_list = []
    for i in range(nat):
        js = np.arange(i + 1, nat)
        if js.size == 0:
            continue
        dx = _minimum_image(x[i] - x[js], box)
        r2 = np.einsum('ij,ij->i', dx, dx)
        m = r2 < rlist2
        for j in js[m]:
            if exclusions and (i, int(j)) in exclusions:
                continue
            idx_i_list.append(i)
            idx_j_list.append(int(j))
    return (np.asarray(idx_i_list, dtype=np.int32),
            np.asarray(idx_j_list, dtype=np.int32))


def exclusions_from_reference(listex: np.ndarray, listmex: np.ndarray) -> set:
    """Convert reference listex/listmex arrays to a set of (i<j) tuples.

    Reference stores, for each atom i, listex[0:listmex[i], i] = excluded j's.
    We canonicalize to (min, max) tuples for symmetric lookup.
    """
    excl = set()
    nat = listmex.shape[0]
    for i in range(nat):
        for k in range(int(listmex[i])):
            j = int(listex[k, i])
            lo, hi = (i, j) if i < j else (j, i)
            if lo != hi:
                excl.add((lo, hi))
    return excl



def build_neighbor_pairs_padded(x, box, rcut, skin, exclusions, capacity):
    """Fixed-size neighbor pairs for recompile-free MD.

    Returns idx_i, idx_j of length EXACTLY `capacity`, padded with self-pairs
    (0,0) that the pair kernel masks out (valid = i != j). Because the shape is
    constant across rebuilds, the jit'd force fn compiles ONCE instead of every
    rebuild — the single biggest M2->M4 speedup.

    Raises if real pair count exceeds capacity (caller should size capacity at
    ~1.2x the expected max, e.g. from the first build).
    """
    ii, jj = build_neighbor_pairs(x, box, rcut, skin=skin, exclusions=exclusions)
    n = ii.size
    if n > capacity:
        raise ValueError(f"neighbor pairs {n} exceed capacity {capacity}; "
                         f"raise capacity (e.g. int({n}*1.2)).")
    if n < capacity:
        pad = capacity - n
        ii = np.concatenate([ii, np.zeros(pad, dtype=np.int32)])
        jj = np.concatenate([jj, np.zeros(pad, dtype=np.int32)])
    return ii.astype(np.int32), jj.astype(np.int32), n
