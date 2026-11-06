"""
polarization_gpu.py
===================
GPU-accelerated polarization SCF iteration via CuPy.

This is the next-biggest hot spot after compute_recip_efield. Per perf_snapshot:
  scf_iteration's inner loop accounts for ~1000 ms / step (4 SCF iters)
  compute_pol_forces is ~500 ms / step (1 call)
  Combined: ~1500 ms / step out of 2075 ms total = 72% of step cost.

Strategy:
  1. Persistent GPU buffers for pair data (rebuilt only when neighbor list
     rebuilds, ~every 30+ steps).
  2. Pair list EXPANDED so each unordered pair (i,j) appears as both
     (i->j) and (j->i). This means each thread only writes to ONE atom's
     accumulator slot (its own iat), avoiding the atomicAdd cost from
     Newton's-third-law symmetric updates.
  3. Dipoles live on GPU across the entire SCF loop — no host-device
     copies between iterations.
  4. Convergence check happens on GPU via a max-reduction on |delta-mu|^2.

Usage:
    from polarization_gpu import GPUSCFSolver, gpu_available
    if gpu_available():
        solver = GPUSCFSolver(state)            # one-time setup
        solver.update_pair_data(state, ...)     # call after neighbor rebuild
        px, py, pz, niter = solver.run_scf(state, elfown, ...)
"""

import numpy as np

try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    cp = None
    _HAS_CUPY = False


def gpu_available():
    if not _HAS_CUPY:
        return False
    try:
        cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# Kernel 1: Compute T·μ contraction for ONE SCF iteration.
# Uses an expanded pair list (each pair appears twice, both orientations).
# One thread per atom; thread iat sums contributions from all its
# neighbors j into (txown, tyown, tzown).
# --------------------------------------------------------------------------
_TMU_KERNEL = r"""
extern "C" __global__
void tmu_iteration(
    int nat,
    const int* __restrict__ pair_start,    // (nat+1,) CSR-style start of each atom's neighbor list
    const int* __restrict__ pair_jat,      // (npairs,) j-atom for each entry
    const double* __restrict__ pair_rx,    // (npairs,) rij = rj - ri
    const double* __restrict__ pair_ry,
    const double* __restrict__ pair_rz,
    const double* __restrict__ pair_r,     // (npairs,) |rij|
    const double* __restrict__ pair_rinv,  // (npairs,) 1/|rij|
    const double* __restrict__ pair_c1r3,  // (npairs,) c1[itype,jtype] * r^3 (precomputed)
    const double* __restrict__ pair_expc1r3, // (npairs,) exp(-c1r3)
    const double* __restrict__ pair_tauval,  // (npairs,) tau-taper value
    double arf,
    const double* __restrict__ px,         // (nat,) current induced dipoles
    const double* __restrict__ py,
    const double* __restrict__ pz,
    double* __restrict__ txown,            // (nat,) output: T·μ accumulator
    double* __restrict__ tyown,
    double* __restrict__ tzown
) {
    int iat = blockIdx.x * blockDim.x + threadIdx.x;
    if (iat >= nat) return;

    double tx = 0.0, ty = 0.0, tz = 0.0;
    int p_start = pair_start[iat];
    int p_end   = pair_start[iat + 1];

    for (int p = p_start; p < p_end; p++) {
        int jat = pair_jat[p];
        double rxij = pair_rx[p];
        double ryij = pair_ry[p];
        double rzij = pair_rz[p];
        double r    = pair_r[p];
        double rinv = pair_rinv[p];
        double zinv = rinv * rinv;

        double c1r3      = pair_c1r3[p];
        double expc1r3v  = pair_expc1r3[p];
        double tauval    = pair_tauval[p];

        double lamda3 = 1.0 - expc1r3v;
        double lamda5 = 1.0 - expc1r3v * (1.0 + c1r3);

        double fact = rinv * zinv * tauval;

        double pjr = 3.0 * zinv * fact *
            (px[jat]*rxij + py[jat]*ryij + pz[jat]*rzij) * lamda5;

        double tfact = arf - fact * lamda3;

        // Contribution from jat dipole to iat (Newton's 3rd law: only this side
        // because the pair list is expanded -- the (j->i) entry handles the other half)
        tx += pjr * rxij + px[jat] * tfact;
        ty += pjr * ryij + py[jat] * tfact;
        tz += pjr * rzij + pz[jat] * tfact;
    }

    txown[iat] = tx;
    tyown[iat] = ty;
    tzown[iat] = tz;
}
"""


# --------------------------------------------------------------------------
# Kernel 2: Update dipoles + compute per-atom |delta-mu|^2 for convergence.
# --------------------------------------------------------------------------
_UPDATE_KERNEL = r"""
extern "C" __global__
void update_dipoles(
    int nat,
    const int* __restrict__ atomtype,
    const double* __restrict__ pol,
    int ncharge,
    const double* __restrict__ elfown_x,
    const double* __restrict__ elfown_y,
    const double* __restrict__ elfown_z,
    const double* __restrict__ txown,
    const double* __restrict__ tyown,
    const double* __restrict__ tzown,
    double arf,
    double* __restrict__ px,    // updated in place
    double* __restrict__ py,
    double* __restrict__ pz,
    double* __restrict__ delta2_per_atom  // (nat,) output: |dmu|^2 each atom
) {
    int iat = blockIdx.x * blockDim.x + threadIdx.x;
    if (iat >= nat) return;

    int it = atomtype[iat];
    double alpha_i = (it < ncharge) ? pol[it] : 0.0;

    double pxn = alpha_i * (elfown_x[iat] + txown[iat] + arf * px[iat]);
    double pyn = alpha_i * (elfown_y[iat] + tyown[iat] + arf * py[iat]);
    double pzn = alpha_i * (elfown_z[iat] + tzown[iat] + arf * pz[iat]);

    double dx = pxn - px[iat];
    double dy = pyn - py[iat];
    double dz = pzn - pz[iat];
    delta2_per_atom[iat] = dx*dx + dy*dy + dz*dz;

    px[iat] = pxn;
    py[iat] = pyn;
    pz[iat] = pzn;
}
"""


class GPUSCFSolver:
    """GPU-accelerated polarization SCF.

    Holds persistent GPU buffers for atom and pair data. Run pattern:
      * On neighbor-list rebuild: call update_pair_data() to refresh the
        expanded pair list and per-pair scalars.
      * Every step: call run_scf() with the current elfown to get
        converged dipoles back.

    All dipole arithmetic stays on GPU across the SCF loop; only the
    initial elfown upload and the final dipole download cross the bus.
    """

    def __init__(self, state):
        if not gpu_available():
            raise RuntimeError("CuPy/CUDA not available")

        self.nat = state.atoms.nat
        self.ncharge = int(state.ff.ncharge)
        self.atomtype_gpu = cp.asarray(state.atoms.atomtype, dtype=cp.int32)
        self.pol_gpu = cp.asarray(state.ff.pol, dtype=cp.float64)

        # Compile kernels once
        self._tmu = cp.RawKernel(_TMU_KERNEL, "tmu_iteration")
        self._update = cp.RawKernel(_UPDATE_KERNEL, "update_dipoles")

        # Per-atom buffers
        nat = self.nat
        self.px_gpu = cp.zeros(nat, dtype=cp.float64)
        self.py_gpu = cp.zeros(nat, dtype=cp.float64)
        self.pz_gpu = cp.zeros(nat, dtype=cp.float64)
        self.txown_gpu = cp.zeros(nat, dtype=cp.float64)
        self.tyown_gpu = cp.zeros(nat, dtype=cp.float64)
        self.tzown_gpu = cp.zeros(nat, dtype=cp.float64)
        self.delta2_gpu = cp.zeros(nat, dtype=cp.float64)
        self.elfown_gpu = cp.zeros((3, nat), dtype=cp.float64)

        self._npairs_alloc = 0
        self._pair_start_gpu = None

    def update_pair_data(self, listpol, listmpol, rx, ry, rz, rr, rrinv,
                         expc1r3, atomtype, c1, ncharge,
                         tau_func_values):
        """Refresh GPU pair data after a neighbor-list rebuild.

        We EXPAND the pair list: each unordered pair (i,j) becomes two
        directed entries (i->j) and (j->i). This makes the GPU kernel
        fully race-free at the cost of 2x memory for pair scalars.

        Parameters
        ----------
        listpol : (max_neigh, nat) int32 — j-atoms for each i, as in CPU
        listmpol : (nat,) int32 — number of neighbors per atom
        rx, ry, rz : (max_neigh, nat) — pair displacement (rj - ri)
        rr, rrinv  : (max_neigh, nat) — |r| and 1/|r|
        expc1r3 : (max_neigh, nat) — pre-computed exp(-c1*r^3)
        atomtype : (nat,) int — for c1[i,j] lookup
        c1 : (ncharge, ncharge) — Thole damping matrix
        tau_func_values : (max_neigh, nat) — pre-computed tau values
        """
        nat = self.nat

        # Build CSR pair list with both directions
        # Pass 1: count pairs per atom (each pair contributes 1 to both i and j)
        listm = np.asarray(listmpol)
        cnt = listm.copy()  # (i,j) contributes 1 to i
        # Now also count (j,i) contributions: for each pair (k, iat) with j-atom = jat,
        # increment cnt[jat]
        for iat in range(nat):
            for k in range(listm[iat]):
                jat = listpol[k, iat]
                cnt[jat] += 1

        pair_start = np.zeros(nat + 1, dtype=np.int32)
        np.cumsum(cnt, out=pair_start[1:])
        npairs_total = int(pair_start[-1])

        # Allocate GPU pair buffers (lazily resize)
        if npairs_total > self._npairs_alloc:
            self._npairs_alloc = int(npairs_total * 1.2 + 1)  # 20% headroom
            self.pair_jat_gpu = cp.empty(self._npairs_alloc, dtype=cp.int32)
            self.pair_rx_gpu  = cp.empty(self._npairs_alloc, dtype=cp.float64)
            self.pair_ry_gpu  = cp.empty(self._npairs_alloc, dtype=cp.float64)
            self.pair_rz_gpu  = cp.empty(self._npairs_alloc, dtype=cp.float64)
            self.pair_r_gpu   = cp.empty(self._npairs_alloc, dtype=cp.float64)
            self.pair_rinv_gpu= cp.empty(self._npairs_alloc, dtype=cp.float64)
            self.pair_c1r3_gpu= cp.empty(self._npairs_alloc, dtype=cp.float64)
            self.pair_expc1r3_gpu = cp.empty(self._npairs_alloc, dtype=cp.float64)
            self.pair_tauval_gpu  = cp.empty(self._npairs_alloc, dtype=cp.float64)

        # Pass 2: fill flat arrays
        # For atom iat, slot order: own neighbors (i->j) first, then (j->i) entries
        # we collect inverse mappings as we go.
        pair_jat_h    = np.empty(npairs_total, dtype=np.int32)
        pair_rx_h     = np.empty(npairs_total, dtype=np.float64)
        pair_ry_h     = np.empty(npairs_total, dtype=np.float64)
        pair_rz_h     = np.empty(npairs_total, dtype=np.float64)
        pair_r_h      = np.empty(npairs_total, dtype=np.float64)
        pair_rinv_h   = np.empty(npairs_total, dtype=np.float64)
        pair_c1r3_h   = np.empty(npairs_total, dtype=np.float64)
        pair_expc1r3_h= np.empty(npairs_total, dtype=np.float64)
        pair_tauval_h = np.empty(npairs_total, dtype=np.float64)

        write_pos = pair_start.copy()  # next free slot for each iat

        for iat in range(nat):
            it = atomtype[iat]
            for k in range(listm[iat]):
                jat = int(listpol[k, iat])
                jt = atomtype[jat]
                rxij = float(rx[k, iat])
                ryij = float(ry[k, iat])
                rzij = float(rz[k, iat])
                r    = float(rr[k, iat])
                rinv = float(rrinv[k, iat])
                ec1  = float(expc1r3[k, iat])
                tau  = float(tau_func_values[k, iat])
                # c1r3 (matching CPU branch)
                if it < ncharge and jt < ncharge:
                    c1r3 = c1[it, jt] * (r**3)
                else:
                    c1r3 = 1.0e6 * (r**3)

                # i -> j direction (sees +rij, dipole at jat)
                p_i = write_pos[iat]
                pair_jat_h[p_i]     = jat
                pair_rx_h[p_i]      = rxij
                pair_ry_h[p_i]      = ryij
                pair_rz_h[p_i]      = rzij
                pair_r_h[p_i]       = r
                pair_rinv_h[p_i]    = rinv
                pair_c1r3_h[p_i]    = c1r3
                pair_expc1r3_h[p_i] = ec1
                pair_tauval_h[p_i]  = tau
                write_pos[iat] += 1

                # j -> i direction (sees -rij, dipole at iat)
                p_j = write_pos[jat]
                pair_jat_h[p_j]     = iat
                pair_rx_h[p_j]      = -rxij
                pair_ry_h[p_j]      = -ryij
                pair_rz_h[p_j]      = -rzij
                pair_r_h[p_j]       = r
                pair_rinv_h[p_j]    = rinv
                pair_c1r3_h[p_j]    = c1r3
                pair_expc1r3_h[p_j] = ec1
                pair_tauval_h[p_j]  = tau
                write_pos[jat] += 1

        # Upload
        self._pair_start_gpu = cp.asarray(pair_start)
        self.pair_jat_gpu[:npairs_total]     = cp.asarray(pair_jat_h)
        self.pair_rx_gpu[:npairs_total]      = cp.asarray(pair_rx_h)
        self.pair_ry_gpu[:npairs_total]      = cp.asarray(pair_ry_h)
        self.pair_rz_gpu[:npairs_total]      = cp.asarray(pair_rz_h)
        self.pair_r_gpu[:npairs_total]       = cp.asarray(pair_r_h)
        self.pair_rinv_gpu[:npairs_total]    = cp.asarray(pair_rinv_h)
        self.pair_c1r3_gpu[:npairs_total]    = cp.asarray(pair_c1r3_h)
        self.pair_expc1r3_gpu[:npairs_total] = cp.asarray(pair_expc1r3_h)
        self.pair_tauval_gpu[:npairs_total]  = cp.asarray(pair_tauval_h)

        self._npairs_active = npairs_total

    def run_scf(self, elfown, atomtype, pol, ncharge, arf, tolpol,
                init_px, init_py, init_pz, max_iter=20):
        """Run the full SCF loop on GPU.

        Parameters
        ----------
        elfown : (3, nat) — host or GPU
        init_px, init_py, init_pz : (nat,) initial dipole guess
        Returns: (px, py, pz, niter) all on host (numpy)
        """
        nat = self.nat

        # Upload elfown and initial dipoles
        if isinstance(elfown, cp.ndarray):
            self.elfown_gpu[:] = elfown
        else:
            self.elfown_gpu[:] = cp.asarray(elfown)
        self.px_gpu[:] = cp.asarray(init_px)
        self.py_gpu[:] = cp.asarray(init_py)
        self.pz_gpu[:] = cp.asarray(init_pz)

        threads = 256
        blocks = (nat + threads - 1) // threads

        elf_x = self.elfown_gpu[0]
        elf_y = self.elfown_gpu[1]
        elf_z = self.elfown_gpu[2]

        niter = 0
        for it in range(1, max_iter + 1):
            niter = it

            # 1) T·μ contraction
            self._tmu(
                (blocks,), (threads,),
                (np.int32(nat),
                 self._pair_start_gpu, self.pair_jat_gpu,
                 self.pair_rx_gpu, self.pair_ry_gpu, self.pair_rz_gpu,
                 self.pair_r_gpu, self.pair_rinv_gpu,
                 self.pair_c1r3_gpu, self.pair_expc1r3_gpu, self.pair_tauval_gpu,
                 cp.float64(arf),
                 self.px_gpu, self.py_gpu, self.pz_gpu,
                 self.txown_gpu, self.tyown_gpu, self.tzown_gpu)
            )

            # 2) Update dipoles + compute |delta|^2
            self._update(
                (blocks,), (threads,),
                (np.int32(nat),
                 self.atomtype_gpu, self.pol_gpu, np.int32(ncharge),
                 elf_x, elf_y, elf_z,
                 self.txown_gpu, self.tyown_gpu, self.tzown_gpu,
                 cp.float64(arf),
                 self.px_gpu, self.py_gpu, self.pz_gpu,
                 self.delta2_gpu)
            )

            # 3) Convergence check (max |delta|^2 over atoms)
            max_d2 = float(cp.max(self.delta2_gpu))
            if max_d2 <= tolpol:
                break

        # Download dipoles
        px = cp.asnumpy(self.px_gpu)
        py = cp.asnumpy(self.py_gpu)
        pz = cp.asnumpy(self.pz_gpu)
        return px, py, pz, niter


def install_gpu_scf(state):
    """Install the GPU SCF solver onto state.gpu_scf if a GPU is available."""
    if not gpu_available():
        state.gpu_scf = None
        return False
    try:
        state.gpu_scf = GPUSCFSolver(state)
        dev_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
        print(f"  [GPU] Polarization SCF solver on {dev_name} ready")
        return True
    except Exception as e:
        print(f"  [GPU] SCF init failed: {e}; using CPU.")
        state.gpu_scf = None
        return False
