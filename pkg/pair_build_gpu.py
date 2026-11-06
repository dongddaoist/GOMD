"""
pair_build_gpu.py
=================
STAGE 1 of the GPU polarization port.

Replaces compute_charge_efield_and_pairs (the 416 ms/step CPU bottleneck) with
a CUDA kernel. Output is numerically identical to the CPU version (within
fp64 noise from atomicAdd reordering on elfown).

Strategy:
  - One CUDA thread per atom iat.
  - Thread iat scans its neighbors j from list_neighbors[k, iat] (k=0..listm[iat]).
  - For each pair, computes geometry + Thole damping + Ewald real-space
    field contribution, writes pair scalars to its own slot, atomicAdds
    field contributions to both elfown[:, iat] and elfown[:, jat].

The tau_func, c1r3, etc. are NOT precomputed here — kept identical to the CPU
function so we can directly compare outputs. (Optimizations stack in later
stages.)

Numerical equivalence:
  - All math is fp64 except table lookups (which use fp64 tables anyway).
  - The only source of difference vs CPU is atomicAdd ordering, which can
    introduce ~1e-12 relative drift. PE should match CPU to ~0.0001 kcal/mol.

Usage:
    from pair_build_gpu import GPUPairBuilder, gpu_available
    if gpu_available():
        builder = GPUPairBuilder(state)         # one-time setup
        builder.refresh_static(state)           # call after neighbor rebuild
        elfown, listpol, listmpol, rx, ry, rz, rr, rrinv, expc1r3 = (
            builder.compute(state)
        )
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


# =============================================================================
# CUDA kernel: pair-build + charge E-field
# =============================================================================
# Notes:
#  - One thread per atom iat.
#  - Each thread does (listm[iat]) iterations over neighbors j.
#  - The pair-scalar arrays (listpol, rx, ..., expc1r3) are written by ONE
#    thread (the iat thread, slot k) - no atomics needed.
#  - elfown is written from BOTH the i-side and the j-side (Newton's 3rd law),
#    so multiple threads can target the same elfown[:, iat] - need atomicAdd.
#  - We use the same tabulated erfc/exp(-x²) approach as the CPU version
#    for numerical equivalence. Tables are passed as kernel arguments.
# =============================================================================

_PAIR_BUILD_KERNEL = r"""
extern "C" __global__
void pair_build_and_field(
    int nat,
    int ncharge,
    int max_neigh,
    double alpha,
    double twoalphapi,
    double rcut2,
    // SCF-precompute params (Stage 2): tau-tapering function
    double rtaper,           // tapering start radius
    double ro,               // outer cutoff
    double tauden,           // tapering denominator
    // Per-atom input
    const double* __restrict__ x,            // (3, nat) row-major: x[k*nat+iat]
    const int*    __restrict__ atomtype,     // (nat,)
    const int*    __restrict__ list_neighbors, // (maxneigh, nat) col-major: [k*nat+iat]
    const int*    __restrict__ listm,        // (nat,) neighbor count per atom
    const double* __restrict__ q,            // (ncharge,)
    const double* __restrict__ c1_flat,      // (ncharge*ncharge,) c1[i,j] = c1_flat[i*ncharge+j]
    const double* __restrict__ box,          // (3,)
    const double* __restrict__ el,           // (3,) box*0.5
    // Tables for Ewald real-space
    const double* __restrict__ erfc_tab,
    const double* __restrict__ exp_negsq_tab,
    // Output (Stage 1 + Stage 2 precompute)
    double* __restrict__ elfown,             // (3, nat) row-major
    int*    __restrict__ listpol,            // (max_neigh, nat) col-major
    int*    __restrict__ listmpol,           // (nat,)
    double* __restrict__ rx_arr,             // (max_neigh, nat) col-major
    double* __restrict__ ry_arr,
    double* __restrict__ rz_arr,
    double* __restrict__ rr_arr,
    double* __restrict__ rrinv_arr,
    double* __restrict__ expc1r3_arr,
    // Stage 2 precompute outputs (iteration-invariant SCF scalars)
    double* __restrict__ c1r3_arr,           // c1[i,j] * r^3
    double* __restrict__ lamda3_arr,         // 1 - exp(-c1*r^3)
    double* __restrict__ lamda5_arr,         // 1 - exp(-c1*r^3) * (1 + c1*r^3)
    double* __restrict__ fact_arr            // rinv*zinv*tauval
) {
    int iat = blockIdx.x * blockDim.x + threadIdx.x;
    if (iat >= nat) return;

    int itype = atomtype[iat];
    double xi0 = x[0*nat + iat];
    double xi1 = x[1*nat + iat];
    double xi2 = x[2*nat + iat];
    double qi = (itype < ncharge) ? q[itype] : 0.0;

    // PBC half-box lengths
    double el0 = el[0], el1 = el[1], el2 = el[2];
    double bx = box[0], by = box[1], bz = box[2];

    int n_total = listm[iat];
    int nneigh = 0;

    // Local accumulators for elfown[:, iat] - one thread updates these,
    // so we accumulate locally and write once at end (faster than atomicAdd
    // for our own atom). The j-side writes still need atomicAdd.
    double e0_iat = 0.0, e1_iat = 0.0, e2_iat = 0.0;

    for (int k = 0; k < n_total; k++) {
        int jat = list_neighbors[k * nat + iat];
        int jtype = atomtype[jat];

        double xij0 = x[0*nat + jat] - xi0;
        double xij1 = x[1*nat + jat] - xi1;
        double xij2 = x[2*nat + jat] - xi2;

        // Minimum image (matches CPU: if abs(d) > el, d -= sign(d) * box)
        if (xij0 >  el0) xij0 -= bx;
        if (xij0 < -el0) xij0 += bx;
        if (xij1 >  el1) xij1 -= by;
        if (xij1 < -el1) xij1 += by;
        if (xij2 >  el2) xij2 -= bz;
        if (xij2 < -el2) xij2 += bz;

        double z = xij0*xij0 + xij1*xij1 + xij2*xij2;
        if (z > rcut2) continue;
        if (z < 1.0e-10) continue;

        double r = sqrt(z);
        double rinv = 1.0 / r;
        double zinv = rinv * rinv;

        // Store pair data in slot nneigh (only this thread writes, no atomic)
        if (nneigh < max_neigh) {
            listpol[nneigh * nat + iat]   = jat;
            rx_arr[nneigh * nat + iat]    = xij0;
            ry_arr[nneigh * nat + iat]    = xij1;
            rz_arr[nneigh * nat + iat]    = xij2;
            rr_arr[nneigh * nat + iat]    = r;
            rrinv_arr[nneigh * nat + iat] = rinv;
            // Thole damping exp(-c1[i,j] * r^3)
            double c1ij;
            if (itype < ncharge && jtype < ncharge) {
                c1ij = c1_flat[itype * ncharge + jtype];
            } else {
                c1ij = 1.0e6;  // matches CPU's "no damping" branch
            }
            double r3 = r * r * r;
            double c1r3_v = c1ij * r3;
            double ec1 = exp(-c1r3_v);
            expc1r3_arr[nneigh * nat + iat] = ec1;

            // Stage 2 precompute: iteration-invariant SCF scalars
            // Matches CPU formulas:
            //   c1r3   = c1[i,j] * r^3
            //   lamda3 = 1 - exp(-c1r3)
            //   lamda5 = 1 - exp(-c1r3) * (1 + c1r3)
            //   tauval = tau_func(r, rtaper, ro, tauden)  [piecewise polynomial]
            //   fact   = rinv * zinv * tauval = tauval / r^3
            c1r3_arr[nneigh * nat + iat]   = c1r3_v;
            lamda3_arr[nneigh * nat + iat] = 1.0 - ec1;
            lamda5_arr[nneigh * nat + iat] = 1.0 - ec1 * (1.0 + c1r3_v);

            // tau_func inlined (matches polarization.tau_func exactly):
            double tau_v;
            if (r <= rtaper) {
                tau_v = 1.0;
            } else if (r >= ro) {
                tau_v = 0.0;
            } else {
                double w = r - rtaper;
                tau_v = 1.0 + w * w * (w / 3.0 + 0.5 * (rtaper - ro)) / tauden;
            }
            fact_arr[nneigh * nat + iat] = rinv * zinv * tau_v;

            nneigh++;
        }

        // Ewald real-space field via tabulated erfc and exp(-(alphar)^2)
        double alphar = alpha * r;
        double erfc_val = 0.0;
        double exp_val = 0.0;
        if (alphar < 6.0) {
            double fpos = alphar * 2730.6666666666665;
            int idx = (int)fpos;
            double frac = fpos - (double)idx;
            erfc_val = erfc_tab[idx] + frac * (erfc_tab[idx+1] - erfc_tab[idx]);
            exp_val  = exp_negsq_tab[idx] + frac * (exp_negsq_tab[idx+1] - exp_negsq_tab[idx]);
        }
        double fact = (erfc_val * rinv + twoalphapi * exp_val) * zinv;

        double qj = (jtype < ncharge) ? q[jtype] : 0.0;
        double facti = fact * qi;
        double factj = fact * qj;

        // E-field at iat from charge qj (this thread's own atom — local accum)
        e0_iat -= factj * xij0;
        e1_iat -= factj * xij1;
        e2_iat -= factj * xij2;

        // E-field at jat from charge qi (Newton's 3rd law — atomic add)
        atomicAdd(&elfown[0*nat + jat], facti * xij0);
        atomicAdd(&elfown[1*nat + jat], facti * xij1);
        atomicAdd(&elfown[2*nat + jat], facti * xij2);
    }

    listmpol[iat] = nneigh;

    // Flush local accumulators for elfown[:, iat] using atomicAdd
    // (other threads may have written to my elfown via the j-side path).
    atomicAdd(&elfown[0*nat + iat], e0_iat);
    atomicAdd(&elfown[1*nat + iat], e1_iat);
    atomicAdd(&elfown[2*nat + iat], e2_iat);
}
"""


# =============================================================================
# 1-4 correction kernel (small, runs once per step)
# =============================================================================
_PAIR_BUILD_14_KERNEL = r"""
extern "C" __global__
void pair_build_14_correction(
    int nat,
    int ncharge,
    double alpha,
    double twoalphapi,
    double redQmufactor,
    const double* __restrict__ x,
    const int*    __restrict__ atomtype,
    const int*    __restrict__ list14,
    const int*    __restrict__ listm14,
    const double* __restrict__ q,
    const double* __restrict__ box,
    const double* __restrict__ el,
    const double* __restrict__ erfc_tab,
    const double* __restrict__ exp_negsq_tab,
    double* __restrict__ elfown
) {
    int iat = blockIdx.x * blockDim.x + threadIdx.x;
    if (iat >= nat) return;

    int itype = atomtype[iat];
    double xi0 = x[0*nat + iat];
    double xi1 = x[1*nat + iat];
    double xi2 = x[2*nat + iat];
    double qi = (itype < ncharge) ? q[itype] : 0.0;

    double el0 = el[0], el1 = el[1], el2 = el[2];
    double bx = box[0], by = box[1], bz = box[2];

    int n14 = listm14[iat];
    double e0_iat = 0.0, e1_iat = 0.0, e2_iat = 0.0;

    for (int k = 0; k < n14; k++) {
        int jat = list14[k * nat + iat];
        int jtype = atomtype[jat];

        double xij0 = x[0*nat + jat] - xi0;
        double xij1 = x[1*nat + jat] - xi1;
        double xij2 = x[2*nat + jat] - xi2;

        if (xij0 >  el0) xij0 -= bx;
        if (xij0 < -el0) xij0 += bx;
        if (xij1 >  el1) xij1 -= by;
        if (xij1 < -el1) xij1 += by;
        if (xij2 >  el2) xij2 -= bz;
        if (xij2 < -el2) xij2 += bz;

        double z = xij0*xij0 + xij1*xij1 + xij2*xij2;
        if (z < 1.0e-10) continue;
        double r = sqrt(z);
        double rinv = 1.0 / r;
        double zinv = rinv * rinv;

        double alphar = alpha * r;
        double erfc_val = 0.0, exp_val = 0.0;
        if (alphar < 6.0) {
            double fpos = alphar * 2730.6666666666665;
            int idx = (int)fpos;
            double frac = fpos - (double)idx;
            erfc_val = erfc_tab[idx] + frac * (erfc_tab[idx+1] - erfc_tab[idx]);
            exp_val  = exp_negsq_tab[idx] + frac * (exp_negsq_tab[idx+1] - exp_negsq_tab[idx]);
        }
        double fact = (erfc_val * rinv + twoalphapi * exp_val) * zinv;
        fact *= (-redQmufactor);

        double qj = (jtype < ncharge) ? q[jtype] : 0.0;
        double facti = fact * qi;
        double factj = fact * qj;

        e0_iat -= factj * xij0;
        e1_iat -= factj * xij1;
        e2_iat -= factj * xij2;

        atomicAdd(&elfown[0*nat + jat], facti * xij0);
        atomicAdd(&elfown[1*nat + jat], facti * xij1);
        atomicAdd(&elfown[2*nat + jat], facti * xij2);
    }

    atomicAdd(&elfown[0*nat + iat], e0_iat);
    atomicAdd(&elfown[1*nat + iat], e1_iat);
    atomicAdd(&elfown[2*nat + iat], e2_iat);
}
"""


class GPUPairBuilder:
    """Builds polarization pair scalars + Ewald real-space E-field on GPU.

    Persistent buffers stay on GPU across steps. Position upload + result
    download is the only host-device traffic per step.
    """

    def __init__(self, state):
        if not gpu_available():
            raise RuntimeError("CuPy/CUDA not available")

        self.nat = state.atoms.nat
        self.ncharge = int(state.ff.ncharge)

        # Static-ish data (refreshed if neighbor list changes)
        self.atomtype_gpu = cp.asarray(state.atoms.atomtype, dtype=cp.int32)
        self.q_gpu = cp.asarray(state.ff.q, dtype=cp.float64)

        # c1 (Thole damping matrix, ncharge x ncharge) is built by
        # init_polarization and stored at state.pol_state.c1. We defer
        # uploading it to refresh_static, which is guaranteed to run after
        # polarization init.
        self.c1_gpu = None

        # Tabulated Ewald functions
        from ewald_tables import ERFC_TABLE, EXP_NEGSQ_TABLE
        self.erfc_tab_gpu = cp.asarray(ERFC_TABLE, dtype=cp.float64)
        self.exp_tab_gpu = cp.asarray(EXP_NEGSQ_TABLE, dtype=cp.float64)

        # Compile kernels
        self._kernel = cp.RawKernel(_PAIR_BUILD_KERNEL, "pair_build_and_field")
        self._kernel_14 = cp.RawKernel(_PAIR_BUILD_14_KERNEL,
                                        "pair_build_14_correction")

        # Output buffers (allocated lazily based on max_neigh seen)
        self._max_neigh_alloc = 0
        self.elfown_gpu = cp.empty((3, self.nat), dtype=cp.float64)
        self.listmpol_gpu = cp.empty(self.nat, dtype=cp.int32)
        self._listpol_gpu = None  # allocated in refresh_static
        self._rx_gpu = None
        self._ry_gpu = None
        self._rz_gpu = None
        self._rr_gpu = None
        self._rrinv_gpu = None
        self._expc1r3_gpu = None
        # Stage 2 precompute outputs
        self._c1r3_gpu = None
        self._lamda3_gpu = None
        self._lamda5_gpu = None
        self._fact_gpu = None

        # Neighbor list views (refreshed on rebuild)
        self._list_neighbors_gpu = None
        self._listm_gpu = None
        self._list14_gpu = None
        self._listm14_gpu = None
        self._has_neighbors = False

    def refresh_static(self, state):
        """Call after init_polarization (so state.ps exists with c1) AND after
        any neighbor-list rebuild (so list_neighbors is current).

        Cheap to call every step — the upload sizes are O(nat * max_neigh).
        """
        # Upload c1 (Thole damping matrix) — small, ~1 KB; safe to refresh
        ps = state.pol_state
        if ps is None or not hasattr(ps, 'c1') or ps.c1 is None:
            raise RuntimeError(
                "GPUPairBuilder.refresh_static called before init_polarization. "
                "state.pol_state.c1 must exist."
            )
        c1_flat = np.ascontiguousarray(ps.c1, dtype=np.float64).flatten()
        # Sanity check — bug-trap for the NaN issue we just hit
        if not np.all(np.isfinite(c1_flat)):
            raise RuntimeError(
                f"state.pol_state.c1 contains non-finite values "
                f"(min={c1_flat.min()}, max={c1_flat.max()}). "
                f"Cannot upload to GPU."
            )
        if self.c1_gpu is None or self.c1_gpu.size != c1_flat.size:
            self.c1_gpu = cp.asarray(c1_flat)
        else:
            self.c1_gpu[:] = cp.asarray(c1_flat)

        # Refresh neighbor lists (these change on rebuild)
        listm = state.neighbors.listm
        list_neighbors = state.neighbors.list_neighbors
        max_neigh = int(listm.max()) + 10  # match CPU's safety margin

        # Resize output buffers if needed
        if max_neigh > self._max_neigh_alloc:
            self._max_neigh_alloc = max_neigh
            self._listpol_gpu = cp.empty((max_neigh, self.nat), dtype=cp.int32)
            self._rx_gpu = cp.empty((max_neigh, self.nat), dtype=cp.float64)
            self._ry_gpu = cp.empty((max_neigh, self.nat), dtype=cp.float64)
            self._rz_gpu = cp.empty((max_neigh, self.nat), dtype=cp.float64)
            self._rr_gpu = cp.empty((max_neigh, self.nat), dtype=cp.float64)
            self._rrinv_gpu = cp.empty((max_neigh, self.nat), dtype=cp.float64)
            self._expc1r3_gpu = cp.empty((max_neigh, self.nat), dtype=cp.float64)
            # Stage 2 precompute buffers
            self._c1r3_gpu = cp.empty((max_neigh, self.nat), dtype=cp.float64)
            self._lamda3_gpu = cp.empty((max_neigh, self.nat), dtype=cp.float64)
            self._lamda5_gpu = cp.empty((max_neigh, self.nat), dtype=cp.float64)
            self._fact_gpu = cp.empty((max_neigh, self.nat), dtype=cp.float64)

        # Upload neighbor list (transposed to col-major form for kernel access)
        # CPU stores list_neighbors[k, iat] in (max_neigh, nat) format.
        # On GPU we want [k*nat + iat] so a copy works directly when ascontiguous.
        ln = np.ascontiguousarray(list_neighbors[:max_neigh, :], dtype=np.int32)
        self._list_neighbors_gpu = cp.asarray(ln)
        self._listm_gpu = cp.asarray(np.asarray(listm, dtype=np.int32))

        # 1-4 list (uses correct field names: list14, listm14 - NOT listex14)
        # Bug fix v2026.05.05k: previously used non-existent listex14 field,
        # which silently disabled the 1-4 correction on GPU.
        has_l14 = (hasattr(state.neighbors, 'list14')
                   and state.neighbors.list14 is not None)
        if has_l14 and state.params.lredQ_mu14:
            list14_arr = np.ascontiguousarray(state.neighbors.list14, dtype=np.int32)
            listm14_arr = np.asarray(state.neighbors.listm14, dtype=np.int32)
            self._list14_gpu = cp.asarray(list14_arr)
            self._listm14_gpu = cp.asarray(listm14_arr)
            self._has_14 = True
            print(f"  [GPU] 1-4 correction enabled: max_l14 = {int(listm14_arr.max())}")
        else:
            self._has_14 = False
            if state.params.lredQ_mu14:
                print(f"  [GPU] WARNING: lredQ_mu14=True but list14 not found; "
                      f"GPU 1-4 correction DISABLED")

        self._has_neighbors = True

    def compute(self, state, keep_on_gpu=False):
        """Run the GPU kernel.

        Parameters
        ----------
        state : SimulationState
        keep_on_gpu : bool
            If True, return GPU (CuPy) arrays instead of downloading.
            Used by Stage 2 GPU SCF to avoid round-trip transfers.

        Returns
        -------
        Either:
            (elfown, listpol, listmpol, rx, ry, rz, rr, rrinv, expc1r3)
        as CPU numpy arrays (keep_on_gpu=False, default), OR

            dict {'elfown': cp_array, 'listpol': cp_array, ...,
                  'c1r3': cp_array, 'lamda3': cp_array, 'lamda5': cp_array,
                  'fact': cp_array}
        as CuPy arrays (keep_on_gpu=True). Note keep_on_gpu mode also includes
        the Stage 2 precomputed scalars.
        """
        if not self._has_neighbors:
            raise RuntimeError("Call refresh_static() first")

        nat = self.nat
        max_neigh = self._max_neigh_alloc

        # Get tau-tapering params from state (needed by the extended kernel).
        # In the CPU code, ro = state.params.rcut and rtaper = state.params.rtaper.
        # tauden lives on the polarization state object.
        ps = state.pol_state
        rtaper = float(state.params.rtaper)
        ro = float(state.params.rcut)
        tauden = float(ps.tauden)

        # Upload positions (small: 200KB)
        x_gpu = cp.asarray(state.atoms.x, dtype=cp.float64)
        box_gpu = cp.asarray(state.atoms.box, dtype=cp.float64)
        el_gpu = cp.asarray(state.atoms.el, dtype=cp.float64)

        # Zero output buffers (safety against downstream consumers reading
        # beyond listmpol[iat]).
        self.elfown_gpu.fill(0.0)
        self.listmpol_gpu.fill(0)
        self._listpol_gpu.fill(-1)
        self._rx_gpu.fill(0.0)
        self._ry_gpu.fill(0.0)
        self._rz_gpu.fill(0.0)
        self._rr_gpu.fill(0.0)
        self._rrinv_gpu.fill(0.0)
        self._expc1r3_gpu.fill(0.0)
        # Stage 2 precompute buffers (always populated by extended kernel)
        self._c1r3_gpu.fill(0.0)
        self._lamda3_gpu.fill(0.0)
        self._lamda5_gpu.fill(0.0)
        self._fact_gpu.fill(0.0)

        # Constants
        alpha = float(state.params.alphai)
        twoalphapi = 2.0 * alpha / np.sqrt(np.pi)
        rcut2 = float(state.params.rcut ** 2)

        # Launch main kernel — note 4 new args (rtaper, ro, tauden, +4 buffers)
        threads = 128
        blocks = (nat + threads - 1) // threads
        self._kernel(
            (blocks,), (threads,),
            (np.int32(nat), np.int32(self.ncharge), np.int32(max_neigh),
             cp.float64(alpha), cp.float64(twoalphapi), cp.float64(rcut2),
             cp.float64(rtaper), cp.float64(ro), cp.float64(tauden),
             x_gpu, self.atomtype_gpu, self._list_neighbors_gpu, self._listm_gpu,
             self.q_gpu, self.c1_gpu, box_gpu, el_gpu,
             self.erfc_tab_gpu, self.exp_tab_gpu,
             self.elfown_gpu, self._listpol_gpu, self.listmpol_gpu,
             self._rx_gpu, self._ry_gpu, self._rz_gpu,
             self._rr_gpu, self._rrinv_gpu, self._expc1r3_gpu,
             self._c1r3_gpu, self._lamda3_gpu, self._lamda5_gpu, self._fact_gpu)
        )

        # 1-4 correction kernel (if applicable)
        if self._has_14:
            self._kernel_14(
                (blocks,), (threads,),
                (np.int32(nat), np.int32(self.ncharge),
                 cp.float64(alpha), cp.float64(twoalphapi),
                 cp.float64(state.params.redQmufactor),
                 x_gpu, self.atomtype_gpu, self._list14_gpu, self._listm14_gpu,
                 self.q_gpu, box_gpu, el_gpu,
                 self.erfc_tab_gpu, self.exp_tab_gpu,
                 self.elfown_gpu)
            )

        cp.cuda.runtime.deviceSynchronize()

        if keep_on_gpu:
            # Stage 2 mode: hand back GPU arrays + precompute scalars.
            # Caller is responsible for not modifying these in place.
            return {
                'elfown':   self.elfown_gpu,
                'listpol':  self._listpol_gpu,
                'listmpol': self.listmpol_gpu,
                'rx':       self._rx_gpu,
                'ry':       self._ry_gpu,
                'rz':       self._rz_gpu,
                'rr':       self._rr_gpu,
                'rrinv':    self._rrinv_gpu,
                'expc1r3':  self._expc1r3_gpu,
                'c1r3':     self._c1r3_gpu,
                'lamda3':   self._lamda3_gpu,
                'lamda5':   self._lamda5_gpu,
                'fact':     self._fact_gpu,
            }

        # Stage 1 default mode: download to host (compatibility w/ CPU SCF)
        elfown   = cp.asnumpy(self.elfown_gpu)
        listpol  = cp.asnumpy(self._listpol_gpu)
        listmpol = cp.asnumpy(self.listmpol_gpu)
        rx_arr   = cp.asnumpy(self._rx_gpu)
        ry_arr   = cp.asnumpy(self._ry_gpu)
        rz_arr   = cp.asnumpy(self._rz_gpu)
        rr_arr   = cp.asnumpy(self._rr_gpu)
        rrinv_arr= cp.asnumpy(self._rrinv_gpu)
        expc1r3_arr = cp.asnumpy(self._expc1r3_gpu)

        return (elfown, listpol, listmpol,
                rx_arr, ry_arr, rz_arr,
                rr_arr, rrinv_arr, expc1r3_arr)


def install_gpu_pair_build(state):
    """Attach a GPUPairBuilder to state.gpu_pair_builder if a GPU is present."""
    if not gpu_available():
        state.gpu_pair_builder = None
        return False
    try:
        state.gpu_pair_builder = GPUPairBuilder(state)
        dev_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
        print(f"  [GPU] Pair builder on {dev_name}: ready (refresh_static needed)")
        return True
    except Exception as e:
        print(f"  [GPU] Pair builder init failed: {e}; using CPU.")
        state.gpu_pair_builder = None
        return False
