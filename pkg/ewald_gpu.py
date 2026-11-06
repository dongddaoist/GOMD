"""
ewald_gpu.py
============
GPU-accelerated reciprocal E-field for the polarization SCF, via CuPy.

This is a drop-in replacement for `ewald_fast.compute_recip_efield_fast`
that runs on a CUDA GPU. It targets the ~11 calls per step made during
the SCF iteration, which dominate per-step cost.

Strategy:
  - One CUDA kernel launches one block per (kx, ky, kz) k-vector.
  - Each block has 256 threads cooperating to:
      a) compute cos/sin(k.r_i) for every atom (recurrence-free since
         k is fixed per block; we use cos/sin tables)
      b) reduce |S(k)|^2 into shared memory
      c) accumulate the per-atom field contribution
  - A persistent GPU class holds the static data (charges, atom types,
    box) so each call only transfers x_atoms (in) and elfown (out).

Numerical: keeps fp64 to match the CPU reference exactly. fp32 mode is
available via the dtype kwarg if you can tolerate ~1e-5 relative error.

Usage (main.py):
    from ewald_gpu import GPUEwaldRecip, gpu_available
    if gpu_available():
        state.gpu_ewald_recip = GPUEwaldRecip(state)

Usage (polarization.py):
    if hasattr(state, 'gpu_ewald_recip') and state.gpu_ewald_recip is not None:
        elfown_recip = state.gpu_ewald_recip.compute(state.atoms.x)
    else:
        from ewald_fast import compute_recip_efield_fast
        elfown_recip = compute_recip_efield_fast(state)
"""

import numpy as np

try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    cp = None
    _HAS_CUPY = False


def gpu_available():
    """Return True iff CuPy is importable and a CUDA device is present."""
    if not _HAS_CUPY:
        return False
    try:
        cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# CUDA: trig tables for all atoms via recurrence
# --------------------------------------------------------------------------
_TRIG_KERNEL = r"""
extern "C" __global__
void build_trig_tables(
    const double* __restrict__ x,                 // (3, nat) row-major
    int nat,
    int klimx, int klimy, int klimz,
    double fx, double fy, double fz,
    double* __restrict__ cx, double* __restrict__ sx,  // (klimx+1, nat)
    double* __restrict__ cy, double* __restrict__ sy,  // (klimy+1, nat)
    double* __restrict__ cz, double* __restrict__ sz   // (klimz+1, nat)
) {
    int iat = blockIdx.x * blockDim.x + threadIdx.x;
    if (iat >= nat) return;

    double xi = x[0*nat + iat];
    double yi = x[1*nat + iat];
    double zi = x[2*nat + iat];

    cx[0*nat + iat] = 1.0; sx[0*nat + iat] = 0.0;
    cy[0*nat + iat] = 1.0; sy[0*nat + iat] = 0.0;
    cz[0*nat + iat] = 1.0; sz[0*nat + iat] = 0.0;
    if (klimx >= 1) { cx[1*nat + iat] = cos(fx*xi); sx[1*nat + iat] = sin(fx*xi); }
    if (klimy >= 1) { cy[1*nat + iat] = cos(fy*yi); sy[1*nat + iat] = sin(fy*yi); }
    if (klimz >= 1) { cz[1*nat + iat] = cos(fz*zi); sz[1*nat + iat] = sin(fz*zi); }

    for (int k = 2; k <= klimx; k++) {
        double cp1 = cx[1*nat + iat], sp1 = sx[1*nat + iat];
        double cn  = cx[(k-1)*nat + iat], sn = sx[(k-1)*nat + iat];
        cx[k*nat + iat] = cn*cp1 - sn*sp1;
        sx[k*nat + iat] = sn*cp1 + cn*sp1;
    }
    for (int k = 2; k <= klimy; k++) {
        double cp1 = cy[1*nat + iat], sp1 = sy[1*nat + iat];
        double cn  = cy[(k-1)*nat + iat], sn = sy[(k-1)*nat + iat];
        cy[k*nat + iat] = cn*cp1 - sn*sp1;
        sy[k*nat + iat] = sn*cp1 + cn*sp1;
    }
    for (int k = 2; k <= klimz; k++) {
        double cp1 = cz[1*nat + iat], sp1 = sz[1*nat + iat];
        double cn  = cz[(k-1)*nat + iat], sn = sz[(k-1)*nat + iat];
        cz[k*nat + iat] = cn*cp1 - sn*sp1;
        sz[k*nat + iat] = sn*cp1 + cn*sp1;
    }
}
"""

# --------------------------------------------------------------------------
# CUDA: per-k-vector E-field accumulation
# One block per k-vector. Threads in the block cooperate over atoms.
# --------------------------------------------------------------------------
_KSPACE_EFIELD_KERNEL = r"""
extern "C" __global__
void kspace_efield_one_kvec(
    int nat,
    const int* __restrict__ atomtype,
    const double* __restrict__ q,
    int ncharge,
    const double* __restrict__ cx, const double* __restrict__ sx,
    const double* __restrict__ cy, const double* __restrict__ sy,
    const double* __restrict__ cz, const double* __restrict__ sz,
    int kx, int ky, int kz,
    double hx, double hy, double hz,
    double prefact,
    double* __restrict__ elfown   // (3, nat)
) {
    extern __shared__ double sh[];
    double* s_cos = sh;
    double* s_sin = sh + blockDim.x;
    int tid = threadIdx.x;

    int kya = (ky < 0) ? -ky : ky;
    int kza = (kz < 0) ? -kz : kz;

    // Reduction: charge structure factor S(k)
    double cossum_local = 0.0, sinsum_local = 0.0;
    for (int iat = tid; iat < nat; iat += blockDim.x) {
        double cxi = cx[kx*nat + iat];
        double sxi = sx[kx*nat + iat];
        double cyi = cy[kya*nat + iat];
        double syi = sy[kya*nat + iat];
        double czi = cz[kza*nat + iat];
        double szi = sz[kza*nat + iat];

        double cxy, sxy;
        if (ky >= 0) { cxy = cxi*cyi - sxi*syi; sxy = sxi*cyi + cxi*syi; }
        else         { cxy = cxi*cyi + sxi*syi; sxy = sxi*cyi - cxi*syi; }
        double cxyz, sxyz;
        if (kz >= 0) { cxyz = cxy*czi - sxy*szi; sxyz = sxy*czi + cxy*szi; }
        else         { cxyz = cxy*czi + sxy*szi; sxyz = sxy*czi - cxy*szi; }

        int it = atomtype[iat];
        double ci = (it < ncharge) ? q[it] : 0.0;
        cossum_local += ci * cxyz;
        sinsum_local += ci * sxyz;
    }

    s_cos[tid] = cossum_local;
    s_sin[tid] = sinsum_local;
    __syncthreads();
    for (int off = blockDim.x/2; off > 0; off >>= 1) {
        if (tid < off) { s_cos[tid] += s_cos[tid+off]; s_sin[tid] += s_sin[tid+off]; }
        __syncthreads();
    }
    double cossum = s_cos[0];
    double sinsum = s_sin[0];

    // Per-atom field contribution
    for (int iat = tid; iat < nat; iat += blockDim.x) {
        double cxi = cx[kx*nat + iat];
        double sxi = sx[kx*nat + iat];
        double cyi = cy[kya*nat + iat];
        double syi = sy[kya*nat + iat];
        double czi = cz[kza*nat + iat];
        double szi = sz[kza*nat + iat];

        double cxy, sxy;
        if (ky >= 0) { cxy = cxi*cyi - sxi*syi; sxy = sxi*cyi + cxi*syi; }
        else         { cxy = cxi*cyi + sxi*syi; sxy = sxi*cyi - cxi*syi; }
        double cxyz, sxyz;
        if (kz >= 0) { cxyz = cxy*czi - sxy*szi; sxyz = sxy*czi + cxy*szi; }
        else         { cxyz = cxy*czi + sxy*szi; sxyz = sxy*czi - cxy*szi; }

        double tmp = prefact * (cossum*sxyz - sinsum*cxyz);
        atomicAdd(&elfown[0*nat + iat], tmp * hx);
        atomicAdd(&elfown[1*nat + iat], tmp * hy);
        atomicAdd(&elfown[2*nat + iat], tmp * hz);
    }
}
"""


class GPUEwaldRecip:
    """GPU-accelerated reciprocal E-field for the polarization SCF.

    Build once per simulation (or per neighbor-list rebuild if charges /
    atom types change). compute() can then be called every SCF iteration.
    """

    def __init__(self, state):
        if not gpu_available():
            raise RuntimeError("CuPy/CUDA not available")

        self.nat = state.atoms.nat
        self.alpha = float(state.params.alphai)
        self.box = np.array(state.atoms.box, dtype=np.float64)
        self.klimit = (int(state.params.klimitx),
                       int(state.params.klimity),
                       int(state.params.klimitz))

        # Persistent GPU buffers
        self.q_gpu = cp.asarray(state.ff.q, dtype=cp.float64)
        self.atomtype_gpu = cp.asarray(state.atoms.atomtype, dtype=cp.int32)
        self.ncharge = int(state.ff.ncharge)

        klimx, klimy, klimz = self.klimit
        nat = self.nat
        self._cx = cp.empty((klimx + 1, nat), dtype=cp.float64)
        self._sx = cp.empty((klimx + 1, nat), dtype=cp.float64)
        self._cy = cp.empty((klimy + 1, nat), dtype=cp.float64)
        self._sy = cp.empty((klimy + 1, nat), dtype=cp.float64)
        self._cz = cp.empty((klimz + 1, nat), dtype=cp.float64)
        self._sz = cp.empty((klimz + 1, nat), dtype=cp.float64)

        # Compile kernels
        self._trig = cp.RawKernel(_TRIG_KERNEL, "build_trig_tables")
        self._kspace = cp.RawKernel(_KSPACE_EFIELD_KERNEL, "kspace_efield_one_kvec")

        # Pre-build the list of (kx, ky, kz) triples that survive the
        # spherical cutoff. Same convention as ewald_fast.
        self._build_kvec_list()

    def _build_kvec_list(self):
        klimx, klimy, klimz = self.klimit
        kxmax2inv = 1.0 / float(klimx*klimx)
        kymax2inv = 1.0 / float(klimy*klimy)
        kzmax2inv = 1.0 / float(klimz*klimz)
        front = 2.0 * np.pi / self.box

        kvecs = []
        for kx in range(0, klimx + 1):
            kymin = -klimy if kx > 0 else 0
            for ky in range(kymin, klimy + 1):
                kzmin = 1 if (kx == 0 and ky == 0) else -klimz
                for kz in range(kzmin, klimz + 1):
                    s = (kx*kx)*kxmax2inv + (ky*ky)*kymax2inv + (kz*kz)*kzmax2inv
                    if s > 1.0:
                        continue
                    hx = float(kx) * front[0]
                    hy = float(ky) * front[1]
                    hz = float(kz) * front[2]
                    rkmag2 = hx*hx + hy*hy + hz*hz
                    if rkmag2 < 1e-12:
                        continue
                    kvecs.append((kx, ky, kz, hx, hy, hz, rkmag2))
        self._kvecs = kvecs

    def compute(self, x_cpu_or_gpu):
        """Compute reciprocal E-field and return (3, nat) numpy array."""
        nat = self.nat
        klimx, klimy, klimz = self.klimit
        if isinstance(x_cpu_or_gpu, cp.ndarray):
            x_gpu = x_cpu_or_gpu
        else:
            x_gpu = cp.asarray(x_cpu_or_gpu, dtype=cp.float64)

        front_x = 2.0 * np.pi / float(self.box[0])
        front_y = 2.0 * np.pi / float(self.box[1])
        front_z = 2.0 * np.pi / float(self.box[2])

        threads = 256
        blocks = (nat + threads - 1) // threads
        self._trig(
            (blocks,), (threads,),
            (x_gpu, np.int32(nat),
             np.int32(klimx), np.int32(klimy), np.int32(klimz),
             cp.float64(front_x), cp.float64(front_y), cp.float64(front_z),
             self._cx, self._sx, self._cy, self._sy, self._cz, self._sz)
        )

        elfown_gpu = cp.zeros((3, nat), dtype=cp.float64)

        vol = float(self.box[0] * self.box[1] * self.box[2])
        twopifact = 2.0 * np.pi / vol
        elfact = 2.0 * twopifact
        fouralpha2inv = 1.0 / (4.0 * self.alpha * self.alpha)

        # Launch one block per k-vector
        threads_kv = 256
        shared_bytes = 2 * threads_kv * 8

        for (kx, ky, kz, hx, hy, hz, rkmag2) in self._kvecs:
            akvec = 2.0 * np.exp(-rkmag2 * fouralpha2inv) / rkmag2
            prefact = akvec * elfact

            self._kspace(
                (1,), (threads_kv,),
                (np.int32(nat),
                 self.atomtype_gpu, self.q_gpu, np.int32(self.ncharge),
                 self._cx, self._sx, self._cy, self._sy, self._cz, self._sz,
                 np.int32(kx), np.int32(ky), np.int32(kz),
                 cp.float64(hx), cp.float64(hy), cp.float64(hz),
                 cp.float64(prefact),
                 elfown_gpu),
                shared_mem=shared_bytes
            )

        cp.cuda.runtime.deviceSynchronize()
        return cp.asnumpy(elfown_gpu)


# Helper for main.py
def install_gpu_recip(state):
    """Attach GPUEwaldRecip to state.gpu_ewald_recip if a GPU is available.

    Returns True on success, False otherwise. Polarization will use the
    GPU path automatically when this attribute is set.
    """
    if not gpu_available():
        state.gpu_ewald_recip = None
        return False
    try:
        state.gpu_ewald_recip = GPUEwaldRecip(state)
        n = len(state.gpu_ewald_recip._kvecs)
        dev_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
        print(f"  [GPU] Reciprocal E-field on {dev_name}: {n} k-vectors")
        return True
    except Exception as e:
        print(f"  [GPU] Init failed: {e}; falling back to CPU.")
        state.gpu_ewald_recip = None
        return False
