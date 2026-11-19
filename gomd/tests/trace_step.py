"""
Kernel-level profile of the fused GOMD MD step (what XLA actually runs).

    python -m gomd.tests.trace_step --input-dir ./md-input --ref-dir ./pkg --polarizable

Builds the step exactly as main_scan does (PCG + predicted start, drift
rebuilds, per-step rcut compaction, shared k-space pass), compiles a lax.scan
of --steps steps, runs it once to warm up, then records one run with the JAX
profiler. Every GPU kernel is mapped back to its HLO instruction and, through
the compiled HLO's metadata, to its JAX name path (jit names and the gomd_*
named scopes) and source line. Reports, per MD step:
  * wall time, summed kernel time, kernel launches, and the gap between them
  * time per gomd_* block, then per jitted function inside each block
  * the top kernels, with their source lines
"""
from __future__ import annotations
import argparse, collections, glob, gzip, json, os, re, sys, time


def parse_hlo_metadata(text):
    """instruction name -> (op_name, source 'file:line')."""
    meta = {}
    pat = re.compile(r'^\s*(?:ROOT\s+)?%?([\w.\-]+)\s*=.*?metadata=\{([^}]*)\}')
    for line in text.splitlines():
        m = pat.match(line)
        if not m:
            continue
        name, md = m.group(1), m.group(2)
        op = re.search(r'op_name="([^"]*)"', md)
        sf = re.search(r'source_file="([^"]*)"', md)
        sl = re.search(r'source_line=(\d+)', md)
        src = (os.path.basename(sf.group(1)) + ':' + sl.group(1)) if sf and sl else ''
        meta[name] = (op.group(1) if op else '', src)
    return meta


def kernel_events(logdir):
    """[(kernel name, hlo_op or '', duration_us)] for the GPU device plane(s)."""
    paths = glob.glob(os.path.join(logdir, '**', '*.xplane.pb'), recursive=True)
    events = []
    try:
        from jax.profiler import ProfileData
        for p in paths:
            pd = ProfileData.from_file(p)
            for plane in pd.planes:
                if not plane.name.startswith('/device:GPU'):
                    continue
                for line in plane.lines:
                    if not line.name.startswith('Stream'):
                        continue
                    for ev in line.events:
                        stats = {k: v for k, v in ev.stats}
                        events.append((ev.name, str(stats.get('hlo_op', '')), ev.duration_ns / 1e3))
        if events:
            return events, 'xplane'
    except Exception as exc:                                          # fall back to perfetto json
        print(f"  (ProfileData unavailable: {type(exc).__name__}: {exc}; trying perfetto json)")
    for p in glob.glob(os.path.join(logdir, '**', '*perfetto*.json*'), recursive=True):
        op = gzip.open if p.endswith('.gz') else open
        with op(p, 'rt') as f:
            tr = json.load(f)
        pids = {e['pid']: e['args']['name'] for e in tr.get('traceEvents', [])
                if e.get('ph') == 'M' and e.get('name') == 'process_name'}
        for e in tr.get('traceEvents', []):
            if e.get('ph') == 'X' and 'GPU' in str(pids.get(e.get('pid'), '')):
                a = e.get('args', {})
                events.append((e['name'], str(a.get('hlo_op', '')), float(e.get('dur', 0))))
    return events, 'perfetto'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', default='./md-input')
    ap.add_argument('--ref-dir', default='./pkg')
    ap.add_argument('--polarizable', action='store_true')
    ap.add_argument('--tolpol', type=float, default=1e-8)
    ap.add_argument('--ensemble', choices=['nve', 'nvt'], default='nve')
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--top', type=int, default=30)
    ap.add_argument('--logdir', default='/tmp/gomd_trace')
    ap.add_argument('--out', default=None)
    ap.add_argument('--command-buffers', action='store_true',
                    help='keep XLA command buffers (CUDA graphs); they hide the individual '
                         'kernels from the trace, so they are off by default here')
    args = ap.parse_args(argv)
    if not args.command_buffers:
        os.environ['XLA_FLAGS'] = (os.environ.get('XLA_FLAGS', '') +
                                   ' --xla_gpu_enable_command_buffer=').strip()
    from jax import config as jc
    jc.update('jax_enable_x64', True)
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

    out_f = open(args.out, 'w') if args.out else None
    def emit(s=''):
        print(s, flush=True)
        if out_f:
            out_f.write(s + '\n'); out_f.flush()

    def block(o):
        for leaf in jax.tree_util.tree_leaves(o):
            if hasattr(leaf, 'block_until_ready'):
                leaf.block_until_ready()
        return o

    # ---------------- the step, as main_scan builds it ----------------
    D = load_m2_system(args.input_dir, args.ref_dir)
    nat = D['nat']; E = D['ensemble']
    Dj = build_Dj(D); Dj['con']['tol'] = 1e-12
    kxn, kyn, kzn = build_kvectors(D['box'], *D['klimit'])
    recip = make_recip(D['box'], D['q'], kxn, kyn, kzn, D['alpha'])
    kx, ky, kz = jnp.asarray(kxn), jnp.asarray(kyn), jnp.asarray(kzn)
    ewself = self_energy(D['q'], D['alpha'])
    ii_h, jj_h = build_neighbor_pairs(np.asarray(D['x']), D['box'], D['rcut'], skin=1.0,
                                      exclusions=D['excl_set'])
    capacity = int(ii_h.size * 1.25)
    ncut0 = int(count_pairs_within(jnp.asarray(D['x']), jnp.asarray(ii_h), jnp.asarray(jj_h),
                                   jnp.asarray(D['box']), float((D['rcut'] + 0.05) ** 2)))
    pol_fn = None
    if args.polarizable:
        from gomd.forces.polarization_full import make_polarization_fn
        D['pol_data']['tolpol'] = args.tolpol
        pol_fn = make_polarization_fn(D, kx, ky, kz, scf_max_iter=300, solver='pcg', recip=recip)
    nhc = (nhc_mod.make_nhc_params(E['Tset'], E['ndof'], E['nnos'], E['qfreq'], D['dt'])
           if args.ensemble == 'nvt' else None)
    parts = make_md_step(Dj, kx, ky, kz, ewself, exclusions_to_flat(D['excl_set'], nat), capacity,
                         100, float((D['rcut'] + 1.0) ** 2), polarization_fn=pol_fn,
                         use_aspc=args.polarizable, nhc_params=nhc, natreal=E['natreal'],
                         skin=1.0, drift_rebuild=True, cut_capacity=int(ncut0 * 1.10), recip=recip)
    x0 = jax.jit(parts['place_dummies'])(Dj['x'])
    ii, jj, cnt = parts['rebuild'](x0)
    x0, pe0, f0, comps0, mu0 = jax.jit(parts['force_fn'])(x0, ii, jj)
    mu = (mu0, mu0, mu0) if mu0 is not None else (jnp.zeros((nat, 3)),) * 3
    nhc0 = ((jnp.asarray(E['xlogs']), jnp.asarray(E['vlogs']), jnp.asarray(E['glogs']))
            if nhc else (jnp.zeros(1), jnp.zeros(1), jnp.zeros(1)))
    S0 = {'x': x0, 'v': Dj['v'], 'f': f0, 'nhc': nhc0, 'step': jnp.array(1)}
    carry0 = parts['init_carry'](S0, x0, ii, jj, cnt, mu, pe0, f0)
    run = make_runner(parts['scan_body'])
    n = args.steps

    emit(f"# GOMD fused-step trace  {time.strftime('%Y-%m-%d %H:%M:%S')}  jax {jax.__version__}  "
         f"{jax.devices()[0].device_kind}")
    emit(f"# {'polarizable' if args.polarizable else 'classical'} {args.ensemble.upper()}, "
         f"{n} steps in lax.scan, tolpol {args.tolpol:g} (PCG + predict), drift rebuilds, "
         f"rcut compaction, shared k-space")
    t0 = time.perf_counter()
    c1, _ = run(carry0, nsteps=n); block(c1)
    emit(f"# compile + first run {time.perf_counter() - t0:.1f} s")
    t0 = time.perf_counter()
    c2, h2 = run(c1, nsteps=n); block(c2)
    wall_untraced = (time.perf_counter() - t0) / n * 1e3

    hlo = run.lower(c1, nsteps=n).compile().as_text()
    meta = parse_hlo_metadata(hlo)

    os.makedirs(args.logdir, exist_ok=True)
    for p in glob.glob(os.path.join(args.logdir, '**', '*'), recursive=True):
        if os.path.isfile(p):
            os.remove(p)
    with jax.profiler.trace(args.logdir, create_perfetto_trace=True):
        t0 = time.perf_counter()
        c3, h3 = run(c2, nsteps=n); block(c3)
        wall = (time.perf_counter() - t0) / n * 1e3
    ev, src = kernel_events(args.logdir)
    if not ev:
        emit('# no GPU kernel events found in the trace'); return 1
    nrebuild = int(np.sum(np.asarray(h3['rebuilt'])))
    scf = float(np.mean(np.asarray(h3['scf_it'])))
    tot = sum(d for _, _, d in ev) / 1e3 / n
    emit(f"# per step: wall {wall_untraced:.2f} ms untraced ({wall:.2f} traced), kernels {tot:.2f} ms, "
         f"{len(ev) / n:.0f} kernel launches; gaps {wall - tot:.2f} ms  "
         f"[{nrebuild} rebuilds in these {n} steps, {scf:.2f} T.mu products/step; events from {src}]")

    def scope_of(op):
        parts_ = op.split('/')
        g = [p for p in parts_ if p.startswith('gomd_')]
        j = [p[4:-1] for p in parts_ if p.startswith('jit(') and p not in ('jit(run)', 'jit(main)')]
        return (g[-1] if g else '(unlabelled)'), (j[-1] if j else '-')

    by_scope = collections.defaultdict(float)
    by_fn = collections.defaultdict(float)
    by_kernel = collections.defaultdict(lambda: [0.0, 0, '', ''])
    unmapped = 0.0
    for kname, hop, d in ev:
        op, srcl = meta.get(hop, ('', ''))
        if not op:
            unmapped += d
        sc, fn = scope_of(op)
        by_scope[sc] += d
        by_fn[(sc, fn)] += d
        k = by_kernel[hop or kname]
        k[0] += d; k[1] += 1; k[2] = op; k[3] = srcl

    emit("\n## time per block (ms per MD step, share of kernel time)")
    for sc, d in sorted(by_scope.items(), key=lambda t: -t[1]):
        emit(f"  {d / 1e3 / n:8.3f}  {d / 1e3 / n / tot:6.1%}  {sc}")
        for (s2, fn), d2 in sorted(by_fn.items(), key=lambda t: -t[1]):
            if s2 == sc and d2 / 1e3 / n >= 0.05:
                emit(f"            {d2 / 1e3 / n:8.3f}  {fn}")
    if unmapped:
        emit(f"  ({unmapped / 1e3 / n:.3f} ms/step of kernels had no HLO metadata)")

    emit(f"\n## top {args.top} kernels (ms per MD step, launches per step, name path, source)")
    for name, (d, c, op, srcl) in sorted(by_kernel.items(), key=lambda t: -t[1][0])[:args.top]:
        path = '/'.join(p for p in op.split('/') if p.startswith(('gomd_', 'jit(')) and
                        p not in ('jit(run)', 'jit(main)'))
        tail = op.split('/')[-1] if op else ''
        emit(f"  {d / 1e3 / n:7.3f} {c / n:5.1f}x  {name[:38]:<38} {path}/{tail}  {srcl}")
    if out_f:
        out_f.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
