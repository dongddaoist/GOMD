"""
perf_snapshot.py
================
Quick standalone profiler. Runs the system through one neighbor-list build
and a few force evaluations, reporting the time for each. Useful before
launching long MD runs.

Usage:
    python perf_snapshot.py
"""

import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def banner(s):
    print("\n" + "=" * 60)
    print(s)
    print("=" * 60)


def main():
    banner("Performance snapshot")

    from input_parser import load_simulation
    from main import setup_system

    cwd = os.getcwd()
    print(f"Loading from {cwd}...")
    t0 = time.time()
    state = load_simulation(cwd)
    print(f"  load:        {(time.time()-t0)*1000:8.0f} ms")

    t0 = time.time()
    setup_system(state)
    t_setup = time.time() - t0
    print(f"  setup_system: {t_setup*1000:8.0f} ms  (incl. JIT compilation)")

    # ---- compute_all_forces timing ----
    banner("compute_all_forces (3 calls; first is JIT warm-up)")
    from forces import compute_all_forces
    timings = []
    for trial in range(3):
        t0 = time.time()
        compute_all_forces(state)
        dt = time.time() - t0
        timings.append(dt)
        print(f"  trial {trial+1}: {dt*1000:8.1f} ms")
    print(f"  mean of last 2: {np.mean(timings[1:])*1000:8.1f} ms/call")

    # ---- build_cell_list timing ----
    banner("build_cell_list (3 calls)")
    try:
        from neighbor_list_fast import build_cell_list_fast as build_cell_list
        which = "FAST (JIT-compiled)"
    except ImportError:
        from neighbor_list import build_cell_list
        which = "ORIGINAL (pure Python)"
    print(f"  using: {which}")
    nl_times = []
    for trial in range(3):
        t0 = time.time()
        build_cell_list(state)
        dt = time.time() - t0
        nl_times.append(dt)
        print(f"  trial {trial+1}: {dt*1000:8.1f} ms")
    print(f"  mean of last 2: {np.mean(nl_times[1:])*1000:8.1f} ms/call")

    # ---- compute_recip_efield timing (route used by polarization) ----
    if state.ff.lpolarizable:
        banner("compute_recip_efield path used by polarization")
        if hasattr(state, 'gpu_ewald_recip') and state.gpu_ewald_recip is not None:
            print("  using GPU")
            f = lambda: state.gpu_ewald_recip.compute(state.atoms.x)
        else:
            try:
                from ewald_fast import compute_recip_efield_fast
                f = lambda: compute_recip_efield_fast(state)
                print("  using ewald_fast (CPU JIT)")
            except ImportError:
                from ewald import compute_recip_efield
                f = lambda: compute_recip_efield(state)
                print("  using ewald (CPU original)")
        rf_times = []
        for trial in range(3):
            t0 = time.time()
            f()
            dt = time.time() - t0
            rf_times.append(dt)
            print(f"  trial {trial+1}: {dt*1000:8.1f} ms")
        print(f"  mean of last 2: {np.mean(rf_times[1:])*1000:8.1f} ms/call")

    banner("Summary")
    print(f"  setup (one-time):           {t_setup:7.2f} s")
    print(f"  forces / call (steady):     {np.mean(timings[1:])*1000:7.1f} ms")
    print(f"  build_cell_list / rebuild:  {np.mean(nl_times[1:])*1000:7.1f} ms")
    print(f"")
    print(f"  Approximate per-step time on a step that DOESN'T rebuild:")
    print(f"    forces + integrator overhead  ~ {np.mean(timings[1:])*1000:.0f} ms")
    print(f"  Approximate per-step time on a step that DOES rebuild:")
    print(f"    add ~{np.mean(nl_times[1:])*1000:.0f} ms once every ~10-50 steps.")


if __name__ == "__main__":
    main()
