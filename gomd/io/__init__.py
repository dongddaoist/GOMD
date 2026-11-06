"""
io — the numpy -> JAX boundary.

assembly_loader.load_m2_system runs the reference package's one-time setup
(parsing mdrun.params, ff.dat, connectivity.dat, coords.inp; spline tables;
connectivity; constraints; dummy sites; polarization arrays) and extracts a
flat dict of JAX-ready arrays. Everything per-step is pure JAX after that.
"""
