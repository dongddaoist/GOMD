md-input3 — graphite electrode | electrolyte | graphite electrode cell (constant-potential task)

Source: the user's WSL run folder ~/run4-resize (job "DME-40ZnOTF-80Pyr14TFSI-interface-300K",
Fortran WMI-MD on 40 MPI ranks). Copied unchanged on 2026-11-18 (md5 checked):
  coords.inp, connectivity.dat, ff.dat, params.h   from run4-resize/input/
  mdrun.params                                     from run4-resize/
The force field is APPLE&P (SystemGenerator 8.0.2, FFversion f1f005_R): proprietary, do not
distribute.

System: 17,544 sites = 3,584 graphite C (Cg) + 480 DME + 40 Zn2+ + 80 OTf- + 80 Pyr14+ + 80 TFSI-
(11,960 electrolyte atoms + 2,000 lone-pair sites); neutral. Box 33.84 x 34.1907 x 131.625 A,
periodic in 3D. Cg: charge 0, polarizability 0; graphene bonds, bends, torsions and out-of-plane
terms (the old run kept the sheets flexible, with thermal velocities).

Electrodes (as the user defined them): atoms 1-1792 = lower electrode, the last 1792 atoms
(15753-17544) = upper electrode. Each is 4 graphene sheets of 448 atoms (8 x 14 rectangular
cells, C-C 1.41 A, armchair along x), 3.43 A apart:
  lower  z = 131.31 (i.e. -0.32 across the z boundary), 3.09, 6.51, 9.95
  upper  z = 117.64, 121.05, 124.48, 127.89
The electrolyte fills z 12.1-115.5 A; the inner sheets are 107.7 A apart. In this 3D-periodic
box the two outer sheets are 3.4 A apart across the z boundary (electrodes back to back).

mdrun.params: NVT 300 K (NHC 1, OmegaQ 0.01), dt 0.5 fs, no SHAKE (tolerance written as the
Fortran literal 1.d-14), Ewald alpha 0.23 with kmax 7 7 16, tolpol 1.0d-9, taper from 11.5 A,
"Resize Box" on with the target 33.84 34.1907 131.625 (the Fortran code would rescale a box of
another size to this; GOMD ignores the flag). params.h raises the parser's array limits
(12 vdW types, 27 charge types, 17,544 sites, ...); pkg reads it since 2026-11-18.

GOMD use (P0, fixed electrodes):
  python -m gomd.main_scan --input-dir md-input3 --ref-dir pkg --polarizable \
      --electrode-lower first:1792 --electrode-upper last:1792 [--slab-height 205] ...
