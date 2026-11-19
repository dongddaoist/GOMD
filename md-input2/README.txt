Lone-pair-free copy of md-input for MLIP runs
made 2026-11-17 23:17 by: python -m gomd.io.make_lp_free --input-dir md-input --out-dir md-input2

system.xyz  extended XYZ, one frame (positions A, velocities m/s, masses g/mol)
  7400 atoms = 8716 sites - 1316 lone-pair sites
  composition H2880 Li36 C1920 N36 O784 F1672 S72
  box 46.956589 x 46.956589 x 46.956589 A, periodic
  charge 0 (total of the source system, lone-pair charges included), spin multiplicity 1
  state: coords.inp at t = 1.5 ps; kinetic temperature 344.70 K over 3N - 3 = 22197 dof
         (no constraints here; the PFF counts 15192 dof with its 7008 SHAKE bonds)
  column 'site' = the atom's index in the source system (lone-pair sites included)

Run MD on it with the MLIP driver:  python -m gomd.mlip_md --input md-input2/system.xyz --mlip uma|mock ...

2026-11-18 00:37  python -m gomd.io.make_subsystem --input-dir md-input --shape sphere --natoms 2000 --out md-input2/sphere_2000.xyz
  sphere_2000.xyz: sphere, radius 18.564 A, 2002 atoms (H774 Li11 C516 N11 O216 F452 S22) in 108 complete molecules (solvent 86, Li+ 11, FSI- 11), 2 counter-ion(s) added for neutrality; charge 0, spin 1; cluster in vacuum, frame 72.98 A (pbc F); closest contact between molecules 1.845 A; T 353.4 K over 6003 dof; read-back OK

