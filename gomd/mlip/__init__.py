"""MLIP (machine-learned interatomic potential) support for GOMD.

client.py   MLIPProvider: GOMD-side callable, x -> (E kcal/mol, F kcal/mol/A)
server.py   standalone process that runs the MLIP (UMA via fairchem, or mock)
elements.py atomic numbers from the force-field masses
compare.py  PFF-vs-MLIP agreement metrics
"""
