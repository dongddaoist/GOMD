def print_energy_breakdown(state):
    print(f"  Bond:   {state.energy.ebond:15.4f}")
    print(f"  Bend:   {state.energy.ebend:15.4f}")
    print(f"  Tors:   {state.energy.etort:15.4f}")
    print(f"  Impr:   {state.energy.eopbs:15.4f}")
    print(f"  Nonbd:  {state.energy.unbd:15.4f}")
    print(f"  Total:  {state.energy.poten:15.4f}")
