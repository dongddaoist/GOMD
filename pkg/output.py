"""
Output Routines for MD Simulation
Includes XYZ trajectory output, md.log, and statistics
"""

import numpy as np
from pathlib import Path
from typing import TextIO, Optional
from datatypes import SimulationState
from constants import GASK_KCAL


class MDLogWriter:
    """
    Write md.log — the primary simulation log file.
    Replaces Fortran fort.65 running-average output.
    
    Standard columns: Time(fs) Temp(K) Pres(atm) BoxX BoxY BoxZ TotalE Pxx
    Full tensor mode adds: Pyy Pzz Pxy Pxz Pyz
    """
    
    def __init__(self, filename: str = "md.log", full_tensor: bool = False):
        self.filename = filename
        self.full_tensor = full_tensor
        self.file: Optional[TextIO] = None
    
    def open(self):
        self.file = open(self.filename, 'w')
        self._write_header()
    
    def close(self):
        if self.file:
            self.file.close()
            self.file = None
    
    def _write_header(self):
        hdr = (f"{'Time(fs)':>12} {'Temp(K)':>10} {'Pres(atm)':>12} "
               f"{'BoxX(A)':>12} {'BoxY(A)':>12} {'BoxZ(A)':>12} "
               f"{'TotalE':>14} {'Pxx':>12}")
        if self.full_tensor:
            hdr += (f" {'Pyy':>12} {'Pzz':>12} "
                    f"{'Pxy':>12} {'Pxz':>12} {'Pyz':>12}")
        self.file.write("#" + "-" * (len(hdr) + 1) + "\n")
        self.file.write("# " + hdr.strip() + "\n")
        self.file.write("#" + "-" * (len(hdr) + 1) + "\n")
    
    def write(self, state: SimulationState, time_fs: float):
        if self.file is None:
            self.open()
        
        temp = state.temp
        pres = state.pressure.pres
        box = state.atoms.box
        toten = state.energy.toten
        
        # Pressure tensor from tvirpo + kinetic contribution
        # Pxx = kinetic_xx + virial_xx (both stored in stress)
        vol = np.prod(box)
        volinv = 1.0 / vol
        conv_vir = 2.2857e4  # virial -> atm conversion
        conv_kin = 45.42     # kinetic -> atm conversion (per dof*T)
        
        # Directional kinetic pressure: P_kin_kk = m*v_kk^2 / V * conversion
        pkin = np.zeros(3)
        for kk in range(3):
            akin2_kk = 0.0
            for iat in range(state.atoms.nat):
                if not state.atoms.ldummy[iat]:
                    akin2_kk += state.atoms.mass[iat] * state.atoms.v[kk, iat]**2
            pkin[kk] = akin2_kk / (4184.0 * 1000.0)  # to kcal/mol
            pkin[kk] = 6.8571e4 * pkin[kk] * volinv   # to atm (3x isotropic factor)
        
        # Full stress tensor: virial + kinetic diagonal
        stress = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                stress[i, j] = conv_vir * state.pressure.tvirpo[i, j] * volinv
            stress[i, i] += pkin[i]
        
        pxx = stress[0, 0]
        
        line = (f"{time_fs:12.2f} {temp:10.3f} {pres:12.3f} "
                f"{box[0]:12.6f} {box[1]:12.6f} {box[2]:12.6f} "
                f"{toten:14.4f} {pxx:12.3f}")
        
        if self.full_tensor:
            pyy = stress[1, 1]
            pzz = stress[2, 2]
            pxy = stress[0, 1]
            pxz = stress[0, 2]
            pyz = stress[1, 2]
            line += (f" {pyy:12.3f} {pzz:12.3f} "
                     f"{pxy:12.3f} {pxz:12.3f} {pyz:12.3f}")
        
        self.file.write(line + "\n")
        self.file.flush()
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class TrajectoryWriter:
    """
    Write trajectory in XYZ format (plain text).
    This replaces the Fortran binary fort.77 format.
    """
    
    def __init__(self, filename: str = "trajectory.xyz"):
        self.filename = filename
        self.file: Optional[TextIO] = None
        self.frame_count = 0
    
    def open(self):
        """Open trajectory file for writing."""
        self.file = open(self.filename, 'w')
        self.frame_count = 0
    
    def close(self):
        """Close trajectory file."""
        if self.file:
            self.file.close()
            self.file = None
    
    def write_frame(self, state: SimulationState, time_fs: float):
        """
        Write a single frame in XYZ format.
        
        Format:
        nat
        Comment line with time and box dimensions
        Element x y z
        ...
        """
        if self.file is None:
            self.open()
        
        nat = state.atoms.nat
        x = state.atoms.x
        box = state.atoms.box
        atomtype = state.atoms.atomtype
        atom_labels = state.ff.atom_labels
        
        # Write number of atoms
        self.file.write(f"{nat}\n")
        
        # Comment line with metadata
        self.file.write(f"Frame {self.frame_count} Time={time_fs:.3f}fs "
                       f"Box={box[0]:.6f},{box[1]:.6f},{box[2]:.6f}\n")
        
        # Write atom coordinates
        for iat in range(nat):
            itype = atomtype[iat]
            label = atom_labels[itype][:2].strip()  # Get element symbol
            
            # Clean up label (remove numbers)
            element = ''.join(c for c in label if not c.isdigit())
            if not element:
                element = label[:2]
            
            self.file.write(f"{element:2s} {x[0, iat]:15.8f} {x[1, iat]:15.8f} {x[2, iat]:15.8f}\n")
        
        self.frame_count += 1
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class VelocityWriter:
    """Write velocities to file."""
    
    def __init__(self, filename: str = "velocities.dat"):
        self.filename = filename
        self.file: Optional[TextIO] = None
    
    def open(self):
        self.file = open(self.filename, 'w')
    
    def close(self):
        if self.file:
            self.file.close()
            self.file = None
    
    def write_frame(self, state: SimulationState, time_fs: float):
        if self.file is None:
            self.open()
        
        nat = state.atoms.nat
        v = state.atoms.v
        
        self.file.write(f"# Time={time_fs:.3f}fs nat={nat}\n")
        for iat in range(nat):
            self.file.write(f"{v[0, iat]:15.8e} {v[1, iat]:15.8e} {v[2, iat]:15.8e}\n")


class StatisticsWriter:
    """
    Write running averages and statistics.
    Equivalent to fort.65 output.
    """
    
    def __init__(self, filename: str = "statistics.dat"):
        self.filename = filename
        self.file: Optional[TextIO] = None
        
        # Accumulators for averages
        self.sumpr = np.zeros(40)
        self.sumsqpr = np.zeros(40)
        self.avgsumpr = np.zeros(40)
        self.avgsumsqpr = np.zeros(40)
        self.kstar = 0
    
    def open(self):
        self.file = open(self.filename, 'w')
        self._write_header()
    
    def close(self):
        if self.file:
            self.file.close()
            self.file = None
    
    def _write_header(self):
        """Write header to statistics file."""
        self.file.write("#" + "-" * 71 + "\n")
        self.file.write("#                         Running Averages\n")
        self.file.write("#\n")
        self.file.write("#     Time    Temp     Pressure     Box   Total E    Hamiltonian, BoxX, BoxY, Pxx, Pyy\n")
        self.file.write("#      fs       K         Atm       Ang   kcal/mol   kcal/mol    Ang    Ang  atm atm\n")
        self.file.write("#" + "-" * 71 + "\n")
    
    def accumulate(self, state: SimulationState, prop: np.ndarray):
        """Accumulate properties for averaging."""
        for i in range(len(prop)):
            self.sumpr[i] += prop[i]
            self.sumsqpr[i] += prop[i] * prop[i]
    
    def write_averages(self, state: SimulationState, time_fs: float, 
                       avgpr: np.ndarray, kave: int, kinit: int):
        """Write averaged properties to file."""
        if self.file is None:
            self.open()
        
        if state.kount > kinit:
            self.kstar += 1
            for i in range(len(avgpr)):
                self.avgsumpr[i] += avgpr[i]
                self.avgsumsqpr[i] += avgpr[i] * avgpr[i]
        
        # Format: time, temp, pressure, box, total_e, hamiltonian, box_x, box_y, pxx, pyy
        self.file.write(f"{time_fs:11.2f}{avgpr[0]:9.2f}{avgpr[1]:12.3f}"
                       f"{avgpr[2]:10.4f}{avgpr[3]:12.2f}{avgpr[4]:12.2f}"
                       f"{avgpr[24]:10.4f}{avgpr[25]:10.4f}"
                       f"{avgpr[15]:9.2f}{avgpr[19]:9.2f}\n")
        self.file.flush()


class StressTensorWriter:
    """Write stress tensor output."""
    
    def __init__(self, filename: str = "stress.dat"):
        self.filename = filename
        self.file: Optional[TextIO] = None
    
    def open(self):
        self.file = open(self.filename, 'w')
        self.file.write("#     Time  Pxx  Pyy  Pzz  Pxy  Pxz  Pyz\n")
    
    def close(self):
        if self.file:
            self.file.close()
            self.file = None
    
    def write_stress(self, time_fs: float, stress: np.ndarray):
        if self.file is None:
            self.open()
        
        self.file.write(f"{time_fs:11.2f}"
                       f"{stress[0,0]:9.1f}{stress[1,1]:9.1f}{stress[2,2]:9.1f}"
                       f"{stress[0,1]:9.1f}{stress[0,2]:9.1f}{stress[1,2]:9.1f}\n")


class DipoleWriter:
    """Write dipole moment output."""
    
    def __init__(self, filename: str = "dipole.dat"):
        self.filename = filename
        self.file: Optional[TextIO] = None
    
    def open(self):
        self.file = open(self.filename, 'w')
        self.file.write("# Time(fs)  Dipole_x  Dipole_y  Dipole_z  ChargeFlux_x  ChargeFlux_y  ChargeFlux_z\n")
    
    def close(self):
        if self.file:
            self.file.close()
            self.file = None
    
    def write_dipole(self, time_fs: float, totdipole: np.ndarray, chargeflux: np.ndarray):
        if self.file is None:
            self.open()
        
        self.file.write(f"{time_fs:11.2f} "
                       f"{totdipole[0]:12.6f} {totdipole[1]:12.6f} {totdipole[2]:12.6f} "
                       f"{chargeflux[0]:12.6f} {chargeflux[1]:12.6f} {chargeflux[2]:12.6f}\n")


def write_restart(state: SimulationState, filename: str = "restart.inp"):
    """
    Write restart file in the same format as coords.inp.
    """
    with open(filename, 'w') as f:
        # Write temperature ramp info if active
        if state.lTempRamp:
            f.write(f"*!T  T  {state.Tset:10.4f}  {state.Tdelta:10.4f}\n")
        
        # Write atomic coordinates and velocities
        nat = state.atoms.nat
        x = state.atoms.x
        v = state.atoms.v
        atomtype = state.atoms.atomtype
        atom_labels = state.ff.atom_labels
        
        for iat in range(nat):
            itype = atomtype[iat]
            label = atom_labels[itype]
            # Parse label to get element and charge index
            elem = label[:3].strip()
            idx = int(label[3:].strip()) if len(label) > 3 else 1
            
            # Position line
            f.write(f"{x[0, iat]:15.8f} {x[1, iat]:15.8f} {x[2, iat]:15.8f}  "
                   f"{elem:3s} {idx:2d}\n")
            
            # Velocity line
            f.write(f"{v[0, iat]:15.8e} {v[1, iat]:15.8e} {v[2, iat]:15.8e}\n")
        
        # Extended Lagrangian variables
        box = state.atoms.box
        f.write(f"{box[0]:15.8f} {box[1]:15.8f} {box[2]:15.8f}\n")
        
        xlogs = state.thermostat.xlogs
        vlogs = state.thermostat.vlogs
        glogs = state.thermostat.glogs
        nnos = state.params.nnos
        
        f.write(" ".join(f"{xlogs[i]:15.8e}" for i in range(nnos)) + "\n")
        f.write(" ".join(f"{vlogs[i]:15.8e}" for i in range(nnos)) + "\n")
        f.write(" ".join(f"{glogs[i]:15.8e}" for i in range(nnos)) + "\n")
        
        xlogv = state.barostat.xlogv
        vlogv = state.barostat.vlogv
        glogv = state.barostat.glogv
        
        f.write(f"{xlogv[0]:15.8e} {xlogv[1]:15.8e} {xlogv[2]:15.8e}\n")
        f.write(f"{vlogv[0]:15.8e} {vlogv[1]:15.8e} {vlogv[2]:15.8e}\n")
        f.write(f"{glogv[0]:15.8e} {glogv[1]:15.8e} {glogv[2]:15.8e}\n")
    
    print(f"Restart file written: {filename}")


def write_final_output(state: SimulationState, filename: str = "final_output.dat"):
    """
    Write final simulation statistics.
    Equivalent to output2() in Fortran.
    """
    with open(filename, 'w') as f:
        f.write("-" * 72 + "\n")
        f.write("              GOMD Simulation Summary\n")
        f.write("-" * 72 + "\n\n")
        
        f.write(f"Total steps completed:        {state.kount}\n")
        f.write(f"Total atoms:                  {state.atoms.nat}\n")
        f.write(f"Real atoms (non-dummy):       {state.atoms.natreal}\n")
        f.write(f"Degrees of freedom:           {state.ndof}\n\n")
        
        f.write("Final energies (kcal/mol):\n")
        f.write(f"  Kinetic energy:             {state.energy.kinen:15.4f}\n")
        f.write(f"  Potential energy:           {state.energy.poten:15.4f}\n")
        f.write(f"  Total energy:               {state.energy.toten:15.4f}\n\n")
        
        f.write("Energy components (kcal/mol):\n")
        f.write(f"  Bond energy:                {state.energy.ebond:15.4f}\n")
        f.write(f"  Bend energy:                {state.energy.ebend:15.4f}\n")
        f.write(f"  Torsion energy:             {state.energy.etort:15.4f}\n")
        f.write(f"  Out-of-plane energy:        {state.energy.eopbs:15.4f}\n")
        f.write(f"  Non-bonded energy:          {state.energy.unbd:15.4f}\n\n")
        
        f.write(f"Final temperature (K):        {state.temp:15.4f}\n")
        f.write(f"Final pressure (atm):         {state.pressure.pres:15.4f}\n\n")
        
        f.write("Final box dimensions (Angstrom):\n")
        f.write(f"  Box X:                      {state.atoms.box[0]:15.6f}\n")
        f.write(f"  Box Y:                      {state.atoms.box[1]:15.6f}\n")
        f.write(f"  Box Z:                      {state.atoms.box[2]:15.6f}\n")
        f.write(f"  Volume:                     {np.prod(state.atoms.box):15.6f}\n\n")
        
        f.write("-" * 72 + "\n")
    
    print(f"Final output written: {filename}")


def print_step_info(state: SimulationState, time_fs: float):
    """Print information about current step to console."""
    print(f"Kount ={state.kount:7d} Box = {state.atoms.box[0]:8.4f} {state.atoms.box[1]:8.4f} {state.atoms.box[2]:8.4f} "
          f"P ={state.pressure.pres:11.2f} T = {state.temp:9.3f} E = {state.energy.toten:11.3f} Ts = {state.Tset:9.3f}")
