# simpleEMS
# Copyright (C) 2026 Mohamoud Ali
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Write the problem file that tells the solver what to solve.

:func:`write_problem` assembles a self-contained GetDP ``.pro`` file: the
equation to solve for the electric field, the material properties of each
region of the mesh from :mod:`~simpleEMS.fem_geometry`, the boundary
conditions, the port excitation, and the results to report. Nothing is solved
here.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .fem_materials import ABC, AIR, EPS0, MU0, PEC

if TYPE_CHECKING:
    from .fem_geometry import Mesh
    from .fem_backend import Problem


def _fmt(x: float) -> str:
    return repr(float(x))


def _dir_vector(direction: str) -> str:
    d = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[direction]
    return f"Vector[{d[0]}., {d[1]}., {d[2]}.]"


def write_problem(
    problem: "Problem",
    mesh: "Mesh",
    workdir: str | Path,
) -> str:
    """
    Write the problem file for ``problem`` as meshed in ``mesh``.

    Parameters
    ----------
    problem : Problem
        The problem to solve, supplying the materials, ports, boundary
        condition, and element order.
    mesh : Mesh
        The generated mesh, supplying the region tag of each part of it.
    workdir : str | Path
        Directory to write the ``.pro`` file into.

    Returns
    -------
    str
        Path to the written ``.pro`` file.
    """
    ports = [mesh.port_regions[p.number] for p in problem.ports]
    nports = len(ports)
    port_numbers = [pm.number for pm in ports]
    active_default = port_numbers[0] if port_numbers else 1
    f0 = float(problem.freqs[0])

    k0_def = "k0[] = 2*Pi*FREQ / c0;"
    freqvar = "FREQ"

    # A wave port's mode is a solved field, not a formula: fem_port_mode writes
    # it as a Gmsh view per solve, and the modal index and impedance come in as
    # runtime constants beside it. Everything below is additive -- a problem
    # with no wave ports emits exactly what it always did.
    wave_ports = [pm for pm in ports if pm.is_wave]
    wave_consts = "".join(
        f",\n  NEFF_RE_{pm.number} = 1.0,   // Re(beta/k0) of port {pm.number}'s mode"
        f",\n  NEFF_IM_{pm.number} = 0.0,   // Im(beta/k0); negative on a lossy line"
        f"\n  ZC_{pm.number} = {_fmt(pm.z0)}   // its characteristic impedance [ohm]"
        for pm in wave_ports
    )
    gmsh_reads = [
        f'      GmshRead[ StrCat[myDir, "mode_{pm.number}.pos"], {pm.number} ] ;'
        for pm in wave_ports
    ]
    single_reads = ("\n".join(gmsh_reads) + "\n") if gmsh_reads else ""

    # Only the source term changes between ports, so assemble and factorise
    # once and rebuild just the right-hand side for each subsequent port.
    resolution_lines = ["CreateDir[Str[myDir]] ;", *gmsh_reads]
    if port_numbers:
        resolution_lines += [
            f"      Evaluate[ $ActivePort = {active_default} ] ;",
            "      Generate[A] ; Solve[A] ;",
            "      PostOperation[Get_SParameters] ;",
        ]
        for number in port_numbers[1:]:
            resolution_lines += [
                f"      Evaluate[ $ActivePort = {number} ] ;",
                "      GenerateRightHandSideGroup[A, Ports] ;",
                "      SolveAgain[A] ;",
                "      PostOperation[Get_SParameters] ;",
            ]
    resolution_op = "\n".join(resolution_lines)

    # Table results are appended: one row per port per solve point, in the
    # Resolution's port order. Callers read the last `nports` rows.
    xs_file = "File >"

    diel_lines = []
    epsr_lines = []
    mur_lines = []  # muR overrides for magnetic dielectrics (default 1)
    nur_lines = []  # nuR = 1/muR per dielectric (+ Air below)
    for name, rid in sorted(mesh.dielectric_regions.items()):
        spec = problem.solids[name]
        d = spec.dielectric
        diel_lines.append(f"  Diel_{name} = Region[{rid}];")
        # This .pro is written for exp(+i w t) (see the module docstring), so
        # a passive lossy dielectric has Im[eps]<0: eps_r*(1 - i*tan_d)
        epsr_lines.append(
            f"  epsR[Diel_{name}] = Complex[{_fmt(d.eps_r)}, "
            f"-{_fmt(d.eps_r * d.tan_d)}];"
        )
        if d.mu_r != 1.0:
            mur_lines.append(f"  muR[Diel_{name}] = {_fmt(d.mu_r)};")
        nur_lines.append(f"  nuR[Diel_{name}] = {_fmt(1.0 / d.mu_r)};")
    nur_lines.append("  nuR[Air] = 1.;")
    diel_region_list = ", ".join(f"Diel_{n}" for n in sorted(mesh.dielectric_regions))

    port_group_lines = []
    for pm in ports:
        port_group_lines.append(f"  Port_{pm.number} = Region[{pm.region}];")
    ports_list = ", ".join(f"Port_{pm.number}" for pm in ports)

    # lossy conductors: surface-impedance (Leontovich) boundary, Zs = Rs*(1+i),
    # Rs = sqrt(pi*f*mu0/sigma) (frequency dependent, hence a Function of freqvar).
    imped_group_lines, imped_fun_lines, imped_eq_lines, imped_names = [], [], [], []
    for j, (rid, sig) in enumerate(mesh.impedance_regions):
        imped_group_lines.append(f"  Imped_{j} = Region[{rid}];")
        imped_names.append(f"Imped_{j}")
        imped_fun_lines.append(f"  Rs_{j}[] = Sqrt[Pi*{freqvar}*mu0/({_fmt(sig)})];")
        imped_fun_lines.append(
            f"  Yimp_{j}[] = eta0 / (Rs_{j}[]*(1 + I[]));  // eta0/Zs, Zs=Rs*(1+i)"
        )
        imped_eq_lines.append(
            f"""      // lossy conductor {j}: surface impedance (sigma={sig:g} S/m)
      Galerkin {{ [ -I[]*k0[]*Yimp_{j}[]*(1/muR[]) * Normal[] /\\ (Normal[] /\\ Dof{{e}}) , {{e}} ] ;
        In Imped_{j} ; Integration I1 ; Jacobian Jac ; }}"""
        )
    imped_extra = (", ".join(imped_names) + ", ") if imped_names else ""

    # symmetry plane: PEC (constrain E_tan=0) or PMC (natural, excluded from ABC)
    sym = getattr(mesh, "sym_region", 0)
    sym_group_line = f"  Sym = Region[{sym}];" if sym else ""
    sym_in_pec = ", Sym" if (sym and mesh.sym_kind == "pec") else ""
    sym_in_tot = ", Sym" if sym else ""

    # A "pec" outer boundary shorts the box instead of absorbing into it: the
    # outer faces join BndPEC and the Silver-Muller term is not emitted. It
    # makes the model a shielded enclosure, which is what a closed structure --
    # a line, a filter -- actually wants, and it is the only boundary whose
    # walls match what the wave port's own 2D mode solve assumes, since
    # fem_port_mode.extract_cross_section treats the cross-section's outline as
    # a PEC wall. It cannot radiate, so a far field is refused under it.
    is_pec_box = getattr(mesh, "boundary", "silver_muller") == "pec"
    abc_in_pec = ", Abc" if is_pec_box else ""
    abc_eq = (
        "      // outer boundary: perfect electric conductor, constrained to\n"
        "      // E_tan = 0 through BndPEC -- a shielded box, nothing to absorb"
        if is_pec_box
        else (
            "      // outer free-space Silver-Muller ABC (first-order "
            "outgoing-wave boundary)\n"
            "      Galerkin { [ -I[]*k0[] * (1/muR[]) * Normal[] /\\ "
            "( Normal[] /\\ Dof{e} ) , {e} ] ;\n"
            "        In Abc ; Integration I1 ; Jacobian Jac ; }"
        )
    )

    # PML: complex coordinate stretching in the outer shell. epsR and nuR (=1/muR)
    # become anisotropic tensors there; cX/cY/cZ = 1 - i*Damp/k0.
    pml = getattr(mesh, "pml_region", 0)
    if pml:
        ib = mesh.inner_bbox
        pml_group_line = f"  Pml = Region[{mesh.pml_region}];"
        domain_pml = ", Pml"
        pml_fun = (
            f"  PmlXmin = {_fmt(ib[0])}; PmlXmax = {_fmt(ib[3])};\n"
            f"  PmlYmin = {_fmt(ib[1])}; PmlYmax = {_fmt(ib[4])};\n"
            f"  PmlZmin = {_fmt(ib[2])}; PmlZmax = {_fmt(ib[5])};\n"
            f"  PmlDelta = {_fmt(mesh.pml_thick)};\n"
            "  DampX[] = ((X[]>=PmlXmax)||(X[]<=PmlXmin)) ? ((X[]>=PmlXmax) ? 1/(PmlDelta-(X[]-PmlXmax)) : 1/(PmlDelta-(PmlXmin-X[]))) : 0;\n"
            "  DampY[] = ((Y[]>=PmlYmax)||(Y[]<=PmlYmin)) ? ((Y[]>=PmlYmax) ? 1/(PmlDelta-(Y[]-PmlYmax)) : 1/(PmlDelta-(PmlYmin-Y[]))) : 0;\n"
            "  DampZ[] = ((Z[]>=PmlZmax)||(Z[]<=PmlZmin)) ? ((Z[]>=PmlZmax) ? 1/(PmlDelta-(Z[]-PmlZmax)) : 1/(PmlDelta-(PmlZmin-Z[]))) : 0;\n"
            "  cX[] = Complex[1, -DampX[]/k0[]];\n"
            "  cY[] = Complex[1, -DampY[]/k0[]];\n"
            "  cZ[] = Complex[1, -DampZ[]/k0[]];\n"
            "  epsR[Pml] = TensorDiag[ cY[]*cZ[]/cX[], cX[]*cZ[]/cY[], cX[]*cY[]/cZ[] ];\n"
            "  nuR[Pml]  = TensorDiag[ cX[]/(cY[]*cZ[]), cY[]/(cX[]*cZ[]), cZ[]/(cX[]*cY[]) ];"
        )
    else:
        pml_group_line = ""
        domain_pml = ""
        pml_fun = ""

    # Radiated power is accounted, not measured: whatever the port accepts and
    # the materials do not dissipate has radiated. A flux integral does not
    # work here -- it reads ~0 behind a PML, and GetDP's surface trace of a
    # Form1 field drops the components the Poynting normal needs.
    # fem_solver.solve_fields_and_power does the arithmetic.
    pcond_q = []
    pcond_op = []
    for j, _ in enumerate(mesh.impedance_regions):
        pcond_q.append(
            f"""      {{ Name Pcond_{j} ; Value {{ Integral {{
        [ 0.5*Re[Yimp_{j}[]]/eta0 * SquNorm[ (Normal[] /\\ {{e}}) /\\ Normal[] ] ] ;
        In Imped_{j} ; Jacobian Jac ; Integration I1 ; }} }} }}"""
        )
        pcond_op.append(
            f"""      Print [ Pcond_{j}[Imped_{j}], OnGlobal, Format Table,
        {xs_file} StrCat[myDir, "Pcond_{j}.txt"] ] ;"""
        )

    # V/I of each port, printed by Get_Power so the field/power solve carries
    # its own accepted power at its own frequency -- the V_<n>/I_<n> files that
    # Get_SParameters writes belong to the sweep and are at whatever frequency
    # it last solved.
    # V_n/I_n are normalised by the register #(n), which only intPort_n sets, so
    # it has to be printed here too -- Get_Power does not run Get_SParameters
    # and would otherwise divide by an unset (zero) register.
    pdrv_op = []
    for pm in ports:
        n = pm.number
        pdrv_op.append(
            f"""      Print [ intPort_{n}[Port_{n}], OnRegion Port_{n}, StoreInRegister ({n}),
        Format Table, {xs_file} StrCat[myDir, "intPort.txt"] ] ;
      Print [ V_{n}[Port_{n}], OnRegion Port_{n}, Format Table,
        {xs_file} StrCat[myDir, "Vdrv_{n}.txt"] ] ;
      Print [ I_{n}[Port_{n}], OnRegion Port_{n}, Format Table,
        {xs_file} StrCat[myDir, "Idrv_{n}.txt"] ] ;"""
        )

    # per-port functions: unit mode direction, sheet relative admittance, incident field
    port_fun_lines = []
    for pm in ports:
        port_fun_lines.append(f"  dir_{pm.number}[] = {_dir_vector(pm.direction)};")
        if pm.is_wave:
            # the mode solved on this port's cross-section, read back by tag
            port_fun_lines.append(
                f"  ePort_{pm.number}[] = ComplexVectorField[XYZ[]]{{{pm.number}}};"
            )
            # A wave port terminates into its own mode, so the boundary term is
            # scaled by the modal wave admittance beta/k0, not by a sheet. It is
            # complex on a lossy line -- beta carries the attenuation -- and
            # GetDP's -setnumber only takes reals, so it arrives as two.
            # Yrel is a Function, not a bare constant: a bare assignment in
            # GetDP is evaluated at parse time and cannot hold a Complex[].
            port_fun_lines.append(
                f"  Yrel_{pm.number}[] = "
                f"Complex[NEFF_RE_{pm.number}, NEFF_IM_{pm.number}];  // beta/k0"
            )
            port_fun_lines.append(
                f"  VMODE_{pm.number} = Sqrt[2*ZC_{pm.number}];"
                f"  // voltage of the 1 W mode"
            )
        else:
            port_fun_lines.append(f"  ePort_{pm.number}[] = dir_{pm.number}[];")
            port_fun_lines.append(
                f"  Yrel_{pm.number}[] = eta0 / ({_fmt(pm.sheet_impedance)});"
                f"  // eta0/Zs, Zs=z0*w/gap"
            )
        # $ActivePort (not the ACTIVE_PORT constant) so the source can be
        # rebuilt per port inside one launch -- see the Resolution below.
        port_fun_lines.append(
            f"  eInc[Port_{pm.number}] = ($ActivePort == {pm.number}) ? "
            f"ePort_{pm.number}[] : Vector[0.,0.,0.];"
        )

    # formulation: port impedance sheet + source, per port. First Galerkin term
    # = a matched impedance sheet (absorbs the reflected wave into Z0); second
    # term = the impressed incident mode, non-zero only on the active port (the
    # factor 2 launches unit incident amplitude into the matched sheet).
    port_eq_lines = []
    for pm in ports:
        label = (
            f"wave port {pm.number}: modal impedance sheet (beta/k0) + modal source"
            if pm.is_wave
            else f"lumped port {pm.number}: resistive sheet (Z0={pm.z0}) + source"
        )
        port_eq_lines.append(
            f"""      // {label}
      Galerkin {{ [ -I[]*k0[]*Yrel_{pm.number}[]*(1/muR[]) * Normal[] /\\ (Normal[] /\\ Dof{{e}}) , {{e}} ] ;
        In Port_{pm.number} ; Integration I1 ; Jacobian Jac ; }}
      Galerkin {{ [ 2*I[]*k0[]*Yrel_{pm.number}[]*(1/muR[]) * Normal[] /\\ (Normal[] /\\ eInc[]) , {{e}} ] ;
        In Port_{pm.number} ; Integration I1 ; Jacobian Jac ; }}"""
        )

    # S-parameter post-quantities. S_nk is the overlap of the solved field with
    # port n's mode: on the driven port the incident mode is subtracted first so
    # the overlap is the *reflected* wave (S_kk); on the others it is the full
    # transmitted wave (S_nk). #(n) is the mode-normalisation stored above.
    # The quantity names carry only the observed port n -- the driven port is
    # the runtime $ActivePort, which cannot appear in a parse-time name.
    sparam_q = []
    for pm in ports:
        n = pm.number
        if pm.is_wave:
            # A solved mode is a vector field with a phase, so the overlap is
            # the full conjugated inner product rather than a projection on one
            # axis. Dividing by the mode's self-overlap makes S independent of
            # how the mode was normalised. V and I follow the lumped forms with
            # the gap replaced by the mode's own voltage.
            sparam_q.append(
                f"""        {{ Name intPort_{n} ;
          Value {{ Integral {{ [ ePort_{n}[] * Conj[ePort_{n}[]] ] ;
            In Port_{n} ; Jacobian Jac ; Integration I1 ; }} }} }}
        {{ Name xS_{n} ;
          Value {{ Integral {{ [ ({{e}} - (($ActivePort == {n}) ? ePort_{n}[] : Vector[0.,0.,0.])) * Conj[ePort_{n}[]] / #({n}) ] ;
            In Port_{n} ; Jacobian Jac ; Integration I1 ; }} }} }}
        {{ Name V_{n} ;
          Value {{ Integral {{ [ VMODE_{n} * ({{e}} * Conj[ePort_{n}[]]) / #({n}) ] ;
            In Port_{n} ; Jacobian Jac ; Integration I1 ; }} }} }}
        {{ Name I_{n} ;
          Value {{ Integral {{ [ (VMODE_{n}/ZC_{n}) * ((2*eInc[] - {{e}}) * Conj[ePort_{n}[]]) / #({n}) ] ;
            In Port_{n} ; Jacobian Jac ; Integration I1 ; }} }} }}"""
            )
            continue
        sparam_q.append(
            f"""        {{ Name intPort_{n} ;
          Value {{ Integral {{ [ (ePort_{n}[]*dir_{n}[]) * (ePort_{n}[]*dir_{n}[]) ] ;
            In Port_{n} ; Jacobian Jac ; Integration I1 ; }} }} }}
        {{ Name xS_{n} ;
          Value {{ Integral {{ [ (({{e}}*dir_{n}[]) - (($ActivePort == {n}) ? (ePort_{n}[]*dir_{n}[]) : 0)) * (ePort_{n}[]*dir_{n}[]) / #({n}) ] ;
            In Port_{n} ; Jacobian Jac ; Integration I1 ; }} }} }}
        {{ Name V_{n} ;
          Value {{ Integral {{ [ {_fmt(pm.gap)} * ({{e}}*dir_{n}[]) / #({n}) ] ;
            In Port_{n} ; Jacobian Jac ; Integration I1 ; }} }} }}
        {{ Name I_{n} ;
          Value {{ Integral {{ [ ({_fmt(pm.gap)}/{_fmt(pm.meshed_impedance)}) * (2*(eInc[]*dir_{n}[]) - ({{e}}*dir_{n}[])) / #({n}) ] ;
            In Port_{n} ; Jacobian Jac ; Integration I1 ; }} }} }}"""
        )

    # intPort is appended like everything else: it is written once per port per
    # solve and is diagnostic only -- xS/V/I read the value through the register
    # #(n), which StoreInRegister sets on the line above, not from the file.
    sparam_op = []
    for pm in ports:
        n = pm.number
        sparam_op.append(
            f"""      Print [ intPort_{n}[Port_{n}], OnRegion Port_{n}, StoreInRegister ({n}),
        Format Table, {xs_file} StrCat[myDir, "intPort.txt"] ];
      Print [ xS_{n}[Port_{n}], OnRegion Port_{n}, Format Table,
        {xs_file} StrCat[myDir, "xS_{n}.txt"] ];
      Print [ V_{n}[Port_{n}], OnRegion Port_{n}, Format Table,
        {xs_file} StrCat[myDir, "V_{n}.txt"] ];
      Print [ I_{n}[Port_{n}], OnRegion Port_{n}, Format Table,
        {xs_file} StrCat[myDir, "I_{n}.txt"] ];"""
        )

    # Assemble the .pro from the pieces built above. A GetDP problem file is a
    # fixed sequence of blocks, each filled in below:
    #   DefineConstant - runtime knobs (FREQ, ACTIVE_PORT, FEorder)
    #   Group          - named regions (dielectrics, air, PEC, ports, ABC/PML)
    #   Function       - material properties epsR/nuR, k0, port admittances, PML
    #   Constraint     - PEC walls force tangential E = 0
    #   FunctionSpace  - the edge-element space for E (+ optional 2nd order)
    #   Formulation    - the weak-form equation (curl-curl + ABC + port terms)
    #   Resolution     - assemble + solve the complex linear system
    #   PostProcessing/PostOperation - extract S-params / fields / power
    pro = f"""// Auto-generated by simpleEMS for '{problem.name}'.
// E-field formulation, lumped resistive-sheet ports (reference Z0), mode-overlap S-parameters.

DefineConstant[
  FREQ = {_fmt(f0)},          // Hz (override per solve with -setnumber FREQ <f>)
  ACTIVE_PORT = {active_default},   // driven port for AnalysisSinglePort only
  FEorder = {int(problem.fe_order)}   // 1 (lowest-order Nedelec) or 2 (add BF_Edge_2E){wave_consts}
];
NbPorts = {nports};
myDir = "output/";

eps0 = {_fmt(EPS0)};
mu0  = {_fmt(MU0)};
c0   = 1/Sqrt[eps0*mu0];
Z0   = Sqrt[mu0/eps0];
eta0 = Z0;

Group {{
{chr(10).join(diel_lines)}
  Air  = Region[{AIR}];
  Pec  = Region[{PEC}];
{chr(10).join(port_group_lines)}
{chr(10).join(imped_group_lines)}
{pml_group_line}
{sym_group_line}
  Abc  = Region[{ABC}];

  Ports  = Region[{{{ports_list}}}];
  BndPEC = Region[{{Pec{sym_in_pec}{abc_in_pec}}}];
  DomainDiel = Region[{{{diel_region_list}}}];
  Domain = Region[{{{diel_region_list}, Air{domain_pml}}}];
  BndAll = Region[{{Pec, {imped_extra}Ports, Abc}}];
  TotAll = Region[{{Domain, {imped_extra}Ports, Abc{sym_in_tot}}}];
}}

Function {{
  I[] = Complex[0., 1.];
{chr(10).join(epsr_lines)}
  epsR[Air] = 1.;
  muR[] = 1.;
{chr(10).join(mur_lines)}
{chr(10).join(nur_lines)}
  {k0_def}
{pml_fun}
{chr(10).join(imped_fun_lines)}
{chr(10).join(port_fun_lines)}
  eInc[Abc] = Vector[0.,0.,0.];
}}

Jacobian {{
  {{ Name Jac ; Case {{
    {{ Region BndAll ; Jacobian Sur ; }}
    {{ Region Domain ; Jacobian Vol ; }}
  }} }}
}}

Integration {{
  {{ Name I1 ; Case {{ {{ Type Gauss ; Case {{
    {{ GeoElement Point ; NumberOfPoints 1 ; }}
    {{ GeoElement Line ; NumberOfPoints 4 ; }}
    {{ GeoElement Triangle ; NumberOfPoints 7 ; }}
    {{ GeoElement Tetrahedron ; NumberOfPoints 15 ; }}
  }} }} }} }}
}}

Constraint {{
  {{ Name ElectricField ; Case {{ {{ Region BndPEC ; Value 0. ; }} }} }}
}}

FunctionSpace {{
  {{ Name eSpace ; Type Form1 ;
    BasisFunction {{
      {{ Name sn ; NameOfCoef en ; Function BF_Edge ; Support TotAll ; Entity EdgesOf[All] ; }}
      If (FEorder == 2)
        {{ Name sn2 ; NameOfCoef en2 ; Function BF_Edge_2E ; Support TotAll ; Entity EdgesOf[All] ; }}
      EndIf
    }}
    Constraint {{
      {{ NameOfCoef en ; EntityType EdgesOf ; NameOfConstraint ElectricField ; }}
      If (FEorder == 2)
        {{ NameOfCoef en2 ; EntityType EdgesOf ; NameOfConstraint ElectricField ; }}
      EndIf
    }}
  }}
}}

Formulation {{
  {{ Name eFormulation ; Type FemEquation ;
    Quantity {{ {{ Name e ; Type Local ; NameOfSpace eSpace ; }} }}
    Equation {{
      // curl-curl "stiffness" term: nuR * (curl E)·(curl w)  [d e = curl e]
      Galerkin {{ [ nuR[] * Dof{{d e}} , {{d e}} ] ;
        In Domain ; Integration I1 ; Jacobian Jac ; }}
      // "mass" term: -k0^2 * epsR * E·w  (together they are the wave equation)
      Galerkin {{ [ -k0[]^2 * epsR[] * Dof{{e}} , {{e}} ] ;
        In Domain ; Integration I1 ; Jacobian Jac ; }}

{abc_eq}

{chr(10).join(imped_eq_lines)}
{chr(10).join(port_eq_lines)}
    }}
  }}
}}

Resolution {{
  // All ports at one frequency, in a single launch: assemble + factorise once,
  // then only the right-hand side is rebuilt per port (see above).
  {{ Name Analysis ;
    System {{ {{ Name A ; NameOfFormulation eFormulation ; Type ComplexValue ; Frequency FREQ ; }} }}
    Operation {{
      {resolution_op}
    }}
  }}
  // One port only, driven by the ACTIVE_PORT constant, keeping the solution for
  // a follow-up -pos run. Used by fem_solver.solve_fields_and_power, which
  // wants the field views at a single driven port rather than an S-matrix.
  {{ Name AnalysisSinglePort ;
    System {{ {{ Name A ; NameOfFormulation eFormulation ; Type ComplexValue ; Frequency FREQ ; }} }}
    Operation {{
      CreateDir[Str[myDir]] ;
{single_reads}      Evaluate[ $ActivePort = ACTIVE_PORT ] ;
      Generate[A] ; Solve[A] ; SaveSolution[A] ;
    }}
  }}
}}

PostProcessing {{
  {{ Name postPro ; NameOfFormulation eFormulation ;
    Quantity {{
{chr(10).join(sparam_q)}
      {{ Name e ; Value {{ Local {{ [ {{e}} ] ; In Domain ; Jacobian Jac ; }} }} }}
      {{ Name h ; Value {{ Local {{ [ I[]*(1/muR[])*{{d e}}/(k0[]*eta0) ] ; In Domain ; Jacobian Jac ; }} }} }}
      // dielectric loss  P = (1/2) w eps0 Im[epsR] |E|^2  integrated over the volume.
      // Over DomainDiel, not Domain: in the PML epsR is a TensorDiag, so the
      // integrand there is a tensor rather than a scalar (and the PML's
      // absorption is not dielectric loss anyway). Air is lossless, so
      // dropping it too costs nothing.
      {{ Name Ploss ; Value {{ Integral {{ [ -Pi*{freqvar}*eps0*Im[epsR[]]*SquNorm[{{e}}] ] ;
        In DomainDiel ; Jacobian Jac ; Integration I1 ; }} }} }}
      // conductor loss, one quantity per distinct sheet conductivity
{chr(10).join(pcond_q)}
    }}
  }}
}}

PostOperation {{
  {{ Name Get_SParameters ; NameOfPostProcessing postPro ;
    Operation {{
{chr(10).join(sparam_op)}
    }}
  }}
  {{ Name Get_Fields ; NameOfPostProcessing postPro ;
    Operation {{
      Print [ e, OnElementsOf Domain, File StrCat[myDir, "e.pos"] ] ;
      Print [ h, OnElementsOf Domain, File StrCat[myDir, "h.pos"] ] ;
    }}
  }}
  {{ Name Get_Power ; NameOfPostProcessing postPro ;
    Operation {{
      Print [ Ploss[DomainDiel], OnGlobal, Format Table, {xs_file} StrCat[myDir, "Ploss.txt"] ] ;
{chr(10).join(pcond_op)}
{chr(10).join(pdrv_op)}
    }}
  }}
}}
"""
    pro_path = Path(workdir).absolute() / f"{problem.name}.pro"
    pro_path.write_text(pro)
    return str(pro_path)


def write_mode_problem(
    problem: "Problem",
    mesh: "Mesh",
    port_number: int,
    workdir: str | Path,
) -> str:
    """
    Write the 2D transverse mode problem for one wave port.

    The port's cross-section carries a guided mode
    ``E = [E_t(x,y) + z_hat E_z(x,y)] exp(-j*beta*z)``. Substituting
    ``e_t = beta*E_t`` and ``phi = j*E_z`` turns the vector wave equation into
    the generalised linear eigenproblem ``A_tt = beta^2 * C``, with

    .. math::

        A_{tt} &= \\int \\nu_r (\\nabla_t \\times w_t)(\\nabla_t \\times e_t)
                  - k_0^2 \\epsilon_r\\, w_t e_t \\\\
        C &= \\int \\nu_r (d\\Psi + \\hat z \\wedge w_t)(d\\Phi + \\hat z \\wedge e_t)
             - k_0^2 \\epsilon_r\\, \\Psi \\Phi

    ``A_tt`` goes in the default (``NoDt``) block and ``-C`` in the ``DtDtDof``
    block, so GetDP's ``GenerateSeparate`` + ``EigenSolve`` pair reports
    ``beta`` itself as the eigenvalue. The ``Form1`` + ``Form1P`` space pair
    follows onelab's ``models/BlochPeriodicWaveguides/rhombus.pro``.

    Parameters
    ----------
    problem : Problem
        The problem being solved, supplying the dielectric materials.
    mesh : Mesh
        The generated mesh, supplying the region tag of each dielectric.
    port_number : int
        One-based number of the port whose mode is solved.
    workdir : str | Path
        Directory to write the ``.pro`` file into.

    Returns
    -------
    str
        Path to the written ``.pro`` file.
    """
    # The cross-section reuses the 3D mesh's material tags, so the same epsR
    # assignment is written here as in write_problem. It is a separate mesh
    # file, so the tags cannot collide.
    diel_lines, epsr_lines, nur_lines = [], [], []
    for name, rid in sorted(mesh.dielectric_regions.items()):
        d = problem.solids[name].dielectric
        diel_lines.append(f"  Diel_{name} = Region[{rid}];")
        # The cross-section carries the same complex permittivity the 3D solve
        # uses, so the mode comes out with the propagation constant of the real,
        # lossy line: beta gains a negative imaginary part, which is the
        # attenuation. This is what Ansys' port solver does -- "the port solver
        # assumes that the wave port you define is connected to a waveguide that
        # has the same cross-section and material properties as the port", and
        # "wave ports calculate [...] complex propagation constant".
        epsr_lines.append(
            f"  epsR[Diel_{name}] = Complex[{_fmt(d.eps_r)}, "
            f"-{_fmt(d.eps_r * d.tan_d)}];"
        )
        nur_lines.append(f"  nuR[Diel_{name}] = {_fmt(1.0 / d.mu_r)};")
    nur_lines.append("  nuR[Air] = 1.;")
    diel_region_list = ", ".join(f"Diel_{n}" for n in sorted(mesh.dielectric_regions))
    xsec_list = f"{diel_region_list}, Air" if diel_region_list else "Air"

    pro = f"""// Auto-generated by simpleEMS: 2D transverse mode of port {port_number} of '{problem.name}'.
// Eigenproblem A_tt = beta^2 * C in the substituted unknowns e_t = beta*E_t, phi = j*E_z.
// GetDP reports the eigenvalue as `w`, which is beta directly; it is also written
// into each $Solution header of the .res file, which is what fem_port_mode reads.

DefineConstant[
  FREQ     = {_fmt(float(problem.freqs[0]))},   // Hz
  NMODES   = 6,      // eigenpairs to compute
  SHIFT_RE = 0.,     // spectral shift, targeted at beta^2 (NOT near 0: the
  SHIFT_IM = 0.      // gradient null space swamps a small shift)
];

myDir = "output/";

eps0 = {_fmt(EPS0)};
mu0  = {_fmt(MU0)};
c0   = 1/Sqrt[eps0*mu0];

Group {{
{chr(10).join(diel_lines)}
  Air  = Region[{AIR}];
  Xsec = Region[{{{xsec_list}}}];   // NB: 'Cross' is a reserved GetDP word
  Wall = Region[{PEC}];             // PEC edges: conductors cut by the port plane,
  Tot  = Region[{{Xsec, Wall}}];    // plus the outer boundary of the mode box
}}

Function {{
  I[]  = Complex[0., 1.];
  EZ[] = Vector[0., 0., 1.];
{chr(10).join(epsr_lines)}
  epsR[Air] = 1.;
{chr(10).join(nur_lines)}
  k0 = 2*Pi*FREQ/c0;
}}

Jacobian {{
  {{ Name Jac ; Case {{ {{ Region All ; Jacobian Vol ; }} }} }}
}}

Integration {{
  {{ Name I1 ; Case {{ {{ Type Gauss ; Case {{
    {{ GeoElement Point ; NumberOfPoints 1 ; }}
    {{ GeoElement Line ; NumberOfPoints 4 ; }}
    {{ GeoElement Triangle ; NumberOfPoints 7 ; }}
  }} }} }} }}
}}

Constraint {{
  {{ Name Et_wall ; Type Assign ; Case {{ {{ Region Wall ; Value 0. ; }} }} }}
  {{ Name Ez_wall ; Type Assign ; Case {{ {{ Region Wall ; Value 0. ; }} }} }}
}}

FunctionSpace {{
  // transverse field e_t: in-plane edge elements
  {{ Name Et_space ; Type Form1 ;
    BasisFunction {{
      {{ Name se ; NameOfCoef ee ; Function BF_Edge ; Support Tot ; Entity EdgesOf[All] ; }}
    }}
    Constraint {{ {{ NameOfCoef ee ; EntityType EdgesOf ; NameOfConstraint Et_wall ; }} }}
  }}
  // longitudinal field phi: perpendicular (out-of-plane) nodal elements
  {{ Name Ez_space ; Type Form1P ;
    BasisFunction {{
      {{ Name sn ; NameOfCoef en ; Function BF_PerpendicularEdge ; Support Tot ; Entity NodesOf[All] ; }}
    }}
    Constraint {{ {{ NameOfCoef en ; EntityType NodesOf ; NameOfConstraint Ez_wall ; }} }}
  }}
}}

Formulation {{
  {{ Name ModeForm ; Type FemEquation ;
    Quantity {{
      {{ Name et ; Type Local ; NameOfSpace Et_space ; }}
      {{ Name ez ; Type Local ; NameOfSpace Ez_space ; }}
    }}
    Equation {{
      // ---- A_tt : the NoDt block ----
      Galerkin {{ [ nuR[] * Dof{{d et}} , {{d et}} ] ;
        In Xsec ; Integration I1 ; Jacobian Jac ; }}
      Galerkin {{ [ -k0^2 * epsR[] * Dof{{et}} , {{et}} ] ;
        In Xsec ; Integration I1 ; Jacobian Jac ; }}

      // ---- -C : the eigenvalue (DtDtDof) block; the minus sign is what makes
      // GetDP's K + lambda^2 M report lambda = beta rather than j*beta ----
      Galerkin {{ DtDtDof [ -nuR[] * Dof{{d ez}} , {{d ez}} ] ;
        In Xsec ; Integration I1 ; Jacobian Jac ; }}
      Galerkin {{ DtDtDof [ -nuR[] * (EZ[] /\\ Dof{{et}}) , {{d ez}} ] ;
        In Xsec ; Integration I1 ; Jacobian Jac ; }}
      Galerkin {{ DtDtDof [ -nuR[] * Dof{{d ez}} , EZ[] /\\ {{et}} ] ;
        In Xsec ; Integration I1 ; Jacobian Jac ; }}
      Galerkin {{ DtDtDof [ -nuR[] * (EZ[] /\\ Dof{{et}}) , EZ[] /\\ {{et}} ] ;
        In Xsec ; Integration I1 ; Jacobian Jac ; }}
      Galerkin {{ DtDtDof [ k0^2 * epsR[] * Dof{{ez}} , {{ez}} ] ;
        In Xsec ; Integration I1 ; Jacobian Jac ; }}
    }}
  }}
}}

Resolution {{
  {{ Name ModeAnalysis ;
    System {{ {{ Name M ; NameOfFormulation ModeForm ; Type ComplexValue ; }} }}
    Operation {{
      CreateDir[Str[myDir]] ;
      GenerateSeparate[M] ;
      EigenSolve[M, NMODES, SHIFT_RE, SHIFT_IM] ;
      SaveSolutions[M] ;
    }}
  }}
}}

PostProcessing {{
  {{ Name postMode ; NameOfFormulation ModeForm ;
    Quantity {{
      {{ Name et ; Value {{ Local {{ [ {{et}} ] ; In Xsec ; Jacobian Jac ; }} }} }}
      {{ Name ez ; Value {{ Local {{ [ {{ez}} ] ; In Xsec ; Jacobian Jac ; }} }} }}
    }}
  }}
}}

PostOperation {{
  // Every eigenpair is written; GetDP stores a complex solution as a pair of
  // real/imag steps, so mode k is steps 2k and 2k+1. fem_port_mode selects.
  {{ Name Get_Mode ; NameOfPostProcessing postMode ;
    Operation {{
      // Format GmshParsed: with the Gmsh kernel linked in, a bare .pos comes
      // out mesh-based or parsed depending on what else the run touched, and
      // fem_port_mode.read_pos_steps reads the parsed form. Pin it.
      Print [ et, OnElementsOf Xsec, Format GmshParsed, File StrCat[myDir, "et_{port_number}.pos"] ] ;
    }}
  }}
}}
"""
    pro_path = Path(workdir).absolute() / f"{problem.name}_mode_{port_number}.pro"
    pro_path.write_text(pro)
    return str(pro_path)
