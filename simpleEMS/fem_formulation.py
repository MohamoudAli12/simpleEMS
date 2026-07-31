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
Generate a self-contained GetDP ``.pro`` for the E-field microwave formulation.

This module is a *code generator*: :func:`write_problem` assembles a text
``.pro`` file that tells GetDP exactly which finite-element problem to solve on
the mesh from :mod:`~simpleEMS.fem_geometry`. Nothing is solved here.

The physics it emits:

* **Unknown / discretisation** -- the electric field ``E`` expanded in
  first-order Nedelec *edge* elements (GetDP ``Type Form1``, ``BF_Edge``; the
  ``FEorder == 2`` branch adds ``BF_Edge_2E`` for 2nd order). Edge elements are
  the standard choice for the vector wave equation (they allow tangential jumps
  and kill spurious modes).
* **Equation** -- the time-harmonic curl-curl wave equation
  ``curl(nuR curl E) - k0^2 epsR E = 0`` in weak (Galerkin) form, complex-valued,
  where ``k0 = w/c0``, ``nuR = 1/muR``. GetDP uses the ``exp(-i w t)`` sign
  convention, so a lossy dielectric has ``epsR = eps_r*(1 + i*tan_d)`` (positive
  imaginary part) -- note this is conjugated back to the engineering convention
  in :func:`~simpleEMS.fem_backend.compute_sim_data`.
* **Truncation** -- a first-order Silver-Muller absorbing boundary on the outer
  air box, or a complex-coordinate-stretched PML shell.
* **Excitation / ports** -- each port sheet carries an impedance term
  (``eta0/Zs``) plus an impressed quasi-TEM mode source; S-parameters come from
  the mode overlap of the solved field on each port.

The driven port is selected by the *runtime* GetDP variable ``$ActivePort``
rather than a parse-time constant, which is what lets the ``Analysis``
resolution walk every port inside a single GetDP launch: only the source term
changes between ports, so the system matrix is assembled and factorised once
and each subsequent port needs nothing but a fresh right-hand side
(``GenerateRightHandSideGroup`` on the port sheets) and a ``SolveAgain``.
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
    Write the GetDP ``.pro`` problem file for ``problem`` on ``mesh``.

    Parameters
    ----------
    problem : Problem
        The FEM problem definition.
    mesh : Mesh
        The generated mesh with its region maps.
    workdir : str | Path
        Directory to write the ``.pro`` file into.

    Returns
    -------
    str
        Absolute path to the written ``.pro`` file.
    """
    ports = [mesh.port_regions[p.number] for p in problem.ports]
    nports = len(ports)
    port_numbers = [pm.number for pm in ports]
    active_default = port_numbers[0] if port_numbers else 1
    f0 = float(problem.freqs[0])

    k0_def = "k0[] = 2*Pi*FREQ / c0;"
    freqvar = "FREQ"

    # --- port loop inside one GetDP launch ------------------------------------
    # Driving port k only changes the source term, which lives entirely on the
    # port sheets: the system matrix is identical for every port at a given
    # frequency. So assemble and factorise once for the first port, then for
    # each remaining port rebuild *only* the right-hand side over the Ports
    # group and re-solve against the existing factorisation.
    #   GenerateRightHandSideGroup[A, Ports]  leaves A untouched (it sets
    #   GetDP's Flag_RHS, which skips the matrix zero/rebuild) and reassembles b
    #   from the terms supported on Ports;
    #   SolveAgain[A]                         reuses the preconditioner, i.e.
    #   the LU factorisation for the default direct solver.
    # This is what makes the driven port a runtime `$ActivePort` rather than a
    # parse-time constant -- a constant could not change between Generate calls.
    resolution_lines = ["CreateDir[Str[myDir]] ;"]
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

    # Every scalar (Format Table) result is appended, not overwritten, so each
    # port of each solve point adds one row. Within a launch the rows follow the
    # Resolution's port order, so the last `nports` rows of xS_<n>.txt are that
    # frequency's S[n, :] (see fem_solver.read_complex_rows).
    # fem_backend._sweep_from_meta clears stale files from any previous run
    # before a sweep starts, so rows never mix across runs.
    xs_file = "File >"

    diel_lines = []
    epsr_lines = []
    mur_lines = []  # muR overrides for magnetic dielectrics (default 1)
    nur_lines = []  # nuR = 1/muR per dielectric (+ Air below)
    for name, rid in sorted(mesh.dielectric_regions.items()):
        spec = problem.solids[name]
        d = spec.dielectric
        diel_lines.append(f"  Diel_{name} = Region[{rid}];")
        # GetDP uses the exp(-i w t) convention, so a passive lossy dielectric
        # has Im[eps]>0: eps_r_complex = eps_r * (1 + i*tan_d)
        epsr_lines.append(
            f"  epsR[Diel_{name}] = Complex[{_fmt(d.eps_r)}, "
            f"{_fmt(d.eps_r * d.tan_d)}];"
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

    # lossy conductors: surface-impedance (Leontovich) boundary, Zs = Rs*(1-i),
    # Rs = sqrt(pi*f*mu0/sigma) (frequency dependent, hence a Function of freqvar).
    imped_group_lines, imped_fun_lines, imped_eq_lines, imped_names = [], [], [], []
    for j, (rid, sig) in enumerate(mesh.impedance_regions):
        imped_group_lines.append(f"  Imped_{j} = Region[{rid}];")
        imped_names.append(f"Imped_{j}")
        imped_fun_lines.append(f"  Rs_{j}[] = Sqrt[Pi*{freqvar}*mu0/({_fmt(sig)})];")
        imped_fun_lines.append(
            f"  Yimp_{j}[] = eta0 / (Rs_{j}[]*(1 - I[]));  // eta0/Zs, Zs=Rs*(1-i)"
        )
        imped_eq_lines.append(
            f"""      // lossy conductor {j}: surface impedance (sigma={sig:g} S/m)
      Galerkin {{ [ I[]*k0[]*Yimp_{j}[]*(1/muR[]) * Normal[] /\\ (Normal[] /\\ Dof{{e}}) , {{e}} ] ;
        In Imped_{j} ; Integration I1 ; Jacobian Jac ; }}"""
        )
    imped_extra = (", ".join(imped_names) + ", ") if imped_names else ""

    # symmetry plane: PEC (constrain E_tan=0) or PMC (natural, excluded from ABC)
    sym = getattr(mesh, "sym_region", 0)
    sym_group_line = f"  Sym = Region[{sym}];" if sym else ""
    sym_in_pec = ", Sym" if (sym and mesh.sym_kind == "pec") else ""
    sym_in_tot = ", Sym" if sym else ""

    # PML: complex coordinate stretching in the outer shell. epsR and nuR (=1/muR)
    # become anisotropic tensors there; cX/cY/cZ = 1 + i*Damp/k0.
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
            "  cX[] = Complex[1, DampX[]/k0[]];\n"
            "  cY[] = Complex[1, DampY[]/k0[]];\n"
            "  cZ[] = Complex[1, DampZ[]/k0[]];\n"
            "  epsR[Pml] = TensorDiag[ cY[]*cZ[]/cX[], cX[]*cZ[]/cY[], cX[]*cY[]/cZ[] ];\n"
            "  nuR[Pml]  = TensorDiag[ cX[]/(cY[]*cZ[]), cY[]/(cX[]*cZ[]), cZ[]/(cX[]*cY[]) ];"
        )
    else:
        pml_group_line = ""
        domain_pml = ""
        pml_fun = ""

    # per-port functions: unit mode direction, sheet relative admittance, incident field
    port_fun_lines = []
    for pm in ports:
        zs = pm.sheet_impedance  # wave: Zc; lumped: z0*w/gap
        kind = "Zc(wave)" if pm.port_type == "wave" else "z0*w/gap(lumped)"
        port_fun_lines.append(f"  dir_{pm.number}[] = {_dir_vector(pm.direction)};")
        port_fun_lines.append(f"  ePort_{pm.number}[] = dir_{pm.number}[];")
        port_fun_lines.append(
            f"  Yrel_{pm.number} = eta0 / ({_fmt(zs)});  // eta0/Zs, Zs={kind}"
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
        port_eq_lines.append(
            f"""      // lumped port {pm.number}: resistive sheet (Z0={pm.z0}) + source
      Galerkin {{ [ I[]*k0[]*Yrel_{pm.number}*(1/muR[]) * Normal[] /\\ (Normal[] /\\ Dof{{e}}) , {{e}} ] ;
        In Port_{pm.number} ; Integration I1 ; Jacobian Jac ; }}
      Galerkin {{ [ -2*I[]*k0[]*Yrel_{pm.number}*(1/muR[]) * Normal[] /\\ (Normal[] /\\ eInc[]) , {{e}} ] ;
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
          Value {{ Integral {{ [ ({_fmt(pm.gap)}/{_fmt(pm.ref_impedance)}) * (2*(eInc[]*dir_{n}[]) - ({{e}}*dir_{n}[])) / #({n}) ] ;
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
  FEorder = {int(problem.fe_order)}   // 1 (lowest-order Nedelec) or 2 (add BF_Edge_2E)
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
  BndPEC = Region[{{Pec{sym_in_pec}}}];
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

      // outer free-space Silver-Muller ABC (first-order outgoing-wave boundary)
      Galerkin {{ [ I[]*k0[] * (1/muR[]) * Normal[] /\\ ( Normal[] /\\ Dof{{e}} ) , {{e}} ] ;
        In Abc ; Integration I1 ; Jacobian Jac ; }}

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
      Evaluate[ $ActivePort = ACTIVE_PORT ] ;
      Generate[A] ; Solve[A] ; SaveSolution[A] ;
    }}
  }}
}}

PostProcessing {{
  {{ Name postPro ; NameOfFormulation eFormulation ;
    Quantity {{
{chr(10).join(sparam_q)}
      {{ Name e ; Value {{ Local {{ [ {{e}} ] ; In Domain ; Jacobian Jac ; }} }} }}
      {{ Name h ; Value {{ Local {{ [ -I[]*(1/muR[])*{{d e}}/(k0[]*eta0) ] ; In Domain ; Jacobian Jac ; }} }} }}
      // dielectric loss  P = (1/2) w eps0 Im[epsR] |E|^2  integrated over the volume.
      // Over DomainDiel, not Domain: in the PML epsR is a TensorDiag, so the
      // integrand there is a tensor rather than a scalar (and the PML's
      // absorption is not dielectric loss anyway). Air is lossless, so
      // dropping it too costs nothing.
      {{ Name Ploss ; Value {{ Integral {{ [ Pi*{freqvar}*eps0*Im[epsR[]]*SquNorm[{{e}}] ] ;
        In DomainDiel ; Jacobian Jac ; Integration I1 ; }} }} }}
      // radiated power = power absorbed by the matched Silver-Muller ABC
      {{ Name Prad ; Value {{ Integral {{ [ 0.5/eta0 * SquNorm[ (Normal[] /\\ {{e}}) /\\ Normal[] ] ] ;
        In Abc ; Jacobian Jac ; Integration I1 ; }} }} }}
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
      Print [ Prad[Abc], OnGlobal, Format Table, {xs_file} StrCat[myDir, "Prad.txt"] ] ;
    }}
  }}
}}
"""
    pro_path = Path(workdir).absolute() / f"{problem.name}.pro"
    pro_path.write_text(pro)
    return str(pro_path)
