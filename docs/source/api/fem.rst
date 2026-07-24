FEM Backend (GetDP)
===================

The FEM backend solves the same structures as the default openEMS FDTD engine
using a Gmsh + GetDP finite-element frequency-domain method. It is selected by
setting ``backend_engine="FEM"`` on the simulation parameters; the pipeline
(``write_and_show_structure`` / ``run_simulation`` / ``compute_sim_data``) then
routes automatically to these modules and still returns a ``SimData``.

The ``getdp`` binary must be on ``PATH``. Run ``simpleems install getdp`` to
install it.

.. automodapi:: simpleEMS.fem_backend
   :no-inheritance-diagram:

.. automodapi:: simpleEMS.fem_geometry
   :no-inheritance-diagram:

.. automodapi:: simpleEMS.fem_formulation
   :no-inheritance-diagram:

.. automodapi:: simpleEMS.fem_solver
   :no-inheritance-diagram:

.. automodapi:: simpleEMS.fem_sweep
   :no-inheritance-diagram:

.. automodapi:: simpleEMS.fem_materials
   :no-inheritance-diagram:

.. automodapi:: simpleEMS.fem_radiation
   :no-inheritance-diagram:
