# DifferentiableMoM.jl

`DifferentiableMoM.jl` is a Julia package for differentiable EFIE-MoM analysis
and inverse design of reactive impedance metasurfaces.

The current scope targets:
- RWG-discretized EFIE forward solves,
- adjoint sensitivities for impedance parameters,
- beam-oriented far-field objectives and optimization,
- validation workflows (including Bempp-cl and sphere Mie benchmarks).

## Installation

### From a local checkout

```julia
import Pkg
Pkg.activate("path/to/DifferentiableMoM.jl")
Pkg.instantiate()
```

### Add by URL

```julia
import Pkg
Pkg.add(url="https://github.com/jake-w-liu/DifferentiableMoM.jl")
```

## Quick Start

From the package root:

```bash
julia --project=. test/runtests.jl
julia --project=. examples/ex_beam_steer.jl
julia --project=. examples/ex_auto_preconditioning.jl
julia --project=. examples/ex_pec_sphere_rcs.jl
julia --project=. examples/ex_pec_sphere_mie_benchmark.jl
```

For OBJ-based workflows:

```bash
julia --project=. examples/ex_repair_obj_mesh.jl input.obj [output.obj]
julia --project=. examples/ex_visualize_simulation_mesh.jl [repaired.obj] [coarse.obj] [output_prefix]
julia --project=. examples/ex_airplane_rcs.jl ../Airplane.obj 3.0 0.001 300
```

For an external sphere mesh:

```bash
julia --project=. examples/ex_pec_sphere_rcs.jl path/to/sphere.obj
julia --project=. examples/ex_pec_sphere_mie_benchmark.jl path/to/sphere.obj 3.0
```

## Minimal Usage

```julia
using DifferentiableMoM

mesh = make_rect_plate(0.1, 0.1, 6, 6)
rwg  = build_rwg(mesh)
k    = 2π / 0.1
Z    = assemble_Z_efie(mesh, rwg, k; quad_order=3)
```

## Key Features

- **Adjoint differentiation:** impedance gradients from explicit
  \(\partial Z / \partial \theta_p\) assembly.
- **Objective operators:** Hermitian PSD \(Q\)-matrix far-field objectives.
- **Optimization:** projected L-BFGS and ratio-objective two-adjoint workflow.
- **Conditioning options:** regularization and optional left preconditioning
  (`:off`, `:on`, `:auto`).
- **Mesh safety:** automatic manifoldness/degeneracy/orientation prechecks.
- **Mesh utilities:** OBJ import/repair/coarsening for practical simulation.
- **Visualization helpers:** mesh wireframe/comparison previews.

## Validation and Reproducibility

- Full regression and consistency suite:
  ```bash
  julia --project=. test/runtests.jl
  ```
- Paper-facing metrics snapshot:
  ```bash
  julia --project=. validation/paper/generate_consistency_report.jl
  ```
- Bempp-cl cross-validation workflows:
  see `validation/bempp/README.md`.

`data/` and `figs/` are generated artifacts and are intentionally git-ignored.

## Repository Layout

- `src/` — solver, adjoint, optimization, verification, visualization
- `examples/` — runnable examples and demos
- `test/` — automated validation and gates
- `validation/` — paper and external cross-validation scripts
- `data/` — generated numeric outputs (not tracked)
- `figs/` — generated figures (not tracked)

## Citation

If this package contributes to your work, please cite:
- J. W. Liu, *DifferentiableMoM.jl: Open Differentiable EFIE-MoM Inverse-Design Pipeline*,
  GitHub repository, 2026.
