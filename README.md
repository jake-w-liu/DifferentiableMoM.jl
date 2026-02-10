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
julia --project=. examples/ex_radiationpatterns_adapter.jl
julia --project=. examples/ex_horn_pattern_import_demo.jl
julia --project=. examples/ex_horn_pattern_import_demo.jl examples/antenna_pattern.csv 28.0 reflector
```

For OBJ-based workflows:

```bash
julia --project=. examples/ex_obj_rcs_pipeline.jl
julia --project=. examples/ex_obj_rcs_pipeline.jl full input.obj [freq_GHz] [scale_to_m] [target_rwg] [tag]
julia --project=. examples/ex_obj_rcs_pipeline.jl full input.mat [freq_GHz] [scale_to_m] [target_rwg] [tag]
julia --project=. examples/ex_obj_rcs_pipeline.jl repair input.obj [output.obj] [scale_to_m]
julia --project=. examples/ex_obj_rcs_pipeline.jl repair input.mat [output.obj] [scale_to_m]
julia --project=. examples/ex_obj_rcs_pipeline.jl plot [data_dir] [out_dir] [tag]
julia --project=. examples/ex_visualize_simulation_mesh.jl [repaired.obj] [coarse.obj] [output_prefix]
```

`examples/demo_aircraft.obj` is a built-in moderate-detail aircraft mesh
(converted from a compact MAT source) so the workflow runs out-of-the-box
without large external assets. You can always pass your own OBJ, or pass a MAT
mesh (`coord` + `facet`) and let the pipeline convert it automatically.


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
- Metrics snapshot:
  ```bash
  julia --project=. validation/paper/generate_consistency_report.jl
  ```
- Bempp-cl cross-validation workflows:
  see `validation/bempp/README.md`.


## Repository Layout

- `src/` — solver, adjoint, optimization, verification, visualization
- `examples/` — runnable examples and demos
- `test/` — automated validation and gates
- `validation/` — external cross-validation scripts


## Citation

If this package contributes to your work, please cite:
- J. W. Liu, *DifferentiableMoM.jl: Open Differentiable EFIE-MoM Inverse-Design Pipeline*,
  GitHub repository, 2026.

## License

This project is released under the MIT License. See `LICENSE`.
