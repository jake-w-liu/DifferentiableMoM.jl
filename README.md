# DifferentiableMoM.jl

`DifferentiableMoM.jl` is a Julia package for differentiable EFIE-MoM analysis
and inverse design of reactive impedance metasurfaces.

The current scope targets:
- RWG-discretized EFIE forward solves (dense, ACA H-matrix, MLFMA),
- adjoint sensitivities for impedance parameters,
- beam-oriented far-field objectives and multi-angle RCS optimization,
- spatial patch assignment for impedance design,
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
julia --project=. test/runtests.jl                     # run all 36 tests
julia --project=. examples/01_pec_plate_basics.jl       # basic EFIE on a plate
julia --project=. examples/02_impedance_optimization.jl # impedance design
julia --project=. examples/03_beamsteering_physical_unitcell.jl # paper beam-steering example
julia --project=. examples/04_pec_sphere_mie.jl         # PEC sphere Mie benchmark
julia --project=. examples/05_solver_methods.jl         # solver methods comparison
julia --project=. examples/05b_aca_scaling.jl           # ACA H-matrix scaling
julia --project=. examples/06_aircraft_rcs.jl           # aircraft RCS with OBJ mesh
julia --project=. examples/07_pattern_feed.jl           # pattern-feed excitation
julia --project=. examples/08_solve_scattering_workflow.jl  # high-level workflow API
julia --project=. examples/09_mom_vs_po.jl              # MoM vs physical optics
julia --project=. examples/10_mlfma_scaling.jl          # MLFMA scaling demo
julia --project=. examples/11_mlfma_finer.jl            # MLFMA with finer mesh
julia --project=. examples/12_plate_rcs_stl_roundtrip.jl    # STL mesh I/O roundtrip
julia --project=. examples/13_sphere_rcs_optimization.jl    # sphere RCS optimization
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

- **Multi-scale solvers:** dense LU/GMRES (small), ACA H-matrix (medium), MLFMA (large N).
- **Adjoint differentiation:** impedance gradients from explicit
  \(\partial Z / \partial \theta_p\) assembly.
- **Objective operators:** Hermitian PSD \(Q\)-matrix far-field objectives.
- **Optimization:** projected L-BFGS, multi-angle backscatter RCS optimization,
  and ratio-objective two-adjoint workflow.
- **Spatial patches:** grid, region-based, and uniform patch assignment for
  impedance design variables.
- **Conditioning options:** near-field sparse preconditioner (ILU),
  regularization, and optional left/right preconditioning.
- **Mesh safety:** automatic manifoldness/degeneracy/orientation prechecks.
- **Mesh I/O:** OBJ, STL, Gmsh MSH import/export with repair and coarsening.
- **Visualization helpers:** mesh wireframe/comparison previews.

## Validation and Reproducibility

### Regression tests (36 sequential tests)

```bash
julia --project=. test/runtests.jl
```

### Mie series — PEC sphere RCS

MoM vs analytical Mie series on an icosphere (phi=0 and phi=90 cuts). Saves
comparison tables to `data/mie_rcs_validation*.csv`.

```bash
julia --project=. validation/mie/validate_mie_rcs.jl
```

### Physical Optics — PO sanity check

Self-contained PO validation: compare the package `solve_po` result against a
MoM reference and the broadside flat-plate analytical formula. Saves summaries
to `data/po_vs_mom_validation*.csv`.

```bash
julia --project=. validation/po/validate_po_vs_pofacets.jl
```

### Bempp-cl cross-validation

Independent full-wave reference using `bempp-cl` (Python). PEC and
impedance-loaded far-field comparisons, convention sweep, operator-aligned
current/phase benchmarks. Requires Python 3.10+, `bempp-cl`, and Gmsh.

```bash
# Generate the Julia beam-steering baseline used by the PEC comparison
julia --project=. validation/paper/run_beam_steering_case.jl

# PEC cross-validation
python validation/bempp/run_pec_cross_validation.py
python validation/bempp/compare_pec_to_julia.py

# Impedance validation matrix (6-case beam-centric acceptance)
python validation/bempp/run_impedance_validation_matrix.py

# Impedance convention sweep
python validation/bempp/sweep_impedance_conventions.py --run-julia

# Operator-aligned benchmark (current + phase + residuals)
python validation/bempp/run_impedance_operator_aligned_benchmark.py \
  --freq-ghz 3.0 --zs-imag-ohm 200 --mesh-mode structured --nx 12 --ny 12 \
  --output-prefix opalign_z200
```

See `validation/bempp/README.md` for full usage and convention override options.

### Cost scaling

Assembly and solve wall-time vs N for dense, ACA, and MLFMA methods.

```bash
julia --project=. validation/scaling/run_cost_scaling.jl
```

### Robustness sweep

Optimization robustness across frequency and incidence-angle perturbations.
Requires `validation/paper/run_beam_steering_case.jl` first.

```bash
julia --project=. validation/robustness/run_robustness_sweep.jl
```

### Paper consistency report

Snapshot of key metrics for manuscript reproducibility. Before running the
report, generate the paper data drivers and Bempp comparison artifacts:

```bash
julia --project=. validation/paper/run_convergence_study.jl
julia --project=. validation/paper/run_beam_steering_case.jl
julia --project=. validation/scaling/run_cost_scaling.jl
julia --project=. validation/robustness/run_robustness_sweep.jl
python validation/bempp/run_pec_cross_validation.py
python validation/bempp/compare_pec_to_julia.py
python validation/bempp/run_impedance_validation_matrix.py
julia --project=. validation/paper/generate_consistency_report.jl
```


## Repository Layout

- `src/` — solver, adjoint, optimization, fast methods, verification, visualization (30 source files)
- `examples/` — numbered runnable examples (`01_`–`23_`) and auxiliary assets
- `test/` — 36 sequential regression tests
- `validation/` — analytical, paper, and external cross-validation workflows
- `data/` — test-generated CSV files for validation


## Citation

If this package contributes to your work, please cite:
- J. W. Liu, *DifferentiableMoM.jl: Open Differentiable EFIE-MoM Inverse-Design Pipeline*,
  GitHub repository, 2026.

## License

This project is released under the MIT License. See `LICENSE`.
