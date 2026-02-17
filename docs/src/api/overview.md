# API Reference Overview

## Purpose

This page provides a complete function-level map of `DifferentiableMoM.jl`, organized by workflow stage. Use it to locate the right function for your task, then follow the cross-references to detailed documentation.

---

## Import

```julia
using DifferentiableMoM
```

All symbols listed below are exported from the top-level module. No additional `using` or `import` statements are needed for normal usage.

---

## API Map by Workflow

The typical simulation pipeline follows these stages in order. Each stage builds on the outputs of the previous one.

### 1) Geometry and Mesh Safety

Define or import the scatterer geometry and ensure the mesh is suitable for MoM simulation.

- **Types:** `TriMesh`, `RWGData`, `PatchPartition`, `SphGrid`, `ScatteringResult`, `Vec3`, `CVec3`, `AbstractPreconditionerData`, `NearFieldPreconditionerData`, `DiagonalPreconditionerData`
  Core data structures used throughout the package. See [types.md](types.md) for field-level documentation.

- **Helpers:** `nvertices`, `ntriangles`
  Quick queries on mesh size.

- **Build/import:** `make_rect_plate`, `make_parabolic_reflector`, `read_obj_mesh`, `write_obj_mesh`
  Create meshes programmatically or load from Wavefront OBJ files.

- **Geometry queries:** `triangle_area`, `triangle_center`, `triangle_normal`, `mesh_unique_edges`, `mesh_wireframe_segments`
  Per-triangle geometry and edge extraction.

- **Quality diagnostics:** `mesh_quality_report`, `mesh_quality_ok`, `assert_mesh_quality`
  Check for degenerate triangles, non-manifold edges, and orientation conflicts before assembly.

- **Repair and coarsening:** `repair_mesh_for_simulation`, `repair_obj_mesh`, `coarsen_mesh_to_target_rwg`, `cluster_mesh_vertices`, `drop_nonmanifold_triangles`
  Fix problematic meshes (from CAD exports, etc.) and reduce mesh density to a target RWG count.

- **Resolution diagnostics:** `mesh_resolution_report`, `mesh_resolution_ok`
  Check whether mesh edge lengths satisfy a frequency-based lambda/N criterion. See [mesh.md](mesh.md).

- **Mesh refinement:** `refine_mesh_to_target_edge`, `refine_mesh_for_mom`
  Uniform midpoint subdivision to meet a target maximum edge length or lambda/N criterion. See [mesh.md](mesh.md).

- **Utilities:** `estimate_dense_matrix_gib`
  Estimate memory cost of the dense MoM matrix before assembly.

### 2) Basis Functions and System Assembly

Construct RWG basis functions on the mesh, then assemble the EFIE system matrix and optional impedance loading.

- **RWG basis:** `build_rwg`, `eval_rwg`, `div_rwg`, `basis_triangles`
  Build edge-based Rao-Wilton-Glisson basis functions and evaluate them at any point. See [rwg.md](rwg.md).

- **Green's function kernels:** `greens`, `greens_smooth`, `grad_greens`
  Free-space scalar Green's function and its smooth/gradient variants. Used internally by EFIE assembly.

- **Quadrature:** `tri_quad_rule`, `tri_quad_points`
  Gaussian quadrature rules on the reference triangle.

- **Singular integration:** `analytical_integral_1overR`, `self_cell_contribution`
  Analytical and hybrid singular integrals for self-cell terms (when source and test triangles overlap).

- **EFIE assembly:** `assemble_Z_efie`
  Build the dense N x N EFIE impedance matrix. This is the core MoM system matrix for PEC scatterers.

- **Matrix-free EFIE operators:** `matrixfree_efie_operator`, `matrixfree_efie_adjoint_operator`, `efie_entry`
  Matrix-free EFIE matvec without dense N x N allocation. Types: `MatrixFreeEFIEOperator`, `MatrixFreeEFIEAdjointOperator`. See [assembly-solve.md](assembly-solve.md) and [types.md](types.md).

- **Impedance loading:** `precompute_patch_mass`, `assemble_Z_impedance`, `assemble_dZ_dtheta`, `assemble_full_Z`
  Add surface impedance loading for design optimization. See [assembly-solve.md](assembly-solve.md).

- **Composite operator:** `ImpedanceLoadedOperator`
  Matrix-free operator wrapping any `AbstractMatrix{ComplexF64}` base (MLFMA, ACA, dense) with sparse impedance perturbation `Z(theta) = Z_base - Sigma_p theta_p M_p`. Enables GMRES-based optimization with fast operators. See [composite-operators.md](composite-operators.md).

- **Spatial patch assignment:** `assign_patches_grid`, `assign_patches_by_region`, `assign_patches_uniform`, `region_halfspace`, `region_sphere`, `region_box`
  Automatic spatial partitioning of mesh triangles into impedance design patches. See [spatial-patches.md](spatial-patches.md).

### 3) Excitation, Solve, and Far Field

Apply an incident field, solve for currents, and compute far-field radiation or scattering patterns.

- **Excitation sources:** See [excitation.md](excitation.md) for the full excitation system.
  - Types: `AbstractExcitation`, `PlaneWaveExcitation`, `PortExcitation`, `DeltaGapExcitation`, `DipoleExcitation`, `LoopExcitation`, `ImportedExcitation`, `PatternFeedExcitation`, `MultiExcitation`
  - Constructors: `make_plane_wave`, `make_delta_gap`, `make_dipole`, `make_loop`, `make_imported_excitation`, `make_pattern_feed`, `make_analytic_dipole_pattern_feed`, `make_multi_excitation`
  - Assembly: `plane_wave_field`, `pattern_feed_field`, `assemble_v_plane_wave`, `assemble_excitation`, `assemble_multiple_excitations`
  - Example scripts: `examples/ex_radiationpatterns_adapter.jl`, `examples/ex_horn_pattern_import_demo.jl`

- **Linear solves (direct):** `solve_forward`, `solve_system`
  Solve the MoM system `Z I = v` using LU factorization (default) or GMRES.

- **Iterative solves (GMRES):** `solve_gmres`, `solve_gmres_adjoint`
  GMRES via Krylov.jl with optional near-field preconditioning. Use these for large problems where direct LU is too slow or memory-intensive.

- **Near-field preconditioner:** `build_nearfield_preconditioner`, `rwg_centers`
  Build a sparse near-field preconditioner that dramatically reduces GMRES iteration counts. Multiple overloads: from dense matrix, abstract matrix, matrix-free operator, or geometry/physics inputs directly. Supports sparse LU (`:lu`) or Jacobi diagonal (`:diag`) factorization. See [assembly-solve.md](assembly-solve.md) for details and performance data.
  - Types: `AbstractPreconditionerData`, `NearFieldPreconditionerData`, `DiagonalPreconditionerData`, `NearFieldOperator`, `NearFieldAdjointOperator`

- **Far-field computation:** `make_sph_grid`, `radiation_vectors`, `compute_farfield`
  Sample the far-field radiation pattern on a spherical grid.

- **Objective (Q-matrix) helpers:** `build_Q`, `apply_Q`, `pol_linear_x`, `cap_mask`, `direction_mask`
  Build Hermitian PSD matrices for quadratic far-field objectives used in optimization. `direction_mask` generalizes `cap_mask` to arbitrary directions for multi-angle RCS optimization.

### 3b) ACA H-Matrix and High-Level Workflow

For large problems (N > ~2000), ACA compression and the `solve_scattering` workflow automate method selection and preconditioner construction.

- **Cluster tree:** `build_cluster_tree`, `cluster_diameter`, `cluster_distance`, `is_admissible`, `is_leaf`, `leaf_nodes`
  Binary space-partitioning tree for H-matrix block structure. Types: `ClusterNode`, `ClusterTree`. See [aca-workflow.md](aca-workflow.md).

- **ACA low-rank approximation:** `aca_lowrank`, `build_aca_operator`
  Partially-pivoted ACA for far-field block compression. Types: `ACAOperator`, `ACAAdjointOperator`, `DenseBlock` (internal), `LowRankBlock` (internal). See [aca-workflow.md](aca-workflow.md).

- **High-level workflow:** `solve_scattering`
  One-call scattering solve with automatic method selection (dense direct, dense GMRES, or ACA GMRES) based on problem size. Returns `ScatteringResult`. See [aca-workflow.md](aca-workflow.md).

### 4) Diagnostics and RCS

Validate simulation results and compute scattering cross sections.

- **Power and conditioning:** `radiated_power`, `projected_power`, `input_power`, `energy_ratio`, `condition_diagnostics`
  Energy-balance checks and matrix conditioning analysis.

- **Radar cross section:** `bistatic_rcs`, `backscatter_rcs`
  Compute bistatic and monostatic RCS from far-field data.

- **Analytical reference:** `mie_s1s2_pec`, `mie_bistatic_rcs_pec`
  Mie series for PEC sphere scattering; use as a validation reference for your MoM results.

### 4b) Physical Optics

High-frequency approximate solver for electrically large problems where full MoM is too expensive.

- **Physical optics solve:** `solve_po`
  Compute PO surface currents and far-field scattering using the tangential magnetic field approximation on illuminated faces. Returns `POResult`. See `src/postprocessing/PhysicalOptics.jl`.

- **Types:** `POResult`
  Result container for PO solutions, analogous to `ScatteringResult` for MoM.

### 5) Differentiable Optimization

Compute gradients via the adjoint method and run impedance optimization.

- **Adjoint primitives:** `compute_objective`, `solve_adjoint`, `solve_adjoint_rhs`, `gradient_impedance`
  The building blocks: evaluate the quadratic objective, solve the adjoint system, and compute the impedance gradient. `solve_adjoint_rhs` accepts a pre-computed RHS for matrix-free Q application or multi-angle objectives. See [adjoint-optimize.md](adjoint-optimize.md).

- **Single-objective optimizers:** `optimize_lbfgs`, `optimize_directivity`
  Projected L-BFGS with box constraints. `optimize_lbfgs` minimizes/maximizes a single quadratic objective; `optimize_directivity` maximizes the ratio of two quadratic objectives (directivity).

- **Multi-angle optimizer:** `optimize_multiangle_rcs`, `build_multiangle_configs`, `AngleConfig`
  Minimize weighted backscatter RCS over multiple incidence angles simultaneously. Supports MLFMA, ACA, and dense base operators via `ImpedanceLoadedOperator`. See [adjoint-optimize.md](adjoint-optimize.md) and the [Multi-Angle RCS chapter](../differentiable-design/05-multiangle-rcs.md).

- **Conditioning helpers:** `make_mass_regularizer`, `make_left_preconditioner`, `select_preconditioner`, `transform_patch_matrices`, `prepare_conditioned_system`
  Advanced: mass-based preconditioning and regularization for ill-conditioned optimization problems.

### 6) Verification and Visualization

Check gradient correctness and visualize meshes.

- **Gradient verification:** `complex_step_grad`, `fd_grad`, `verify_gradient`
  Compare adjoint gradients against complex-step and finite-difference references. Essential for validating new objective functions or modified adjoint code. See [verification.md](verification.md).

- **Mesh visualization:** `plot_mesh_wireframe`, `plot_mesh_comparison`, `save_mesh_preview`
  Lightweight 3D wireframe plots for mesh inspection. See [visualization.md](visualization.md).

---

## Recommended Reading Order

For a first read-through of the API documentation, follow this order:

1. **[types.md](types.md)** — Core data structures (`TriMesh`, `RWGData`, `SphGrid`, `ScatteringResult`, matrix-free operators, etc.)
2. **[mesh.md](mesh.md)** and **[rwg.md](rwg.md)** — Geometry creation, mesh quality, resolution diagnostics, refinement, and RWG basis construction
3. **[assembly-solve.md](assembly-solve.md)** — EFIE assembly (dense and matrix-free), impedance loading, direct/GMRES solvers, and near-field preconditioning
4. **[aca-workflow.md](aca-workflow.md)** — ACA H-matrix compression, cluster trees, and the `solve_scattering` high-level workflow
5. **[farfield-rcs.md](farfield-rcs.md)** — Far-field patterns, Q-matrices, `direction_mask`, RCS, and Mie validation
6. **[composite-operators.md](composite-operators.md)** — `ImpedanceLoadedOperator` for fast-operator optimization
7. **[spatial-patches.md](spatial-patches.md)** — Automatic spatial patch assignment
8. **[adjoint-optimize.md](adjoint-optimize.md)** — Adjoint gradients, L-BFGS optimization, and multi-angle RCS
7. **[verification.md](verification.md)** — Gradient correctness checks
8. **[excitation.md](excitation.md)** — Extended excitation system (plane waves, ports, dipoles, imported fields, pattern feeds)

---

## Notes on Stability

- All exported function names and signatures listed here are stable for current tutorial and validation workflows.
- Internal helper methods in `src/` may evolve; rely on the exported API for forward compatibility.

---

## Exercises

- **Basic:** Trace one example script (e.g., `examples/ex_plate_rwg.jl`) and classify every called function into one of the workflow groups above.
- **Challenge:** Replace a manual matrix solve (`Z \ v`) in a script with `prepare_conditioned_system` + `solve_system` while preserving the output.
