# API Reference Overview

## Purpose

Provide a function-level map of `DifferentiableMoM.jl` so you can move from
concepts to implementation quickly.

---

## Import

```julia
using DifferentiableMoM
```

All documented symbols below are exported from the top-level module.

---

## API Map by Workflow

### 1) Geometry and Mesh Safety

- Types: `TriMesh`, `RWGData`, `PatchPartition`, `SphGrid`, `Vec3`, `CVec3`
- Helpers: `nvertices`, `ntriangles`
- Build/import: `make_rect_plate`, `make_parabolic_reflector`, `read_obj_mesh`, `write_obj_mesh`
- Geometry: `triangle_area`, `triangle_center`, `triangle_normal`,
  `mesh_unique_edges`, `mesh_wireframe_segments`
- Quality: `mesh_quality_report`, `mesh_quality_ok`, `assert_mesh_quality`
- Repair/coarsen: `repair_mesh_for_simulation`, `repair_obj_mesh`,
  `coarsen_mesh_to_target_rwg`, `cluster_mesh_vertices`,
  `drop_nonmanifold_triangles`
- Utilities: `estimate_dense_matrix_gib`

### 2) Basis and Assembly

- RWG: `build_rwg`, `eval_rwg`, `div_rwg`, `basis_triangles`
- Greens kernels: `greens`, `greens_smooth`, `grad_greens`
- Quadrature: `tri_quad_rule`, `tri_quad_points`
- Singular integration: `analytical_integral_1overR`, `self_cell_contribution`
- EFIE: `assemble_Z_efie`
- Impedance blocks: `precompute_patch_mass`, `assemble_Z_impedance`,
  `assemble_dZ_dtheta`, `assemble_full_Z`

### 3) Excitation, Solve, and Far Field

- **Excitation**: See detailed API in `excitation.md`
  and fundamentals derivation in
  `fundamentals/06-excitation-theory-and-usage.md`
  - `AbstractExcitation`, `PlaneWaveExcitation`, `PortExcitation`, `DeltaGapExcitation`,
    `DipoleExcitation`, `LoopExcitation`, `ImportedExcitation`,
    `PatternFeedExcitation`, `MultiExcitation`
  - `make_plane_wave`, `make_delta_gap`, `make_dipole`, `make_loop`,
  `make_imported_excitation`, `make_pattern_feed`,
  `make_analytic_dipole_pattern_feed`, `make_multi_excitation`
  - `plane_wave_field`, `pattern_feed_field`, `assemble_v_plane_wave`,
  `assemble_excitation`, `assemble_multiple_excitations`
- Practical pattern-import demos:
  `examples/ex_radiationpatterns_adapter.jl`,
  `examples/ex_horn_pattern_import_demo.jl`
- Linear solves: `solve_forward`, `solve_system`
- Far-field: `make_sph_grid`, `radiation_vectors`, `compute_farfield`
- Q/objective helpers: `build_Q`, `apply_Q`, `pol_linear_x`, `cap_mask`

### 4) Diagnostics and RCS

- Power/conditioning: `radiated_power`, `projected_power`, `input_power`,
  `energy_ratio`, `condition_diagnostics`
- RCS: `bistatic_rcs`, `backscatter_rcs`
- Analytical reference: `mie_s1s2_pec`, `mie_bistatic_rcs_pec`

### 5) Differentiable Optimization

- Adjoint pieces: `compute_objective`, `solve_adjoint`, `gradient_impedance`
- Optimizers: `optimize_lbfgs`, `optimize_directivity`
- Conditioning helpers:
  `make_mass_regularizer`, `make_left_preconditioner`,
  `select_preconditioner`, `transform_patch_matrices`,
  `prepare_conditioned_system`

### 6) Verification and Visualization

- Gradient checks: `complex_step_grad`, `fd_grad`, `verify_gradient`
- Mesh plots: `plot_mesh_wireframe`, `plot_mesh_comparison`,
  `save_mesh_preview`

---

## Recommended Reading Order

1. `types.md`
2. `mesh.md` and `rwg.md`
3. `assembly-solve.md`
4. `farfield-rcs.md`
5. `adjoint-optimize.md`
6. `verification.md`
7. `excitation.md` (new extended excitation system)

---

## Notes on Stability

- API names are stable for current tutorial/validation workflows.
- Some algorithmic internals may evolve; rely on exported functions rather than
  internal helper methods in `src/`.

---

## Exercises

- Basic: trace one example script and classify every called function into one
  of the workflow groups above.
- Challenge: replace a manual matrix solve in a script with
  `prepare_conditioned_system` + `solve_system` while preserving output.
