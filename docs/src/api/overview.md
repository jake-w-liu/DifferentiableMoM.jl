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

- Types: `TriMesh`, `PatchPartition`
- Build/import: `make_rect_plate`, `read_obj_mesh`, `write_obj_mesh`
- Quality: `mesh_quality_report`, `mesh_quality_ok`, `assert_mesh_quality`
- Repair/coarsen: `repair_mesh_for_simulation`, `repair_obj_mesh`,
  `coarsen_mesh_to_target_rwg`

### 2) Basis and Assembly

- RWG: `build_rwg`, `eval_rwg`, `div_rwg`, `basis_triangles`
- EFIE: `assemble_Z_efie`
- Impedance blocks: `precompute_patch_mass`, `assemble_Z_impedance`,
  `assemble_dZ_dtheta`, `assemble_full_Z`

### 3) Excitation, Solve, and Far Field

- Excitation: `assemble_v_plane_wave`
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
