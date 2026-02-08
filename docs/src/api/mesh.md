# API: Mesh Utilities

## Purpose

Reference for mesh generation, OBJ IO, quality diagnostics, repair, and
coarsening.

---

## Creation and IO

### `make_rect_plate(Lx, Ly, Nx, Ny)`

Creates a triangulated rectangular plate in the `xy` plane centered at origin.

### `read_obj_mesh(path)`

Reads an OBJ file (`v`, `f`) and fan-triangulates polygon faces if needed.

### `write_obj_mesh(path, mesh; header="...")`

Writes `TriMesh` to OBJ using triangle faces.

---

## Geometry Helpers

- `triangle_area(mesh, t)`
- `triangle_center(mesh, t)`
- `triangle_normal(mesh, t)`
- `mesh_unique_edges(mesh)`
- `mesh_wireframe_segments(mesh)`

---

## Quality Checks

### `mesh_quality_report(mesh; area_tol_rel=1e-12, check_orientation=true)`

Returns a named tuple with counts of:

- invalid triangles,
- degenerate triangles,
- boundary edges,
- non-manifold edges,
- orientation conflicts.

### `mesh_quality_ok(report; allow_boundary=true, require_closed=false)`

Boolean gate from report values.

### `assert_mesh_quality(mesh; ...)`

Throws detailed error if mesh fails checks.

---

## Repair and Coarsening

### `repair_mesh_for_simulation(mesh; kwargs...)`

Pipeline for:

1. dropping invalid/degenerate triangles,
2. fixing orientation conflicts,
3. enforcing manifold constraints (optional strict mode),
4. re-checking mesh suitability.

Returns repaired mesh plus before/after diagnostics.

### `repair_obj_mesh(input_path, output_path; kwargs...)`

OBJ-level wrapper around `repair_mesh_for_simulation`.

### `coarsen_mesh_to_target_rwg(mesh, target_rwg; kwargs...)`

Voxel-cluster and repair loop to approach target RWG count with manifold-safe
cleanup.

### `cluster_mesh_vertices(mesh, h)`

Single voxel clustering pass.

### `drop_nonmanifold_triangles(mesh; max_passes=8)`

Iteratively removes triangles attached to non-manifold edges.

### `estimate_dense_matrix_gib(N)`

Estimates dense `ComplexF64` memory (`16*N^2` bytes) in GiB.

---

## Typical Safe Pipeline

```julia
mesh = read_obj_mesh("input.obj")
rep = repair_mesh_for_simulation(mesh; allow_boundary=true, strict_nonmanifold=true)
mesh_ok = rep.mesh
coarse = coarsen_mesh_to_target_rwg(mesh_ok, 400)
mesh_sim = coarse.mesh
```

---

## Notes

- Use `require_closed=true` for closed PEC bodies where boundary edges are not
  physically acceptable.
- Keep `allow_boundary=true` for open surfaces (e.g., plates).

---

## Code Mapping

- Full implementation: `src/Mesh.jl`
- Mesh tutorials: `examples/ex_repair_obj_mesh.jl`, `examples/ex_airplane_rcs.jl`

---

## Exercises

- Basic: run `mesh_quality_report` before/after repair on one OBJ.
- Challenge: quantify RWG count vs `estimate_dense_matrix_gib` during
  coarsening sweep.
