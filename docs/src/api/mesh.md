# API: Mesh Utilities

## Purpose

Reference for mesh generation, OBJ IO, quality diagnostics, repair, and
coarsening.

---

## Creation and IO

### `make_rect_plate(Lx, Ly, Nx, Ny)`

Creates a triangulated rectangular plate in the `xy` plane centered at origin.

**Parameters:**
- `Lx::Real`: plate length in x‑direction (meters)
- `Ly::Real`: plate length in y‑direction (meters)
- `Nx::Int`: number of cells along x (≥1)
- `Ny::Int`: number of cells along y (≥1)

**Returns:** `TriMesh` with `(Nx+1)*(Ny+1)` vertices and `2*Nx*Ny` triangles.

---

### `make_parabolic_reflector(D, f, Nr, Nphi; center=Vec3(0,0,0))`

Creates an open parabolic reflector mesh aligned with `+z`, with surface

```math
z = \frac{x^2 + y^2}{4f}, \qquad x^2 + y^2 \le (D/2)^2.
```

**Parameters:**
- `D::Real`: aperture diameter (meters)
- `f::Real`: focal length (meters)
- `Nr::Int`: number of radial rings (≥2)
- `Nphi::Int`: number of azimuth samples per ring (≥3)
- `center::Vec3=Vec3(0,0,0)`: reflector apex location

**Returns:** `TriMesh` with `1 + Nr*Nphi` vertices and
`Nphi + 2*(Nr-1)*Nphi` triangles.

Use `allow_boundary=true` when checking/open-surface solving.

**Example:**
```julia
mesh = make_parabolic_reflector(0.30, 0.105, 8, 28)
report = assert_mesh_quality(mesh; allow_boundary=true, require_closed=false)
println((nvertices(mesh), ntriangles(mesh)))
```

---

### `read_obj_mesh(path)`

Reads a triangle mesh from a Wavefront OBJ file.

**Parameters:**
- `path::AbstractString`: path to OBJ file

**Returns:** `TriMesh`

**Supported records:**
- `v x y z` (vertex coordinates)
- `f i j k ...` (polygon faces; polygons are fan‑triangulated)

Texture/normal indices (`f v/t/n`) are ignored. Positive and negative OBJ
vertex indices are supported.

---

### `write_obj_mesh(path, mesh; header="Exported by DifferentiableMoM")`

Writes a `TriMesh` to a Wavefront OBJ file using triangle faces.

**Parameters:**
- `path::AbstractString`: output file path
- `mesh::TriMesh`: triangle mesh
- `header::AbstractString="Exported by DifferentiableMoM"`: comment written at top of file

**Returns:** `path` (the input path as a string).

---

## Geometry Helpers

### `triangle_area(mesh, t)`

Compute the area of triangle `t` in the mesh.

**Parameters:**
- `mesh::TriMesh`: triangle mesh
- `t::Int`: triangle index (1‑based)

**Returns:** `Float64` area in m².

---

### `triangle_center(mesh, t)`

Compute the centroid of triangle `t`.

**Parameters:**
- `mesh::TriMesh`: triangle mesh
- `t::Int`: triangle index

**Returns:** `Vec3` centroid coordinates in meters.

---

### `triangle_normal(mesh, t)`

Compute the outward unit normal of triangle `t`.

**Parameters:**
- `mesh::TriMesh`: triangle mesh
- `t::Int`: triangle index

**Returns:** `Vec3` unit normal vector.

---

### `mesh_unique_edges(mesh)`

Return the unique undirected edges of a triangle mesh as a vector of
`(i, j)` vertex‑index pairs with `i < j`.

**Parameters:** `mesh::TriMesh`

**Returns:** `Vector{Tuple{Int,Int}}`.

---

### `mesh_wireframe_segments(mesh)`

Build line‑segment arrays for lightweight 3D wireframe visualization.

**Parameters:** `mesh::TriMesh`

**Returns:** named tuple `(x, y, z, n_edges)` where each edge contributes
`(p1, p2, NaN)` to each coordinate vector, suitable for `Plots.path3d`.

---

## Quality Checks

### `mesh_quality_report(mesh; area_tol_rel=1e-12, check_orientation=true)`

Compute mesh‑quality diagnostics for a triangle surface mesh.

**Parameters:**
- `mesh::TriMesh`: triangle mesh
- `area_tol_rel::Float64=1e-12`: relative tolerance for degenerate triangle detection (scaled by bounding‑box diagonal)
- `check_orientation::Bool=true`: check winding‑consistency across interior edges

**Returns:** named tuple with fields:
- `n_vertices`, `n_triangles`, `n_edges_total`, `n_interior_edges`, `n_boundary_edges`, `n_nonmanifold_edges`, `n_orientation_conflicts`
- `n_invalid_triangles`, `n_degenerate_triangles`
- `invalid_triangles`, `degenerate_triangles` (indices)
- `area_tol_abs` (absolute tolerance used)

---

### `mesh_quality_ok(report; allow_boundary=true, require_closed=false)`

Return `true` if a mesh‑quality report passes hard checks.

**Parameters:**
- `report`: named tuple returned by `mesh_quality_report`
- `allow_boundary::Bool=true`: allow boundary edges
- `require_closed::Bool=false`: require closed surface (no boundary edges)

**Returns:** `Bool`

**Checks:**
- no invalid triangles
- no degenerate triangles
- no non‑manifold edges
- no orientation conflicts
- boundary edges allowed unless `allow_boundary=false` or `require_closed=true`

---

### `assert_mesh_quality(mesh; allow_boundary=true, require_closed=false, area_tol_rel=1e-12)`

Run mesh‑quality checks and throw a detailed error if the mesh is unsuitable.

**Parameters:**
- `mesh::TriMesh`: triangle mesh
- `allow_boundary::Bool=true`: allow boundary edges
- `require_closed::Bool=false`: require closed surface
- `area_tol_rel::Float64=1e-12`: relative tolerance for degenerate triangle detection

**Returns:** the computed quality report on success.

---

## Repair and Coarsening

### `repair_mesh_for_simulation(mesh; allow_boundary=true, require_closed=false, area_tol_rel=1e-12, drop_invalid=true, drop_degenerate=true, fix_orientation=true, strict_nonmanifold=true)`

Repair a triangle mesh so it can pass solver prechecks.

**Pipeline:**
1. Optionally remove invalid/degenerate triangles (`drop_invalid`, `drop_degenerate`)
2. Fix orientation conflicts across interior edges (`fix_orientation`)
3. Enforce manifold constraints (optional strict mode, `strict_nonmanifold`)
4. Re‑check mesh suitability

**Returns:** named tuple containing:
- `mesh::TriMesh`: repaired mesh
- `before`, `cleaned`, `after`: quality reports before cleaning, after cleaning, and after orientation fix
- `removed_invalid`, `removed_degenerate`: indices of removed triangles
- `flipped_triangles`: indices of triangles whose orientation was flipped
- `area_tol_abs`: absolute area tolerance used

---

### `repair_obj_mesh(input_path, output_path; kwargs...)`

Read an OBJ mesh, repair it for solver prechecks, and write a repaired OBJ.

**Parameters:**
- `input_path::AbstractString`: path to input OBJ file
- `output_path::AbstractString`: path to output OBJ file
- `kwargs...`: forwarded to `repair_mesh_for_simulation`

**Returns:** same metadata as `repair_mesh_for_simulation`, plus `output_path`.

---

### `coarsen_mesh_to_target_rwg(mesh, target_rwg; max_iters=10, allow_boundary=true, require_closed=false, area_tol_rel=1e-12, strict_nonmanifold=true)`

Auto‑coarsen a mesh by voxel clustering to approach a target RWG count.
Each candidate mesh is non‑manifold cleaned and repaired before RWG counting.

**Parameters:**
- `mesh::TriMesh`: input mesh
- `target_rwg::Int`: target number of RWG basis functions (>0)
- `max_iters::Int=10`: maximum clustering iterations
- `allow_boundary::Bool=true`: allow boundary edges in repaired mesh
- `require_closed::Bool=false`: require closed surface
- `area_tol_rel::Float64=1e-12`: relative area tolerance
- `strict_nonmanifold::Bool=true`: enforce strict manifold constraints during repair

**Returns:** named tuple `(mesh, rwg_count, target_rwg, best_gap, iterations)`.

---

### `cluster_mesh_vertices(mesh, h)`

Voxel‑cluster a mesh using cubic cell size `h`, replacing all vertices in each
cell by their centroid and remapping triangles. Degenerate and duplicate
triangles created by remapping are removed.

**Parameters:**
- `mesh::TriMesh`: input mesh
- `h::Float64`: voxel cell size (>0)

**Returns:** `TriMesh` (clustered mesh).

---

### `drop_nonmanifold_triangles(mesh; max_passes=8)`

Iteratively remove triangles attached to non‑manifold edges (edges with more
than two incident triangles).

**Parameters:**
- `mesh::TriMesh`: input mesh
- `max_passes::Int=8`: maximum iteration count

**Returns:** `TriMesh` with only manifold/boundary edges.

---

### `estimate_dense_matrix_gib(N)`

Estimate memory (GiB) for a dense complex `N × N` matrix with `ComplexF64`
entries (16 bytes per entry).

**Parameters:** `N::Integer`: matrix dimension

**Returns:** `Float64` estimated GiB.

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
