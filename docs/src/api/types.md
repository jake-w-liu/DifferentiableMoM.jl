# API: Types

## Purpose

Reference for core data structures used across `DifferentiableMoM.jl`. Understanding these types is essential for advanced usage and custom workflows.

---

## `TriMesh` – Triangle Mesh

```julia
struct TriMesh
    xyz::Matrix{Float64}   # (3, Nv) vertex coordinates [m]
    tri::Matrix{Int}       # (3, Nt) 1‑based vertex indices per triangle
end
```

**Fields:**

| Field | Type | Dimensions | Description |
|-------|------|------------|-------------|
| `xyz` | `Matrix{Float64}` | `(3, Nv)` | Vertex coordinates: `xyz[:, i]` is `[x, y, z]` of vertex `i` in meters. |
| `tri` | `Matrix{Int}` | `(3, Nt)` | Triangle connectivity: `tri[:, j]` contains three 1‑based vertex indices of triangle `j`. |

**Helper functions:**

- `nvertices(mesh::TriMesh) -> Int`: Number of vertices.
- `ntriangles(mesh::TriMesh) -> Int`: Number of triangles.

**Constructors:** Typically created via `make_rect_plate`, `read_obj_mesh`, or manually from arrays.

**Example:**

```julia
xyz = [0.0 1.0 1.0 0.0;   # x coordinates
       0.0 0.0 1.0 1.0;   # y coordinates  
       0.0 0.0 0.0 0.0]   # z coordinates
tri = [1 2; 2 3; 3 4; 4 1]'  # two triangles forming a square
mesh = TriMesh(xyz, tri)
println("Vertices: ", nvertices(mesh), ", triangles: ", ntriangles(mesh))
```

---

## `RWGData` – RWG Basis Data

Stores edge‑to‑triangle connectivity and geometric factors for Rao–Wilton–Glisson basis functions. Constructed by `build_rwg`.

```julia
struct RWGData
    mesh::TriMesh
    nedges::Int
    tplus::Vector{Int}          # T⁺ triangle index for each edge
    tminus::Vector{Int}         # T⁻ triangle index for each edge
    evert::Matrix{Int}          # (2, nedges) edge vertex indices
    vplus_opp::Vector{Int}      # opposite vertex in T⁺
    vminus_opp::Vector{Int}     # opposite vertex in T⁻
    len::Vector{Float64}        # edge length [m]
    area_plus::Vector{Float64}  # area of T⁺ [m²]
    area_minus::Vector{Float64} # area of T⁻ [m²]
end
```

**Fields:**

| Field | Type | Dimensions | Description |
|-------|------|------------|-------------|
| `mesh` | `TriMesh` | – | Reference mesh (unchanged). |
| `nedges` | `Int` | – | Number of interior edges = number of RWG basis functions. |
| `tplus`, `tminus` | `Vector{Int}` | `nedges` | Triangle indices of the two triangles sharing the edge. Convention: `tplus` is the triangle where the basis vector points from vertex `evert[1]` to `evert[2]`. |
| `evert` | `Matrix{Int}` | `(2, nedges)` | Edge vertex indices: `evert[1, n]` and `evert[2, n]` are the two vertices of edge `n`. |
| `vplus_opp`, `vminus_opp` | `Vector{Int}` | `nedges` | Vertex index opposite the edge in triangle `tplus` / `tminus`. |
| `len` | `Vector{Float64}` | `nedges` | Edge length in meters. |
| `area_plus`, `area_minus` | `Vector{Float64}` | `nedges` | Areas of the two support triangles in m². |

**Constructors:** Use `build_rwg(mesh; precheck=true, allow_boundary=true, require_closed=false)`.

**Example:**

```julia
mesh = make_rect_plate(0.1, 0.1, 3, 3)
rwg = build_rwg(mesh; precheck=true)
println("RWG basis count: ", rwg.nedges)
println("Edge length of first basis: ", rwg.len[1], " m")
println("Support triangles: ", rwg.tplus[1], " (T⁺), ", rwg.tminus[1], " (T⁻)")
```

---

## `PatchPartition` – Impedance Patch Mapping

Maps each triangle to a design patch for impedance parameterization.

```julia
struct PatchPartition
    tri_patch::Vector{Int}      # length Nt: patch id for each triangle
    P::Int                      # number of patches
end
```

**Fields:**

| Field | Type | Dimensions | Description |
|-------|------|------------|-------------|
| `tri_patch` | `Vector{Int}` | `Nt` | `tri_patch[t]` is the patch index (1‑based) of triangle `t`. |
| `P` | `Int` | – | Total number of patches (maximum of `tri_patch`). |

**Typical usage:** One patch per triangle (`P = Nt`) for fine‑grained control, or coarser grouping for reduced design space.

**Example:**

```julia
Nt = ntriangles(mesh)
# One patch per triangle
tri_patch = collect(1:Nt)
partition = PatchPartition(tri_patch, Nt)
```

---

## `SphGrid` – Spherical Far‑Field Grid

Sampling directions and quadrature weights for far‑field computations.

```julia
struct SphGrid
    rhat::Matrix{Float64}       # (3, NΩ) unit direction vectors
    theta::Vector{Float64}      # polar angles [rad]
    phi::Vector{Float64}        # azimuthal angles [rad]
    w::Vector{Float64}          # quadrature weights [sr]
end
```

**Fields:**

| Field | Type | Dimensions | Description |
|-------|------|------------|-------------|
| `rhat` | `Matrix{Float64}` | `(3, NΩ)` | Unit direction vectors: `rhat[:, q] = [sinθ cosφ, sinθ sinφ, cosθ]`. |
| `theta` | `Vector{Float64}` | `NΩ` | Polar angle θ ∈ [0, π] measured from +z axis. |
| `phi` | `Vector{Float64}` | `NΩ` | Azimuthal angle φ ∈ [0, 2π) measured from +x axis. |
| `w` | `Vector{Float64}` | `NΩ` | Quadrature weights for integration over the sphere: ∫ f dΩ ≈ Σ f(θ_q, φ_q) w_q. |

**Constructors:** Created by `make_sph_grid(Ntheta, Nphi)`.

**Example:**

```julia
grid = make_sph_grid(180, 72)  # 1° θ resolution, 5° φ resolution
println("Number of directions: ", length(grid.w))
println("First direction: θ = ", grid.theta[1], " rad, φ = ", grid.phi[1], " rad")
println("Corresponding unit vector: ", grid.rhat[:, 1])
```

---

## Vector Type Aliases

| Alias | Definition | Purpose |
|-------|------------|---------|
| `Vec3` | `SVector{3,Float64}` | 3‑component real vector for geometry and physics. |
| `CVec3` | `SVector{3,ComplexF64}` | 3‑component complex vector for phasor fields. |

These are used throughout kernel evaluations and field representations for performance and clarity.

**Example:**

```julia
using StaticArrays
v = Vec3(1.0, 2.0, 3.0)        # real vector
cv = CVec3(1.0 + 2.0im, 0.0, 0.0)  # complex vector
```

---

## Internal Details

### Memory Layout

- `TriMesh` stores vertices and triangles as dense matrices for cache‑efficient access.
- `RWGData` fields are `Vector` or `Matrix` with `nedges` columns; all lengths are in meters, areas in m².
- `SphGrid` weights `w` are scaled such that `sum(w) ≈ 4π` (full sphere) or `2π` (hemisphere).

### Performance Notes

- All types are immutable (`struct`) except for their array contents.
- Functions that modify meshes (repair, coarsening) return new `TriMesh` instances.
- `RWGData` holds a reference to the original `mesh`; ensure the mesh is not mutated after RWG construction.

---

## Code Mapping

| Type | Source File | Primary Users |
|------|-------------|---------------|
| `TriMesh`, `RWGData`, `PatchPartition`, `SphGrid` | `src/Types.jl` | All assembly, solve, and post‑processing modules |
| Vector aliases | `src/Types.jl` | `src/Geometry.jl`, `src/Greens.jl`, `src/FarField.jl` |

---

## Exercises

### Basic

1. Create a `TriMesh` for a right‑triangle with vertices (0,0,0), (1,0,0), (0,1,0). Verify `ntriangles` returns 1.
2. Build `RWGData` for a 2×2 plate. Print `tplus`, `tminus`, and `len` for each edge. Confirm each edge is interior (both `tplus` and `tminus` non‑zero).

### Practical

1. Generate a `SphGrid` with 90 θ points and 36 φ points. Compute the total weight `sum(grid.w)` and compare with 4π (full sphere) or 2π (hemisphere, if θ limited).
2. Create a `PatchPartition` that groups triangles into 4 patches based on quadrant of their centroid (x>0, y>0; x>0, y<0; etc.). Use `triangle_center` from `Mesh.jl`.

### Advanced

1. Write a function that extracts all edges of a `TriMesh` (including boundary edges) and compares with `RWGData`’s interior edges. Use `mesh_unique_edges`.
2. Implement a custom spherical grid with Gauss–Legendre quadrature in θ and uniform φ. Compare integration accuracy for ∫ cos²θ dΩ with the built‑in `make_sph_grid`.
