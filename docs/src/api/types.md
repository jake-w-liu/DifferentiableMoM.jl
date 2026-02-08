# API: Types

## Purpose

Reference for core data structures used across the package.

---

## `TriMesh`

```julia
struct TriMesh
    xyz::Matrix{Float64}   # (3, Nv)
    tri::Matrix{Int}       # (3, Nt), 1-based indices
end
```

Helpers:

- `nvertices(mesh)`
- `ntriangles(mesh)`

---

## `RWGData`

Stores edge-to-triangle connectivity and geometric factors for RWG basis:

- `nedges`
- `tplus`, `tminus`
- `evert`
- `vplus_opp`, `vminus_opp`
- `len`
- `area_plus`, `area_minus`

Constructed via `build_rwg(mesh; ...)`.

---

## `PatchPartition`

```julia
struct PatchPartition
    tri_patch::Vector{Int}
    P::Int
end
```

Maps each triangle to one patch index for impedance design.

---

## `SphGrid`

```julia
struct SphGrid
    rhat::Matrix{Float64}
    theta::Vector{Float64}
    phi::Vector{Float64}
    w::Vector{Float64}
end
```

Construct with `make_sph_grid`.

---

## Aliases

- `Vec3 = SVector{3,Float64}`
- `CVec3 = SVector{3,ComplexF64}`

Used heavily in geometry/kernel routines.

---

## Code Mapping

- Definitions: `src/Types.jl`
- Primary users: `src/Mesh.jl`, `src/RWG.jl`, `src/FarField.jl`

---

## Exercises

- Basic: inspect dimensions of each field for one built mesh and grid.
