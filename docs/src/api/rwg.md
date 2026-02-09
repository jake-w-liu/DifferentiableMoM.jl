# API: RWG Utilities

## Purpose

Reference for RWG basis construction and evaluation functions.

---

## `build_rwg(mesh; precheck=true, allow_boundary=true, require_closed=false, area_tol_rel=1e-12)`

Builds `RWGData` from interior edges of a triangle mesh.
Each interior edge shared by two triangles defines one RWG basis function.

**Parameters:**
- `mesh::TriMesh`: triangle mesh
- `precheck::Bool=true`: run mesh-quality checks before basis construction
- `allow_boundary::Bool=true`: allow boundary edges in mesh
- `require_closed::Bool=false`: require closed surface (no boundary edges)
- `area_tol_rel::Float64=1e-12`: relative tolerance for degenerate triangle detection

**Returns:** `RWGData` containing edge‑to‑triangle connectivity and geometric factors.

**Fields of `RWGData`:** See `types.md` for detailed field descriptions.

---

## `eval_rwg(rwg, n, r, t)`

Evaluates RWG basis function `n` at point `r` on triangle `t`.

**Parameters:**
- `rwg::RWGData`: RWG basis data
- `n::Int`: basis function index (1‑based)
- `r::Vec3`: evaluation point (in meters)
- `t::Int`: triangle index (must be either `rwg.tplus[n]` or `rwg.tminus[n]`)

**Returns:** `Vec3` basis vector value (V/m scale). Returns zero vector if triangle `t` is not in support of basis `n`.

**Formula:**
- For `t == rwg.tplus[n]`: `f_n(r) = (l_n / (2 A⁺)) * (r - r_opp⁺)`
- For `t == rwg.tminus[n]`: `f_n(r) = (l_n / (2 A⁻)) * (r_opp⁻ - r)`

where `l_n` is edge length, `A⁺`/`A⁻` are triangle areas, and `r_opp⁺`/`r_opp⁻` are opposite vertex positions.

---

## `div_rwg(rwg, n, t)`

Returns piecewise‑constant surface divergence of basis `n` on triangle `t`.

**Parameters:**
- `rwg::RWGData`: RWG basis data
- `n::Int`: basis function index
- `t::Int`: triangle index (must be either `rwg.tplus[n]` or `rwg.tminus[n]`)

**Returns:** `Float64` divergence value.
- For `t == rwg.tplus[n]`: `∇⋅f_n = l_n / A⁺`
- For `t == rwg.tminus[n]`: `∇⋅f_n = -l_n / A⁻`

Used in scalar (charge/divergence) EFIE terms.

---

## `basis_triangles(rwg, n)`

Returns the two triangle indices supporting basis function `n`.

**Parameters:**
- `rwg::RWGData`: RWG basis data
- `n::Int`: basis function index

**Returns:** tuple `(tplus, tminus)` where `tplus = rwg.tplus[n]`, `tminus = rwg.tminus[n]`.

---

## Typical Usage

```julia
rwg = build_rwg(mesh; precheck=true)
tp, tm = basis_triangles(rwg, 1)
println("Edge length: ", rwg.len[1])
println("Divergence on T+: ", div_rwg(rwg, 1, tp))
println("Divergence on T-: ", div_rwg(rwg, 1, tm))
```

---

## Code Mapping

- Implementation: `src/RWG.jl`
- Geometry helpers used internally: `src/Mesh.jl` (`triangle_area`, `triangle_center`, etc.)

---

## Exercises

- Basic: print support triangles and edge length for first 5 RWG bases.
