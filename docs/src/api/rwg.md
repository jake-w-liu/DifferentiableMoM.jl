# API: RWG Utilities

## Purpose

Reference for RWG basis construction and evaluation functions.

---

## `build_rwg(mesh; ...)`

Builds `RWGData` from a triangle mesh.

Key options:

- `precheck=true` to run mesh-quality gate,
- `allow_boundary`, `require_closed` to enforce topology policy.

Returns interior-edge RWG basis only.

---

## `eval_rwg(rwg, n, r, t)`

Evaluates RWG basis function `n` at point `r` on triangle `t`.
Returns `Vec3`.

If triangle `t` is not in support of basis `n`, returns zero vector.

---

## `div_rwg(rwg, n, t)`

Returns piecewise-constant surface divergence of basis `n` on triangle `t`.

Used in scalar (charge/divergence) EFIE terms.

---

## `basis_triangles(rwg, n)`

Returns `(tplus, tminus)` support triangle indices for basis `n`.

---

## Typical Usage

```julia
rwg = build_rwg(mesh; precheck=true)
tp, tm = basis_triangles(rwg, 1)
println(div_rwg(rwg, 1, tp), " ", div_rwg(rwg, 1, tm))
```

---

## Code Mapping

- Implementation: `src/RWG.jl`
- Geometry helpers used internally: `src/Mesh.jl`

---

## Exercises

- Basic: print support triangles and edge length for first 5 RWG bases.
