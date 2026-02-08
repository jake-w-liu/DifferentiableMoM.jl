# API: Assembly and Solve

## Purpose

Reference for forward-system assembly and linear-solve helpers.

---

## EFIE Assembly

### `assemble_Z_efie(mesh, rwg, k; quad_order=3, eta0=...)`

Builds dense EFIE matrix for RWG basis.
Includes self-term singular handling through internal branching.

---

## Impedance Assembly

### `precompute_patch_mass(mesh, rwg, part; quad_order=3)`

Precomputes patch mass matrices `Mp`.

### `assemble_Z_impedance(Mp, theta)`

Builds impedance block from `Mp` and parameter vector:

```math
\mathbf Z_{\mathrm{imp}}=-\sum_p \theta_p \mathbf M_p.
```

For reactive loading, pass complex coefficients in `theta` (e.g., `1im .* θ`)
or use `assemble_full_Z(...; reactive=true)` for the common real-parameter
reactive workflow.

### `assemble_full_Z(Z_efie, Mp, theta; reactive=false)`

Convenience full system assembly:

```math
\mathbf Z = \mathbf Z_{\mathrm{EFIE}} - \sum_p c_p \mathbf M_p,
```

with `c_p = θ_p` or `iθ_p` in reactive mode.

---

## Linear Solves

### `solve_forward(Z, v)`

Direct solve for forward system.

### `solve_system(Z, rhs)`

General direct solve helper.

---

## Conditioning Helpers

- `make_mass_regularizer(Mp)`
- `make_left_preconditioner(Mp; eps_rel=...)`
- `select_preconditioner(Mp; mode=:off|:on|:auto, ...)`
- `transform_patch_matrices(Mp; ...)`
- `prepare_conditioned_system(Z_raw, rhs; ...)`

Use these when enabling optional regularization/preconditioning.

---

## Minimal Pattern

```julia
Zef = assemble_Z_efie(mesh, rwg, k)
Mp = precompute_patch_mass(mesh, rwg, part)
Z = assemble_full_Z(Zef, Mp, theta; reactive=true)
I = solve_forward(Z, v)
```

---

## Code Mapping

- EFIE kernel and assembly: `src/EFIE.jl`
- Impedance blocks: `src/Impedance.jl`
- Solves and conditioning: `src/Solve.jl`

---

## Exercises

- Basic: confirm `assemble_full_Z(..., theta=zeros(P))` matches `Z_efie`.
