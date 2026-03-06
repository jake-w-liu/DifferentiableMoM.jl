# Singular Integration

## Purpose

This chapter documents the singular-integration path used by `DifferentiableMoM.jl` for EFIE self terms. The goal is strict code-formulation correspondence to:

- `src/basis/Greens.jl`
- `src/assembly/SingularIntegrals.jl`
- `src/assembly/EFIE.jl`

It focuses on what is implemented now: singularity extraction for **exact self-triangle pairs** (`tm == tn`).

---

## Learning Goals

After this chapter, you should be able to:

1. Explain why EFIE self terms need special treatment.
2. Derive the `G = G_smooth + 1/(4πR)` split used in code.
3. Describe how `analytical_integral_1overR` is used inside `self_cell_contribution`.
4. Map the self/non-self branching in `assemble_Z_efie` to the formulas.
5. Identify current limitations (what is not yet special-cased).

---

## 1. Where the Singularity Appears

In mixed-potential EFIE assembly, each entry uses integrals of the form

```math
\iint f_m(\mathbf r)\, f_n(\mathbf r')\, G(\mathbf r,\mathbf r')\, dS\, dS',
\qquad
G(\mathbf r,\mathbf r') = \frac{e^{-ikR}}{4\pi R},\; R=|\mathbf r-\mathbf r'|.
```

When source and test points approach each other (`R -> 0`), `1/R` is singular. For the EFIE implementation in this package:

- singular extraction is applied when the two integration triangles are the same (`tm == tn`),
- otherwise standard product quadrature is used.

This logic is in `src/assembly/EFIE.jl` (`_efie_entry`).

---

## 2. Kernel Splitting Used in Code

The Green kernel is split as

```math
G(\mathbf r,\mathbf r')
=
\underbrace{\frac{e^{-ikR}-1}{4\pi R}}_{G_{\text{smooth}}}
+
\underbrace{\frac{1}{4\pi R}}_{G_{\text{sing}}}.
```

### 2.1 Smooth Part

`greens_smooth(r, rp, k)` computes

```math
G_{\text{smooth}} = \frac{e^{-ikR}-1}{4\pi R},
\qquad
\lim_{R\to 0} G_{\text{smooth}} = -\frac{i k}{4\pi}.
```

In code, the small-`R` limit is handled explicitly (`abs(R) < 1e-14`) and otherwise uses `expm1(-im*k*R)/(4πR)` for numerical stability.

### 2.2 Singular Part

The singular part is handled through

```math
S(\mathbf P) = \int_T \frac{1}{|\mathbf P-\mathbf r'|}\, dS',
```

computed by `analytical_integral_1overR(P, V1, V2, V3)`.

---

## 3. Analytical Inner Integral `analytical_integral_1overR`

`analytical_integral_1overR` evaluates the triangle integral by summing edge-log contributions:

```math
S(\mathbf P) = \sum_{\text{edges }i} d_i
\log\!\left(\frac{\ell_{B_i}+R_{B_i}}{\ell_{A_i}+R_{A_i}}\right).
```

The implementation uses oriented edge tangents and in-plane outward normals derived from the triangle normal.

Practical behavior in code:

- degenerate triangle (`|n_T|` tiny): returns `0.0`,
- near-zero perpendicular distance to an edge (`|d_i| < 1e-15`): skips that edge contribution,
- near-zero log numerator/denominator: safely skipped.

This is implemented exactly in `src/assembly/SingularIntegrals.jl`.

---

## 4. Self-Cell Contribution Algorithm

`self_cell_contribution(...)` returns the self-triangle integral contribution for `(vec_part - scl_part)` before multiplying by `-iωμ0`.

It is decomposed into:

```math
I_{\text{self}} = I_{\text{smooth}} + I_{\text{singular}}.
```

### 4.1 Smooth Product-Quadrature Part

For the same triangle, it still performs product quadrature with `greens_smooth`:

```math
I_{\text{smooth}}
=\sum_{q_m,q_n}
\left[f_m\cdot f_n\,G_{\text{smooth}} - \frac{(\nabla\cdot f_m)(\nabla\cdot f_n)}{k^2}G_{\text{smooth}}\right]
 w_{q_m}w_{q_n}(2A)^2.
```

### 4.2 Singular Part

For each outer quadrature point `r_m`:

1. compute `S = analytical_integral_1overR(r_m, V1, V2, V3)`,
2. set `inner_scalar = S/(4π)`,
3. scalar singular term:

```math
\text{scl}_{\text{sing}} = \frac{(\nabla\cdot f_m)(\nabla\cdot f_n)}{k^2}\,\frac{S}{4\pi},
```

4. vector singular leading term:

```math
\text{vec}_{\text{lead}} = (f_m(r_m)\cdot f_n(r_m))\,\frac{S}{4\pi},
```

5. vector remainder (bounded integrand):

```math
\text{vec}_{\text{rem}}
= \int_T f_m(r_m)\cdot\frac{f_n(r')-f_n(r_m)}{4\pi|r_m-r'|}\, dS'.
```

The remainder is computed by quadrature and skips the `R=0` sample explicitly.

---

## 5. How EFIE Assembly Uses It

In `_efie_entry` (`src/assembly/EFIE.jl`):

- if `tm == tn`: call `self_cell_contribution(...)`,
- else: use non-self product quadrature with `greens(rm, rn, k)`.

Final scaling is applied after accumulation:

```math
Z_{mn} = -i\,\omega\mu_0\,(\text{accumulated }(\text{vec}-\text{scl})).
```

with `omega_mu0 = k * eta0`.

---

## 6. Current Scope and Limitations

The current singular treatment is intentionally narrow:

1. **Implemented**: exact self-cell extraction for `tm == tn`.
2. **Not special-cased**: edge-touching / vertex-touching non-self triangle pairs.
3. **Not implemented**: dedicated near-singular quadrature switching or Duffy-transform path.

So, statements like “adjacent pairs are singular-extracted” are not correct for the current code path.

---

## 7. Practical Verification Checks

A minimal self-consistency workflow is:

```julia
using DifferentiableMoM, LinearAlgebra

mesh = make_rect_plate(0.1, 0.1, 4, 4)
rwg = build_rwg(mesh)
k = 2π / 0.1

# Dense EFIE build should complete with finite values
Z = assemble_Z_efie(mesh, rwg, k; mesh_precheck=false)
@assert all(isfinite, real.(Z))
@assert all(isfinite, imag.(Z))

# Symmetry-style sanity (integration noise tolerated)
rel_sym = norm(Z - transpose(Z)) / max(norm(Z), 1e-30)
println("relative transpose-mismatch = ", rel_sym)
```

For package-level regression, use:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

---

## 8. Code Mapping

| Concept | Source | Function |
|---|---|---|
| Full free-space kernel | `src/basis/Greens.jl` | `greens` |
| Smooth split kernel | `src/basis/Greens.jl` | `greens_smooth` |
| Analytical `∫ 1/R` over triangle | `src/assembly/SingularIntegrals.jl` | `analytical_integral_1overR` |
| Self-cell regularized integral | `src/assembly/SingularIntegrals.jl` | `self_cell_contribution` |
| Self/non-self branch | `src/assembly/EFIE.jl` | `_efie_entry` |
| Top-level dense EFIE assembly | `src/assembly/EFIE.jl` | `assemble_Z_efie` |

---

## 9. Exercises

1. Numerically verify `greens_smooth(r, rp, k)` approaches `-im*k/(4π)` as `|r-rp| -> 0`.
2. For a fixed triangle and point `P`, compare `analytical_integral_1overR` against high-order numerical integration with `P` slightly off-plane.
3. Measure how EFIE matrix entries change when increasing `quad_order` from `3` to `7` on a small mesh.

---

## 10. Chapter Checklist

- [ ] Explain why `1/R` needs special handling for self terms.
- [ ] Derive and interpret `G = G_smooth + 1/(4πR)`.
- [ ] Map `analytical_integral_1overR` inputs to triangle geometry.
- [ ] Describe `self_cell_contribution` as smooth + singular + remainder parts.
- [ ] Identify that only `tm == tn` is singular-extracted in current code.
- [ ] Locate the exact self/non-self branch in `_efie_entry`.

---

## 11. Further Reading

- Graglia, R. D. (1993). Numerical integration of shape functions times 3D Green's function on triangles.
- Khayat, M. A., & Wilton, D. R. (2005). Numerical evaluation of singular and near-singular potential integrals.
- Gibson, W. C. *The Method of Moments in Electromagnetics* (singular-integral treatment chapters).

---

*Next: [Conditioning and Preconditioning](05-conditioning-preconditioning.md) covers system conditioning diagnostics and solver preconditioning pathways.*
