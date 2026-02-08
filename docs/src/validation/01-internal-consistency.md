# Internal Consistency

## Purpose

Document the package’s internal correctness checks that do not require external
solvers: residual, objective consistency, energy balance, and reciprocity-style
sanity behavior.

---

## Learning Goals

After this chapter, you should be able to:

1. Run internal consistency checks on a new case.
2. Identify which failure mode corresponds to which likely bug class.
3. Decide whether results are trustworthy before cross-validation.

---

## 1) Core Internal Gates

Recommended minimum gates:

1. **Linear residual**: $\|\mathbf Z\mathbf I-\mathbf v\|/\|\mathbf v\|$.
2. **Objective consistency**: direct angular integration vs
   $\mathbf I^\dagger\mathbf Q\mathbf I$.
3. **Energy ratio**: `P_rad / P_in` near expected value for tested scenario.
4. **Condition diagnostics**: finite and reasonable matrix condition estimate.

---

## 2) Energy Consistency in Practice

For PEC benchmarks, energy ratio near unity is a strong indicator that EFIE
assembly + far-field quadrature + excitation scaling are mutually consistent.

Use:

```julia
ratio = energy_ratio(I, v, E_ff, grid; eta0=η0)
```

---

## 3) Objective Consistency

A useful check:

```math
\mathbf I^\dagger \mathbf Q \mathbf I
\approx
\sum_{q\in\mathcal D}
w_q\,
\left|\mathbf p_q^\dagger \mathbf E^\infty(\hat{\mathbf r}_q)\right|^2.
```

If this fails, suspect mismatch in grid weights, polarization projection, or
`Q` assembly.

---

## 4) Diagnostic Script Entry Points

- convergence/energy trend: `examples/ex_convergence.jl`
- paper consistency aggregation: `validation/paper/generate_consistency_report.jl`

Generated artifacts in `data/` provide reproducible numerical evidence.

---

## 5) Interpreting Failures

- Large residual only: linear system or assembly issue.
- Good residual but bad energy: far-field or integration inconsistency.
- Good energy but poor objective consistency: `Q`-construction/polarization issue.
- Intermittent failures with geometry import: mesh quality pipeline issue.

---

## Code Mapping

- Diagnostics: `src/Diagnostics.jl`
- Q and far field: `src/QMatrix.jl`, `src/FarField.jl`
- Forward solve path: `src/Solve.jl`, `src/EFIE.jl`, `src/Excitation.jl`

---

## Exercises

- Basic: compute all four gates for one PEC plate case.
- Challenge: intentionally perturb angular weights and observe which gates fail.
