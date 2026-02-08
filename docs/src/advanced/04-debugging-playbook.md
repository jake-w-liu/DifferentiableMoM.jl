# Debugging Playbook

## Purpose

When a MoM run fails, the fastest path is a fixed diagnostic order.

This playbook gives a practical sequence that moves from cheap checks to
expensive checks, so you can isolate root causes quickly and reproducibly.

---

## Learning Goals

After this chapter, you should be able to:

1. Identify likely failure class from symptoms.
2. Run minimal checks that isolate mesh, units, solver, or objective issues.
3. Produce a compact reproducibility packet for bug reports.

---

## One-Page Triage Order

Use this order every time:

1. **Mesh sanity** (topology, degeneracy, orientation).
2. **Units sanity** (geometry scale vs wavelength).
3. **Linear system sanity** (residual + conditioning).
4. **Far-field sanity** (transversality + power normalization).
5. **Objective sanity** (`I†QI` vs direct angular sum).
6. **Gradient sanity** (adjoint vs finite difference).
7. **External comparison** (only after 1–6 pass).

Skipping earlier steps often causes false debugging conclusions.

---

## Symptom → Likely Cause Map

### A) RWG build or assembly fails immediately

Likely causes:

- invalid triangles,
- degenerate triangles,
- non-manifold edges,
- orientation conflicts.

First checks:

```julia
rep = mesh_quality_report(mesh)
mesh_quality_ok(rep; allow_boundary=true, require_closed=false)
```

If needed:

```julia
fixed = repair_mesh_for_simulation(mesh; allow_boundary=true, require_closed=false)
```

---

### B) Solve completes but residual is large

Likely causes:

- bad conditioning,
- wrong units causing extreme electrical size,
- corrupted assembly inputs.

Check:

```julia
r_rel = norm(Z*I - v) / max(norm(v), 1e-30)
```

Then inspect singular values/condition:

```julia
diag = condition_diagnostics(Z)
```

If needed, enable regularization/preconditioning and compare behavior.

---

### C) Pattern looks nonphysical

Likely causes:

- far-field vector assembly mismatch,
- polarization projector mismatch,
- grid normalization issue.

Check transversality:

```math
\hat{\mathbf r}\cdot\mathbf E^\infty \approx 0.
```

Check energy ratio:

```julia
rho = energy_ratio(I, v, E_ff, grid)
```

For lossless PEC benchmarks, `rho` should be close to 1.

---

### D) Objective values seem inconsistent

Likely causes:

- mismatch between `Q` construction and direct angular integration,
- wrong mask/polarization alignment.

Check:

```math
\mathbf I^\dagger\mathbf Q\mathbf I
\approx
\sum_q w_q\,|\mathbf p_q^\dagger\mathbf E_q^\infty|^2.
```

If this fails, debug `Q` and far-field projection before optimizer logic.

---

### E) Adjoint gradient does not match finite difference

Likely causes:

- inconsistent forward/adjoint operators (especially with conditioning),
- wrong derivative blocks,
- poor finite-difference step.

Use:

- central differences with sensible `h`,
- and check the same conditioned operator in both forward/adjoint paths.

For quadratic objectives with conjugation, do not rely on complex-step for full
end-to-end gradients.

---

## Units and Scale Debugging (High-Impact)

Many “mysterious” failures are actually unit mistakes.

Quick test:

1. compute geometric length scale `L`,
2. compute wavelength `λ`,
3. inspect ratio `L/λ`.

If `L/λ` is wildly different from your intended regime, stop and fix geometry
scaling before further debugging.

---

## Preconditioning Debugging Rules

When conditioning is enabled:

- apply the same conditioned operator in forward and adjoint solves,
- transform derivative blocks consistently,
- compare against unconditioned results on a small benchmark.

A useful regression check is solution invariance for left preconditioning:
solving `ZI=v` and `M^{-1}ZI=M^{-1}v` should yield matching `I` (within
numerical tolerance).

---

## Cross-Validation Debugging (After Internal Checks)

If internal checks pass but external mismatch remains:

1. verify convention alignment (time sign, phase, polarization, units),
2. verify geometry/mesh and excitation are truly matched,
3. compare beam-centric features first (main beam, sidelobe location/level),
4. only then inspect null-region residuals.

Null-heavy aggregate dB errors can overstate practical mismatch for
beam-steering objectives.

---

## Minimal Reproducibility Packet (for Issues)

When filing or sharing a bug, include:

- input mesh (or script that generates it),
- frequency, polarization, incidence definition,
- mesh-quality report,
- `N` and memory estimate,
- residual, condition diagnostic, energy ratio,
- exact commit hash and run command,
- generated CSV outputs.

This short packet usually reduces debug time by an order of magnitude.

---

## Recommended Commands

Full internal regression:

```bash
julia --project=. test/runtests.jl
```

Mesh repair sanity:

```bash
julia --project=. examples/ex_repair_obj_mesh.jl input.obj repaired.obj
```

Large OBJ smoke run:

```bash
julia --project=. examples/ex_airplane_rcs.jl ../Airplane.obj 3.0 0.001 300
```

---

## Code Mapping

- Mesh diagnostics/repair/coarsening: `src/Mesh.jl`
- Assembly/solve/conditioning: `src/EFIE.jl`, `src/Solve.jl`
- Far field, Q, and diagnostics: `src/FarField.jl`, `src/QMatrix.jl`,
  `src/Diagnostics.jl`
- Gradient verification: `src/Adjoint.jl`, `src/Verification.jl`
- End-to-end gates: `test/runtests.jl`

---

## Exercises

- Basic: intentionally flip triangle winding on a subset and trace which checks
  fail first.
- Practical: apply a wrong geometry scale by `×1000`, run preflight, and
  document which metrics reveal the issue fastest.
- Challenge: create a “debug report template” from the reproducibility packet
  and use it on one failing test case.
