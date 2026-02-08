# Large-Problem Strategy

## Purpose

The current package prioritizes **reference-correct dense EFIE solves**.
That is ideal for validation and research reproducibility, but dense MoM has a
hard scaling wall.

This chapter teaches a practical strategy for “as large as possible” runs
within the current implementation, and explains what to change when your target
problem is too large for direct dense algebra.

---

## Learning Goals

After this chapter, you should be able to:

1. Estimate memory/time cost from RWG unknown count before assembling anything.
2. Decide whether to solve directly, precondition, or simplify geometry.
3. Build a reproducible large-mesh workflow (repair → coarsen → solve → audit).

---

## Why Dense MoM Hits a Wall

For a dense complex matrix with `N` RWG unknowns:

- storage scales as `O(N^2)`,
- direct factorization scales as roughly `O(N^3)`.

In this package, matrix entries are `ComplexF64` (16 bytes), so a raw dense
matrix needs approximately

```math
\text{bytes} \approx 16N^2.
```

The helper function

```julia
estimate_dense_matrix_gib(N)
```

implements this estimate in GiB.

> Important: factorization and temporary buffers need extra memory beyond raw
> matrix storage, so treat the estimate as a **lower bound**, not a full budget.

---

## Practical Preflight Checklist

Before any expensive assembly:

1. Build RWG once (`rwg = build_rwg(mesh)`), get `N = rwg.nedges`.
2. Compute `estimate_dense_matrix_gib(N)`.
3. Check mesh quality (`mesh_quality_report`) and repair if needed.
4. Decide target unknown count you can afford.
5. Only then run full assembly/solve.

A typical pattern:

```julia
report = mesh_quality_report(mesh)
mesh_ok = mesh_quality_ok(report; allow_boundary=true, require_closed=false)

rwg = build_rwg(mesh; precheck=true, allow_boundary=true)
N = rwg.nedges
mem_gib = estimate_dense_matrix_gib(N)
```

---

## Cost-Control Lever 1: Geometry Coarsening

For complex CAD/OBJ platforms, use

```julia
coarsen_mesh_to_target_rwg(mesh, target_rwg)
```

which iterates:

1. vertex clustering,
2. removal of non-manifold triangles,
3. orientation/degeneracy repair,
4. RWG recount against the target.

This is the most effective current lever for large runs, because it reduces
both assembly and solve cost.

### Choosing `target_rwg`

- Start from available memory and wall-time budget.
- Pick a conservative `target_rwg` (e.g., 200–800) for first pass.
- Increase until accuracy/cost tradeoff is acceptable.

Always save repaired and coarsened meshes so results are reproducible.

---

## Cost-Control Lever 2: Scenario Simplification

Even with fixed mesh size, runtime grows with number of scenarios.

Common multipliers:

- many frequencies,
- many incidence angles,
- dense angular sampling for far field,
- optimization iterations.

To control cost, stage your study:

1. coarse screening grid,
2. shortlist promising settings,
3. high-resolution rerun only on finalists.

---

## What Preconditioning Does Here

The package supports optional mass-based left preconditioning
(`:off`, `:on`, `:auto`, or user matrix) and optional mass-based
regularization.

Preconditioning improves conditioning of the linear algebra problem, but does
**not** remove dense `O(N^2)` storage or `O(N^3)` direct-solve scaling.

In other words:

- it helps numerical behavior,
- it does not magically make dense direct solves “large-scale”.

For this reason, `:auto` is conservative and intended as a safety/default hint
for larger or iterative-style workflows.

---

## Current Implementation Envelope

Within this repository, the reliable path for large geometry is:

1. mesh repair,
2. controlled coarsening,
3. dense direct solve,
4. strong validation diagnostics.

If your project needs much larger electrical size, the next algorithmic step is
matrix-free fast methods (e.g., FMM/MLFMM) with iterative Krylov solvers.
That extension is compatible with the same EFIE/adjoint formulations but is not
part of the current release.

---

## Recommended Workflow (Command Level)

For complex OBJ platforms:

```bash
julia --project=. examples/ex_repair_obj_mesh.jl ../Airplane.obj ../Airplane_repaired.obj
julia --project=. examples/ex_airplane_rcs.jl ../Airplane.obj 3.0 0.001 300
```

For auto-preconditioning behavior and settings:

```bash
julia --project=. examples/ex_auto_preconditioning.jl
```

---

## Interpreting “Good Enough” at Large Size

When you coarsen aggressively, do not expect exact RCS amplitude preservation at
all angles. Prefer a **tiered acceptance test**:

1. mesh quality and solver residual pass,
2. internal consistency checks pass,
3. key engineering observables are stable (main lobe, monostatic sample,
   trend vs parameter changes),
4. optional external cross-check on selected cases.

This makes cost-vs-fidelity tradeoffs explicit and reproducible.

---

## Code Mapping

- Memory estimate and coarsening: `src/Mesh.jl`
- Conditioning helpers: `src/Solve.jl`
- Optimization-time conditioning options: `src/Optimize.jl`
- Large OBJ demo: `examples/ex_airplane_rcs.jl`
- Auto-preconditioning example: `examples/ex_auto_preconditioning.jl`

---

## Exercises

- Basic: pick three `target_rwg` values and report how
  `estimate_dense_matrix_gib(N)` changes.
- Practical: run airplane RCS with two coarsening levels and compare monostatic
  RCS plus solve time.
- Challenge: design a two-stage sweep (coarse then refined) for frequency,
  and justify your compute budget choices.
