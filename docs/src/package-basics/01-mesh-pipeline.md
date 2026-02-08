# Mesh Pipeline

## Purpose

Explain the complete geometry path used before MoM assembly:
import/generate mesh, validate topology, repair when needed, and coarsen to a
computationally feasible simulation mesh.

---

## Learning Goals

After this chapter, you should be able to:

1. Run mesh quality checks and interpret their outputs.
2. Repair problematic OBJ meshes for solver safety.
3. Coarsen while preserving key geometric features.

---

## 1) Native Mesh Type

`TriMesh` stores:

- `xyz::Matrix{Float64}` with shape `(3,Nv)`,
- `tri::Matrix{Int}` with shape `(3,Nt)`.

Helper accessors:

- `nvertices(mesh)`
- `ntriangles(mesh)`

---

## 2) Two Entry Paths

1. Analytical plate generation:

```julia
mesh = make_rect_plate(Lx, Ly, Nx, Ny)
```

2. OBJ import:

```julia
mesh = read_obj_mesh("platform.obj")
```

`read_obj_mesh` supports polygon faces by fan triangulation.

---

## 3) Quality Diagnostics

Run:

```julia
report = mesh_quality_report(mesh)
ok = mesh_quality_ok(report; allow_boundary=true, require_closed=false)
```

Key failure indicators:

- invalid triangle indices,
- degenerate triangles,
- non-manifold edges,
- orientation conflicts.

Use `assert_mesh_quality` to enforce a hard gate.

---

## 4) Repair Pathway

For external meshes:

```julia
rep = repair_mesh_for_simulation(mesh;
    allow_boundary=true,
    strict_nonmanifold=true,
    fix_orientation=true,
)
mesh_rep = rep.mesh
```

OBJ-level wrapper:

```julia
repair_obj_mesh("in.obj", "out.obj")
```

This pipeline removes invalid/degenerate faces, resolves orientation
inconsistencies, and iteratively drops non-manifold offenders when requested.

---

## 5) Coarsening to a Target RWG Count

Dense MoM complexity quickly becomes expensive, so practical runs often target a
maximum unknown count.

```julia
coarse = coarsen_mesh_to_target_rwg(mesh_rep, target_rwg)
mesh_sim = coarse.mesh
```

Internally, coarsening combines vertex clustering and repeated quality cleanup.

---

## 6) Pre-Solve Checklist

Before `build_rwg` and EFIE assembly:

1. `mesh_quality_ok` is true for your boundary policy.
2. `rwg = build_rwg(mesh_sim; precheck=true, ...)` succeeds.
3. `estimate_dense_matrix_gib(rwg.nedges)` fits your memory budget.

---

## 7) Minimal End-to-End Mesh Prep

```julia
using DifferentiableMoM

mesh0 = read_obj_mesh("Airplane.obj")
rep = repair_mesh_for_simulation(mesh0; allow_boundary=true, strict_nonmanifold=true)
coarse = coarsen_mesh_to_target_rwg(rep.mesh, 500)
mesh = coarse.mesh

rwg = build_rwg(mesh; precheck=true, allow_boundary=true, require_closed=false)
println("RWG unknowns = ", rwg.nedges)
```

---

## Code Mapping

- Mesh IO/repair/coarsening: `src/Mesh.jl`
- RWG precheck integration: `src/RWG.jl`
- Demo workflows: `examples/ex_repair_obj_mesh.jl`,
  `examples/ex_airplane_rcs.jl`

---

## Exercises

- Basic: compare quality reports before/after repair for one OBJ.
- Challenge: produce three coarsened meshes and plot RWG count vs memory
  estimate.
