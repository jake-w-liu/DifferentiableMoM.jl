# Complex OBJ Platforms

## Purpose

Describe robust workflows for importing and simulating complex CAD/OBJ
platforms (e.g., aircraft-like geometry) within the package’s dense-MoM
envelope.

---

## Learning Goals

After this chapter, you should be able to:

1. Diagnose OBJ geometry issues that break simulation.
2. Repair/coarsen meshes to a stable simulation-ready state.
3. Run platform RCS heuristics with reproducible preprocessing.

---

## 1) Common OBJ Failure Modes

- non-manifold edges,
- inconsistent triangle orientation,
- degenerate/duplicate faces,
- unrealistic scale (mm interpreted as m),
- disconnected fragments.

The mesh-quality pipeline is designed to expose these early.

---

## 2) Recommended Platform Workflow

1. `read_obj_mesh(...)`
2. `repair_mesh_for_simulation(...)`
3. optional `coarsen_mesh_to_target_rwg(...)`
4. `build_rwg(...; precheck=true)`
5. forward solve and RCS extraction

Use visualization at steps 2 and 3 for geometric sanity.

---

## 3) Reproducibility Pattern

Always save intermediate artifacts:

- repaired OBJ,
- coarsened OBJ used for simulation,
- preview plots,
- metadata (`Nv`, `Nt`, RWG count, frequency).

This makes platform studies auditable and repeatable.

---

## 4) Performance Reality

Complex platforms often require coarsening for dense solves.
Tradeoff decisions should be made explicitly by tracking:

- unknown count,
- memory estimate,
- key observable stability (not just visual similarity).

---

## Code Mapping

- OBJ import and repair: `src/Mesh.jl`
- Visualization: `src/Visualization.jl`
- Platform example: `examples/ex_airplane_rcs.jl`

---

## Exercises

- Basic: run repair+quality report on one external OBJ.
- Challenge: compare two coarsening levels and discuss RCS-fidelity tradeoff.
