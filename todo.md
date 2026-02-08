# Documentation Roadmap (Pedagogical User Manual + API Reference)

## Objective
Create a complete, beginner-friendly manual that teaches:
1. Method of Moments (MoM) for EM scattering,
2. Differentiable/adjoint inverse design,
3. Practical use of `DifferentiableMoM.jl` APIs and workflows.

Audience assumption: users know basic EM but not computational EM.

---

## Scope and Deliverables

### D1 — Pedagogical Manual
- Foundations: IE/EFIE, RWG, singular integration, conditioning.
- Solver use: mesh handling, assembly, solve, far field, RCS.
- Differentiable design: adjoint derivation, gradients, optimization.
- Validation: internal checks, sphere-vs-Mie, Bempp cross-validation.
- Advanced usage: OBJ pipelines, large/iterative settings, troubleshooting.

### D2 — Tutorial Track (Runnable)
- Step-by-step tutorials mapped to examples/scripts.
- “Run now” commands with expected outputs/plots.
- Exercises (basic/challenge) in each tutorial.

### D3 — Complete API Reference
- Every exported API documented with:
  - signature,
  - arguments (types/units),
  - return values,
  - assumptions and failure conditions,
  - minimal usage snippet.

### D4 — Documentation Infrastructure
- `docs/` scaffold with structured pages.
- Build system (`Documenter.jl`) and page navigation.
- CI gate for docs build and doctest-style snippet checks.

---

## Information Architecture

### Part I — Computational EM Foundations
1. Why integral equations for open-region scattering
2. EFIE and boundary conditions (PEC / impedance)
3. MoM discretization and RWG basis
4. Singular/near-singular integration essentials
5. Conditioning and preconditioning intuition

### Part II — Package Fundamentals
1. Mesh import/repair/quality precheck/coarsening
2. Forward pipeline: mesh → RWG → Z/v → I
3. Far-field operators, Q matrices, and RCS outputs
4. Visualization and interpretation of simulation meshes

### Part III — Differentiable Inverse Design
1. Adjoint method from constrained objective
2. Impedance sensitivities and gradient assembly
3. Ratio objectives and two-adjoint workflow
4. L-BFGS workflow and practical tuning

### Part IV — Validation and Trustworthiness
1. Energy, reciprocity, objective consistency checks
2. Gradient verification (FD/complex-step context)
3. External validation: Bempp beam-centric comparisons
4. Sphere RCS vs Mie benchmark and acceptance thresholds

### Part V — Advanced Workflows
1. Complex OBJ platforms (e.g., aircraft mesh)
2. Auto-preconditioning and large-problem strategy
3. Robustness studies and interpretation
4. Common failure modes and debugging playbook

### Part VI — API Reference
- Types
- Mesh utilities
- RWG and basis evaluation
- EFIE assembly / excitation / solve
- Far-field / diagnostics / RCS
- Adjoint / optimization
- Mie utilities
- Visualization
- Verification helpers

### Appendices
- Conventions and units
- Symbol glossary
- Suggested reading path
- FAQ

---

## Pedagogical Template per Chapter
- Motivation (what problem this solves)
- Minimal theory (only necessary math)
- Code mapping (`src` modules + examples)
- Runnable command(s)
- Expected outputs/checkpoints
- Common mistakes and fixes
- Exercises (basic + challenge)

---

## Module-to-Docs Mapping
- `Types.jl` → core data structures and dimensional conventions
- `Mesh.jl` → mesh import/repair/quality/coarsening workflow
- `RWG.jl` → basis construction and interpretation
- `Quadrature.jl`, `Greens.jl`, `SingularIntegrals.jl` → numerical integration core
- `EFIE.jl`, `Excitation.jl`, `Solve.jl` → forward simulation pipeline
- `FarField.jl`, `QMatrix.jl`, `Diagnostics.jl` → objectives, radiation metrics, RCS
- `Adjoint.jl`, `Optimize.jl` → differentiable design and optimization
- `Verification.jl`, `Mie.jl` → trust checks and benchmark references
- `Visualization.jl` → geometry sanity and presentation utilities

---

## Milestones
- **M1:** docs scaffold + navigation + style guide
- **M2:** Part I foundations + first two tutorials
- **M3:** Part II/III + optimization tutorial track
- **M4:** Part IV validation + benchmark interpretation chapters
- **M5:** Full API reference + cross-links + CI docs gate

---

## Definition of Done
- New user can follow reading path from zero MoM background to running inverse design.
- All tutorial commands execute on clean environment.
- All exported APIs have complete docs.
- Docs build passes in CI.
- Manual and API links are cross-referenced and internally consistent.

---

## Immediate Execution Plan
1. Scaffold `docs/` with a complete chapter/page skeleton and navigation tree.
2. Add placeholder pages for all manual parts, tutorials, API, and appendices.
3. Hook up a reproducible docs build entry (`docs/make.jl`).
4. Populate chapters iteratively, starting from foundations and quickstart tutorials.
