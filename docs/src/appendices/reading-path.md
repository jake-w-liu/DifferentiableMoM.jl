# Suggested Reading Paths

## Path A: New to Computational EM

**Prerequisite check**: If you need refreshers on vector calculus, complex numbers, or linear algebra, start with [Mathematical Prerequisites](mathematical-prerequisites.md).

1. Getting Started
2. Part I — Foundations (all chapters)
3. Part II — Package Fundamentals
4. Tutorial 01 (First PEC Plate)
5. Tutorial 04 (Sphere–Mie benchmark)

Goal: build numerical intuition before optimization.

---

## Path B: Interested in Inverse Design

1. Part I Chapters 2–5
2. Part III — Differentiable Design (all chapters)
3. Tutorial 02 (Adjoint gradient check)
4. Tutorial 03 (Beam steering design)
5. Part IV — Validation

Goal: trust gradients, then optimize.

---

## Path C: CAD/Platform Scattering User

1. Part II Chapter 1 (Mesh pipeline)
2. Tutorial 06 (OBJ repair workflow)
3. Tutorial 05 (Airplane RCS)
4. Part V Chapters 1, 2, and 4

Goal: robust imported-mesh simulation with realistic resource limits.

---

## Path D: API-First Developer

1. API Overview
2. API Types, Mesh, RWG
3. API Assembly/Solve + Far-Field/RCS
4. API Adjoint/Optimization + Verification
5. Source modules in `src/` as needed

Goal: implement custom workflows quickly.

---

## Path E: Validation-Focused Reviewer

1. Part IV — Validation
2. Tutorial 04 (Sphere–Mie)
3. Tutorial 03 (Beam steering)
4. API Verification

Goal: reproduce consistency gates and benchmark evidence.
