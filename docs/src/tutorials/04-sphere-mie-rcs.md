# Tutorial: Sphere-Mie RCS

## Purpose

Run a canonical PEC sphere benchmark and compare MoM-computed RCS with Mie
reference values.

---

## Learning Goals

After this tutorial, you should be able to:

1. Generate sphere RCS data from MoM workflow.
2. Compute/reference Mie RCS in the same scenario.
3. Diagnose mismatch sources (mesh, scaling, sampling, implementation).

---

## 1) Run Benchmark Script

```bash
julia --project=. examples/ex_pec_sphere_mie_benchmark.jl
```

Outputs include benchmark summaries under `data/`.

---

## 2) Standalone Sphere MoM Run

If you only want package-side RCS without Mie comparison:

```bash
julia --project=. examples/ex_pec_sphere_rcs.jl
```

---

## 3) Comparison Steps

1. Ensure same radius/frequency.
2. Use same angular sampling for plotted comparisons.
3. Compare both linear and dB scales.
4. Refine mesh before concluding algorithmic mismatch.

---

## 4) Interpretation Guide

- Good trend but amplitude offset: often discretization/coarseness issue.
- Strong shape mismatch after refinement: investigate implementation.
- Local spikes around nulls: may be numerically sensitive and should be
  interpreted carefully.

---

## Code Mapping

- Mie reference: `src/Mie.jl`
- RCS diagnostics: `src/Diagnostics.jl`
- Example workflows: `examples/ex_pec_sphere_mie_benchmark.jl`,
  `examples/ex_pec_sphere_rcs.jl`

---

## Exercises

- Basic: run benchmark at one mesh level and inspect summary CSV.
- Challenge: repeat at higher mesh density and compare error trend.
