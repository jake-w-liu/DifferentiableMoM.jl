# Sphere-vs-Mie Benchmark

## Purpose

Use a canonical PEC sphere benchmark against analytical Mie theory to validate
far-field/RCS behavior independently of cross-package comparisons.

---

## Learning Goals

After this chapter, you should be able to:

1. Run the sphere benchmark workflow.
2. Compare MoM and Mie RCS trends and error summaries.
3. Distinguish discretization limitations from implementation bugs.

---

## 1) Why Sphere-vs-Mie Matters

For a PEC sphere, Mie theory provides an analytical reference.
This gives a stronger baseline check for:

- far-field computation path,
- RCS post-processing path,
- angular trend correctness.

---

## 2) Run the Benchmark

```bash
julia --project=. examples/ex_pec_sphere_mie_benchmark.jl
```

Typical generated artifacts:

- `data/sphere_mie_benchmark_summary.csv`
- `data/sphere_monostatic_rcs.csv`

---

## 3) Theory Reference Call

Analytical helpers in package:

- `mie_s1s2_pec(...)`
- `mie_bistatic_rcs_pec(...)`

These are used by benchmark scripts for reference curves/values.

---

## 4) Interpreting Discrepancies

If MoM vs Mie mismatch is large:

1. check mesh quality and sphere discretization density,
2. confirm radius/frequency units,
3. verify angular sampling and polarization definitions,
4. inspect whether discrepancy is global or localized to sharp features.

Large persistent mismatch after refinement may indicate implementation issues.

---

## 5) Suggested Acceptance Pattern

Use trend-based acceptance:

- qualitative curve shape alignment first,
- then tighten quantitative thresholds with mesh refinement.

This avoids over-trusting single coarse-mesh numbers.

---

## Code Mapping

- Mie reference functions: `src/Mie.jl`
- RCS diagnostics: `src/Diagnostics.jl`
- Sphere workflows: `examples/ex_pec_sphere_mie_benchmark.jl`,
  `examples/ex_pec_sphere_rcs.jl`

---

## Exercises

- Basic: rerun benchmark at two mesh densities and compare error trend.
- Challenge: add one CI threshold based on summary metrics and justify its
  value.
