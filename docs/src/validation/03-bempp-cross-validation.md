# Bempp Cross-Validation

## Purpose

Provide external consistency evidence against an independent BEM code
(`Bempp-cl`) using matched scenarios and beam-centric comparison metrics.

---

## Learning Goals

After this chapter, you should be able to:

1. Run package-to-Bempp comparison scripts.
2. Interpret beam-centric agreement metrics.
3. Avoid misleading conclusions from null-dominated global residuals.

---

## 1) Comparison Philosophy

For steering workflows, we prioritize beam-centric metrics on aligned cuts:

- main-beam angle difference,
- main-beam level difference,
- strongest sidelobe angle/level differences.

These metrics map directly to steering utility.

---

## 2) Script Entry Points

Main workflow scripts in `validation/bempp/`:

- `run_pec_cross_validation.py`
- `run_impedance_cross_validation.py`
- `run_impedance_validation_matrix.py`
- `compare_pec_to_julia.py`
- `compare_impedance_to_julia.py`

Refer to `validation/bempp/README.md` for dependencies and run order.

---

## 3) Reproducible Run Pattern

Typical pattern:

1. generate Julia reference case,
2. run Bempp case with matched settings,
3. compute aligned metrics and save report artifacts.

Always match:

- frequency,
- incidence direction/polarization,
- geometry and scale,
- angular sampling.

---

## 4) Metric Interpretation

Good agreement in main-beam direction/level with modest sidelobe deviation is
often acceptable in practical external consistency checks, especially when deep
null regions dominate global dB residuals.

If beam-centric metrics degrade strongly, inspect convention mismatch first
(time sign, normalization, geometry scaling) before concluding solver bugs.

---

## 5) Known Practical Caveat

Different codes may implement impedance weak forms and quadrature details
differently. Cross-code impedance-loaded comparisons are therefore interpreted
as external consistency checks, not strict identity proofs.

---

## Code Mapping

- Python comparison workflows: `validation/bempp/*.py`
- Julia reference generation: `validation/bempp/run_impedance_case_julia_reference.jl`
- Report aggregation: `validation/paper/generate_consistency_report.jl`

---

## Exercises

- Basic: run one PEC cross-validation case and inspect the generated report.
- Challenge: run a multi-impedance matrix and summarize worst-case
  beam-centric deltas.
