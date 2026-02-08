# Tutorial: Beam Steering Design

## Purpose

Run the full differentiable inverse-design workflow for a reactive impedance
sheet using a directivity-ratio objective.

---

## Learning Goals

After this tutorial, you should be able to:

1. Configure target and total `Q` matrices for steering.
2. Run projected L-BFGS ratio optimization.
3. Inspect convergence trace and final pattern metrics.

---

## 1) Run the Scripted Workflow

From project root:

```bash
julia --project=. examples/ex_beam_steer.jl
```

This script assembles the optimization problem and writes data/figure artifacts.

---

## 2) Core Configuration Knobs

Most influential settings:

- mesh/patch resolution (`N` unknowns, `P` parameters),
- target cone width and steering direction,
- box bounds on impedance parameters,
- initialization (phase ramp strongly recommended for off-broadside steering).

---

## 3) Optimization Call Pattern

```julia
theta_opt, trace = optimize_directivity(
    Z_efie, Mp, v, Q_target, Q_total, theta0;
    reactive=true,
    maxiter=300,
    tol=1e-6,
    lb=fill(-500.0, P),
    ub=fill( 500.0, P),
)
```

---

## 4) Post-Run Checks

1. inspect objective trace monotonic trend,
2. compare PEC vs optimized patterns on the same angular grid,
3. compute target-angle gain and peak-angle shift.

---

## 5) Practical Tips

- If optimization stalls near broadside, use stronger asymmetry in initial map.
- If line search struggles, check gradient scale and objective normalization.
- Keep same quadrature/grid settings for fair before/after comparisons.

---

## Code Mapping

- Ratio optimizer: `src/Optimize.jl`
- Q construction: `src/QMatrix.jl`
- Far-field/pattern extraction: `src/FarField.jl`, `src/Diagnostics.jl`
- End-to-end example: `examples/ex_beam_steer.jl`

---

## Exercises

- Basic: run nominal steering case and report final objective value.
- Challenge: narrow target cone and compare resulting sidelobe behavior.
