# Robustness Studies

## Purpose

An optimized design at one nominal condition can fail when frequency or
illumination changes.

This chapter explains how to run and interpret robustness sweeps in this
package, and how to distinguish:

- **performance at the design point**, and
- **stability around the design point**.

---

## Learning Goals

After this chapter, you should be able to:

1. Define robustness metrics aligned with your beam objective.
2. Run the provided frequency/incidence perturbation workflow.
3. Diagnose whether degradation is expected detuning or likely implementation
   error.

---

## Robustness Is a Multi-Scenario Question

In beam-steering mode, the core objective is a ratio:

```math
J(\theta;\xi)
= 
\frac{\mathbf I(\theta;\xi)^\dagger \mathbf Q_{\text{target}}(\xi)\,\mathbf I(\theta;\xi)}
{\mathbf I(\theta;\xi)^\dagger \mathbf Q_{\text{total}}(\xi)\,\mathbf I(\theta;\xi)},
```

where `ξ` denotes scenario settings (frequency, incidence angle, etc.).

A design optimized only for one scenario `ξ0` usually becomes **narrowband**:
small perturbations can move the global maximum away from the target direction,
even if target-direction gain remains high.

---

## Metrics You Should Track

The package robustness workflow reports both objective-space and pattern-space
quantities:

1. `J_opt_pct`: optimized design ratio under scenario `ξ`.
2. `J_pec_pct`: PEC baseline ratio under the same `ξ`.
3. `gain_target_dB`: optimized-minus-PEC directivity at target angle.
4. `peak_theta_opt_deg`: actual peak angle on the selected cut.
5. `peak_opt_dBi`: peak value on that cut.

Together these answer two different questions:

- **Does the design still help at target?** (`gain_target_dB`)
- **Does the design still peak at target?** (`peak_theta_opt_deg`)

---

## Run the Sweep

From project root:

```bash
julia --project=. validation/robustness/run_robustness_sweep.jl
```

Output:

- `data/robustness_sweep.csv`

with scenario rows and the metrics listed above.

---

## Interpreting a Typical Detuning Pattern

A common pattern is:

- target-angle gain stays positive,
- but global peak shifts toward broadside at `+Δf`.

This is usually **not** a solver bug by itself.
It is expected when the impedance map was optimized at one frequency and then
reused unchanged at another frequency.

Why this happens physically:

- phase progression needed for steering depends on `k = 2π/λ`,
- reactive loading is dispersive in effective phase response,
- fixed map + changed `k` ⇒ wrong phase slope for steering target.

So “still better than PEC at target” can coexist with “global maximum moved away
from target.”

---

## When to Suspect an Implementation Problem

Treat robustness degradation as suspicious only if one or more of these occur:

- internal consistency checks fail at the same scenario,
- residuals grow abnormally,
- energy/objective consistency breaks,
- very small perturbations produce discontinuous/nonphysical jumps,
- PEC reference itself behaves inconsistently under the same sweep.

If those checks are clean, robustness loss is usually a design-bandwidth issue,
not a numerical correctness issue.

---

## Toward Robust Design (Not Just Robustness Evaluation)

Current scripts evaluate a fixed optimized map under scenario perturbations.
To *design for robustness*, use a multi-scenario objective such as:

```math
J_{\mathrm{mean}}(\theta)=\frac{1}{S}\sum_{s=1}^{S}J(\theta;\xi_s),
\qquad
J_{\mathrm{worst}}(\theta)=\min_s J(\theta;\xi_s).
```

For smooth aggregates (like weighted means), gradients are assembled by summing
scenario gradients.

This can be built with existing primitives by looping over scenarios, solving
forward/adjoint per scenario, and combining gradients before one optimizer
update.

---

## Recommended Reporting Style

For publications and engineering reports, avoid a single “robust/not robust”
label.
Report both:

- target-angle gain retention,
- peak-angle drift,
- objective ratio drift.

That gives a transparent picture of bandwidth vs steering accuracy.

---

## Code Mapping

- Robustness sweep script: `validation/robustness/run_robustness_sweep.jl`
- Directivity optimization core: `src/Optimize.jl`
- EFIE / excitation / far field: `src/EFIE.jl`, `src/Excitation.jl`,
  `src/FarField.jl`
- Diagnostics for interpretation: `src/Diagnostics.jl`

---

## Exercises

- Basic: run the sweep and identify which metric first indicates detuning.
- Practical: add one extra frequency point and compare trend of
  `peak_theta_opt_deg`.
- Challenge: prototype a 3-scenario mean objective and compare its robustness
  against the single-scenario design.
