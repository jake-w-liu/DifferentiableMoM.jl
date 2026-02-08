# Optimization Workflow

## Purpose

Provide the end-to-end practical loop for inverse design in this package,
including objective setup, solver calls, line search behavior, and diagnostic
checks.

---

## Learning Goals

After this chapter, you should be able to:

1. Configure a reproducible optimization run.
2. Interpret optimization traces (`J`, `|g|`) correctly.
3. Apply practical safeguards (bounds, initialization, conditioning).

---

## 1) Core Iteration Structure

At iteration ``k``:

1. assemble ``\mathbf Z(\theta^{(k)})``,
2. solve forward for ``\mathbf I^{(k)}``,
3. solve adjoint system(s),
4. compute gradient ``\mathbf g^{(k)}``,
5. update ``\theta`` using projected L-BFGS with backtracking.

---

## 2) Two Optimizers

- `optimize_lbfgs` for single quadratic objective.
- `optimize_directivity` for ratio objective (`Q_target` / `Q_total`).

Both support:

- box constraints via projection (`lb`, `ub`),
- optional regularization/preconditioning,
- reactive/resistive parameterization switch.

---

## 3) Initialization Matters

For off-broadside steering, symmetric initial maps often stall in symmetric
solutions. Use a phase-ramp initialization (or any physically informed
asymmetric seed) to help convergence toward steered solutions.

---

## 4) Monitoring and Stopping

Trace entries include:

- objective value `J`,
- gradient norm `|g|`.

Typical stopping criteria:

- gradient norm below tolerance,
- max-iteration cap,
- no meaningful objective improvement.

Always inspect final pattern/metrics, not only objective history.

---

## 5) Practical Run Template

```julia
theta_opt, trace = optimize_directivity(
    Z_efie, Mp, v, Q_target, Q_total, theta0;
    maxiter=300,
    tol=1e-6,
    m_lbfgs=10,
    reactive=true,
    lb=fill(-500.0, length(theta0)),
    ub=fill( 500.0, length(theta0)),
    preconditioning=:auto,
    iterative_solver=true,
)
```

---

## 6) Post-Optimization Checklist

1. Verify optimization trace is stable (no erratic spikes).
2. Recompute far field and beam metrics independently.
3. Run at least one perturbation scenario (frequency/incidence) for sensitivity.
4. Keep seed and settings logged for reproducibility.

---

## Code Mapping

- L-BFGS and ratio workflow: `src/Optimize.jl`
- Gradient kernels: `src/Adjoint.jl`
- Forward assembly/solve: `src/Solve.jl`, `src/EFIE.jl`
- Example run: `examples/ex_beam_steer.jl`

---

## Exercises

- Basic: run one optimization with and without box constraints.
- Challenge: compare convergence behavior for two different initial seeds and
  explain differences in the final beam pattern.
