# DifferentiableMoM.jl

Julia implementation of a differentiable EFIE-MoM workflow for inverse design of reactive impedance metasurfaces.

## Quick Start

From this folder:

```bash
julia --project=. test/runtests.jl
julia --project=. examples/ex_beam_steer.jl
julia --project=. plot.jl
julia --project=. validation/paper/generate_consistency_report.jl
```

## Repository Layout

- `src/`: core solver and optimization implementation
- `examples/`: reproducible study scripts
- `test/`: verification and consistency tests
- `data/`: **tracked numerical outputs** used by manuscript tables/claims
- `figs/`: **generated figures** (ignored by git; regenerate with `julia --project=. plot.jl`)
- `validation/bempp/`: independent Bempp-cl cross-validation scripts

## Data vs Figures Policy

- `data/` should be committed and reviewed as source-of-truth for reported metrics.
- `figs/` is treated as generated output and is intentionally not tracked.

## Bempp Cross-Validation

See `validation/bempp/README.md` for setup and commands.
For acceptance-focused impedance external checks, use
`validation/bempp/run_impedance_validation_matrix.py`.
For convention-reconciliation sweeps, use
`validation/bempp/sweep_impedance_conventions.py`.

## Paper Consistency Snapshot

Generate a manuscript-facing metrics snapshot from tracked CSV outputs:

```bash
julia --project=. validation/paper/generate_consistency_report.jl
```

This writes:
- `data/paper_metrics_snapshot.csv`
- `data/paper_consistency_report.md`
