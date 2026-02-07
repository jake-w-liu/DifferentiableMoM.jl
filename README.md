# DifferentiableMoM.jl

Julia implementation of a differentiable EFIE-MoM workflow for inverse design of reactive impedance metasurfaces.

## Quick Start

From this folder:

```bash
julia --project=. test/runtests.jl
julia --project=. examples/ex_beam_steer.jl
julia --project=. plot.jl
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
