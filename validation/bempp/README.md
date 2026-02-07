# Bempp-cl Cross-Validation Scripts

This folder provides an independent PEC full-wave reference using `bempp-cl`,
then compares it against Julia outputs in `data/beam_steer_farfield.csv`.

## Scope

- Implemented here: PEC cross-validation of far-field directivity.
- Not implemented here: impedance-sheet optimization in Bempp.

The goal is to add an independent-solver check for the PEC baseline used by the
paper's beam-steering study.

## Prerequisites

- Python 3.10+
- `bempp-cl`
- `numpy`
- Gmsh available on the system path (used by BEM mesh generation)

Example install:

```bash
pip install -r validation/bempp/requirements.txt
```

## Run

From project root (`DifferentiableMoM.jl/`):

1) Generate Bempp PEC reference:

```bash
python validation/bempp/run_pec_cross_validation.py
```

2) Compare Bempp vs Julia PEC far field:

```bash
python validation/bempp/compare_pec_to_julia.py
```

Optional threshold gate:

```bash
python validation/bempp/compare_pec_to_julia.py --max-rmse-db 2.0 --max-abs-db 6.0
```

### Impedance-Loaded (Single Case)

Generate Julia and Bempp fields for a chosen impedance case:

```bash
julia --project=. validation/bempp/run_impedance_case_julia_reference.jl \
  --freq-ghz 3.0 --theta-ohm 200 --theta-inc-deg 0 --phi-inc-deg 0 \
  --output-prefix impedance

python validation/bempp/run_impedance_cross_validation.py \
  --freq-ghz 3.0 --zs-imag-ohm 200 --theta-inc-deg 0 --phi-inc-deg 0 \
  --output-prefix impedance

python validation/bempp/compare_impedance_to_julia.py \
  --output-prefix impedance --target-theta-deg 30
```

### Impedance-Loaded Validation Matrix (Acceptance Guide)

Run the 6-case impedance matrix and produce aggregate summary artifacts:

```bash
python validation/bempp/run_impedance_validation_matrix.py
```

Dry-run command preview:

```bash
python validation/bempp/run_impedance_validation_matrix.py --dry-run
```

Optional convention overrides (for reconciliation experiments):

```bash
python validation/bempp/run_impedance_validation_matrix.py \
  --bempp-op-sign minus \
  --bempp-rhs-cross e_cross_n \
  --bempp-rhs-sign 1.0 \
  --bempp-phase-sign plus \
  --bempp-zs-scale 1.0
```

### Impedance Convention Sweep

Sweep Bempp convention variants against a fixed Julia impedance reference:

```bash
python validation/bempp/sweep_impedance_conventions.py --run-julia
```

Outputs:

- `data/impedance_convention_sweep.csv`
- `data/impedance_convention_sweep.md`
- `data/impedance_convention_sweep.json`

## Outputs

- `data/bempp_pec_farfield.csv`
- `data/bempp_pec_cut_phi0.csv`
- `data/bempp_pec_metadata.json`
- `data/bempp_cross_validation_report.json`
- `data/bempp_cross_validation_report.md`
- `data/impedance_validation_matrix_summary.csv` (matrix run)
- `data/impedance_validation_matrix_summary.md` (matrix run)
- `data/impedance_validation_matrix_summary.json` (matrix run)

## Notes

- Angular sampling is matched to the Julia grid (`180 x 72`) with centered bins.
- Comparison keys are matched on `(theta_deg, phi_deg)` rounded to `1e-6` deg.
- Small dB-level differences are expected because discretization and kernels differ
  between implementations.
