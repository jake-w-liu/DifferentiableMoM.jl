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

## Outputs

- `data/bempp_pec_farfield.csv`
- `data/bempp_pec_cut_phi0.csv`
- `data/bempp_pec_metadata.json`
- `data/bempp_cross_validation_report.json`
- `data/bempp_cross_validation_report.md`

## Notes

- Angular sampling is matched to the Julia grid (`180 x 72`) with centered bins.
- Comparison keys are matched on `(theta_deg, phi_deg)` rounded to `1e-6` deg.
- Small dB-level differences are expected because discretization and kernels differ
  between implementations.
