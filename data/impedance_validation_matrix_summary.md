# Impedance Validation Matrix Summary

## Bempp Convention Configuration
- op-sign: `minus`
- rhs-cross: `e_cross_n`
- rhs-sign: `1.0`
- phase-sign: `plus`
- zs-scale: `0.002654418727984993`

## Case Results
| Case | f (GHz) | Zs imag (ohm) | theta_inc (deg) | RMSE (dB) | Mean |ΔD| near target (dB) | Max |ΔD| (dB) |
|---|---:|---:|---:|---:|---:|---:|
| case01_z50_n0_f3p00 | 3.00 | 50.0 | 0.0 | 4.720 | 2.700 | 21.394 |
| case02_z100_n0_f3p00 | 3.00 | 100.0 | 0.0 | 2.701 | 0.641 | 14.415 |
| case03_z200_n0_f3p00 | 3.00 | 200.0 | 0.0 | 6.237 | 4.017 | 23.664 |
| case04_z300_n0_f3p00 | 3.00 | 300.0 | 0.0 | 8.309 | 3.622 | 25.050 |
| case05_z200_n5_f3p00 | 3.00 | 200.0 | 5.0 | 6.331 | 2.946 | 24.483 |
| case06_z200_n0_f3p06 | 3.06 | 200.0 | 0.0 | 6.237 | 4.017 | 23.656 |

## Acceptance Gates
- Cases with near-target mean |ΔD| <= 1.5 dB: 1/6
- Cases with global RMSE <= 2.5 dB: 0/6
- Cases with max |ΔD| <= 25 dB: 5/6

- Matrix gate status (>=4/6 for target and RMSE, all <=25 dB max): FAIL
