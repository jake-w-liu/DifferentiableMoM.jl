# Impedance Validation Matrix Summary

## Case Results
| Case | f (GHz) | Zs imag (ohm) | theta_inc (deg) | RMSE (dB) | Mean |ΔD| near target (dB) | Max |ΔD| (dB) |
|---|---:|---:|---:|---:|---:|---:|
| case01_z50_n0_f3p00 | 3.00 | 50.0 | 0.0 | 7.801 | 2.821 | 41.874 |
| case02_z100_n0_f3p00 | 3.00 | 100.0 | 0.0 | 6.024 | 3.377 | 36.240 |
| case03_z200_n0_f3p00 | 3.00 | 200.0 | 0.0 | 9.783 | 6.897 | 43.119 |
| case04_z300_n0_f3p00 | 3.00 | 300.0 | 0.0 | 8.311 | 4.240 | 42.423 |
| case05_z200_n5_f3p00 | 3.00 | 200.0 | 5.0 | 7.999 | 5.161 | 32.775 |
| case06_z200_n0_f3p06 | 3.06 | 200.0 | 0.0 | 9.783 | 6.897 | 43.119 |

## Acceptance Gates
- Cases with near-target mean |ΔD| <= 1.5 dB: 0/6
- Cases with global RMSE <= 2.5 dB: 0/6
- Cases with max |ΔD| <= 25 dB: 0/6

- Matrix gate status (>=4/6 for target and RMSE, all <=25 dB max): FAIL
