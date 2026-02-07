# Paper Consistency Report

- Generated: 2026-02-07 17:28:32
- Repository commit: `f0e832f`

## Check Summary
- [PASS] Mesh-sweep max gradient error <= 3e-6
- [PASS] Minimum energy ratio >= 0.98
- [PASS] Nominal J_opt > nominal J_PEC
- [PASS] PEC near-target mean |ΔD| <= 0.5 dB
- [PASS] Impedance near-target mean |ΔD| <= 3.0 dB
- [PASS] Cost assembly increases with N
- [PASS] Cost solve increases with N
- [PASS] Cost per-iteration increases with N

## Key Metrics
| Metric | Value |
|---|---:|
| Max gradient rel. error (reference) | 2.864e-07 |
| Max gradient rel. error (mesh sweep) | 2.829e-06 |
| Minimum energy ratio | 0.981 |
| Nominal J_opt | 13.249 % |
| Nominal J_PEC | 0.246 % |
| Nominal improvement factor | 53.91 x |
| Gain at target angle (nearest 29.5 deg) | 22.667 dB |
| J_opt at +2% frequency | 4.773 % |

## External Cross-Validation Metrics
| Case | Mean |ΔD| (global) | Mean |ΔD| near target |
|---|---:|---:|
| PEC (Bempp vs Julia) | 0.711 dB | 0.337 dB |
| Impedance (Bempp vs Julia) | 2.641 dB | 2.603 dB |

## Impedance Matrix Gates
- Cases with near-target mean |ΔD| <= 1.5 dB: 0/6
- Cases with RMSE <= 2.5 dB: 0/6
- Cases with max |ΔD| <= 25 dB: 0/6
- Matrix gate status: FAIL
