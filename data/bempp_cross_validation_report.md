# Bempp vs Julia PEC Cross-Validation

## Global Error Metrics
- Mean delta (Bempp - Julia): 0.1095 dB
- Mean absolute delta: 0.7112 dB
- RMSE: 1.2316 dB
- Max absolute delta: 10.6636 dB

## Phi=0 Cut Metrics
- Mean absolute delta: 0.6917 dB
- RMSE: 1.0614 dB
- Max absolute delta: 4.8951 dB

## Directional Slices
- Near 0 deg: nearest theta = 0.5 deg, mean abs delta = 0.0232 dB, max abs delta = 0.0235 dB
- Near 30 deg: nearest theta = 29.5 deg, mean abs delta = 0.3375 dB, max abs delta = 1.0851 dB

## Notes
- Julia reference columns: `data/beam_steer_farfield.csv` -> `dir_pec_dBi`
- Bempp reference columns: `data/bempp_pec_farfield.csv` -> `dir_bempp_dBi`
- Grid match uses rounded `(theta_deg, phi_deg)` keys to 1e-6 deg.
