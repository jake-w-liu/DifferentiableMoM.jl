# Appendix: Units and Conventions

## Quick Reference

- Time convention: $e^{+i\omega t}$
- Green function: $e^{-ikR}/(4\pi R)$
- Geometry units: meters
- Frequency: Hz
- Wavenumber: $k=2\pi f/c_0$
- Free-space impedance: $\eta_0\approx 376.73\ \Omega$

---

## Angular Coordinates

- $\theta$: polar angle from +z (`0` to `π`)
- $\phi$: azimuth (`0` to `2π`)

Spherical grids use midpoint quadrature with
$w_q=\sin\theta_q\,\Delta\theta\,\Delta\phi$.

---

## Common dB Conversions

- Power/area quantity to dB: `10*log10(x)`
- Field-amplitude quantity to dB: `20*log10(x)`

RCS is area-like, so use `10*log10(σ)`.

---

## Matrix Notation

- $\mathbf Z$: MoM system matrix
- $\mathbf I$: unknown current coefficients
- $\mathbf v$: excitation vector
- $(\cdot)^\dagger$: conjugate transpose

In code, adjoint solve uses `Z'`.

---

## Related Chapters

- Getting Started → Conventions
- Foundations → EFIE and Boundary Conditions
