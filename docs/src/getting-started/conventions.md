# Conventions and Units

## Purpose

Make all numerical experiments unambiguous by documenting sign conventions,
units, and notation choices used throughout `DifferentiableMoM.jl`.

---

## Learning Goals

After this chapter, you should be able to:

1. Match package outputs with external references using consistent conventions.
2. Avoid unit/scale errors in geometry, frequency, and RCS interpretation.
3. Interpret objective and gradient signs correctly.

---

## 1) Time Convention

The package uses the $e^{+i\omega t}$ convention.
Therefore the scalar free-space Green function is

```math
G(\mathbf r,\mathbf r')
=
\frac{e^{-ik|\mathbf r-\mathbf r'|}}{4\pi |\mathbf r-\mathbf r'|}.
```

If a reference uses $e^{-i\omega t}$, phase signs will flip.

---

## 2) Geometric Units

- Lengths are in **meters**.
- Frequency is in **Hz**.
- Wavelength is $\lambda_0=c_0/f$.
- Wavenumber is $k=2\pi/\lambda_0$ (rad/m).

Imported OBJ meshes are treated as coordinate values in meters unless you
rescale explicitly before solving.

---

## 3) Electrical Quantities

- Surface current density: A/m.
- Surface impedance: $\Omega$.
- Far-field electric field: V/m (up to the usual asymptotic normalization in
  the chosen formulation).
- Directivity/RCS values in scripts are often reported in dB units for plotting.

---

## 4) Matrix and Vector Notation

- Current coefficients: $\mathbf I\in\mathbb C^N$.
- MoM system: $\mathbf Z\mathbf I=\mathbf v$.
- Conjugate transpose in equations: $(\cdot)^\dagger$.
- In Julia code, conjugate dot products are done with `dot(a,b)` and adjoint
  solves use `Z' \\ rhs`.

---

## 5) Objective Sign Conventions

Quadratic objective:

```math
J = \mathbf I^\dagger \mathbf Q \mathbf I,\qquad \mathbf Q\succeq 0.
```

For ratio objectives, the code maximizes

```math
J_{\mathrm{ratio}} =
\frac{\mathbf I^\dagger\mathbf Q_t\mathbf I}
{\mathbf I^\dagger\mathbf Q_{\mathrm{tot}}\mathbf I}.
```

L-BFGS is implemented internally as minimization, so maximize-mode uses a sign
flip of objective gradient in optimizer internals.

---

## 6) Angle Conventions

Spherical grids follow:

- $\theta\in[0,\pi]$ polar angle from +z,
- $\phi\in[0,2\pi)$ azimuth angle in the x-y plane.

`make_sph_grid(Ntheta,Nphi)` uses midpoint sampling with weights
$w_q=\sin\theta_q\,\Delta\theta\,\Delta\phi$.

---

## 7) Cross-Validation Conventions

When comparing with external solvers:

1. match time convention,
2. match geometry scale,
3. match excitation polarization and incidence,
4. compare physically aligned metrics (beam-centric metrics are often more
   informative than null-dominated global dB residuals).

---

## Code Mapping

- Green kernel and convention: `src/basis/Greens.jl`
- Spherical grid definition: `src/postprocessing/FarField.jl`
- Objective/adjoint implementation: `src/optimization/Adjoint.jl`, `src/optimization/Optimize.jl`
- Units in diagnostics/RCS: `src/postprocessing/Diagnostics.jl`

---

## Exercises

- Basic: verify that changing frequency by 2% changes $k$ by 2%.
- Challenge: reproduce one case with an external solver and list every
  convention you had to align.
