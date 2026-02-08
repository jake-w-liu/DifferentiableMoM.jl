# Mathematical Prerequisites

## Purpose

This appendix provides concise refreshers on the mathematical concepts used throughout `DifferentiableMoM.jl`. It is designed for readers who need to bridge gaps between theoretical electromagnetics and computational implementation.

---

## Learning Goals

After this appendix, you should be able to:

1. Recognize and interpret vector calculus operators in the MoM context.
2. Understand surface integral formulations and their discretization.
3. Follow the mathematical derivations in the main documentation.

---

## 1) Vector Calculus Essentials

### Gradient, Divergence, and Curl

For a scalar field `φ(r)` and vector field `F(r)`:

```math
\nabla φ = \left(\frac{\partial φ}{\partial x}, \frac{\partial φ}{\partial y}, \frac{\partial φ}{\partial z}\right)
```

```math
\nabla \cdot \mathbf F = \frac{\partial F_x}{\partial x} + \frac{\partial F_y}{\partial y} + \frac{\partial F_z}{\partial z}
```

```math
\nabla \times \mathbf F = \begin{vmatrix}
\hat{\mathbf x} & \hat{\mathbf y} & \hat{\mathbf z} \\
\frac{\partial}{\partial x} & \frac{\partial}{\partial y} & \frac{\partial}{\partial z} \\
F_x & F_y & F_z
\end{vmatrix}
```

### Surface Divergence

For a vector field tangent to a surface `Γ` with unit normal `n̂`:

```math
\nabla_s \cdot \mathbf F = \nabla \cdot \mathbf F - \hat{\mathbf n} \cdot (\nabla \times (\mathbf F \times \hat{\mathbf n}))
```

**Physical meaning**: measures how much field "flows out" along the surface, not through it.

---

## 2) Surface Integrals and Parametrization

### Triangle Parametrization

A triangle with vertices `r₁, r₂, r₃` can be parametrized using barycentric coordinates `(u, v)`:

```math
\mathbf r(u, v) = \mathbf r_1 + u(\mathbf r_2 - \mathbf r_1) + v(\mathbf r_3 - \mathbf r_1)
```

where `u ≥ 0, v ≥ 0, u + v ≤ 1`.

**Area calculation**: `A = |(r₂ - r₁) × (r₃ - r₁)| / 2`

### ASCII Diagram: Triangle Geometry and Barycentric Coordinates

```
    Triangle with vertices r₁, r₂, r₃
    Barycentric coordinates (u, v):
    
          r₃
          │\
          │ \
          │  \
          │   \
          │    \
          │     \
          │      \
          │       \
          │        \
          │         \
        r₁───────────r₂
    
    Any point in triangle: r(u,v) = r₁ + u(r₂ - r₁) + v(r₃ - r₁)
    
    Constraints: u ≥ 0, v ≥ 0, u + v ≤ 1
    
    Special points:
    - (0,0) → r₁
    - (1,0) → r₂  
    - (0,1) → r₃
    - (0.5,0) → midpoint of edge r₁-r₂
    - (0,0.5) → midpoint of edge r₁-r₃
    - (0.5,0.5) → not in triangle! (u+v=1 > 1)
    - (1/3,1/3) → centroid
    
    Area A = |(r₂ - r₁) × (r₃ - r₁)| / 2
    Jacobian |J| = 2A for integration
```

### Surface Integral Transformation

```math
\int_T f(\mathbf r) \, dS = \int_0^1 \int_0^{1-u} f(\mathbf r(u, v)) \, |\mathbf J| \, dv \, du
```

where `|J| = 2A` is the Jacobian determinant for the triangle.

---

## 3) Complex Numbers and Phasors

### Time-Harmonic Representation

With `e^{+iωt}` convention:

```math
\mathbf E(\mathbf r, t) = \text{Re}\{\tilde{\mathbf E}(\mathbf r) e^{+i\omega t}\}
```

where `tilde{E}(r)` is the complex phasor amplitude.

### Complex Conjugate Operations

For complex vectors `a, b`:

```math
\mathbf a \cdot \mathbf b^* = \sum_i a_i b_i^*
```

```math
\|\mathbf a\|^2 = \mathbf a \cdot \mathbf a^*
```

---

## 4) Linear Algebra Concepts

### Matrix Condition Number

```math
\kappa(\mathbf A) = \|\mathbf A\| \cdot \|\mathbf A^{-1}\|
```

**Interpretation**: measures sensitivity of solutions to perturbations. Large `κ` indicates ill-conditioning.

### Hermitian Matrices

Matrix `H` is Hermitian if `H = H†` (conjugate transpose). Properties:

- Eigenvalues are real
- Eigenvectors are orthogonal
- `x† H x` is real for any complex vector `x`

---

## 5) Special Functions in Electromagnetics

### Free-Space Green's Function

```math
G(\mathbf r, \mathbf r') = \frac{e^{-ik|\mathbf r - \mathbf r'|}}{4\pi |\mathbf r - \mathbf r'|}
```

**Key properties**:
- Singular at `r = r'`
- Satisfies Helmholtz equation: `(∇² + k²)G = -δ(r - r')`
- Represents outgoing wave solution

### Wavenumber and Wave Parameters

```math
k = \frac{2\pi}{\lambda} = \frac{\omega}{c}
```

```math
\eta_0 = \sqrt{\frac{\mu_0}{\epsilon_0}} \approx 376.73 \, \Omega
```

---

## 6) Quadrature and Numerical Integration

### Gaussian Quadrature on Triangles

Reference triangle integration:

```math
\int_{T_{\text{ref}}} f(u, v) \, du \, dv \approx \sum_{q=1}^{N_q} w_q f(u_q, v_q)
```

**Physical triangle mapping**: multiply by Jacobian `2A`.

### Order of Accuracy

- Linear functions: exact with 1-point quadrature
- Quadratic functions: need 3+ points for exactness
- Singular integrals: require special treatment

---

## 7) Method of Moments Concepts

### Inner Product Definition

```math
\langle \mathbf f, \mathbf g \rangle = \int_\Gamma \mathbf f(\mathbf r) \cdot \mathbf g^*(\mathbf r) \, dS
```

### Galerkin Method

Choose testing functions equal to basis functions: `w_m = f_m`. This leads to symmetric matrices for self-adjoint operators.

### Matrix Assembly Pattern

```math
Z_{mn} = \langle f_m, \mathcal T[f_n] \rangle
```

where `T` is the integral operator (e.g., EFIE operator).

---

## 8) Common Mathematical Identities

### Vector Triple Product

```math
\mathbf a \times (\mathbf b \times \mathbf c) = \mathbf b(\mathbf a \cdot \mathbf c) - \mathbf c(\mathbf a \cdot \mathbf b)
```

### Divergence Theorem (Surface Form)

```math
\int_\Gamma \nabla_s \cdot \mathbf F \, dS = \oint_{\partial \Gamma} \mathbf F \cdot \hat{\mathbf t} \, dl
```

where `t̂` is the tangent to the boundary curve.

---

## Code Mapping

- Vector operations: `src/Geometry.jl`
- Quadrature rules: `src/Quadrature.jl`
- Green's function: `src/Greens.jl`
- Linear algebra utilities: `src/Solve.jl`

---

## Exercises

- Basic: verify the triangle area formula for a right triangle with vertices (0,0,0), (1,0,0), (0,1,0).
- Challenge: derive the surface divergence expression for a vector field tangent to a sphere.

---

## Quick Reference

| Symbol | Meaning | Typical Units |
|--------|---------|---------------|
| `∇` | Gradient operator | 1/m |
| `∇·` | Divergence operator | 1/m |
| `∇×` | Curl operator | 1/m |
| `∇s·` | Surface divergence | 1/m |
| `k` | Wavenumber | rad/m |
| `η₀` | Free-space impedance | Ω |
| `G` | Green's function | 1/m |
| `κ` | Condition number | dimensionless |