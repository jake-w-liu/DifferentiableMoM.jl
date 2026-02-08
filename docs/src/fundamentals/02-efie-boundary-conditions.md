# EFIE and Boundary Conditions

## Purpose

This chapter derives the exact EFIE form used in the package and connects each
term to implementation.

---

## Learning Goals

After this chapter, you should be able to:

1. Derive the PEC and impedance-sheet equations from tangential boundary
   conditions.
2. Map the continuous EFIE terms to the assembled matrix blocks.

---

## Step 1: Time Convention and Green's Function

The package uses the $e^{+i\omega t}$ time convention.
With that choice, the free-space scalar Green's function is:

```math
G(\mathbf r,\mathbf r')
=
\frac{e^{-ik|\mathbf r-\mathbf r'|}}{4\pi|\mathbf r-\mathbf r'|}.
```

This is implemented by `greens` in `src/Greens.jl`.

### Sign sanity check from the wave equation

Under $e^{+i\omega t}$, time derivatives map as

```math
\frac{\partial}{\partial t}\rightarrow +i\omega,\qquad
\frac{\partial^2}{\partial t^2}\rightarrow -\omega^2.
```

For Helmholtz form $(\nabla^2+k^2)u=-\delta$, the outgoing fundamental solution
is $e^{-ikR}/(4\pi R)$, matching the implementation.  
If a reference uses $e^{-i\omega t}$, the exponential sign flips.

---

## Step 2: Field Decomposition and Tangential Projection

Write total field as

```math
\mathbf E^{\mathrm{tot}}
=
\mathbf E^{\mathrm{inc}}+\mathbf E^{\mathrm{sca}}.
```

On a surface, only tangential field enters EFIE boundary conditions:

```math
\mathbf E_t = \mathbf P_t\mathbf E,
\qquad
\mathbf P_t=\mathbf I-\hat{\mathbf n}\hat{\mathbf n}^{T}.
```

The current unknown is tangential and expanded in RWG basis later.

---

## Step 3: EFIE Operator Form

For surface current `J`, the scattered field operator is

```math
\mathbf E^{\mathrm{sca}}(\mathbf r)
=
-i\omega\mu_0
\int_{\Gamma}
\left[\mathbf I+\frac{1}{k^2}\nabla\nabla\right]
G(\mathbf r,\mathbf r')\,\mathbf J(\mathbf r')\,dS'.
```

Testing with RWG basis and integrating by parts leads to the mixed-potential
MoM entry:

```math
Z_{mn}^{\mathrm{EFIE}}
=
-i\omega\mu_0
\left[
\iint \mathbf f_m(\mathbf r)\cdot\mathbf f_n(\mathbf r')\,G\,dS\,dS'
-
\frac{1}{k^2}
\iint (\nabla_s\cdot\mathbf f_m)(\nabla_s'\cdot\mathbf f_n)\,G\,dS\,dS'
\right].
```

Here $\nabla_s\cdot$ is the surface divergence.  
This exact `(vector part) - (scalar/divergence part)` structure is what
`assemble_Z_efie` computes in `src/EFIE.jl`.

### Derivation sketch of the mixed-potential split

Start from

```math
\left[\mathbf I+\frac{1}{k^2}\nabla\nabla\right]G\,\mathbf J.
```

Project/test with tangential basis and integrate by parts on the surface:

1. The identity part gives the vector-potential coupling
   $\iint f_m\cdot f_n\,G$.
2. The $\nabla\nabla$ part transfers derivatives onto basis divergences,
   producing $-(1/k^2)\iint (\nabla_s\cdot f_m)(\nabla_s'\cdot f_n)\,G$.

This is why assembly naturally separates into a vector term and a scalar
charge-related term.

---

## Step 4: PEC Boundary Condition

PEC requires tangential total electric field to vanish:

```math
\mathbf E_t^{\mathrm{tot}}=\mathbf 0.
```

Substitute decomposition:

```math
\mathbf E_t^{\mathrm{sca}}=-\mathbf E_t^{\mathrm{inc}}.
```

After Galerkin testing:

```math
\mathbf Z_{\mathrm{EFIE}}\mathbf I=\mathbf v,
\qquad
v_m=-\langle\mathbf f_m,\mathbf E_t^{\mathrm{inc}}\rangle.
```

This is the baseline system used in PEC examples.

### Three-line algebra check

```math
\mathbf E_t^{\mathrm{tot}}=0
\;\Rightarrow\;
\mathbf E_t^{\mathrm{inc}}+\mathbf E_t^{\mathrm{sca}}=0
\;\Rightarrow\;
\mathbf E_t^{\mathrm{sca}}=-\mathbf E_t^{\mathrm{inc}}.
```

Then test both sides with `f_m` to obtain `Z I = v`.

---

## Step 5: Impedance-Sheet Boundary Condition

For sheet impedance `Z_s`:

```math
\mathbf E_t^{\mathrm{tot}}=Z_s\mathbf J.
```

Substitute total field decomposition:

```math
\mathbf E_t^{\mathrm{sca}}-Z_s\mathbf J=-\mathbf E_t^{\mathrm{inc}}.
```

Testing gives:

```math
\left(\mathbf Z_{\mathrm{EFIE}}+\mathbf Z_{\mathrm{imp}}\right)\mathbf I=\mathbf v,
```

with

```math
(\mathbf Z_{\mathrm{imp}})_{mn}
=
-\langle \mathbf f_m, Z_s\mathbf f_n\rangle.
```

In patch form,

```math
\mathbf Z_{\mathrm{imp}}
=
-\sum_{p=1}^{P} \theta_p\mathbf M_p
\quad\text{(resistive)}
\qquad\text{or}\qquad
-\sum_{p=1}^{P} (i\theta_p)\mathbf M_p
\quad\text{(reactive)}.
```

This is assembled in `assemble_full_Z` (`src/Solve.jl`) using patch mass
matrices from `src/Impedance.jl`.

### Why the impedance term has a minus sign

From

```math
\mathbf E_t^{\mathrm{sca}}-Z_s\mathbf J=-\mathbf E_t^{\mathrm{inc}},
```

the impedance contribution appears on the **left** as `-Z_s J`.
Testing therefore yields `-<f_m,Z_s f_n>`, so the matrix block enters with a
negative sign.

---

## Implementation Sign Check

`src/EFIE.jl` uses

```julia
omega_mu0 = k * eta0
Z[m, n] = -1im * omega_mu0 * val
```

which is consistent with

```math
\omega\mu_0 = k\eta_0.
```

This check is important when cross-validating with codes that use
`exp(-iωt)` convention.

---

## Minimal Forward Assembly Example

```julia
using DifferentiableMoM

freq = 3e9
c0   = 299792458.0
k    = 2π / (c0/freq)

mesh = make_rect_plate(0.1, 0.1, 6, 6)
rwg  = build_rwg(mesh)
Zef  = assemble_Z_efie(mesh, rwg, k; quad_order=3)
```

At this point `Zef` is the PEC EFIE operator. Add impedance via
`assemble_full_Z(...)`.

### Quick consistency check for users

If you set all impedance parameters to zero, `assemble_full_Z` should reduce to
the same PEC operator (up to floating-point roundoff). This is a good first
debug check for custom workflows.

---

## Code Mapping

- Green's function: `src/Greens.jl`
- EFIE matrix assembly: `src/EFIE.jl`
- Excitation vector assembly: `src/Excitation.jl`
- Impedance blocks and derivatives: `src/Impedance.jl`
- Full loaded system: `src/Solve.jl`

---

## Exercises

- Basic: derive the PEC equation from `E_t^{tot}=0` in three algebra steps.
- Derivation check: starting from the impedance BC, show why the impedance
  matrix enters with a negative sign.
