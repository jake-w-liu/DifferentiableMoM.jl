# Excitation Theory and Usage

## Purpose

This chapter gives a full physics-to-code explanation of excitation modeling in
`DifferentiableMoM.jl`.

In EFIE-MoM, most confusion comes from one question:

- Are we solving for the source itself, or for the scatterer response to a
  known source field?

In this package, excitation models are used to build the **known incident
field** on the scattering mesh and then assemble the EFIE right-hand side (RHS).
They are not separate source-radiation solvers unless you provide such a field
from outside.

---

## Learning Goals

After this chapter, you should be able to:

1. Derive exactly where excitation enters the EFIE linear system.
2. Map each excitation type to its physical assumptions and limitations.
3. Decide when to use `PlaneWave`, `Dipole`, `Loop`,
   `ImportedExcitation`, or `PatternFeed`.
4. Interpret the package checks (amplitude/polarization/phase) as physical
   validation, not just software tests.

---

## 1. Conventions and Notation

The package uses the $e^{+i\omega t}$ time convention.

Under this convention:

```math
\widetilde{\mathbf{E}}(\mathbf{r},t)=\Re\{\mathbf{E}(\mathbf{r})e^{+i\omega t}\},
\qquad
\mathbf{G}(\mathbf{r},\mathbf{r}')\propto e^{-ikR}
```

with $R = |\mathbf{r}-\mathbf{r}'|$ and $k = \omega\sqrt{\mu_0\epsilon_0}$.

Main symbols used in this chapter:

- $\Gamma$: scattering surface
- $\mathbf{f}_m$: RWG testing function
- $\mathbf{J}$: unknown induced surface current on $\Gamma$
- $\mathbf{E}_{\mathrm{inc}}$: known incident electric field phasor
- $\mathbf{Z}\mathbf{I}=\mathbf{v}$: MoM linear system

---

## 2. From Boundary Condition to RHS

## 2.1 PEC case

For PEC:

```math
\mathbf{E}^{\mathrm{tot}}_t=\mathbf{E}^{\mathrm{inc}}_t+\mathbf{E}^{\mathrm{sca}}_t=0
\quad\Rightarrow\quad
\mathbf{E}^{\mathrm{sca}}_t=-\mathbf{E}^{\mathrm{inc}}_t
```

Write the EFIE operator as $\mathcal{T}[\mathbf{J}] = \mathbf{E}_{\mathrm{sca},t}$.
Galerkin testing gives:

```math
\langle \mathbf{f}_m,\,\mathcal{T}[\mathbf{J}]\rangle=-\langle \mathbf{f}_m,\,\mathbf{E}^{\mathrm{inc}}\rangle
```

After RWG expansion $\mathbf{J} = \sum_n I_n\mathbf{f}_n$:

```math
\sum_n Z_{mn} I_n = v_m
```

with

```math
Z_{mn}=\langle \mathbf{f}_m,\,\mathcal{T}[\mathbf{f}_n]\rangle,
\qquad
v_m=-\langle \mathbf{f}_m,\,\mathbf{E}^{\mathrm{inc}}\rangle
```

So excitation enters only through $v_m$.

## 2.2 Impedance boundary case

For impedance-sheet modeling:

```math
\mathbf{E}^{\mathrm{tot}}_t = Z_s\mathbf{J}
```

which yields

```math
\mathcal{T}[\mathbf{J}]-Z_s\mathbf{J}=-\mathbf{E}^{\mathrm{inc}}_t
```

The **matrix** changes because of $-Z_s\mathbf{J}$, but the excitation term is
still the same projection of $-\mathbf{E}_{\mathrm{inc}}$.

---

## 3. Discrete RHS Assembly Used in Code

For each RWG basis index $m$, supported on triangles $T_m^+$ and $T_m^-$:

```math
v_m=-\sum_{T\in\{T_m^+,T_m^-\}}\int_T \mathbf{f}_m(\mathbf{r})\cdot \mathbf{E}^{\mathrm{inc}}(\mathbf{r})\,dS
```

Using reference-triangle quadrature:

```math
v_m\approx-
\sum_{T}\sum_q w_q
\,\mathbf{f}_m(\mathbf{r}_{T,q})\cdot \mathbf{E}^{\mathrm{inc}}(\mathbf{r}_{T,q})\,(2A_T)
```

where $2A_T$ is the Jacobian factor for affine triangle mapping.

This expression is exactly what the excitation assembly functions implement.

## 3.1 Quadrature order and `min_quad_order`

`ImportedExcitation` exposes `min_quad_order`, which is combined with the
user-supplied assembly quadrature:

```math
\text{quad}_{\mathrm{eff}}=
\max\big(\text{user\_quad},\text{min\_quad\_order}\big).
```

Internally, the code uses supported triangle rules of orders $1,3,4,7$, so
the effective order is rounded up to the nearest supported rule.

This is a numerical-resolution control, not a physical model change.

---

## 4. Excitation Models: Physics, Math, and Use Cases

## 4.1 Plane wave (`PlaneWaveExcitation`)

Model:

```math
\mathbf{E}^{\mathrm{inc}}(\mathbf{r})=\mathbf{p}\,E_0\,e^{-ik\hat{\mathbf{k}}\cdot \mathbf{r}}
```

with $\mathbf{p}$ the polarization vector and $\mathbf{k}_{\mathrm{vec}} = k\hat{\mathbf{k}}$.

Use when:

- RCS and canonical scattering.
- Cross-validation against Mie or external full-wave references.

Notes:

- `assemble_v_plane_wave(...)` remains for backward compatibility and routes to
  unified excitation assembly.

---

## 4.2 Port and delta-gap (`PortExcitation`, `DeltaGapExcitation`)

These are **lumped feed approximations** on RWG edges.

Delta-gap model in package:

```math
v_m\approx\frac{V}{g}\,\delta_{m,m_{\mathrm{gap}}}
```

where $g$ is the specified physical gap length.

Port model in current implementation:

- applies edge-local voltage scaling over listed RWG edges,
- uses edge-length normalization for those edges,
- stores `impedance` metadata but does not solve full waveguide/coax modal
  excitation.

Use when:

- feed prototyping,
- gap-driven examples,
- optimization where exact port modal fields are not yet required.

This is not yet a full network-port field model.

---

## 4.3 Electric/magnetic dipole (`DipoleExcitation`)

The package evaluates closed-form dipole fields and projects them to RHS.

For an electric dipole moment $\mathbf{p}$ at source position $\mathbf{r}_0$, with
$R = |\mathbf{r}-\mathbf{r}_0|$ and $\hat{\mathbf{R}}=(\mathbf{r}-\mathbf{r}_0)/R$:

```math
\mathbf{E}_{\mathrm{elec}}(\mathbf{r})=
\frac{e^{-ikR}}{4\pi\epsilon_0 R}
\left[
  k^2\,\hat{\mathbf{R}}\times(\mathbf{p}\times\hat{\mathbf{R}})
  +\left(\frac{1}{R^2}-i\frac{k}{R}\right)
   \left(3\hat{\mathbf{R}}(\hat{\mathbf{R}}\cdot\mathbf{p})-\mathbf{p}\right)
\right]
```

For a magnetic dipole moment $\mathbf{m}$:

```math
\mathbf{E}_{\mathrm{mag}}(\mathbf{r})=
\frac{\eta_0}{4\pi}
\left(\frac{k}{R^2}+i\frac{k^2}{R}\right)e^{-ikR}(\hat{\mathbf{R}}\times\mathbf{m})
```

Use when:

- physically interpretable localized sources,
- near/far-field transition checks,
- polarization sanity tests.

---

## 4.4 Loop (`LoopExcitation`)

Small loop current source is mapped to equivalent magnetic dipole:

```math
\mathbf{m} = I\,\pi a^2\,\hat{\mathbf{n}}
```

and then uses the magnetic-dipole field expression.

Use when:

- electrically small loop approximations are valid,
- comparing $E_\theta$ and $E_\phi$ polarization channels versus electric
  dipole sources.

---

## 4.5 Imported/distributed source (`ImportedExcitation`)

This model is often misused, so scope is explicit.

It is an **incident-field generator for RHS assembly**, not a separate IE/FEM
source-radiation solve.

Supported semantics:

1. `kind=:electric_field`

```math
\mathbf{E}^{\mathrm{inc}}(\mathbf{r})=\mathbf{S}(\mathbf{r})
```

2. `kind=:surface_current_density`

```math
\mathbf{E}^{\mathrm{inc}}(\mathbf{r})\approx\eta_{\mathrm{equiv}}\mathbf{J}_s(\mathbf{r})
```

The second is a local equivalent-sheet mapping useful for aperture/sheet-like
approximations.

It is **not** a rigorous substitute for full radiation from arbitrary volumetric
currents.

Implementation safety checks:

- output must be a 3-component numeric vector,
- tuple/vector/SVector returns are accepted and converted,
- non-finite components throw explicit errors.

Use when:

- you already know $\mathbf{E}_{\mathrm{inc}}$ on or near the scatterer,
- custom deterministic field prescriptions are needed.

---

## 4.6 Imported Cartesian field (via `ImportedExcitation`)

Model:

```math
\mathbf{E}^{\mathrm{inc}}(\mathbf{r})=\mathbf{E}_{\mathrm{imported}}(\mathbf{r})
```

where $\mathbf{E}_{\mathrm{imported}}$ comes from external
simulation/measurement interpolation.

Use when:

- an external solver gives a trustworthy near field,
- you need strict control of phase and polarization in Cartesian components.

How to use it in practice:

1. Build an interpolator (or analytic function) that returns a **complex 3-vector**
   for any point `r` on the scattering surface.
2. Pass that function to `ImportedExcitation(...; kind=:electric_field)`.
3. Assemble the RHS exactly as with other excitations.

Example:

```julia
using DifferentiableMoM

# Example imported field: x-polarized plane wave
freq = 3.0e9
k = 2π * freq / 299792458.0
E_imported(r) = CVec3(exp(-1im * k * r[3]), 0.0 + 0im, 0.0 + 0im)

exc = ImportedExcitation(E_imported; kind=:electric_field, min_quad_order=3)
v   = assemble_excitation(mesh, rwg, exc; quad_order=3)
```

Important implementation note:

- `ImportedExcitation(kind=:electric_field)` and
  `ImportedExcitation(kind=:surface_current_density, eta_equiv=...)`
  share the same EFIE RHS projection path.
- For matching effective incident fields, they produce identical RHS physics.
- The distinction is modeling intent: direct imported field vs. local
  equivalent-sheet map $\mathbf{E}_{\mathrm{inc}} \approx \eta_{\mathrm{equiv}}\mathbf{J}_s$.

---

## 4.7 Imported spherical feed pattern (`PatternFeedExcitation`)

This model imports complex $E_\theta$ and $E_\phi$ pattern data on a
$(\theta,\phi)$ grid.

Field model:

```math
\mathbf{E}(\mathbf{r})=
\frac{e^{-ikR}}{R}
\left(F_\theta(\theta,\phi)\,\hat{\boldsymbol\theta}
+F_\phi(\theta,\phi)\,\hat{\boldsymbol\phi}\right)
```

with $R = |\mathbf{r}-\mathbf{r}_{\mathrm{phase\_center}}|$.

Numerical details in implementation:

- bilinear interpolation on $(\theta,\phi)$,
- non-periodic bracketing in $\theta$, periodic wrapping in $\phi$,
- explicit convention handling:
  - if imported data use $e^{-i\omega t}$, coefficients are conjugated before
    use,
- requires both complex component patterns ($E_\theta, E_\phi$), not
  power-only gain.

Use when:

- horn/array feed illumination of scatterers or reflectors,
- cross-tool workflows with spherical pattern exports.

---

## 4.8 Superposition (`MultiExcitation`)

Because the EFIE system is linear in excitation:

```math
\mathbf{v}=\sum_j w_j\,\mathbf{v}_j
```

where $\mathbf{v}_j$ are RHS vectors from individual excitation models.

Use when:

- multi-source studies,
- weighted scenario synthesis,
- parameter sweeps with shared matrix $\mathbf{Z}$ and multiple RHS columns.

---

## 5. Polarization and Phase: Why Power Alone Is Not Enough

Dipole and loop patterns can have similar power envelopes but different vector
polarization channels.

For example, on the selected validation cut:

- electric dipole dominant channel is $E_\theta$,
- loop (equivalent magnetic dipole) dominant channel is $E_\phi$,
- relative phase relation is checked near $\pm 90^\circ$ after branch
  handling.

This is why the CI gate includes:

- co-pol pattern error,
- cross-pol leakage,
- phase-consistency metrics.

A pure power-only comparison would miss this physics.

---

## 6. Practical Source-Selection Rules

Choose by what you actually know:

- Known far-field illuminator: plane wave.
- Edge/gap feed approximation: delta-gap or port.
- Canonical localized source: dipole/loop.
- Known incident field expression in space:
  `ImportedExcitation(...; kind=:electric_field)`.
- External Cartesian incident field data:
  `ImportedExcitation(...; kind=:electric_field)`.
- External spherical component pattern ($E_\theta, E_\phi$): pattern feed.
- Multiple illuminations: multi-excitation.

If you only know gain/directivity (no complex field phase), you do not have
sufficient data for rigorous `PatternFeedExcitation`.

---

## 7. Validation Gates You Should Interpret Physically

The test suite includes dedicated excitation checks:

1. Plane-wave path-consistency assembly equivalence.
2. Delta-gap scaling with gap length.
3. Imported-excitation semantics consistency ($\mathbf{E}$ vs $\eta \mathbf{J}$).
4. Dipole and loop far-field pattern agreement.
5. Cross-polarization suppression for the intended canonical channel.
6. Dipole-loop phase consistency check (near $\pm 90^\circ$).
7. Pattern-feed amplitude/phase/convention consistency.

These are not arbitrary software checks; they confirm that excitation
mathematics and implementation remain aligned.

---

## 8. Common Failure Modes and Fixes

1. **Using `ImportedExcitation` as a source solver**  
   Fix: use it only as known $\mathbf{E}_{\mathrm{inc}}$ (or local
   $\eta\mathbf{J}$ approximation).

2. **Using power-only pattern files**  
   Fix: provide complex $E_\theta$ and $E_\phi$ components.

3. **Ignoring time-convention mismatch across tools**  
   Fix: use `convention=:exp_minus_iwt` when needed.

4. **Wrong phase center for feed patterns**  
   Fix: set physical phase center explicitly.

5. **Comparing only scalar power**  
   Fix: also inspect vector polarization and phase channels.

---

## 9. Code Mapping

Primary implementation:

- `src/Excitation.jl`

Related demos:

- `examples/ex_dipole_loop_farfield_pattern.jl`
- `examples/ex_pattern_feed_dipole_validation.jl`
- `examples/ex_horn_pattern_import_demo.jl`
- `examples/ex_radiationpatterns_adapter.jl`

---

## 10. Minimal End-to-End Pattern

All excitation models follow the same computational sequence:

1. define incident field model,
2. assemble RHS $\mathbf{v}$ by triangle quadrature projection,
3. solve $\mathbf{Z}\mathbf{I}=\mathbf{v}$,
4. compute currents, fields, and diagnostics.

This architectural uniformity is what makes excitation extensibility practical
without changing the core EFIE and adjoint machinery.

---

## Summary

Excitation in `DifferentiableMoM.jl` is best understood as **controlled
construction of known incident fields on the scattering surface**, followed by
consistent Galerkin projection into EFIE RHS space.

When model scope is respected (especially for distributed and imported sources),
results are physically interpretable and validated by dedicated amplitude,
polarization, and phase gates.
