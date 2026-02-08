# MoM and RWG Discretization

## Purpose

This chapter derives how the continuous EFIE becomes a finite linear system
using Galerkin MoM with RWG basis functions.

---

## Learning Goals

After this chapter, you should be able to:

1. Derive `Z I = v` from weighted-residual testing.
2. Understand RWG geometry and divergence properties.
3. Connect each derivation step to assembly routines in the package.

---

## RWG Intuition Primer (Minimal Background)

Before the derivation, here is the geometric intuition.

### Visual Description

Imagine a mesh surface made of triangular patches. An RWG basis function is attached to **one interior edge** shared by exactly two triangles:

- **"Plus" triangle `T+`**: current flows **away from** the shared edge toward the opposite vertex
- **"Minus" triangle `T-`**: current flows **toward** the shared edge from the opposite vertex

**Picture this**: Think of the shared edge as a "bridge" between two triangles. Current flows across this bridge in a continuous manner - what leaves one triangle enters the other.

### Geometric Construction

For each interior edge with length `ℓ_n`:

1. **Identify the two triangles**: `T+` and `T-` sharing this edge
2. **Locate opposite vertices**: `r_{+,opp}` in `T+`, `r_{-,opp}` in `T-`
3. **Current flow pattern**: 
   - In `T+`: linear flow from edge toward `r_{+,opp}`
   - In `T-`: linear flow from `r_{-,opp}` toward edge

### Why This Design Works

1. **Local support**: Each basis affects only its two triangles → sparse assembly structure
2. **Tangentiality**: Current stays on surface (no normal component) → satisfies boundary conditions
3. **Continuity**: Current leaving `T+` equals current entering `T-` → charge conservation
4. **Linear variation**: Simple to integrate analytically → efficient quadrature

### Physical Interpretation

Each RWG coefficient `I_n` represents the **strength of current flow** through edge `n`. The complete surface current is a superposition of all edge flows:

```math
\mathbf J(\mathbf r) = \sum_{n=1}^{N} I_n \mathbf f_n(\mathbf r)
```

**Key insight**: Instead of describing current at every point, we describe it by how much flows through each edge. This is the discretization that makes MoM computationally tractable.

### Concrete Example

Consider a rectangular plate divided into 2×2 triangles (4 interior edges). Each edge will have:
- One RWG basis function
- One current coefficient `I_n`
- Support on exactly 2 triangles

The total current pattern is determined by these 4 coefficients, not by values at every point on the surface.

### ASCII Diagram: RWG Basis Function Geometry

```
                        RWG Basis Function - Edge n
                        Shared edge length = ℓ_n
                      
                        T+ triangle          T- triangle
                           ▲                     ▼
                          ╱ ╲                   ╱ ╲
                         ╱   ╲                 ╱   ╲
                        ╱     ╲               ╱     ╲
                       ╱       ╲             ╱       ╲
                      ╱         ╲           ╱         ╲
                     ╱           ╲         ╱           ╲
                    ╱             ╲       ╱             ╲
                   ╱               ╲     ╱               ╲
                  ╱                 ╲   ╱                 ╲
                 ╱                   ╲ ╱                   ╲
                ╱                     ●                     ╲
               ╱                     ╱ ╲                     ╲
              ╱                     ╱   ╲                     ╲
             ╱                     ╱     ╲                     ╲
            ╱                     ╱       ╲                     ╲
           ╱                     ╱         ╲                     ╲
          ╱                     ╱           ╲                     ╲
         ▼─────────────────────●─────────────●─────────────────────►
        p₁                    edge n         p₂                    Current
         (vertex)             (length ℓ_n)    (vertex)             flow direction
        
        
        Legend:
        ● = Edge endpoints (p₁, p₂)
        ▲ = Current flows AWAY from edge (T+ triangle)
        ▼ = Current flows TOWARD edge (T- triangle)
        ► = Linear current variation within triangle
        T+ area = A_n^+, T- area = A_n^-
```

### Current Flow Visualization

**Triangle T+ (plus triangle)**:
```
        Opposite vertex r_{+,opp}
                 ▲
                ╱ ╲
               ╱   ╲
              ╱     ╲   Current flows radially outward
             ╱       ╲   from shared edge
            ╱         ╲
           ╱           ╲
          ╱             ╲
         ╱               ╲
        ▼─────────────────►
     edge n            edge n
```

**Triangle T- (minus triangle)**:
```
        Opposite vertex r_{-,opp}
                 ▼
                ╱ ╲
               ╱   ╲
              ╱     ╲   Current flows radially inward
             ╱       ╲   toward shared edge
            ╱         ╲
           ╱           ╲
          ╱             ╲
         ╱               ╲
        ►─────────────────▲
     edge n            edge n
```

### Current Continuity at Shared Edge

```
    T+ triangle        |        T- triangle
                       |
        ▲              |              ▼
       ╱ ╲             |             ╱ ╲
      ╱   ╲            |            ╱   ╲
     ╱     ╲           |           ╱     ╲
    ╱       ╲          |          ╱       ╲
   ╱         ╲         |         ╱         ╲
  ╱           ╲        |        ╱           ╲
 ╱             ╲       |       ╱             ╲
╱               ╲      |      ╱               ╲
▼───────────────►│◄─────────────►───────────────▲
    edge n       │       edge n
                 │
    Current OUT  │  Current IN
    of T+        │  to T-
    equals       │  equals
    Current IN   │  Current OUT
    to T-        │  of T+
```

**Key insight**: The same current that leaves triangle T+ through edge n enters triangle T- through the same edge - ensuring charge conservation.

---

## Step 1: Weighted-Residual Setup

Write the operator equation generically as

```math
\mathcal T[\mathbf J] - Z_s\mathbf J = -\mathbf E_t^{\mathrm{inc}}.
```

Choose trial expansion

```math
\mathbf J(\mathbf r)\approx\sum_{n=1}^{N} I_n\mathbf f_n(\mathbf r),
```

and test with the same basis (Galerkin):

```math
\left\langle \mathbf f_m,
\mathcal T\left[\sum_n I_n\mathbf f_n\right]-Z_s\sum_n I_n\mathbf f_n
\right\rangle
=
-\langle\mathbf f_m,\mathbf E_t^{\mathrm{inc}}\rangle.
```

Linearity gives

```math
\sum_{n=1}^{N} Z_{mn}I_n = v_m,
```

with

```math
Z_{mn}
=
\langle\mathbf f_m,\mathcal T[\mathbf f_n]\rangle
-
\langle\mathbf f_m,Z_s\mathbf f_n\rangle,
\qquad
v_m=-\langle\mathbf f_m,\mathbf E_t^{\mathrm{inc}}\rangle.
```

That is the matrix system solved in code.

---

## Step 2: RWG Basis on an Interior Edge

Each RWG basis is tied to one interior edge shared by triangles `T+` and `T-`.

```math
\mathbf f_n(\mathbf r)=
\begin{cases}
\dfrac{\ell_n}{2A_n^+}(\mathbf r-\mathbf r_{+,\mathrm{opp}}), & \mathbf r\in T^+,\\[6pt]
\dfrac{\ell_n}{2A_n^-}(\mathbf r_{-,\mathrm{opp}}-\mathbf r), & \mathbf r\in T^-,\\[6pt]
\mathbf 0, & \text{otherwise.}
\end{cases}
```

### Geometric Interpretation of the Formula

**Visual breakdown**:
- `ℓ_n`: Edge length (the "bridge" width)
- `A_n^±`: Triangle areas (affects current density)
- `r - r_{+,opp}`: Vector from opposite vertex to current point in `T+`
- `r_{-,opp} - r`: Vector from current point to opposite vertex in `T-`

**Physical meaning**:
- In `T+`: Current flows **radially outward** from the shared edge
- In `T-`: Current flows **radially inward** toward the shared edge
- The scaling `ℓ_n/(2A_n^±)` ensures proper current density

### Continuity at the Shared Edge

**Why continuity matters**: Surface current cannot have discontinuous normal components across interior edges - that would imply infinite charge accumulation.

**Mathematical proof**: At any point on the shared edge, the normal component of current from `T+` equals the normal component from `T-` (with opposite sign convention), ensuring zero net flux across the edge.

**Intuitive picture**: Imagine water flowing through a pipe (the edge). What flows out one side must flow in the other side - no water can accumulate at the pipe junction.

### Step-by-Step Construction

1. **Pick an interior edge** with endpoints `p₁, p₂`
2. **Identify the two triangles** sharing this edge
3. **Find opposite vertices** `v₊` in `T+`, `v₋` in `T-`
4. **Define current flow directions**:
   - In `T+`: from edge toward `v₊`
   - In `T-`: from `v₋` toward edge
5. **Scale by geometry**: `ℓ_n/(2A_n^±)` ensures correct units

### Concrete Numerical Example

Consider an edge with:
- Length: `ℓ_n = 0.01 m`
- Triangle areas: `A_n^+ = 6×10⁻⁵ m²`, `A_n^- = 8×10⁻⁵ m²`
- Scaling factors: `ℓ_n/(2A_n^+) = 83.3`, `ℓ_n/(2A_n^-) = 62.5`

The current density is higher in the smaller triangle (conservation of current).

### Tiny geometric meaning of each symbol

- ``\ell_n``: length of the shared interior edge.
- `A_n^+, A_n^-`: areas of the two support triangles.
- ``\mathbf r_{+,\mathrm{opp}}, \mathbf r_{-,\mathrm{opp}}``: vertices opposite
  the shared edge in `T+` and `T-`.

The linear form in ``\mathbf r`` means the basis varies smoothly over each
triangle, while the sign flip between `T+` and `T-` enforces continuity across
the common edge.

---

## Step 3: Why Divergence Is Simple (and Useful)

On each support triangle, RWG is affine in position, so its divergence is
constant:

```math
\nabla_s\cdot\mathbf f_n=
\begin{cases}
\ell_n/A_n^+, & T^+,\\
-\ell_n/A_n^-, & T^-.
\end{cases}
```

This property directly simplifies the scalar-potential (charge) term in EFIE
assembly, because `div_rwg` can be reused at quadrature points without
recomputation.

### One-line derivation idea

On one triangle, `f_n(r)=a+Br` is affine. Divergence of an affine vector field
is constant (`tr(B)`), which gives the piecewise constants above.

---

## Step 4: Discrete EFIE Block Structure

Using the mixed-potential form from Chapter 2, matrix entries are assembled as

```math
Z_{mn}^{\mathrm{EFIE}}=-i\omega\mu_0\left[V_{mn}-\frac{1}{k^2}S_{mn}\right],
```

with

```math
V_{mn}=\iint \mathbf f_m\cdot\mathbf f_n\,G\,dS\,dS',
\qquad
S_{mn}=\iint (\nabla_s\cdot\mathbf f_m)(\nabla_s'\cdot\mathbf f_n)\,G\,dS\,dS'.
```

In implementation, this appears exactly as
`vec_part - scl_part` in `src/EFIE.jl`.

### Element-pair assembly viewpoint

For each test/source basis pair `(m,n)`, assembly loops over their support
triangles:

1. Evaluate quadrature points on test triangle.
2. Evaluate quadrature points on source triangle.
3. Accumulate vector part `f_m·f_n G`.
4. Accumulate scalar part `div(f_m) div(f_n) G`.

Because each RWG has only two support triangles, local geometry stays compact
even though global coupling is dense.

---

## Step 5: Quadrature and Triangle Mapping

All triangle integrals use reference-triangle Gaussian quadrature:

```math
\int_{T} f(\mathbf r)\,dS
\approx
\sum_{q=1}^{N_q} w_q\,f(\mathbf r_q)\,(2A_T).
```

The factor `2A_T` is the Jacobian from reference triangle to physical triangle.

- Quadrature rule: `tri_quad_rule` in `src/Quadrature.jl`
- Physical points: `tri_quad_points` in `src/Quadrature.jl`

### Why `2A_T` appears

The reference triangle has area `1/2`. Mapping to a physical triangle of area
`A_T` multiplies integrals by Jacobian determinant `2A_T`.
This factor is a frequent source of hidden scaling bugs; keeping it explicit is
important.

---

## Mesh Preconditions Before RWG Build

RWG construction assumes valid topology/geometry. The package precheck detects:

- out-of-range or repeated triangle indices,
- degenerate triangles,
- non-manifold edges,
- interior-edge orientation conflicts.

Use:

```julia
report = mesh_quality_report(mesh)
ok = mesh_quality_ok(report; allow_boundary=true, require_closed=false)
```

`build_rwg(mesh; precheck=true)` calls this automatically.

---

## Minimal Inspection Example

```julia
using DifferentiableMoM

mesh = make_rect_plate(0.1, 0.1, 4, 4)
rwg  = build_rwg(mesh)

println("RWG unknowns = ", rwg.nedges)
tp, tm = basis_triangles(rwg, 1)
println("Basis 1 support triangles: ", (tp, tm))
println("div on T+ = ", div_rwg(rwg, 1, tp))
println("div on T- = ", div_rwg(rwg, 1, tm))
```

Try printing `A_n^+ * div_plus` and `A_n^- * div_minus`; magnitudes should be
equal and signs opposite, matching `±ℓ_n`.

---

## Code Mapping

- RWG geometry and evaluation: `src/RWG.jl`
- Mesh geometry and precheck: `src/Mesh.jl`
- Quadrature rules: `src/Quadrature.jl`
- EFIE assembly using RWG/quadrature data: `src/EFIE.jl`

---

## Exercises

- Basic: derive `Σ_n Z_{mn}I_n=v_m` from the tested residual equation.
- Derivation check: prove that RWG divergence is constant on each support
  triangle.
