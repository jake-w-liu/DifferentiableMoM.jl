# Singular Integration

## Purpose

Self interactions are the hardest part of EFIE assembly: when source and test
triangles coincide, the Green kernel behaves like `1/R`.

This chapter explains, step by step, how the package removes that singularity
numerically while preserving physical consistency.

---

## Learning Goals

After this chapter, you should be able to:

1. Explain why direct product quadrature fails for self terms.
2. Derive the smooth+singular kernel split used in code.
3. Understand the semi-analytical inner integral used for `1/R`.

---

## Step 1: Identify the Singular Kernel

The free-space kernel is

```math
G(\mathbf r,\mathbf r')=
\frac{e^{-ikR}}{4\pi R},
\qquad
R=|\mathbf r-\mathbf r'|.
```

For `tm == tn`, points on the same triangle can satisfy `R -> 0`, so naive
quadrature samples a sharply singular integrand.

---

## Step 2: Algebraic Split of the Kernel

Write

```math
e^{-ikR}=1+\left(e^{-ikR}-1\right),
```

therefore

```math
G
=
\underbrace{\frac{e^{-ikR}-1}{4\pi R}}_{G_{\mathrm{smooth}}}
+
\underbrace{\frac{1}{4\pi R}}_{G_{\mathrm{sing}}}.
```

### Why This Split Works

**Problem**: Direct quadrature fails when `R → 0` because `1/R` becomes infinite.

**Solution**: Separate the kernel into:
- `G_smooth`: Bounded (finite) at `R = 0`
- `G_sing`: Pure `1/R` singularity that we can treat analytically

### Numerical Example

Consider frequency `f = 3 GHz` (so `k ≈ 62.8 rad/m`):

**At R = 1 mm = 0.001 m**:
- `G_sing = 1/(4π × 0.001) ≈ 79.6`
- `G_smooth = (e^{-ik×0.001} - 1)/(4π × 0.001) ≈ -ik/(4π) ≈ -5.0i`

**At R = 0.01 mm = 1×10⁻⁵ m**:
- `G_sing = 1/(4π × 1×10⁻⁵) ≈ 7958`
- `G_smooth ≈ -5.0i` (essentially unchanged!)

**Key insight**: `G_smooth` stays bounded while `G_sing` grows without bound.

### ASCII Diagram: Kernel Split Visualization

```
    Original Green's function G(R) = e^{-ikR}/(4πR)
    
    As R → 0:
    
    ┌─────────────────────────────────────────────────────────┐
    │                         G(R)                            │
    │        = G_smooth(R) + G_sing(R)                        │
    │                                                         │
    │  G_smooth(R) = (e^{-ikR} - 1)/(4πR)                     │
    │              ≈ -ik/(4π) when R is small                 │
    │              (BOUNDED - finite value at R=0)            │
    │                                                         │
    │  G_sing(R) = 1/(4πR)                                    │
    │            → ∞ as R → 0                                 │
    │            (SINGULAR - analytical treatment needed)     │
    └─────────────────────────────────────────────────────────┘

    Numerical behavior near R=0:
    
    R (m)       G_sing(R)      G_smooth(R)      G(R)
    ─────────────────────────────────────────────────────
    0.001       79.6           -5.0i            ~74.6 - 5.0i
    0.0001      795.8          -5.0i           ~790.8 - 5.0i  
    0.00001     7957.7         -5.0i          ~7952.7 - 5.0i
    0            ∞             -5.0i               ∞
```

### Mathematical Verification

`G_smooth` is bounded as `R -> 0`:

```math
\lim_{R\to 0}\frac{e^{-ikR}-1}{R}=-ik
\quad\Rightarrow\quad
\lim_{R\to 0}G_{\mathrm{smooth}}=-\frac{ik}{4\pi}.
```

### Small-series check (why the limit is finite)

Using ``e^{-ikR}=1-ikR-\tfrac{k^2R^2}{2}+O(R^3)``,
```math
\frac{e^{-ikR}-1}{R}
=
-ik-\frac{k^2R}{2}+O(R^2),
```
so no ``1/R`` singularity remains in ``G_{\mathrm{smooth}}``.

### Practical Consequence

When implementing quadrature:
- **Smooth part**: Use standard Gaussian quadrature (integrand is well-behaved)
- **Singular part**: Use analytical integration (avoid quadrature near singularity)

Implementation:

- `greens_smooth` in `src/Greens.jl`
- self branch in `self_cell_contribution` (`src/SingularIntegrals.jl`)

---

## Step 3: Semi-Analytical Inner Integral

For each outer point `P` on a triangle `T`, define

```math
S(P)=\int_T\frac{1}{|P-r'|}dS'.
```

### Why We Need This

When computing `∫∫ f(P)·f(r')·G(P,r') dS dS'`, the inner integral becomes singular when `P` and `r'` are on the same triangle.

**Direct quadrature fails**: If `P` is a quadrature point and we integrate over the same triangle, some integrand samples will have `R ≈ 0`, causing numerical overflow.

### Semi-Analytical Solution

The code evaluates `S(P)` with an edge-log closed form
(`analytical_integral_1overR`):

```math
S(P)=\sum_{i=1}^{3} d_i\,\log\!\frac{\ell_{B_i}+R_{B_i}}{\ell_{A_i}+R_{A_i}}.
```

### Step-by-Step Geometry

**For each triangle edge**:
1. **Project point `P`** onto the edge line
2. **Measure distances**:
   - `d_i`: Perpendicular distance from `P` to edge `i`
   - `ℓ_{A_i}, ℓ_{B_i}`: Distances along edge from projection to vertices
   - `R_{A_i}, R_{B_i}`: 3D distances from `P` to edge vertices
3. **Compute logarithmic contribution** for that edge
4. **Sum over all 3 edges**

### ASCII Diagram: Geometry for Analytical 1/R Integration

```
    Triangle edge i with vertices A and B
    Point P projected onto edge line at point P_proj
    
           P (observation point)
           │
           │ d_i (perpendicular distance)
           │
    A┼─────┼─────┼B
     │     │     │
     │ ℓ_Ai│     │ ℓ_Bi
     │     │     │
     ◄─────►     ◄─────►
     
    Where:
    - A, B: edge vertices
    - P: observation point on triangle (or nearby)
    - P_proj: projection of P onto edge line
    - d_i = distance(P, edge line) = |P - P_proj|
    - ℓ_Ai = distance(P_proj, A) along edge
    - ℓ_Bi = distance(P_proj, B) along edge  
    - R_Ai = distance(P, A) = √(d_i² + ℓ_Ai²)
    - R_Bi = distance(P, B) = √(d_i² + ℓ_Bi²)
    
    Contribution from edge i:
    d_i × log[(ℓ_Bi + R_Bi)/(ℓ_Ai + R_Ai)]
```

### Complete Triangle Integration

```
          P
          │
          │
          │
     ┌────┼────┐
     │    │    │
     │    │    │
     │    │    │
    A●────┼────●B
     │    │    │
     │    │    │
     │    │    │
     └────┼────┘
          │
          │
          │
          ●C (third vertex)
          
    Total S(P) = sum over edges (A-B, B-C, C-A)
```

### Numerical Example

Consider a right triangle with vertices at (0,0,0), (1,0,0), (0,1,0), and point `P = (0.2, 0.2, 0)`:

**Edge 1**: from (0,0,0) to (1,0,0)
- `d₁ = 0.2` (distance to x-axis)
- `ℓ_{A₁} = 0.2`, `ℓ_{B₁} = 0.8`
- `R_{A₁} = √(0.2² + 0.2²) ≈ 0.283`, `R_{B₁} = √(0.8² + 0.2²) ≈ 0.825`
- Contribution: `0.2 × log((0.8+0.825)/(0.2+0.283)) ≈ 0.2 × log(1.625/0.483) ≈ 0.2 × 1.22 ≈ 0.244`

**Repeat for other edges and sum** to get `S(P)`.

### Why This Works

The logarithmic form comes from integrating `1/R` over a line segment, which has a known analytical solution. By decomposing the triangle integral into edge contributions, we avoid the singularity entirely.

### Geometry meaning of the logarithmic terms

Each edge contributes one logarithmic endpoint expression measured from
projected distances of `P` to that edge segment. Summing all three edge
contributions gives the full triangle integral of `1/R`.

---

## Step 4: Vector-Part Decomposition (Why It Stays Bounded)

For the vector term, the code uses

```math
\mathbf f_n(r')=\mathbf f_n(P)+\left[\mathbf f_n(r')-\mathbf f_n(P)\right].
```

Insert this into

```math
\int_T \frac{\mathbf f_m(P)\cdot\mathbf f_n(r')}{4\pi|P-r'|}dS'.
```

You get two pieces:

1. leading singular piece using `S(P)` directly,
2. remainder with bounded numerator `f_n(r')-f_n(P)=O(|r'-P|)`, so
   `(difference)/|r'-P|` is finite.

That second part is safely integrated with standard quadrature.

### Order argument near `r'=P`

For linear basis functions, Taylor expansion gives

```math
\mathbf f_n(r')-\mathbf f_n(P)=\nabla \mathbf f_n(P)\,(r'-P)+O(|r'-P|^2),
```

so numerator is `O(R)`. Dividing by `R` yields `O(1)`, i.e., bounded.

---

## Step 5: Scalar-Part Handling

The scalar/divergence term uses constant RWG **surface** divergences on each
triangle, so
its singular contribution is directly proportional to `S(P)`:

```math
\text{scalar singular part}
\propto
(\nabla\cdot\mathbf f_m)(\nabla\cdot\mathbf f_n)\,S(P).
```

This is exactly what `self_cell_contribution` accumulates.

### Why scalar part is sensitive

The scalar block couples basis divergences and is often numerically sensitive
to mesh quality and quadrature consistency. Accurate `1/R` treatment there is
crucial for stable charge behavior and energy balance.

---

## Why This Is Necessary for Correctness

Without correct self handling, typical symptoms are:

- energy imbalance,
- unstable mesh-refinement trends,
- unreliable gradients.

The implemented split+semi-analytical strategy is a main reason the package
passes its energy and gradient consistency checks.

---

## Minimal Practical Check

Run the convergence workflow and inspect energy-ratio trend:

```bash
julia --project=. examples/ex_convergence.jl
```

A stable implementation should keep PEC `P_rad/P_in` close to 1 with
refinement.

For debugging, compare a run with self extraction temporarily disabled versus
enabled; the extracted version should show clearly improved energy consistency.

---

## Code Mapping

- Kernel split primitives: `src/Greens.jl`
- Analytical `1/R` integral: `analytical_integral_1overR` in
  `src/SingularIntegrals.jl`
- Full self-cell accumulation: `self_cell_contribution` in
  `src/SingularIntegrals.jl`
- Self/non-self branching: `assemble_Z_efie` in `src/EFIE.jl`

---

## Exercises

- Basic: derive `lim_{R->0} G_smooth = -ik/(4π)`.
- Derivation check: show why `f_n(r')-f_n(P)` cancels the `1/R` singularity in
  the vector remainder term.
