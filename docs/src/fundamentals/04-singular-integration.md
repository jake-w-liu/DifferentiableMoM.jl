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

Why this helps:

- `G_smooth` is bounded as `R -> 0`:

```math
\lim_{R\to 0}\frac{e^{-ikR}-1}{R}=-ik
\quad\Rightarrow\quad
\lim_{R\to 0}G_{\mathrm{smooth}}=-\frac{ik}{4\pi}.
```

- The explicit `1/R` part is isolated for special treatment.

Implementation:

- `greens_smooth` in `src/Greens.jl`
- self branch in `self_cell_contribution` (`src/SingularIntegrals.jl`)

### Small-series check (why the limit is finite)

Using $e^{-ikR}=1-ikR-\tfrac{k^2R^2}{2}+O(R^3)$,

```math
\frac{e^{-ikR}-1}{R}
=
-ik-\frac{k^2R}{2}+O(R^2),
```

so no $1/R$ singularity remains in $G_{\mathrm{smooth}}$.

---

## Step 3: Semi-Analytical Inner Integral

For each outer point `P` on a triangle `T`, define

```math
S(P)=\int_T\frac{1}{|P-r'|}dS'.
```

The code evaluates `S(P)` with an edge-log closed form
(`analytical_integral_1overR`):

```math
S(P)=\sum_{i=1}^{3} d_i\,\log\!\frac{\ell_{B_i}+R_{B_i}}{\ell_{A_i}+R_{A_i}}.
```

This avoids brute-force quadrature on a singular integrand.

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
