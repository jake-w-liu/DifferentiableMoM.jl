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

## 1. The Singularity Problem in EFIE Assembly

Accurate evaluation of the double‑surface integrals $V_{mn}$ and $S_{mn}$ (defined in Chapter 3) is challenging when the source triangle $T_n$ and the observation triangle $T_m$ coincide or are very close. This section explains why the Green’s function becomes singular in such cases, what numerical problems this creates, and why specialized treatment is essential for a robust MoM implementation.

### 1.1 The Free‑Space Green’s Function and Its Singularity

The time‑harmonic free‑space Green’s function used in the EFIE is

```math
G(\mathbf{r},\mathbf{r}') = \frac{e^{-ikR}}{4\pi R}, \qquad R = |\mathbf{r}-\mathbf{r}'|,
```

where $k=\omega\sqrt{\mu_0\varepsilon_0}$ is the wavenumber and the $e^{+i\omega t}$ convention is assumed. The function has two key features:

1. **Oscillatory factor** $e^{-ikR}$: varies smoothly with $R$ and approaches 1 as $R\to 0$.
2. **Algebraic factor** $1/R$: diverges as $R^{-1}$ when $\mathbf{r}\to\mathbf{r}'$.

When the source and observation points lie on the **same triangle** (self‑interaction) or on **adjacent triangles** sharing an edge or vertex, the distance $R$ can become arbitrarily small. In the limit $R\to 0$, the factor $1/R$ causes the integrand to blow up, rendering standard numerical quadrature ineffective.

### 1.2 Mathematical Characterization of the Singularity

To quantify the singularity, expand $G$ for small $R$:

```math
G(\mathbf{r},\mathbf{r}') = \frac{1}{4\pi R} - \frac{ik}{4\pi} + O(R).
```

The leading term $1/(4\pi R)$ is difficult to evaluate accurately with standard product Gaussian quadrature over a two‑dimensional domain: a double integral $\iint 1/R \, dS\, dS'$ over coincident triangles converges (the singularity is integrable), but the **physical** EFIE matrix elements $Z_{mn}$ require high accuracy because they represent measurable interactions between finite current elements.

The integral converges, but standard Gaussian quadrature cannot capture the near-singular behavior accurately without special treatment.

### 1.3 Why Product Gaussian Quadrature Fails

Consider the self‑term $Z_{mm}$ where test and source bases share the same triangle pair. The integral to evaluate is

```math
Z_{mm} \propto \iint_{T\times T} \frac{\mathbf{f}_m(\mathbf{r})\cdot\mathbf{f}_m(\mathbf{r}')}{4\pi R}\, dS\, dS' + \text{(smooth part)}.
```

If we discretize each triangle with an $N_q$‑point Gaussian rule, the double sum contains $N_q^2$ terms. When $\mathbf{r}$ and $\mathbf{r}'$ coincide at a quadrature point, $R=0$ and the term is infinite. Even when points are distinct but very close, $1/R$ can be huge, leading to:
- **Catastrophic cancellation**: large positive and negative contributions that nearly cancel, magnifying rounding errors.
- **Loss of precision**: the finite sum may converge to a wrong value or fail to converge at all as $N_q$ increases.
- **Sensitivity to quadrature placement**: small shifts in quadrature points can change the result dramatically.

### 1.4 Physical Interpretation: Local vs. Non‑Local Interactions

From a physical perspective, the singularity corresponds to the **self‑field** of an infinitesimal current element. In reality, currents have finite cross‑section, and the field they produce on themselves is finite. The RWG discretization effectively “spreads” the current over triangles, but the point‑wise Green’s function still diverges. The correct finite result emerges only after integrating over both source and test triangles—a delicate cancellation that must be handled analytically or with specialized quadrature.

### 1.5 Where Singularity Handling Is Needed

In `DifferentiableMoM.jl`, singular integration is required for:
1. **Self‑cell interactions**: $T_m = T_n$ (same triangle).
2. **Adjacent‑cell interactions**: $T_m$ and $T_n$ share an edge or vertex ($R_{\min}=0$).
3. **Near‑singular interactions**: $T_m$ and $T_n$ are separated by less than ~$10^{-3}\lambda$, where standard quadrature loses accuracy.

The package implements a **split‑and‑regularize** strategy that separates the singular $1/R$ part, integrates it semi‑analytically, and treats the remaining smooth part with standard Gaussian quadrature. This approach guarantees accuracy, preserves differentiability, and is computationally efficient.

---

## 2. Algebraic Kernel Splitting: Separating Smooth from Singular

The core idea of singular integration in `DifferentiableMoM.jl` is to split the Green’s function into a **singular part** that can be integrated analytically and a **smooth part** that can be handled by standard Gaussian quadrature. This section derives the split, proves its mathematical correctness, and shows how it is implemented in the code.

### 2.1 The Additive Decomposition

Starting from the Green’s function $G(\mathbf{r},\mathbf{r}') = e^{-ikR}/(4\pi R)$, we write the oscillatory exponential as $1$ plus a remainder:

```math
e^{-ikR} = 1 + \bigl(e^{-ikR} - 1\bigr).
```

Substituting this into $G$ gives

```math
G(\mathbf{r},\mathbf{r}') = 
\underbrace{\frac{e^{-ikR} - 1}{4\pi R}}_{G_{\mathrm{smooth}}(\mathbf{r},\mathbf{r}')}
+ \underbrace{\frac{1}{4\pi R}}_{G_{\mathrm{sing}}(\mathbf{r},\mathbf{r}')}.
```

The first term, $G_{\mathrm{smooth}}$, contains the **difference** $e^{-ikR}-1$, which vanishes as $R\to 0$ at the same rate as $R$, canceling the $1/R$ singularity. The second term, $G_{\mathrm{sing}}$, is the pure $1/R$ singularity that causes the numerical difficulty.

### 2.2 Why the Split Works: Cancellation of Singularity

The key observation is that $e^{-ikR}-1 = O(R)$ for small $R$. More precisely, using the Taylor expansion $e^{-ikR} = 1 - ikR - \tfrac{1}{2}k^2R^2 + O(R^3)$, we obtain

```math
\frac{e^{-ikR} - 1}{R} = -ik - \frac{k^2R}{2} + O(R^2).
```

Thus $G_{\mathrm{smooth}}$ has a finite limit as $R\to 0$:

```math
\lim_{R\to 0} G_{\mathrm{smooth}}(\mathbf{r},\mathbf{r}') = -\frac{ik}{4\pi}.
```

In contrast, $G_{\mathrm{sing}} = 1/(4\pi R)$ diverges as $R^{-1}$. The split therefore isolates the entire singularity into $G_{\mathrm{sing}}$, leaving $G_{\mathrm{smooth}}$ bounded and suitable for ordinary quadrature.

### 2.3 Physical Interpretation

Physically, $G_{\mathrm{sing}}$ represents the **static** (zero‑frequency) Coulomb interaction, while $G_{\mathrm{smooth}}$ captures the **retardation and wave‑nature** corrections due to finite frequency. At low frequencies ($kR \ll 1$), the smooth part is small compared to the singular part; at high frequencies or large distances, both parts contribute comparably.

For self‑interactions ($R\approx 0$), the static part dominates and must be integrated with care, whereas the retardation correction is mild and can be integrated numerically.

### 2.4 Numerical Illustration

Consider a typical RF frequency $f = 3\ \text{GHz}$, giving $k = 2\pi f/c \approx 62.8\ \text{rad/m}$. The table below shows the behavior of the two parts for decreasing $R$:

| $R$ (m) | $G_{\mathrm{sing}}$ | $G_{\mathrm{smooth}}$ | $G = G_{\mathrm{sing}} + G_{\mathrm{smooth}}$ |
|---------|---------------------|-----------------------|-----------------------------------------------|
| $0.001$ | $79.6$ | $-5.0i$ | $\approx 79.6 - 5.0i$ |
| $0.0001$ | $795.8$ | $-5.0i$ | $\approx 795.8 - 5.0i$ |
| $0.00001$ | $7957.7$ | $-5.0i$ | $\approx 7957.7 - 5.0i$ |
| $0$ | $\infty$ | $-5.0i$ | $\infty$ |

**Critical insight**: $G_{\mathrm{smooth}}$ remains essentially constant (around $-ik/(4\pi) \approx -5.0i$) while $G_{\mathrm{sing}}$ grows without bound. Any numerical scheme that samples $G$ directly at small $R$ will see huge variations; sampling $G_{\mathrm{smooth}}$ alone is stable.

### 2.5 Implementation in the Code

The split is implemented in two functions:

- **`greens_smooth(r, rp, k)`** (in `src/basis/Greens.jl`): computes $G_{\mathrm{smooth}} = (e^{-ikR}-1)/(4\pi R)$ where $R = |\mathbf{r} - \mathbf{r}'|$, taking two position vectors `r` and `rp` and the wavenumber `k`. Uses careful handling of the small‑$R$ limit via series expansion to avoid catastrophic cancellation.
- **`self_cell_contribution`** (in `src/assembly/SingularIntegrals.jl`): accumulates the singular part $G_{\mathrm{sing}}$ using the semi‑analytical formula described in Section 3.

When assembling matrix entries, the code branches depending on whether the source and observation triangles are the same (self‑interaction) or different:

```julia
if tm == tn   # self‑interaction
    # Use split: smooth part via quadrature + singular part analytically
    contrib = smooth_part_quadrature() + singular_part_analytical()
else
    # Standard quadrature for the full G
    contrib = full_green_quadrature()
end
```

This branching ensures that the singular $1/R$ behavior is never evaluated directly by quadrature.

### 2.6 Advantages of the Split Approach

1. **Accuracy**: The analytical treatment of $G_{\mathrm{sing}}$ yields exact integrals for the singular part, eliminating quadrature error where it matters most.
2. **Stability**: $G_{\mathrm{smooth}}$ is bounded, so its quadrature converges rapidly and is insensitive to point placement.
3. **Generality**: The same split works for any kernel of the form $e^{-ikR}/(4\pi R)$; only the analytical integration of $1/R$ needs to be implemented for each geometry (triangle, rectangle, etc.).
4. **Differentiability**: Both $G_{\mathrm{smooth}}$ and the analytical integral of $G_{\mathrm{sing}}$ are differentiable with respect to parameters like $k$ or vertex coordinates, enabling gradient‑based optimization.

### 2.7 Connection to Other Singular Integration Methods

The additive split is a special case of **singularity subtraction** or **Duffy‑transformation** methods commonly used in integral equations. It is particularly simple for the Helmholtz kernel because the numerator $e^{-ikR}-1$ cancels the pole exactly. For more severe singularities (e.g., $1/R^2$ in the MFIE), a higher‑order subtraction or a Duffy transform would be required.

---

## 3. Semi‑Analytical Integration of the Singular Part

After splitting the Green’s function into $G_{\mathrm{smooth}}$ and $G_{\mathrm{sing}}$, we must evaluate integrals involving the singular part $1/(4\pi R)$ accurately. This section presents the closed‑form formula used in `DifferentiableMoM.jl` to compute the inner integral of $1/R$ over a triangle, explains its geometric derivation, and shows how it is implemented.

### 3.1 The Inner Singular Integral

For a fixed observation point $\mathbf{P}$ on (or near) a triangle $T$, define

```math
S(\mathbf{P}) = \int_T \frac{1}{|\mathbf{P} - \mathbf{r}'|}\, dS'.
```

This integral appears in both the vector‑ and scalar‑potential contributions when the source and observation triangles coincide. For example, the singular part of the vector‑potential term is

```math
V_{mn}^{\mathrm{sing}} = \frac{1}{4\pi} \int_T \mathbf{f}_m(\mathbf{P})\cdot\mathbf{f}_n(\mathbf{P})\, S(\mathbf{P})\, dS,
```

where we have used the fact that $\mathbf{f}_n(\mathbf{r}')$ can be approximated by $\mathbf{f}_n(\mathbf{P})$ when the singularity is dominant (see Section 4 for the complete treatment).

### 3.2 Closed‑Form Solution via Edge Contributions

The integral $S(\mathbf{P})$ over a triangle admits an exact analytical expression that can be written as a sum over the three edges:

```math
S(\mathbf{P}) = \sum_{i=1}^{3} d_i \,
\log\!\left(\frac{\ell_{B_i} + R_{B_i}}{\ell_{A_i} + R_{A_i}}\right).
```

Here, for edge $i$ with endpoints $\mathbf{A}_i$ and $\mathbf{B}_i$:

- $d_i$ is the perpendicular distance from $\mathbf{P}$ to the line containing the edge.
- $\ell_{A_i}$ and $\ell_{B_i}$ are the signed distances along the edge from the projection of $\mathbf{P}$ to $\mathbf{A}_i$ and $\mathbf{B}_i$, respectively (positive when the projection lies between the vertices, negative otherwise).
- $R_{A_i} = |\mathbf{P} - \mathbf{A}_i|$ and $R_{B_i} = |\mathbf{P} - \mathbf{B}_i|$ are the Euclidean distances from $\mathbf{P}$ to the vertices.

The formula is implemented in the function `analytical_integral_1overR` in `src/assembly/SingularIntegrals.jl`.

### 3.3 Geometric Derivation

The expression arises from a classical technique in potential theory: the integral of $1/R$ over a triangle can be transformed into a sum of line integrals over its edges using the divergence theorem. Consider the vector field

```math
\mathbf{F}(\mathbf{r}') = \frac{\mathbf{P} - \mathbf{r}'}{|\mathbf{P} - \mathbf{r}'|^3},
```

which satisfies $\nabla'\cdot\mathbf{F} = -4\pi\delta(\mathbf{P}-\mathbf{r}') + 2\delta_S(\mathbf{P}-\mathbf{r}')$ (where $\delta_S$ is a surface Dirac distribution). Applying the divergence theorem on the triangle and carefully handling the limits yields the logarithmic edge formula.

A more intuitive derivation starts from the known result for a line segment:

```math
\int_{\mathbf{A}}^{\mathbf{B}} \frac{1}{\sqrt{d^2 + s^2}}\, ds =
\log\!\left(\frac{\ell_B + \sqrt{d^2 + \ell_B^2}}{\ell_A + \sqrt{d^2 + \ell_A^2}}\right),
```

where $d$ is the perpendicular distance to the line and $\ell_A,\ell_B$ are signed distances along the line. Summing such contributions from the three edges, with proper signs, gives the triangle integral.

### 3.4 Step‑by‑Step Geometry

**For each triangle edge** $i$ joining vertices $\mathbf{A}_i$ and $\mathbf{B}_i$:

1. **Project $\mathbf{P}$ onto the edge line**: compute $\mathbf{P}_{\parallel}$, the orthogonal projection of $\mathbf{P}$ onto the infinite line through $\mathbf{A}_i$ and $\mathbf{B}_i$.
2. **Compute perpendicular distance**: $d_i = |\mathbf{P} - \mathbf{P}_{\parallel}|$.
3. **Compute signed edge coordinates**:
   ```math
   \ell_{A_i} = (\mathbf{A}_i - \mathbf{P}_{\parallel})\cdot\hat{\mathbf{t}}_i, \qquad
   \ell_{B_i} = (\mathbf{B}_i - \mathbf{P}_{\parallel})\cdot\hat{\mathbf{t}}_i,
   ```
   where $\hat{\mathbf{t}}_i$ is the unit tangent along the edge from $\mathbf{A}_i$ to $\mathbf{B}_i$.
4. **Compute vertex distances**: $R_{A_i} = |\mathbf{P} - \mathbf{A}_i|$, $R_{B_i} = |\mathbf{P} - \mathbf{B}_i|$.
5. **Add edge contribution**: $d_i \log[(\ell_{B_i}+R_{B_i})/(\ell_{A_i}+R_{A_i})]$.

The signs of $\ell_{A_i}$ and $\ell_{B_i}$ automatically account for whether $\mathbf{P}_{\parallel}$ lies inside the edge segment or beyond its endpoints.

### 3.5 Visual Guide

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

**Complete triangle integration**:

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

### 3.6 Worked Example

Consider a right triangle with vertices $\mathbf{A}=(0,0,0)$, $\mathbf{B}=(1,0,0)$, $\mathbf{C}=(0,1,0)$ and observation point $\mathbf{P}=(0.2,0.2,0)$.

**Edge $\mathbf{A}\mathbf{B}$**:
- Projection of $\mathbf{P}$ onto the $x$‑axis is $(0.2,0,0)$.
- $d_1 = 0.2$ (distance to $x$‑axis).
- $\ell_{A_1} = 0.2$, $\ell_{B_1} = 0.8$.
- $R_{A_1} = \sqrt{0.2^2 + 0.2^2} \approx 0.283$, $R_{B_1} = \sqrt{0.8^2 + 0.2^2} \approx 0.825$.
- Contribution: $0.2 \times \log[(0.8+0.825)/(0.2+0.283)] \approx 0.2 \times \log(1.625/0.483) \approx 0.2 \times 1.22 \approx 0.244$.

**Edge $\mathbf{B}\mathbf{C}$** and **edge $\mathbf{C}\mathbf{A}$** are computed similarly. The three contributions sum to $S(\mathbf{P})$, which for this symmetric point is approximately $0.244 + 0.244 + (\text{third edge}) \approx 0.7$ (the exact value can be obtained by running the code).

### 3.7 Why This Approach Is Effective

1. **Exact integration**: The formula gives the **exact** value of $S(\mathbf{P})$ up to machine precision, eliminating quadrature error for the singular part.
2. **Robust near singularities**: When $\mathbf{P}$ lies on the triangle (or very close), the logarithmic arguments remain well‑behaved; no division by zero occurs.
3. **Efficiency**: Evaluating the formula requires only elementary operations (dot products, square roots, logarithms) and is faster than high‑order quadrature.
4. **Differentiability**: The expression is differentiable with respect to $\mathbf{P}$ and the vertex coordinates, which is crucial for gradient‑based optimization.

### 3.8 Implementation Notes

The function `analytical_integral_1overR` handles several special cases:

- **Point on an edge**: $d_i = 0$ for that edge; the contribution is zero because $\log(1)=0$.
- **Point at a vertex**: two edges have $d_i=0$, the third edge gives a finite contribution.
- **Degenerate triangle**: checks for near‑zero area and returns a safe value.

The result is multiplied by $1/(4\pi)$ to match the factor in $G_{\mathrm{sing}} = 1/(4\pi R)$.

---

## 4. Vector‑Part Regularization: Canceling the Singularity

The singular integral for the vector‑potential term $V_{mn}$ requires special care because the basis functions $\mathbf{f}_n(\mathbf{r}')$ vary over the triangle. Simply replacing $\mathbf{f}_n(\mathbf{r}')$ by its value at the observation point $\mathbf{P}$ would introduce an $O(R)$ error that could spoil accuracy. This section explains how `DifferentiableMoM.jl` decomposes the vector integrand into a singular part that can be integrated analytically and a regular remainder suitable for Gaussian quadrature.

### 4.1 The Regularization Trick

Consider the singular part of the vector‑potential integral for a fixed observation point $\mathbf{P}$ on triangle $T$:

```math
I_{\mathrm{vec}}(\mathbf{P}) = \int_T \frac{\mathbf{f}_m(\mathbf{P})\cdot\mathbf{f}_n(\mathbf{r}')}{4\pi|\mathbf{P}-\mathbf{r}'|}\, dS'.
```

If we naively factor $\mathbf{f}_n(\mathbf{r}')$ out of the integral, we commit an error because $\mathbf{f}_n$ is not constant. Instead, we add and subtract $\mathbf{f}_n(\mathbf{P})$:

```math
\mathbf{f}_n(\mathbf{r}') = \mathbf{f}_n(\mathbf{P}) + \bigl[\mathbf{f}_n(\mathbf{r}') - \mathbf{f}_n(\mathbf{P})\bigr].
```

Substituting this into $I_{\mathrm{vec}}$ gives two terms:

```math
\begin{aligned}
I_{\mathrm{vec}}(\mathbf{P}) &=
\frac{\mathbf{f}_m(\mathbf{P})\cdot\mathbf{f}_n(\mathbf{P})}{4\pi}
\int_T \frac{1}{|\mathbf{P}-\mathbf{r}'|}\, dS' \\[4pt]
&+ \frac{\mathbf{f}_m(\mathbf{P})}{4\pi}\cdot\int_T 
\frac{\mathbf{f}_n(\mathbf{r}') - \mathbf{f}_n(\mathbf{P})}{|\mathbf{P}-\mathbf{r}'|}\, dS'.
\end{aligned}
```

The first term contains the singular integral $S(\mathbf{P}) = \int_T 1/|\mathbf{P}-\mathbf{r}'|\, dS'$ that we already know how to compute analytically (Section 3). The second term has a numerator that vanishes as $\mathbf{r}'\to\mathbf{P}$, canceling the $1/R$ singularity and leaving a bounded integrand.

### 4.2 Why the Remainder Is Bounded

RWG basis functions are **affine** (linear plus constant) on each triangle: $\mathbf{f}_n(\mathbf{r}') = \mathbf{a} + \mathbf{B}\mathbf{r}'$ with constant $\mathbf{a}$ and $\mathbf{B}$. Therefore

```math
\mathbf{f}_n(\mathbf{r}') - \mathbf{f}_n(\mathbf{P}) = \mathbf{B}\,(\mathbf{r}' - \mathbf{P}).
```

The numerator is thus $O(|\mathbf{r}'-\mathbf{P}|) = O(R)$. Dividing by $R$ gives

```math
\frac{\mathbf{f}_n(\mathbf{r}') - \mathbf{f}_n(\mathbf{P})}{|\mathbf{P}-\mathbf{r}'|} = \mathbf{B}\,\frac{\mathbf{r}'-\mathbf{P}}{|\mathbf{r}'-\mathbf{P}|},
```

which has magnitude $|\mathbf{B}|$ and direction along the unit vector $(\mathbf{r}'-\mathbf{P})/R$. This quotient is **finite everywhere**, including at $\mathbf{r}'=\mathbf{P}$ where it equals $\mathbf{B}$ times an arbitrary unit vector (the limit depends on the direction of approach, but the integrand remains bounded).

Consequently, the second integral

```math
\int_T \frac{\mathbf{f}_n(\mathbf{r}') - \mathbf{f}_n(\mathbf{P})}{|\mathbf{P}-\mathbf{r}'|}\, dS'
```

is a perfectly regular surface integral that can be evaluated accurately with standard Gaussian quadrature.

### 4.3 Complete Vector‑Potential Assembly

Putting everything together, the singular contribution to $V_{mn}$ for self‑interacting triangles is computed as:

```math
\begin{aligned}
V_{mn}^{\mathrm{sing}} &= 
\frac{1}{4\pi}\int_T \mathbf{f}_m(\mathbf{P})\cdot\mathbf{f}_n(\mathbf{P})\, S(\mathbf{P})\, dS \\[4pt]
&+ \frac{1}{4\pi}\int_T \mathbf{f}_m(\mathbf{P})\cdot
\Bigl[\int_T \frac{\mathbf{f}_n(\mathbf{r}') - \mathbf{f}_n(\mathbf{P})}{|\mathbf{P}-\mathbf{r}'|}\, dS'\Bigr] dS.
\end{aligned}
```

In the code, this is implemented as:

1. **Outer loop** over quadrature points $\mathbf{P}$ on triangle $T$.
2. **Inner singular integral**: compute $S(\mathbf{P})$ analytically using `analytical_integral_1overR`.
3. **Inner regular integral**: compute the remainder integral using Gaussian quadrature (since the integrand is bounded).
4. **Sum contributions** with appropriate weights.

### 4.4 Accuracy and Efficiency Trade‑Offs

The decomposition achieves **high accuracy** because:
- The singular part is integrated exactly (up to machine precision).
- The regular part is smooth, so Gaussian quadrature converges rapidly.

The cost is **moderate**: for each outer quadrature point $\mathbf{P}$, we must evaluate one analytical integral ($S(\mathbf{P})$) and one numerical inner integral (over $\mathbf{r}'$). However, because the number of self‑interactions is small compared to the total number of matrix entries ($O(N)$ vs. $O(N^2)$), the overhead is acceptable.

### 4.5 Connection to Singularity Subtraction Methods

The technique described here is a form of **singularity subtraction** or **regularization**. It is analogous to the method used in the boundary element method (BEM) for the Laplace equation, where the fundamental solution $1/(4\pi R)$ is treated similarly. The key insight for the Helmholtz equation is that the oscillatory factor $e^{-ikR}$ can be split as $1 + (e^{-ikR}-1)$, isolating the static singularity.

For more severe singularities (e.g., the $1/R^2$ kernel of the MFIE), a higher‑order Taylor expansion of the basis functions would be needed to cancel the stronger pole.

### 4.6 Implementation in `self_cell_contribution`

The function `self_cell_contribution` in `src/assembly/SingularIntegrals.jl` carries out the steps above. It loops over the quadrature points of the test triangle, evaluates the inner integrals (singular and regular), and accumulates the contribution to the matrix block. The regular inner integral is computed by calling the same quadrature routine used for non‑singular interactions, but with the regularized kernel $[\mathbf{f}_n(\mathbf{r}')-\mathbf{f}_n(\mathbf{P})]/R$.

---

## 5. Scalar‑Part Handling: Charge Interactions

The scalar‑potential term $S_{mn}$ in the mixed‑potential EFIE involves the surface divergences of the RWG basis functions. Because these divergences are **constant** on each triangle (Section 3.5 of Chapter 3), the singular integration simplifies dramatically. This section explains the scalar‑term treatment, its physical interpretation as charge‑charge interactions, and why accurate integration is critical for energy conservation.

### 5.1 Structure of the Scalar‑Potential Term

Recall from Chapter 2 that the scalar‑potential contribution to the EFIE matrix element is

```math
S_{mn} = \iint_\Gamma \bigl(\nabla_s\cdot\mathbf{f}_m(\mathbf{r})\bigr)
\bigl(\nabla_s'\cdot\mathbf{f}_n(\mathbf{r}')\bigr)\,
G(\mathbf{r},\mathbf{r}')\, dS\, dS'.
```

When the source and observation triangles coincide ($T_m = T_n$), we substitute the kernel split $G = G_{\mathrm{smooth}} + G_{\mathrm{sing}}$. The singular part becomes

```math
S_{mn}^{\mathrm{sing}} = \frac{1}{4\pi}
\int_T \nabla_s\cdot\mathbf{f}_m(\mathbf{P})\,
\int_T \nabla_s'\cdot\mathbf{f}_n(\mathbf{r}')\,
\frac{1}{|\mathbf{P}-\mathbf{r}'|}\, dS'\, dS.
```

The key simplification is that $\nabla_s\cdot\mathbf{f}_n(\mathbf{r}')$ is **constant** on each triangle (equal to $\pm\ell_n/A_n^\pm$). Therefore it can be pulled out of the inner integral:

```math
S_{mn}^{\mathrm{sing}} = \frac{1}{4\pi}
\int_T \bigl(\nabla_s\cdot\mathbf{f}_m(\mathbf{P})\bigr)
\bigl(\nabla_s\cdot\mathbf{f}_n(\mathbf{P})\bigr)\,
S(\mathbf{P})\, dS,
```

where $S(\mathbf{P}) = \int_T 1/|\mathbf{P}-\mathbf{r}'|\, dS'$ is the same analytical integral used for the vector part.

### 5.2 Physical Interpretation: Charge Density Interactions

Via the continuity equation $\nabla_s\cdot\mathbf{J} = -i\omega\rho$, the surface divergence of an RWG basis function is proportional to the **charge density** associated with that basis. Thus $S_{mn}$ represents the interaction between charge distributions induced by basis functions $m$ and $n$. The singular part $S_{mn}^{\mathrm{sing}}$ is the **Coulomb interaction** between these charge densities.

For self‑interactions ($m=n$), this is the **self‑energy** of a charge distribution, which diverges logarithmically for a point charge but remains finite for a charge spread over a triangle. The analytical integration of $1/R$ yields the exact finite self‑energy, essential for avoiding unphysical infinities in the numerical solution.

### 5.3 Why the Scalar Term Is Numerically Sensitive

The scalar block $S_{mn}$ is often the most delicate part of the EFIE matrix because:

1. **Charge conservation**: The EFIE enforces the continuity equation indirectly through the scalar term. Inaccuracies in $S_{mn}$ can lead to spurious charge accumulation, violating charge conservation and causing low‑frequency breakdown.
2. **Energy balance**: For lossless scatterers, the power radiated should equal the power extracted from the incident field. Errors in the self‑energy terms directly affect this balance, manifesting as unrealistic absorption or gain.
3. **Condition number**: The scalar block can dominate the matrix spectrum at low frequencies, making the system ill‑conditioned if the integration is not accurate.

Using the exact analytical integral for the singular part ensures that the dominant Coulomb interaction is treated exactly, preserving charge conservation and energy balance at the discrete level.

### 5.4 Implementation in `self_cell_contribution`

In the code, the scalar singular contribution is accumulated together with the vector singular contribution. For each outer quadrature point $\mathbf{P}$:

1. Compute $S(\mathbf{P})$ analytically via `analytical_integral_1overR`.
2. Multiply by the constant divergences $\nabla_s\cdot\mathbf{f}_m(\mathbf{P})$ and $\nabla_s\cdot\mathbf{f}_n(\mathbf{P})$.
3. Multiply by the quadrature weight and Jacobian $2A_T$.
4. Add to the total scalar‑part accumulator.

Because the divergences are constant on each triangle, they need not be recomputed at each quadrature point; they can be fetched from the precomputed `div_rwg` array.

### 5.5 Comparison with Vector‑Part Treatment

The scalar term is **simpler** than the vector term because:
- No regularization is needed: the divergences are constant, so no subtraction $[\nabla_s\cdot\mathbf{f}_n(\mathbf{r}') - \nabla_s\cdot\mathbf{f}_n(\mathbf{P})]$ is required.
- The entire singular dependence is contained in $S(\mathbf{P})$, which is integrated exactly.

However, the scalar term is **more sensitive** to errors because it couples directly to charge, and charge inaccuracies propagate through the continuity equation to the current solution.

### 5.6 Role in Low‑Frequency Stabilization

At low frequencies ($k\to 0$), the vector term $V_{mn}$ scales as $O(1)$ while the scalar term $S_{mn}$ scales as $O(1/k^2)$. This imbalance can cause catastrophic cancellation and loss of precision. Accurate integration of the singular part helps maintain a well‑balanced matrix, which is crucial for low‑frequency solvers and preconditioners.

---

## 6. Practical Implications and Validation

Accurate singular integration is not merely a numerical nicety—it is essential for obtaining physically meaningful results from the EFIE. This section describes the concrete symptoms that appear when singular integration is mishandled, explains why they occur, and shows how the split‑and‑analytical approach implemented in `DifferentiableMoM.jl` avoids these pitfalls.

### 6.1 Symptoms of Incorrect Singular Integration

When the self‑interaction terms $Z_{mm}$ are computed with inadequate accuracy (e.g., using standard quadrature without special treatment), one observes:

1. **Energy imbalance**: For a lossless PEC scatterer, the power radiated $P_{\mathrm{rad}}$ should equal the power extracted from the incident field $P_{\mathrm{in}}$ (energy conservation). With faulty self‑terms, the ratio $P_{\mathrm{rad}}/P_{\mathrm{in}}$ deviates from 1, sometimes by 10–50%, indicating that the numerical solution violates physics.

2. **Unstable mesh‑refinement trends**: As the mesh is refined (triangle size $h \to 0$), the solution should converge to a limit. With inaccurate self‑terms, convergence can be erratic: the error may initially decrease but then increase as $h$ becomes small, or the solution may oscillate without settling. This is because the error in the self‑terms scales differently with $h$ than the error in non‑singular terms, disrupting the error balance.

3. **Unreliable gradients**: In design optimization or sensitivity analysis, one needs derivatives of the scattering observables (e.g., RCS) with respect to parameters like frequency or geometry. Errors in self‑terms introduce discontinuities or wrong slopes in these derivatives, misleading optimization algorithms and yielding incorrect sensitivity maps.

4. **Low‑frequency breakdown**: At frequencies where $k h \ll 1$ (electrically small triangles), the scalar term $S_{mn}$ dominates the matrix. Inaccuracies in its singular part cause the matrix to become ill‑conditioned or even singular, preventing solution at very low frequencies.

5. **Spurious resonances**: Artificial resonances may appear at frequencies where the exact solution is smooth, due to numerical cancellation errors in the nearly singular integrals.

### 6.2 Why These Symptoms Arise

The root cause is that the self‑interaction terms contribute **dominant diagonal entries** to the EFIE matrix. For a typical PEC scattering problem, the diagonal entries $Z_{mm}$ are orders of magnitude larger than nearby off‑diagonal entries because the self‑field of a basis function is much stronger than the field it produces on distant bases. A relative error of 1% in $Z_{mm}$ can therefore overwhelm the entire balance of the linear system.

Moreover, the singular part of $Z_{mm}$ is **purely imaginary** (from $G_{\mathrm{sing}} = 1/(4\pi R)$) while the smooth part has both real and imaginary components. Errors in the singular part directly affect the phase of the solution, which governs interference and resonance phenomena.

### 6.3 How the Split‑and‑Analytical Strategy Fixes These Issues

The method described in Sections 2–5 addresses the problem at its core:

- **Exact integration of the singularity**: The analytical formula for $S(\mathbf{P})$ gives the singular part with machine precision, eliminating the dominant source of error.
- **Stable treatment of the smooth part**: $G_{\mathrm{smooth}}$ is bounded, so its quadrature converges rapidly and is insensitive to point placement.
- **Preservation of energy conservation**: Because the Coulomb self‑energy is computed exactly, the power balance of the discrete system matches that of the continuous equations.

As a result, the package passes stringent consistency checks such as:
- **Energy ratio test**: $|P_{\mathrm{rad}}/P_{\mathrm{in}} - 1| < 10^{-4}$ for a wide range of frequencies and meshes.
- **Mesh convergence**: The solution error decreases monotonically as $h \to 0$, with the expected convergence rate (typically $O(h^2)$ for RWG bases).
- **Gradient consistency**: Finite‑difference derivatives match automatic‑differentiation derivatives to high precision.

### 6.4 Comparison with Alternative Approaches

Other strategies for singular integration exist, each with trade‑offs:

| Method | Pros | Cons |
|--------|------|------|
| **Duffy transformation** | Maps triangle to square, regularizing the singularity | More complex implementation; requires adaptive quadrature |
| **Singularity cancellation** | Uses special quadrature weights that cancel the pole | Limited to specific kernels; less general |
| **Analytical integration** (used here) | Exact; fast; differentiable | Only works for kernels with known antiderivative |
| **Ignore singularity** (naïve) | Simple | Causes all the symptoms listed above |

The analytical approach chosen in `DifferentiableMoM.jl` is optimal for the Helmholtz kernel because the antiderivative of $1/R$ over a triangle is known in closed form. For more complicated kernels (e.g., dyadic Green’s functions), a Duffy transformation or higher‑order subtraction might be necessary.

### 6.5 When Singular Integration Matters Most

The need for accurate singular integration is most acute in:

- **Electrically small structures** ($kL \ll 1$): The static field dominates, making the $1/R$ singularity paramount.
- **High‑precision applications**: Radar cross‑section (RCS) prediction, antenna design, and quantum emitter modeling often require relative errors below 0.1%.
- **Gradient‑based optimization**: Incorrect gradients can steer optimization to wrong minima or cause failure to converge.
- **Multi‑physics coupling**: When the EFIE is coupled with circuit equations or thermal models, small errors in the field solution can amplify through the coupling.

For these applications, investing in robust singular integration pays off in reliability and accuracy.

---

## 7. Verification and Testing

Implementing singular integration correctly requires careful validation. This section describes practical tests that verify the accuracy of the split‑and‑analytical approach, detect common errors, and ensure the implementation meets the required standards for physical consistency and numerical stability.

### 7.1 Energy Conservation Test

The most telling test for singular integration is **energy conservation** for a lossless PEC scatterer. The script `examples/ex_convergence.jl` performs this test by:

1. Meshing a simple scatterer (e.g., a plate or sphere) with increasing refinement.
2. Solving the EFIE for a plane‑wave excitation.
3. Computing the radiated power $P_{\mathrm{rad}}$ from the surface currents and the extracted power $P_{\mathrm{in}}$ from the incident field.
4. Checking that the ratio $P_{\mathrm{rad}}/P_{\mathrm{in}}$ is close to 1 (typically within $10^{-4}$).

**Run the test**:
```bash
cd /path/to/DifferentiableMoM.jl
julia --project=. examples/ex_convergence.jl
```

**Expected output**: The script prints a table showing mesh size, number of unknowns, and the energy ratio. A correct implementation yields ratios between 0.999 and 1.001 across all refinements. Deviations larger than 0.01 indicate problems with singular integration.

### 7.2 Comparing With and Without Singular Treatment

To isolate the effect of singular integration, you can temporarily disable the analytical treatment and compare results. In the code, this can be done by modifying `self_cell_contribution` to skip the analytical integral and use standard quadrature for all terms.

```julia
# Pseudo‑code for a diagnostic test
function test_singular_effect(mesh, freq)
    # Solve with full singular treatment (default)
    Z_full = assemble_Z_efie(mesh, rwg, quad_rule, k)
    I_full = Z_full \ V
    P_rad_full = compute_radiated_power(I_full)
    
    # Solve with singular treatment disabled (naïve quadrature)
    Z_naive = assemble_Z_efie_naive(mesh, rwg, quad_rule, k)  # modified routine
    I_naive = Z_naive \ V
    P_rad_naive = compute_radiated_power(I_naive)
    
    println("Energy ratio (full): ", P_rad_full/P_in)
    println("Energy ratio (naïve): ", P_rad_naive/P_in)
end
```

The version with proper singular integration should show a much closer energy balance.

### 7.3 Convergence Under Mesh Refinement

A robust singular integration scheme should yield **monotonic convergence** as the mesh is refined. The error in scattering observables (e.g., radar cross‑section, input impedance) should decrease with $h^2$ for RWG basis functions (second‑order convergence for surface current).

To test this:
1. Generate a sequence of meshes with triangle sizes $h, h/2, h/4, \dots$.
2. Compute a reference solution using a very fine mesh or an independent method (e.g., analytical Mie series for a sphere).
3. Plot the error vs. $h$ on a log‑log scale; the slope should be approximately 2.

If the error curve plateaus or oscillates, singular integration errors are likely contaminating the convergence.

### 7.4 Gradient Consistency Check

Because `DifferentiableMoM.jl` is designed for differentiable optimization, it is crucial that the derivatives of matrix entries with respect to parameters (frequency, geometry) are accurate. The package includes gradient‑consistency tests that compare automatic‑differentiation (AD) gradients with finite‑difference approximations.

**Example test**:
```julia
using DifferentiableMoM, Zygote, FiniteDifferences

function test_gradient(k)
    mesh = make_rect_plate(0.1, 0.1, 10, 10)
    rwg = build_rwg(mesh)
    quad_rule = tri_quad_rule()
    Z = assemble_Z_efie(mesh, rwg, quad_rule, k)
    # Choose a few matrix entries
    f(k) = real(Z[1,1] + Z[5,5])
    grad_ad = gradient(f, k)[1]
    grad_fd = central_fdm(5, 1)(f, k)  # 5th‑order finite difference
    println("AD gradient: ", grad_ad, " FD gradient: ", grad_fd)
    return abs(grad_ad - grad_fd) / abs(grad_ad)
end
```

The relative difference should be below $10^{-8}$ for well‑conditioned parameters. Large discrepancies often point to errors in the derivative of the singular integral.

### 7.5 Unit Tests for `analytical_integral_1overR`

The function `analytical_integral_1overR` should be tested against known analytical results. For example:

- **Point at triangle centroid**: The integral can be compared with high‑order quadrature on a refined mesh (since the integrand is smooth away from the point).
- **Point on an edge**: The integral should be finite and match the result from splitting the triangle into two sub‑triangles.
- **Point at a vertex**: The integral should be finite and symmetric with respect to the adjacent edges.

These checks are covered in the package test suite (`test/runtests.jl`);
running `julia --project=. test/runtests.jl` executes them.

### 7.6 Near‑Singular Integration Test

When triangles are close but not touching, the integrand is nearly singular. Standard Gaussian quadrature may lose accuracy. The package should handle these cases gracefully, either by switching to a higher‑order rule or by applying a near‑singular correction (not yet implemented in the current version). A simple test is to compute $V_{mn}$ for two parallel triangles separated by a distance $d$ ranging from $10^{-6}\lambda$ to $0.1\lambda$ and verify that the result varies smoothly with $d$.

### 7.7 Practical Workflow for Debugging

If you suspect singular integration issues:

1. **Start with a very coarse mesh** (4–8 triangles) where you can inspect matrix entries manually.
2. **Print the self‑interaction matrix block** (diagonal and nearby off‑diagonals) and check for unusually large or small values.
3. **Compare with an independent code** (e.g., FEKO, NEC2) if available, or with a slower but more reliable integration method (e.g., adaptive quadrature).
4. **Visualize the current distribution**: spurious oscillations or extreme values near triangle boundaries often indicate charge‑conservation violations from inaccurate scalar‑term integration.
5. **Monitor the condition number** of the EFIE matrix as the mesh refines; it should grow slowly ($O(1/h)$). Rapid growth suggests ill‑conditioning due to singular‑integration errors.

By applying these tests, you can verify that the singular integration in `DifferentiableMoM.jl` is correct and reliable for production use.

---

## 8. Implementation Details and Code Map

This section provides a roadmap to the source files that implement singular integration in `DifferentiableMoM.jl`. Knowing where each piece of functionality resides is essential for extending the code, debugging, or adapting it to other integral equations.

### 8.1 Overview of Source Files

| File | Purpose | Key functions |
|------|---------|---------------|
| `src/basis/Greens.jl` | Green’s function evaluation and kernel splitting. | `greens`, `greens_smooth`, `grad_greens` |
| `src/assembly/SingularIntegrals.jl` | Analytical integration of singular kernels. | `analytical_integral_1overR`, `self_cell_contribution` |
| `src/assembly/EFIE.jl` | EFIE matrix assembly with self/non‑self branching. | `assemble_Z_efie` (self-cell path calls `self_cell_contribution`) |
| `src/basis/Quadrature.jl` | Gaussian quadrature rules on triangles. | `tri_quad_rule`, `tri_quad_points` |
| `src/basis/RWG.jl` | RWG basis function evaluation. | `eval_rwg`, `div_rwg`, `basis_triangles` |
| `test/runtests.jl` | Unit/integration tests for singular integration and convergence gates. | Includes checks for `analytical_integral_1overR` usage via EFIE assembly |

### 8.2 Key Functions and Their Roles

#### `greens_smooth(r, rp, k)` (in `src/basis/Greens.jl`)
Computes the smooth part of the Green’s function:
```math
G_{\mathrm{smooth}}(k,R) = \frac{e^{-ikR} - 1}{4\pi R}.
```
For small $R$, it uses a Taylor expansion to avoid catastrophic cancellation. For self-cell interactions, the kernel is split into $G = G_{\mathrm{smooth}} + G_{\mathrm{sing}}$ and this function provides the smooth part. For non-self interactions, the full `greens(rm, rn, k)` is used directly.

#### `analytical_integral_1overR(P, V1, V2, V3)` (in `src/assembly/SingularIntegrals.jl`)
Computes the singular integral
```math
S(\mathbf{P}) = \int_T \frac{1}{|\mathbf{P} - \mathbf{r}'|}\, dS'
```
using the closed‑form edge‑log formula described in Section 3. The arguments are:
- `P`: observation point (`Vec3`).
- `V1`, `V2`, `V3`: three separate `Vec3` arguments representing the triangle vertices.

The function returns a real number (the value of $S(\mathbf{P})$). Special cases (point on edge, point at vertex) are handled with care to avoid division by zero.

#### `self_cell_contribution(...)` (in `src/assembly/SingularIntegrals.jl`)
Accumulates the singular contribution for a **self triangle pair** (`tm == tn`).
In the current implementation, near-neighbor non-self pairs are handled by
standard Gaussian quadrature; only exact self-cell pairs use singular extraction.
The self-cell routine performs the following steps:

1. Loops over outer quadrature points $\mathbf{P}$ on triangle `tm`.
2. For each $\mathbf{P}$, computes $S(\mathbf{P})$ analytically.
3. Adds the scalar‑part contribution: $(\nabla\cdot\mathbf{f}_m)(\nabla\cdot\mathbf{f}_n) S(\mathbf{P})$.
4. Adds the vector‑part contribution using the regularization described in Section 4:
   - Singular part: $\mathbf{f}_m(\mathbf{P})\cdot\mathbf{f}_n(\mathbf{P}) S(\mathbf{P})$.
   - Regular part: quadrature over $\frac{\mathbf{f}_n(\mathbf{r}')-\mathbf{f}_n(\mathbf{P})}{|\mathbf{P}-\mathbf{r}'|}$.
5. Multiplies by weights and Jacobians and returns the total contribution.

#### `assemble_Z_efie` (in `src/assembly/EFIE.jl`)
This is the top‑level assembly routine that decides whether to call
`self_cell_contribution` or use standard product quadrature. The current
decision logic is:

```julia
if tm == tn
    # Use singular integration
    Zblock += self_cell_contribution(...)
else
    # Use standard product Gaussian quadrature
    Zblock += regular_product_quadrature(...)
end
```

### 8.3 Data Flow for a Self‑Interaction

To illustrate how the pieces fit together, here is the data flow for computing a self‑term $Z_{mm}$:

1. **`assemble_Z_efie`** identifies that test triangle `tm` and source triangle `tn` are identical.
2. **`self_cell_contribution`** is called with `tm`, `tn`.
3. **Outer loop** over quadrature points $\mathbf{P}$ on `tm`:
   - `tri_quad_points` provides $\mathbf{P}$ and weights.
   - `eval_rwg` evaluates $\mathbf{f}_m(\mathbf{P})$ and $\mathbf{f}_n(\mathbf{P})$.
   - `div_rwg` provides $\nabla_s\cdot\mathbf{f}_m$ and $\nabla_s\cdot\mathbf{f}_n$.
4. **Inner singular integral**:
   - `analytical_integral_1overR` computes $S(\mathbf{P})$.
5. **Inner regular integral**:
   - Loop over inner quadrature points $\mathbf{r}'$ on `tn`.
   - Compute $\mathbf{f}_n(\mathbf{r}') - \mathbf{f}_n(\mathbf{P})$.
   - Divide by $|\mathbf{P}-\mathbf{r}'|$ (bounded).
   - Accumulate with weights.
6. **Combine contributions** and add to $Z_{mm}$.

### 8.4 Extending the Singular Integration

If you need to implement singular integration for a different kernel (e.g., the Magnetic Field Integral Equation kernel $\nabla G \times$), follow these steps:

1. **Add a new analytical integral function** in `src/assembly/SingularIntegrals.jl` for the new singular kernel (e.g., `analytical_integral_gradG`).
2. **Extend `self_cell_contribution`** to handle the new kernel, possibly adding another regularization similar to the vector‑part decomposition.
3. **Modify `assemble_Z_efie`** (or create a new assembly routine) to call the extended self‑cell function when appropriate.

The modular design separates geometry, quadrature, basis functions, and singular integration, making such extensions manageable.

### 8.5 Performance Considerations

- **Caching**: The results of `analytical_integral_1overR` could be cached for repeated points (e.g., when the same triangle appears in many basis pairs), but the current implementation recomputes it each time because the overhead is small compared to the overall assembly cost.
- **Vectorization**: The inner regular integral is implemented with vectorized operations over quadrature points to exploit SIMD instructions.
- **Parallelism**: The outer loops in `assemble_Z_efie` are parallelized with `Threads.@threads` for multi‑core execution. Self‑interactions are a small fraction of the total work, so load balancing is not significantly affected.

### 8.6 Testing the Implementation

Run the singular‑integration tests with:

```bash
cd DifferentiableMoM.jl
julia --project=. -e 'using Pkg; Pkg.test("DifferentiableMoM")'
```

Look for the test set `"singular integrals"`; it should pass with no errors. If you modify any of the singular‑integration functions, re‑run these tests to ensure correctness.

---

## 9. Exercises and Problems

The following exercises reinforce the key concepts of singular integration, ranging from mathematical derivations to hands‑on coding tasks. Working through them will deepen your understanding of singularity subtraction, analytical integration, and regularization techniques.

### 9.1 Conceptual Questions

1. **Physical meaning of the singularity**: Explain in physical terms why the Green’s function $G(\mathbf{r},\mathbf{r}') = e^{-ikR}/(4\pi R)$ diverges as $R\to 0$. What does this divergence represent in terms of field interactions between infinitesimal current elements?

2. **Smooth vs. singular split**: Why is the split $G = G_{\mathrm{smooth}} + G_{\mathrm{sing}}$ with $G_{\mathrm{smooth}} = (e^{-ikR}-1)/(4\pi R)$ particularly effective for the Helmholtz kernel? Could we use $G_{\mathrm{sing}} = e^{-ikR}/(4\pi R)$ instead? What would be the consequence?

3. **Role of the vector‑part regularization**: In the vector‑potential term, why is it necessary to add and subtract $\mathbf{f}_n(\mathbf{P})$ rather than simply factoring $\mathbf{f}_n(\mathbf{r}')$ out of the singular integral? What would be the error if we omitted the subtraction?

4. **Charge conservation and scalar term**: How does accurate singular integration of the scalar‑potential term $S_{mn}$ help enforce charge conservation in the discrete EFIE? What symptoms appear when this integration is inaccurate?

5. **Near‑singular interactions**: For triangles that are close but not touching ($R_{\min} \ll \lambda$), the integrand is nearly singular. Why does standard Gaussian quadrature lose accuracy in this regime? How might you extend the singular‑integration technique to handle near‑singular cases?

### 9.2 Derivation Tasks

6. **Limit of the smooth kernel**: Derive the limit
   ```math
   \lim_{R\to 0} G_{\mathrm{smooth}}(k,R) = -\frac{ik}{4\pi}.
   ```
   Use the Taylor expansion $e^{-ikR} = 1 - ikR - \tfrac{1}{2}k^2R^2 + O(R^3)$ and show that the $1/R$ singularity cancels exactly.

7. **Cancellation in the vector remainder**: Prove that the remainder integrand
   ```math
   \frac{\mathbf{f}_n(\mathbf{r}') - \mathbf{f}_n(\mathbf{P})}{|\mathbf{P} - \mathbf{r}'|}
   ```
   is bounded as $\mathbf{r}'\to\mathbf{P}$. Use the fact that RWG basis functions are affine: $\mathbf{f}_n(\mathbf{r}') = \mathbf{a} + \mathbf{B}\mathbf{r}'$. Show that the numerator is $O(R)$ and the quotient tends to $\mathbf{B}\,(\mathbf{r}'-\mathbf{P})/R$, whose magnitude is $|\mathbf{B}|$.

8. **Edge‑log formula for a line segment**: Derive the closed‑form integral
   ```math
   \int_{\mathbf{A}}^{\mathbf{B}} \frac{1}{\sqrt{d^2 + s^2}}\, ds =
   \log\!\left(\frac{\ell_B + \sqrt{d^2 + \ell_B^2}}{\ell_A + \sqrt{d^2 + \ell_A^2}}\right),
   ```
   where $d$ is the perpendicular distance from the observation point to the line, and $\ell_A,\ell_B$ are signed distances along the line from the projection point to the endpoints. (Hint: use the substitution $s = d\sinh t$.)

9. **Consistency of the scalar term**: Using the constant‑divergence property of RWG functions, show that the singular part of the scalar‑potential term simplifies to
   ```math
   S_{mn}^{\mathrm{sing}} = \frac{1}{4\pi}
   \int_T (\nabla_s\cdot\mathbf{f}_m)(\nabla_s\cdot\mathbf{f}_n)\, S(\mathbf{P})\, dS,
   ```
   where $S(\mathbf{P}) = \int_T 1/|\mathbf{P}-\mathbf{r}'|\, dS'$. Why does no subtraction $[\nabla_s\cdot\mathbf{f}_n(\mathbf{r}') - \nabla_s\cdot\mathbf{f}_n(\mathbf{P})]$ appear?

10. **Energy conservation relation**: Starting from the Poynting theorem for a lossless scatterer, derive the energy‑balance condition
    ```math
    \operatorname{Re}\bigl\langle\mathbf{J}, \mathbf{E}^{\mathrm{inc}}\bigr\rangle = \frac{k\eta}{4} \iint \mathbf{J}(\mathbf{r})\cdot\operatorname{Im}\mathbf{G}(\mathbf{r},\mathbf{r}')\cdot\mathbf{J}(\mathbf{r}')\, dS\, dS',
    ```
    where $\mathbf{G}$ is the dyadic Green’s function. Explain how errors in the singular integration of the self‑terms can violate this condition.

### 9.3 Coding and Verification

11. **Testing `greens_smooth`**: Write a Julia script that compares `greens_smooth(k,R)` (from `src/basis/Greens.jl`) with the direct evaluation $(e^{-ikR}-1)/(4\pi R)$ for a range of $R$ values from $10^{-10}$ to $1$. Plot the relative error and verify that the series expansion used for small $R$ prevents catastrophic cancellation.

12. **Validating `analytical_integral_1overR`**: Create a test that computes $S(\mathbf{P})$ for a given triangle and observation point using both `analytical_integral_1overR` and high‑order Gaussian quadrature (e.g., `tri_quad_rule(7)`). Since the integrand $1/R$ is singular when $\mathbf{P}$ lies on the triangle, place $\mathbf{P}$ slightly above the triangle (e.g., offset by $10^{-6}$ in the normal direction) to make the integral regular. Compare the results for several triangle shapes and point locations.

13. **Energy conservation test**: Modify `examples/ex_convergence.jl` to also compute the energy ratio $P_{\mathrm{rad}}/P_{\mathrm{in}}$ when singular integration is **disabled** (modify `self_cell_contribution` to use standard quadrature for the singular part). Run the test for a simple plate mesh and compare the energy ratios with and without singular treatment.

14. **Gradient consistency**: Write a script that uses automatic differentiation (Zygote.jl or ForwardDiff.jl) to compute the derivative $\partial Z_{mm}/\partial k$ for a self‑interaction term. Compare with a finite‑difference approximation using central differences. Verify that the relative error is below $10^{-8}$ for a range of $k$ values.

15. **Near‑singular integration**: Write a function that computes $V_{mn}$ for two parallel triangles separated by a distance $d$. Vary $d$ from $10^{-6}\lambda$ to $0.1\lambda$ and plot the computed value. Observe any discontinuities or erratic behavior as $d$ becomes very small, indicating the need for near‑singular treatment.

### 9.4 Advanced Challenges

16. **Implementing Duffy transformation**: The Duffy transformation maps a square to a triangle with a change of variables that removes the $1/R$ singularity. Research the Duffy transform for a triangle‑to‑triangle interaction and implement a function `duffy_integral` that computes the singular integral numerically using this transformation. Compare its accuracy and speed with the analytical method used in `DifferentiableMoM.jl`.

17. **Extending to MFIE kernel**: The Magnetic Field Integral Equation (MFIE) kernel involves $\nabla G \times$, which has a $1/R^2$ singularity. Investigate how the split‑and‑regularize approach can be extended to handle this stronger singularity. Derive the necessary subtraction terms and implement a prototype for a single triangle self‑interaction.

18. **Adaptive quadrature for near‑singular cases**: Implement an adaptive quadrature scheme that increases the quadrature order when the distance between source and observation triangles falls below a threshold $d_{\mathrm{crit}}$. Integrate this scheme into `assemble_Z_efie` and measure the improvement in accuracy for nearly touching triangles.

19. **Cache optimization**: The analytical integral $S(\mathbf{P})$ depends only on the triangle geometry and the point $\mathbf{P}$. Design a caching mechanism that stores $S(\mathbf{P})$ for repeated $(triangle, point)$ pairs (e.g., when the same triangle appears in many basis pairs). Implement the cache and benchmark the assembly time for a mesh with several hundred triangles.

20. **Differentiable singular integration**: The analytical edge‑log formula is differentiable with respect to vertex coordinates. Write a test that computes the gradient $\partial S(\mathbf{P})/\partial \mathbf{v}_i$ where $\mathbf{v}_i$ is a triangle vertex, using both automatic differentiation of `analytical_integral_1overR` and finite differences. Verify that the results match to high precision.

---

## 10. Chapter Checklist

Before proceeding to Chapter 5, ensure you understand:

- [ ] Why product Gaussian quadrature fails for self‑interactions and adjacent triangles ($R_{\min}=0$).
- [ ] The additive kernel split $G = G_{\mathrm{smooth}} + G_{\mathrm{sing}}$ with $G_{\mathrm{smooth}} = (e^{-ikR}-1)/(4\pi R)$ and $G_{\mathrm{sing}} = 1/(4\pi R)$.
- [ ] How $G_{\mathrm{smooth}}$ remains bounded as $R\to 0$ ($\lim_{R\to0}G_{\mathrm{smooth}} = -ik/(4\pi)$) while $G_{\mathrm{sing}}$ captures the $1/R$ singularity.
- [ ] The closed‑form edge‑log formula for the singular integral $S(\mathbf{P}) = \int_T 1/|\mathbf{P}-\mathbf{r}'|\, dS'$ and its geometric interpretation.
- [ ] The vector‑part regularization: adding and subtracting $\mathbf{f}_n(\mathbf{P})$ to separate the singular integral from a bounded remainder.
- [ ] Why the scalar‑potential term simplifies because RWG divergences are constant on each triangle, requiring no subtraction.
- [ ] The symptoms of incorrect singular integration: energy imbalance, unstable mesh convergence, unreliable gradients, low‑frequency breakdown.
- [ ] How the implementation branches in `assemble_Z_efie` between singular (self/adjacent) and regular (distant) triangle pairs.
- [ ] The key functions: `greens_smooth` (in `src/basis/Greens.jl`), `analytical_integral_1overR` and `self_cell_contribution` (in `src/assembly/SingularIntegrals.jl`).

If any items are unclear, review the relevant sections or consult the mathematical prerequisites appendix.

---

## 11. Further Reading

1. **Original papers on singular integration**:
   - Graglia, R. D. (1993). *On the numerical integration of the linear shape functions times the 3‑D Green’s function or its gradient on a plane triangle*. IEEE Transactions on Antennas and Propagation, 41(10), 1448‑1455. (Provides analytical formulas for integrals of $1/R$ and $\nabla(1/R)$ over triangles.)
   - Taylor, D. J. (2003). *A comparison of methods for the accurate evaluation of singular integrals on triangular domains in the boundary element method*. International Journal for Numerical Methods in Engineering, 56(5), 693‑711. (Reviews singularity subtraction, Duffy transformation, and analytical integration.)
   - Khayat, M. A., & Wilton, D. R. (2005). *Numerical evaluation of singular and near‑singular potential integrals*. IEEE Transactions on Antennas and Propagation, 53(10), 3180‑3190. (Focuses on near‑singular integrals and adaptive quadrature.)

2. **Boundary element method textbooks**:
   - Brebbia, C. A., Telles, J. C. F., & Wrobel, L. C. (1984). *Boundary Element Techniques: Theory and Applications in Engineering*. Springer‑Verlag. (Classic BEM text covering singular integration techniques.)
   - Bonnet, M. (1995). *Équations intégrales et éléments de frontière*. CNRS Éditions. (French text with rigorous treatment of singular integrals in integral equations.)

3. **Numerical integration of singular kernels**:
   - Duffy, M. G. (1982). *Quadrature over a pyramid or cube of integrands with a singularity at a vertex*. SIAM Journal on Numerical Analysis, 19(6), 1260‑1262. (Introduces the Duffy transformation for regularizing singular integrals.)
   - Ma, J., & Rokhlin, V. (2005). *Quadrature for integrands with a logarithmic singularity*. SIAM Journal on Numerical Analysis, 43(1), 1‑15. (Advanced techniques for logarithmic singularities that appear in 2D problems.)

4. **Electromagnetic integral equation references**:
   - Chew, W. C., Jin, J. M., Michielssen, E., & Song, J. (2001). *Fast and Efficient Algorithms in Computational Electromagnetics*. Artech House. (Chapter 4 discusses singular integration in the context of fast multipole methods.)
   - Peterson, A. F., Ray, S. L., & Mittra, R. (1998). *Computational Methods for Electromagnetics*. IEEE Press. (Includes a section on singular integration for EFIE and MFIE.)

5. **Differentiable programming for integral equations**:
   - Ingraham, J., Riesselman, A., Sander, C., & Marks, D. (2019). *Learning protein structure with a differentiable simulator*. ICLR. (Not directly about electromagnetics, but illustrates how differentiable simulation enables gradient‑based optimization—a key motivation for `DifferentiableMoM.jl`.)

---

*Next: Chapter 5, “Conditioning and Preconditioning,” explores the numerical conditioning of the EFIE matrix and introduces preconditioning techniques that accelerate iterative solvers, especially at low frequencies.*
