# SingularIntegrals.jl — Singularity extraction for EFIE self-cell integrals
#
# When source and observation triangles coincide (self-cell), the Green's
# function G = exp(-ikR)/(4πR) has a 1/R singularity that standard Gaussian
# quadrature cannot resolve.  We split G = G_smooth + 1/(4πR) and compute:
#   - Smooth part: standard product quadrature with G_smooth (bounded)
#   - Singular part: semi-analytical (outer quadrature, analytical inner)

export analytical_integral_1overR, self_cell_contribution

"""
    analytical_integral_1overR(P, V1, V2, V3)

Compute ∫_T 1/|P - r'| dS' analytically for observation point P coplanar
with the flat triangle T = (V1, V2, V3).

Uses the divergence-theorem identity ∇'·((r'-P)/R) = 1/R in 2D, giving:
  ∫_T 1/R dS' = Σ_edges d_i × log[(l_B + R_B)/(l_A + R_A)]

where for each directed edge A → B:
  d_i = (A - P) · n̂   (signed perpendicular distance from P to edge line)
  l_A = (A - P) · ℓ̂,  l_B = (B - P) · ℓ̂   (tangential projections)
  R_A = |P - A|,  R_B = |P - B|
  ℓ̂ = edge tangent,  n̂ = cross(ℓ̂, n̂_T) = outward edge normal in the plane
"""
function analytical_integral_1overR(P::Vec3, V1::Vec3, V2::Vec3, V3::Vec3)
    n_T = cross(V2 - V1, V3 - V1)
    n_norm = norm(n_T)
    if n_norm < 1e-30
        return 0.0
    end
    n_T = n_T / n_norm

    edges = ((V1, V2), (V2, V3), (V3, V1))
    result = 0.0

    for (A, B) in edges
        edge_vec = B - A
        edge_len = norm(edge_vec)
        if edge_len < 1e-30
            continue
        end
        lhat = edge_vec / edge_len
        nhat = cross(lhat, n_T)

        d_i = dot(A - P, nhat)

        # When |d_i| ≈ 0, P is on the edge line and d*log → 0
        if abs(d_i) < 1e-15
            continue
        end

        l_A = dot(A - P, lhat)
        l_B = dot(B - P, lhat)
        R_A = norm(P - A)
        R_B = norm(P - B)

        denom = l_A + R_A
        numer = l_B + R_B
        if abs(denom) < 1e-30 || abs(numer) < 1e-30
            continue
        end

        result += d_i * log(numer / denom)
    end

    return result
end

"""
    self_cell_contribution(mesh, rwg, n, tm,
                           quad_pts_tm, rwg_vals_m, rwg_vals_n,
                           div_m, div_n, Am, wq, k)

Compute the EFIE self-cell integral for basis functions m, n on the same
triangle tm using singularity extraction.

Returns the value (vec_part - scl_part), not yet multiplied by -iωμ₀.

The integral splits as:
  I = I_smooth  (Nq×Nq product quadrature with G_smooth)
    + I_singular (outer Nq-point quadrature, analytical inner ∫ 1/R dS')
"""
function self_cell_contribution(
    mesh::TriMesh, rwg::RWGData,
    n::Int, tm::Int,
    quad_pts_tm::Vector{Vec3},
    rwg_vals_m::Vector{Vec3},
    rwg_vals_n::Vector{Vec3},
    div_m::Float64, div_n::Float64,
    Am::Float64, wq, k)

    Nq = length(wq)
    CT = complex(typeof(real(k)))

    V1 = _mesh_vertex(mesh, mesh.tri[1, tm])
    V2 = _mesh_vertex(mesh, mesh.tri[2, tm])
    V3 = _mesh_vertex(mesh, mesh.tri[3, tm])

    # ── Smooth part: standard product quadrature with G_smooth ──
    val_smooth = zero(CT)
    for qm in 1:Nq
        rm = quad_pts_tm[qm]
        fm = rwg_vals_m[qm]
        for qn in 1:Nq
            rn = quad_pts_tm[qn]
            fn = rwg_vals_n[qn]

            Gs = greens_smooth(rm, rn, k)
            vec_part = dot(fm, fn) * Gs
            scl_part = div_m * div_n * Gs / (k^2)
            weight = wq[qm] * wq[qn] * (2 * Am) * (2 * Am)
            val_smooth += (vec_part - scl_part) * weight
        end
    end

    # ── Singular part: semi-analytical ──
    # For each outer quad point r_qm, compute:
    #   S = ∫_T 1/|r_qm - r'| dS'  (analytical)
    #
    # Vector part: ∫_T f_m·f_n/(4πR) dS'
    #   = f_m·f_n(r_qm) × S/(4π)
    #     + ∫_T f_m·[f_n(r') - f_n(r_qm)]/(4πR) dS'  (regular, standard quad)
    #
    # Scalar part: div_m × div_n × S / (4πk²)
    inv4pi = 1.0 / (4π)

    val_singular = zero(CT)
    for qm in 1:Nq
        rm = quad_pts_tm[qm]
        fm = rwg_vals_m[qm]

        S = analytical_integral_1overR(rm, V1, V2, V3)
        inner_scalar = inv4pi * S

        # Scalar potential singular part
        scl_sing = div_m * div_n * inner_scalar / (k^2)

        # Vector potential singular part: leading term
        fn_at_rm = eval_rwg(rwg, n, rm, tm)
        vec_lead = dot(fm, fn_at_rm) * inner_scalar

        # Vector potential singular part: remainder (bounded integrand)
        vec_rem = zero(CT)
        for qn in 1:Nq
            rn = quad_pts_tm[qn]
            fn = rwg_vals_n[qn]

            R_vec = rm - rn
            R = sqrt(dot(R_vec, R_vec))
            if R < 1e-14
                continue  # [f_n(rn) - f_n(rm)] = 0 when rn = rm
            end

            delta_fn = fn - fn_at_rm
            vec_rem += dot(fm, delta_fn) * (inv4pi / R) * wq[qn] * (2 * Am)
        end

        outer_weight = wq[qm] * (2 * Am)
        val_singular += ((vec_lead + vec_rem) - scl_sing) * outer_weight
    end

    return val_smooth + val_singular
end
