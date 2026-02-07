# FarField.jl — Far-field radiation pattern computation
#
# E∞(r̂) = (ik η₀)/(4π) r̂ × [r̂ × ∫ J(r') exp(ik r̂·r') dS']
#        = Σ_n I_n g_n(r̂)

export make_sph_grid, radiation_vectors, compute_farfield

"""
    make_sph_grid(Ntheta, Nphi)

Create a spherical grid using a uniform midpoint rule in θ and φ, with
quadrature weights w = sin(θ) dθ dφ.
Returns a SphGrid.
"""
function make_sph_grid(Ntheta::Int, Nphi::Int)
    # Simple uniform grid (sufficient for moderate resolution)
    dtheta = π / Ntheta
    dphi   = 2π / Nphi

    NΩ = Ntheta * Nphi
    rhat  = zeros(3, NΩ)
    theta = zeros(NΩ)
    phi   = zeros(NΩ)
    w     = zeros(NΩ)

    idx = 0
    for it in 1:Ntheta
        θ = (it - 0.5) * dtheta
        for ip in 1:Nphi
            φ = (ip - 0.5) * dphi
            idx += 1
            theta[idx] = θ
            phi[idx]   = φ
            rhat[1, idx] = sin(θ) * cos(φ)
            rhat[2, idx] = sin(θ) * sin(φ)
            rhat[3, idx] = cos(θ)
            w[idx] = sin(θ) * dtheta * dphi
        end
    end

    return SphGrid(rhat, theta, phi, w)
end

"""
    radiation_vectors(mesh, rwg, grid, k; quad_order=3, eta0=376.730313668)

Compute the per-basis radiation vectors g_n(r̂_q) for all basis functions
and all grid directions.

Returns G_mat of size (3*NΩ, N) such that
  G_mat[(3*(q-1)+1):(3*q), n] = g_n(r̂_q) ∈ C³
"""
function radiation_vectors(mesh::TriMesh, rwg::RWGData, grid::SphGrid, k;
                           quad_order::Int=3, eta0=376.730313668)
    N = rwg.nedges
    NΩ = length(grid.w)
    prefactor = 1im * k * eta0 / (4π)

    xi, wq = tri_quad_rule(quad_order)
    Nq = length(wq)

    G_mat = zeros(ComplexF64, 3 * NΩ, N)

    for n in 1:N
        # Compute ∫ f_n(r') exp(ik r̂·r') dS' for each r̂
        # by summing over the two support triangles
        for t in (rwg.tplus[n], rwg.tminus[n])
            A = triangle_area(mesh, t)
            pts = tri_quad_points(mesh, t, xi)

            for q_surf in 1:Nq
                rp = pts[q_surf]
                fn = eval_rwg(rwg, n, rp, t)
                wt = wq[q_surf] * 2 * A

                for q_dir in 1:NΩ
                    rh = Vec3(grid.rhat[:, q_dir])
                    phase = exp(1im * k * dot(rh, rp))

                    # Accumulate: N_n(r̂) = ∫ f_n exp(ik r̂·r') dS'
                    contrib = fn * (wt * phase)

                    # Apply r̂ × (r̂ × N) = (r̂·N)r̂ - N
                    rh_cross_N_cross = rh * dot(rh, contrib) - contrib

                    # g_n = prefactor * r̂ × (r̂ × N)
                    idx = 3 * (q_dir - 1)
                    G_mat[idx+1, n] += prefactor * rh_cross_N_cross[1]
                    G_mat[idx+2, n] += prefactor * rh_cross_N_cross[2]
                    G_mat[idx+3, n] += prefactor * rh_cross_N_cross[3]
                end
            end
        end
    end

    return G_mat
end

"""
    compute_farfield(G_mat, I_coeffs, NΩ)

Compute the far-field E∞(r̂_q) = Σ_n I_n g_n(r̂_q) for all grid points.
Returns a (3, NΩ) complex matrix.
"""
function compute_farfield(G_mat::Matrix{ComplexF64}, I_coeffs::Vector{ComplexF64}, NΩ::Int)
    # G_mat is (3*NΩ, N), I_coeffs is (N,)
    E_flat = G_mat * I_coeffs   # (3*NΩ,)
    E = reshape(E_flat, 3, NΩ)
    return E
end
