# DensityInterpolation.jl — Per-triangle density design variables for topology optimization
#
# Material interpolation for PEC/void metasurface topology optimization.
# Each triangle t has density ρ_t ∈ [0,1]:
#   ρ = 1 → PEC (metal), no penalty
#   ρ = 0 → void, large impedance penalty kills surface currents
#
# Penalty model (SIMP):
#   Z_total = Z_efie + Σ_t (1 - ρ̄_t^p) * Z_max * M_t
#
# where M_t[m,n] = ∫_t f_m · f_n dS is the triangle mass matrix,
# p is the SIMP penalization power, and Z_max is the void penalty.
#
# Following Tucek, Capek, Jelinek (IEEE TAP 2023) approach.

export precompute_triangle_mass, assemble_Z_penalty, assemble_dZ_drhobar
export DensityConfig

"""
    DensityConfig

Configuration for density-based topology optimization.
"""
struct DensityConfig
    p::Float64          # SIMP penalization power (typically 3)
    Z_max::Float64      # void penalty impedance, real (large, e.g. 1000*eta0)
    vf_target::Float64  # target volume fraction (metal fraction)
end

"""
    DensityConfig(; p=3.0, Z_max_factor=1000.0, eta0=376.730313668, vf_target=0.5)

Construct DensityConfig with default parameters.
"""
function DensityConfig(; p::Float64=3.0, Z_max_factor::Float64=1000.0,
                       eta0::Float64=376.730313668, vf_target::Float64=0.5)
    return DensityConfig(p, Z_max_factor * eta0, vf_target)
end

"""
    precompute_triangle_mass(mesh, rwg; quad_order=3)

Precompute per-triangle mass matrices M_t[m,n] = ∫_t f_m · f_n dS
for all triangles t = 1:Nt.

Returns a vector of sparse matrices, one per triangle.
Only basis functions with support on triangle t have nonzero entries in M_t.
"""
function precompute_triangle_mass(mesh::TriMesh, rwg::RWGData; quad_order::Int=3)
    N = rwg.nedges
    Nt = ntriangles(mesh)

    xi, wq = tri_quad_rule(quad_order)
    Nq = length(wq)

    # Precompute quad points and areas
    quad_pts = [tri_quad_points(mesh, t, xi) for t in 1:Nt]
    areas = [triangle_area(mesh, t) for t in 1:Nt]

    # Map triangle → basis functions with support
    tri_to_basis = [Int[] for _ in 1:Nt]
    for n in 1:N
        push!(tri_to_basis[rwg.tplus[n]], n)
        push!(tri_to_basis[rwg.tminus[n]], n)
    end

    # Build sparse mass matrix for each triangle
    Mt = [spzeros(Float64, N, N) for _ in 1:Nt]

    for t in 1:Nt
        A = areas[t]
        basis_on_t = tri_to_basis[t]

        for bi in eachindex(basis_on_t)
            m = basis_on_t[bi]
            for bj in eachindex(basis_on_t)
                n = basis_on_t[bj]

                val = 0.0
                for q in 1:Nq
                    rq = quad_pts[t][q]
                    fm = eval_rwg(rwg, m, rq, t)
                    fn = eval_rwg(rwg, n, rq, t)
                    val += wq[q] * dot(fm, fn)
                end
                val *= 2 * A  # reference-to-physical Jacobian

                Mt[t][m, n] += val
            end
        end
    end

    return Mt
end

"""
    assemble_Z_penalty(Mt, rho_bar, config)

Assemble the density penalty matrix:

    Z_penalty = Σ_t (1 - ρ̄_t^p) * Z_max * M_t

where ρ̄ are the filtered/projected densities.

When ρ̄_t = 1 (metal): penalty contribution = 0
When ρ̄_t = 0 (void):  penalty contribution = Z_max * M_t (large impedance)
"""
function assemble_Z_penalty(Mt::Vector{<:AbstractMatrix},
                            rho_bar::AbstractVector{<:Real},
                            config::DensityConfig)
    Nt = length(Mt)
    @assert length(rho_bar) == Nt "rho_bar length ($(length(rho_bar))) must match number of triangles ($Nt)"
    N = size(Mt[1], 1)
    CT = ComplexF64

    Z_pen = zeros(CT, N, N)
    for t in 1:Nt
        penalty = (1 - rho_bar[t]^config.p) * config.Z_max
        Z_pen .+= penalty .* Mt[t]
    end

    return Z_pen
end

"""
    assemble_dZ_drhobar(Mt, rho_bar, config, t)

Compute the derivative ∂Z_penalty/∂ρ̄_t for triangle t:

    ∂Z/∂ρ̄_t = -p * ρ̄_t^(p-1) * Z_max * M_t

This is exact and closed-form (no finite differences needed).
"""
function assemble_dZ_drhobar(Mt::Vector{<:AbstractMatrix},
                             rho_bar::AbstractVector{<:Real},
                             config::DensityConfig, t::Int)
    return -config.p * rho_bar[t]^(config.p - 1) * config.Z_max * Mt[t]
end
