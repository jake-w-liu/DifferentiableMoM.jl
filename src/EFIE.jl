# EFIE.jl — Dense and matrix-free EFIE operators
#
# Z_mn = <f_m, T[f_n]>  (EFIE operator only, without impedance sheet)
#
# Uses mixed-potential form:
#   Z_mn = -iωμ₀ [ ∫∫ f_m(r)·f_n(r') G(r,r') dS dS'
#                   - (1/k²) ∫∫ (∇·f_m)(∇'·f_n) G(r,r') dS dS' ]

export assemble_Z_efie
export MatrixFreeEFIEOperator, MatrixFreeEFIEAdjointOperator
export matrixfree_efie_operator, matrixfree_efie_adjoint_operator
export efie_entry

struct EFIEApplyCache{TK, Tω}
    mesh::TriMesh
    rwg::RWGData
    k::TK
    omega_mu0::Tω
    wq::Vector{Float64}
    Nq::Int
    quad_pts::Vector{Vector{Vec3}}
    areas::Vector{Float64}
    tri_ids::Matrix{Int}                 # (2, N)
    div_vals::Matrix{Float64}            # (2, N)
    rwg_vals::Vector{NTuple{2,Vector{Vec3}}}
end

function _build_efie_cache(mesh::TriMesh, rwg::RWGData, k;
                           quad_order::Int=3, eta0::Float64=376.730313668)
    N = rwg.nedges
    Nt = ntriangles(mesh)

    omega_mu0 = k * eta0   # ωμ₀ = k η₀

    xi, wq = tri_quad_rule(quad_order)
    Nq = length(wq)

    # Precompute quadrature points and areas for all triangles
    quad_pts = Vector{Vector{Vec3}}(undef, Nt)
    areas = Vector{Float64}(undef, Nt)
    for t in 1:Nt
        quad_pts[t] = tri_quad_points(mesh, t, xi)
        areas[t] = triangle_area(mesh, t)
    end

    tri_ids = zeros(Int, 2, N)
    div_vals = zeros(Float64, 2, N)
    rwg_vals = Vector{NTuple{2,Vector{Vec3}}}(undef, N)

    for n in 1:N
        tp = rwg.tplus[n]
        tm = rwg.tminus[n]
        tri_ids[1, n] = tp
        tri_ids[2, n] = tm
        div_vals[1, n] = div_rwg(rwg, n, tp)
        div_vals[2, n] = div_rwg(rwg, n, tm)

        vals_p = [eval_rwg(rwg, n, quad_pts[tp][q], tp) for q in 1:Nq]
        vals_m = [eval_rwg(rwg, n, quad_pts[tm][q], tm) for q in 1:Nq]
        rwg_vals[n] = (vals_p, vals_m)
    end

    return EFIEApplyCache(mesh, rwg, k, omega_mu0, wq, Nq, quad_pts, areas, tri_ids, div_vals, rwg_vals)
end

@inline function _efie_entry(cache::EFIEApplyCache, m::Int, n::Int)
    CT = ComplexF64
    val = zero(CT)

    @inbounds for itm in 1:2
        tm = cache.tri_ids[itm, m]
        Am = cache.areas[tm]
        dvm = cache.div_vals[itm, m]
        fm_vals = _rwg_vals(cache, m, itm)

        for itn in 1:2
            tn = cache.tri_ids[itn, n]
            An = cache.areas[tn]
            dvn = cache.div_vals[itn, n]
            fn_vals = _rwg_vals(cache, n, itn)

            if tm == tn
                # Self-cell singularity extraction
                val += self_cell_contribution(
                    cache.mesh, cache.rwg, n, tm,
                    cache.quad_pts[tm],
                    fm_vals,
                    fn_vals,
                    dvm,
                    dvn,
                    Am,
                    cache.wq,
                    cache.k,
                )
            else
                # Non-self product quadrature
                for qm in 1:cache.Nq
                    rm = cache.quad_pts[tm][qm]
                    fm = fm_vals[qm]

                    for qn in 1:cache.Nq
                        rn = cache.quad_pts[tn][qn]
                        fn = fn_vals[qn]

                        G = greens(rm, rn, cache.k)

                        vec_part = dot(fm, fn) * G
                        scl_part = dvm * dvn * G / (cache.k^2)
                        weight = cache.wq[qm] * cache.wq[qn] * (2 * Am) * (2 * An)

                        val += (vec_part - scl_part) * weight
                    end
                end
            end
        end
    end

    return -1im * cache.omega_mu0 * val
end

@inline function _rwg_vals(cache::EFIEApplyCache, n::Int, it::Int)
    if it == 1
        return cache.rwg_vals[n][1]
    end
    return cache.rwg_vals[n][2]
end

"""
    assemble_Z_efie(mesh, rwg, k; quad_order=3, mesh_precheck=true, allow_boundary=true, require_closed=false)

Assemble the dense EFIE matrix `Z_efie ∈ C^{N×N}`.
`k` is the wavenumber (can be complex for complex-step).
"""
function assemble_Z_efie(mesh::TriMesh, rwg::RWGData, k;
                         quad_order::Int=3, eta0=376.730313668,
                         mesh_precheck::Bool=true,
                         allow_boundary::Bool=true,
                         require_closed::Bool=false,
                         area_tol_rel::Float64=1e-12)
    if mesh_precheck
        assert_mesh_quality(mesh;
            allow_boundary=allow_boundary,
            require_closed=require_closed,
            area_tol_rel=area_tol_rel,
        )
    end

    cache = _build_efie_cache(mesh, rwg, k; quad_order=quad_order, eta0=eta0)
    N = rwg.nedges
    CT = ComplexF64
    Z = zeros(CT, N, N)

    @inbounds for m in 1:N
        for n in 1:N
            Z[m, n] = _efie_entry(cache, m, n)
        end
    end

    return Z
end

"""
Matrix-free EFIE operator `A` such that `A * x` computes the EFIE matvec
without allocating a dense `N×N` matrix.
"""
struct MatrixFreeEFIEOperator{T, TC<:EFIEApplyCache} <: AbstractMatrix{T}
    cache::TC
end

"""
Adjoint matrix-free EFIE operator `Aᴴ` used by GMRES adjoint solves.
"""
struct MatrixFreeEFIEAdjointOperator{T, TO<:MatrixFreeEFIEOperator{T}} <: AbstractMatrix{T}
    op::TO
end

Base.size(A::MatrixFreeEFIEOperator) = (A.cache.rwg.nedges, A.cache.rwg.nedges)
Base.eltype(::MatrixFreeEFIEOperator{T}) where {T} = T

Base.size(A::MatrixFreeEFIEAdjointOperator) = size(A.op)
Base.eltype(::MatrixFreeEFIEAdjointOperator{T}) where {T} = T

function efie_entry(A::MatrixFreeEFIEOperator, m::Int, n::Int)
    return _efie_entry(A.cache, m, n)
end

function Base.getindex(A::MatrixFreeEFIEOperator, i::Int, j::Int)
    return efie_entry(A, i, j)
end

function Base.getindex(A::MatrixFreeEFIEAdjointOperator, i::Int, j::Int)
    return conj(efie_entry(A.op, j, i))
end

function LinearAlgebra.mul!(y::AbstractVector{T}, A::MatrixFreeEFIEOperator{T}, x::AbstractVector) where {T<:Number}
    N = size(A, 1)
    length(x) == N || throw(DimensionMismatch("x length $(length(x)) != $N"))
    length(y) == N || throw(DimensionMismatch("y length $(length(y)) != $N"))

    fill!(y, zero(T))
    @inbounds for m in 1:N
        acc = zero(T)
        for n in 1:N
            acc += efie_entry(A, m, n) * x[n]
        end
        y[m] = acc
    end
    return y
end

function Base.:*(A::MatrixFreeEFIEOperator{T}, x::AbstractVector) where {T<:Number}
    y = zeros(T, size(A, 1))
    mul!(y, A, x)
    return y
end

function LinearAlgebra.mul!(y::AbstractVector{T}, A::MatrixFreeEFIEAdjointOperator{T}, x::AbstractVector) where {T<:Number}
    N = size(A, 1)
    length(x) == N || throw(DimensionMismatch("x length $(length(x)) != $N"))
    length(y) == N || throw(DimensionMismatch("y length $(length(y)) != $N"))

    fill!(y, zero(T))
    @inbounds for n in 1:N
        acc = zero(T)
        for m in 1:N
            acc += conj(efie_entry(A.op, m, n)) * x[m]
        end
        y[n] = acc
    end
    return y
end

function Base.:*(A::MatrixFreeEFIEAdjointOperator{T}, x::AbstractVector) where {T<:Number}
    y = zeros(T, size(A, 1))
    mul!(y, A, x)
    return y
end

LinearAlgebra.adjoint(A::MatrixFreeEFIEOperator{T}) where {T<:Number} =
    MatrixFreeEFIEAdjointOperator{T,typeof(A)}(A)

"""
    matrixfree_efie_operator(mesh, rwg, k; kwargs...)

Create a matrix-free EFIE operator for GMRES matvecs without dense `Z` allocation.
"""
function matrixfree_efie_operator(mesh::TriMesh, rwg::RWGData, k;
                                  quad_order::Int=3, eta0=376.730313668,
                                  mesh_precheck::Bool=true,
                                  allow_boundary::Bool=true,
                                  require_closed::Bool=false,
                                  area_tol_rel::Float64=1e-12)
    if mesh_precheck
        assert_mesh_quality(mesh;
            allow_boundary=allow_boundary,
            require_closed=require_closed,
            area_tol_rel=area_tol_rel,
        )
    end
    cache = _build_efie_cache(mesh, rwg, k; quad_order=quad_order, eta0=eta0)
    return MatrixFreeEFIEOperator{ComplexF64,typeof(cache)}(cache)
end

"""
    matrixfree_efie_adjoint_operator(A)

Return matrix-free adjoint operator `Aᴴ` for Krylov adjoint solves.
"""
matrixfree_efie_adjoint_operator(A::MatrixFreeEFIEOperator) = adjoint(A)
