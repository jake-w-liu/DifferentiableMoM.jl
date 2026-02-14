# NearFieldPreconditioner.jl — Near-field sparse preconditioner for EFIE systems
#
# Builds a sparse approximation of Z by retaining only entries corresponding
# to RWG basis functions within a near-field cutoff distance. The LU
# factorization of this sparse matrix serves as a preconditioner for GMRES.
#
# This is the standard and effective approach for preconditioning dense MoM
# systems: near-field interactions dominate the matrix structure, while
# far-field entries decay as 1/r and contribute less to conditioning.
#
# Key advantage: the near-field preconditioner explicitly captures the spatial
# coupling structure of the Green's function, giving N-independent iteration
# counts for GMRES.

using SparseArrays
using IncompleteLU

export NearFieldPreconditionerData,
       ILUPreconditionerData,
       DiagonalPreconditionerData,
       AbstractPreconditionerData,
       build_nearfield_preconditioner,
       NearFieldOperator,
       NearFieldAdjointOperator,
       rwg_centers

abstract type AbstractPreconditionerData end

"""
    NearFieldPreconditionerData <: AbstractPreconditionerData

Stores the sparse-LU near-field preconditioner.
"""
struct NearFieldPreconditionerData <: AbstractPreconditionerData
    Z_nf_fac::SparseArrays.UMFPACK.UmfpackLU{ComplexF64, Int64}
    cutoff::Float64
    nnz_ratio::Float64
end

"""
    DiagonalPreconditionerData <: AbstractPreconditionerData

Stores a Jacobi/diagonal preconditioner as the inverse diagonal entries.
"""
struct DiagonalPreconditionerData <: AbstractPreconditionerData
    dinv::Vector{ComplexF64}
    cutoff::Float64
    nnz_ratio::Float64
end

"""
    ILUPreconditionerData <: AbstractPreconditionerData

Stores an incomplete LU (ILU) factorization of the near-field sparse matrix.
Uses IncompleteLU.jl's Crout ILU with drop tolerance τ.

Compared to full sparse LU (`NearFieldPreconditionerData`):
- Much less fill-in → feasible for large N with moderate nnz%
- Slightly weaker preconditioner → more GMRES iterations
- Memory: controlled by τ (smaller τ = more fill = better preconditioner)
"""
struct ILUPreconditionerData <: AbstractPreconditionerData
    ilu_fac::IncompleteLU.ILUFactorization{ComplexF64, Int64}
    cutoff::Float64
    nnz_ratio::Float64
    tau::Float64
end

"""
    rwg_centers(mesh, rwg)

Compute the center point of each RWG basis function, defined as the average
of the centroids of its two supporting triangles.

Returns a Vector{Vec3} of length N.
"""
function rwg_centers(mesh::TriMesh, rwg::RWGData)
    N = rwg.nedges
    centers = Vector{Vec3}(undef, N)
    for n in 1:N
        c_plus  = triangle_center(mesh, rwg.tplus[n])
        c_minus = triangle_center(mesh, rwg.tminus[n])
        centers[n] = 0.5 * (c_plus + c_minus)
    end
    return centers
end

@inline function _cell_key(r::Vec3, inv_cell::Float64)
    return (
        floor(Int, r[1] * inv_cell),
        floor(Int, r[2] * inv_cell),
        floor(Int, r[3] * inv_cell),
    )
end

function _nearfield_triplets_bruteforce(centers::Vector{Vec3}, cutoff::Float64, getvalue)
    N = length(centers)
    I_idx = Int[]
    J_idx = Int[]
    V_val = ComplexF64[]

    if N == 0
        return I_idx, J_idx, V_val
    end

    if cutoff <= 0
        sizehint!(I_idx, N)
        sizehint!(J_idx, N)
        sizehint!(V_val, N)
        @inbounds for m in 1:N
            push!(I_idx, m)
            push!(J_idx, m)
            push!(V_val, ComplexF64(getvalue(m, m)))
        end
        return I_idx, J_idx, V_val
    end

    if !isfinite(cutoff)
        sizehint!(I_idx, N * N)
        sizehint!(J_idx, N * N)
        sizehint!(V_val, N * N)
        @inbounds for m in 1:N
            for n in 1:N
                push!(I_idx, m)
                push!(J_idx, n)
                push!(V_val, ComplexF64(getvalue(m, n)))
            end
        end
        return I_idx, J_idx, V_val
    end

    cutoff2 = cutoff * cutoff
    @inbounds for m in 1:N
        cm = centers[m]
        for n in 1:N
            δ = cm - centers[n]
            d2 = dot(δ, δ)
            if m == n || d2 <= cutoff2
                push!(I_idx, m)
                push!(J_idx, n)
                push!(V_val, ComplexF64(getvalue(m, n)))
            end
        end
    end
    return I_idx, J_idx, V_val
end

function _nearfield_triplets_spatial(centers::Vector{Vec3}, cutoff::Float64, getvalue)
    N = length(centers)
    I_idx = Int[]
    J_idx = Int[]
    V_val = ComplexF64[]

    if N == 0
        return I_idx, J_idx, V_val
    end

    if cutoff <= 0 || !isfinite(cutoff)
        return _nearfield_triplets_bruteforce(centers, cutoff, getvalue)
    end

    inv_cell = 1.0 / cutoff
    buckets = Dict{NTuple{3,Int}, Vector{Int}}()
    @inbounds for i in 1:N
        key = _cell_key(centers[i], inv_cell)
        push!(get!(() -> Int[], buckets, key), i)
    end

    cutoff2 = cutoff * cutoff
    est_pairs = max(N, min(N * N, 27 * N))
    sizehint!(I_idx, est_pairs)
    sizehint!(J_idx, est_pairs)
    sizehint!(V_val, est_pairs)

    @inbounds for m in 1:N
        cm = centers[m]
        key = _cell_key(cm, inv_cell)
        for dz in -1:1
            for dy in -1:1
                for dx in -1:1
                    key_n = (key[1] + dx, key[2] + dy, key[3] + dz)
                    n_list = get(buckets, key_n, nothing)
                    n_list === nothing && continue
                    for n in n_list
                        if m == n
                            push!(I_idx, m)
                            push!(J_idx, n)
                            push!(V_val, ComplexF64(getvalue(m, n)))
                        else
                            δ = cm - centers[n]
                            d2 = dot(δ, δ)
                            if d2 <= cutoff2
                                push!(I_idx, m)
                                push!(J_idx, n)
                                push!(V_val, ComplexF64(getvalue(m, n)))
                            end
                        end
                    end
                end
            end
        end
    end
    return I_idx, J_idx, V_val
end

function _build_diagonal_preconditioner_data(getvalue, N::Int, cutoff::Float64)
    diag_entries = Vector{ComplexF64}(undef, N)
    maxabs = 0.0
    @inbounds for i in 1:N
        zii = ComplexF64(getvalue(i, i))
        diag_entries[i] = zii
        ai = abs(zii)
        if ai > maxabs
            maxabs = ai
        end
    end

    floor_abs = 1e-10 * max(maxabs, 1.0)
    @inbounds for i in 1:N
        if abs(diag_entries[i]) < floor_abs
            diag_entries[i] = floor_abs + 0im
        end
    end

    dinv = similar(diag_entries)
    @inbounds for i in 1:N
        dinv[i] = inv(diag_entries[i])
    end

    nnz_ratio = N == 0 ? 0.0 : (N / (N * N))
    return DiagonalPreconditionerData(dinv, cutoff, nnz_ratio)
end

function _build_nearfield_preconditioner_from_entries(mesh::TriMesh, rwg::RWGData, cutoff::Float64,
                                                       getvalue;
                                                       neighbor_search::Symbol=:spatial,
                                                       factorization::Symbol=:lu,
                                                       ilu_tau::Float64=1e-3)
    N = rwg.nedges

    if factorization == :diag
        return _build_diagonal_preconditioner_data(getvalue, N, cutoff)
    elseif factorization ∉ (:lu, :ilu)
        error("Invalid factorization: $factorization (expected :lu, :ilu, or :diag)")
    end

    centers = rwg_centers(mesh, rwg)

    I_idx, J_idx, V_val = if neighbor_search == :spatial
        _nearfield_triplets_spatial(centers, cutoff, getvalue)
    elseif neighbor_search == :bruteforce
        _nearfield_triplets_bruteforce(centers, cutoff, getvalue)
    else
        error("Invalid neighbor_search: $neighbor_search (expected :spatial or :bruteforce)")
    end

    Z_nf = sparse(I_idx, J_idx, V_val, N, N)
    nnz_ratio = nnz(Z_nf) / max(N * N, 1)

    if factorization == :ilu
        ilu_fac = IncompleteLU.ilu(Z_nf, τ = ilu_tau)
        return ILUPreconditionerData(ilu_fac, cutoff, nnz_ratio, ilu_tau)
    else
        Z_nf_fac = lu(Z_nf)
        return NearFieldPreconditionerData(Z_nf_fac, cutoff, nnz_ratio)
    end
end

"""
    build_nearfield_preconditioner(Z, mesh, rwg, cutoff)

Build a near-field sparse preconditioner for the MoM matrix Z.

Retains entries Z[m,n] where the distance between the centers of RWG basis
functions m and n is ≤ `cutoff`. The sparse matrix is then LU-factorized.

# Arguments
- `Z::Matrix{ComplexF64}`: the full N×N MoM matrix
- `mesh::TriMesh`: the triangle mesh
- `rwg::RWGData`: RWG basis function data
- `cutoff::Float64`: distance cutoff (e.g., 0.5λ to 2λ)
- `neighbor_search`: `:spatial` (default) or `:bruteforce` reference mode

# Returns
A `NearFieldPreconditionerData` containing the factorized near-field matrix.
"""
function build_nearfield_preconditioner(Z::Matrix{<:Number}, mesh::TriMesh,
                                         rwg::RWGData, cutoff::Float64;
                                         neighbor_search::Symbol=:spatial,
                                         factorization::Symbol=:lu,
                                         ilu_tau::Float64=1e-3)
    N = rwg.nedges
    size(Z, 1) == N && size(Z, 2) == N ||
        throw(DimensionMismatch("Z has size $(size(Z)), expected ($N, $N) for RWG basis"))
    return _build_nearfield_preconditioner_from_entries(mesh, rwg, cutoff,
        (m, n) -> Z[m, n];
        neighbor_search=neighbor_search,
        factorization=factorization,
        ilu_tau=ilu_tau,
    )
end

"""
    build_nearfield_preconditioner(A, mesh, rwg, cutoff)

Build a near-field sparse preconditioner directly from an operator `A`
without allocating a full dense matrix.
"""
function build_nearfield_preconditioner(A::AbstractMatrix{<:Number}, mesh::TriMesh,
                                         rwg::RWGData, cutoff::Float64;
                                         neighbor_search::Symbol=:spatial,
                                         factorization::Symbol=:lu,
                                         ilu_tau::Float64=1e-3)
    N = rwg.nedges
    size(A, 1) == N && size(A, 2) == N ||
        throw(DimensionMismatch("A has size $(size(A)), expected ($N, $N) for RWG basis"))
    return _build_nearfield_preconditioner_from_entries(mesh, rwg, cutoff,
        (m, n) -> A[m, n];
        neighbor_search=neighbor_search,
        factorization=factorization,
        ilu_tau=ilu_tau,
    )
end

"""
    build_nearfield_preconditioner(A::MatrixFreeEFIEOperator, cutoff)

Convenience overload for building a near-field preconditioner directly from
the matrix-free EFIE operator cache.
"""
function build_nearfield_preconditioner(A::MatrixFreeEFIEOperator, cutoff::Float64;
                                         neighbor_search::Symbol=:spatial,
                                         factorization::Symbol=:lu,
                                         ilu_tau::Float64=1e-3)
    return build_nearfield_preconditioner(A, A.cache.mesh, A.cache.rwg, cutoff;
        neighbor_search=neighbor_search,
        factorization=factorization,
        ilu_tau=ilu_tau,
    )
end

"""
    build_nearfield_preconditioner(mesh, rwg, k, cutoff; kwargs...)

Build a near-field sparse preconditioner directly from EFIE geometry/physics
inputs, without dense EFIE assembly.
"""
function build_nearfield_preconditioner(mesh::TriMesh, rwg::RWGData, k, cutoff::Float64;
                                         quad_order::Int=3,
                                         eta0::Float64=376.730313668,
                                         mesh_precheck::Bool=true,
                                         allow_boundary::Bool=true,
                                         require_closed::Bool=false,
                                         area_tol_rel::Float64=1e-12,
                                         neighbor_search::Symbol=:spatial,
                                         factorization::Symbol=:lu,
                                         ilu_tau::Float64=1e-3)
    A = matrixfree_efie_operator(mesh, rwg, k;
        quad_order=quad_order,
        eta0=eta0,
        mesh_precheck=mesh_precheck,
        allow_boundary=allow_boundary,
        require_closed=require_closed,
        area_tol_rel=area_tol_rel,
    )
    return build_nearfield_preconditioner(A, cutoff;
        neighbor_search=neighbor_search,
        factorization=factorization,
        ilu_tau=ilu_tau,
    )
end

"""
    build_nearfield_preconditioner(Z_nf_sparse; factorization=:lu, ilu_tau=1e-3)

Build a preconditioner directly from a pre-assembled sparse near-field matrix.
Skips the distance-based neighbor search (useful for MLFMA where Z_near is
already assembled with the correct sparsity pattern from the octree).
"""
function build_nearfield_preconditioner(Z_nf::SparseMatrixCSC{ComplexF64,Int};
                                         factorization::Symbol=:lu,
                                         ilu_tau::Float64=1e-3)
    N = size(Z_nf, 1)
    nnz_ratio = nnz(Z_nf) / max(N * N, 1)

    if factorization == :ilu
        ilu_fac = IncompleteLU.ilu(Z_nf, τ = ilu_tau)
        return ILUPreconditionerData(ilu_fac, Inf, nnz_ratio, ilu_tau)
    elseif factorization == :lu
        Z_nf_fac = lu(Z_nf)
        return NearFieldPreconditionerData(Z_nf_fac, Inf, nnz_ratio)
    elseif factorization == :diag
        dinv = Vector{ComplexF64}(undef, N)
        maxabs = 0.0
        @inbounds for i in 1:N
            zii = Z_nf[i, i]
            dinv[i] = zii
            ai = abs(zii)
            if ai > maxabs; maxabs = ai; end
        end
        floor_abs = 1e-10 * max(maxabs, 1.0)
        @inbounds for i in 1:N
            if abs(dinv[i]) < floor_abs
                dinv[i] = floor_abs + 0im
            end
            dinv[i] = inv(dinv[i])
        end
        return DiagonalPreconditionerData(dinv, Inf, nnz_ratio)
    else
        error("Invalid factorization: $factorization (expected :lu, :ilu, or :diag)")
    end
end

"""
    NearFieldOperator

Callable wrapper for use with Krylov.jl as a preconditioner.
Applies Z_nf⁻¹ v.
"""
struct NearFieldOperator{PType<:AbstractPreconditionerData}
    P::PType
end

@inline _preconditioner_size(P::NearFieldPreconditionerData) = size(P.Z_nf_fac, 1)
@inline _preconditioner_size(P::DiagonalPreconditionerData) = length(P.dinv)
@inline _preconditioner_size(P::ILUPreconditionerData) = size(P.ilu_fac.L, 1)

@inline function _apply_preconditioner!(y::StridedVector{ComplexF64}, P::NearFieldPreconditionerData)
    ldiv!(P.Z_nf_fac, y)
    return y
end

@inline function _apply_preconditioner!(y::StridedVector{ComplexF64}, P::DiagonalPreconditionerData)
    @inbounds @simd for i in eachindex(y)
        y[i] *= P.dinv[i]
    end
    return y
end

@inline function _apply_preconditioner!(y::StridedVector{ComplexF64}, P::ILUPreconditionerData)
    ldiv!(P.ilu_fac, y)
    return y
end

@inline function _apply_preconditioner_adjoint!(y::StridedVector{ComplexF64}, P::NearFieldPreconditionerData)
    ldiv!(adjoint(P.Z_nf_fac), y)
    return y
end

@inline function _apply_preconditioner_adjoint!(y::StridedVector{ComplexF64}, P::DiagonalPreconditionerData)
    @inbounds @simd for i in eachindex(y)
        y[i] *= conj(P.dinv[i])
    end
    return y
end

@inline function _apply_preconditioner_adjoint!(y::StridedVector{ComplexF64}, P::ILUPreconditionerData)
    # (LU)⁻ᴴ y = U⁻ᴴ L⁻ᴴ y
    # Step 1: solve Uᴴ z = y  (Uᴴ is lower triangular)
    ldiv!(adjoint(UpperTriangular(P.ilu_fac.U)), y)
    # Step 2: solve Lᴴ x = z  (Lᴴ is unit upper triangular)
    ldiv!(adjoint(UnitLowerTriangular(P.ilu_fac.L)), y)
    return y
end

Base.size(op::NearFieldOperator) = (_preconditioner_size(op.P), _preconditioner_size(op.P))
Base.eltype(::NearFieldOperator) = ComplexF64

function Base.:*(op::NearFieldOperator, v::AbstractVector)
    y = Vector{ComplexF64}(v)
    _apply_preconditioner!(y, op.P)
    return y
end

function LinearAlgebra.mul!(y::StridedVector{ComplexF64}, op::NearFieldOperator, x::StridedVector{ComplexF64})
    length(y) == length(x) || throw(DimensionMismatch("x length $(length(x)) != $(length(y))"))
    if y !== x
        copyto!(y, x)
    end
    _apply_preconditioner!(y, op.P)
    return y
end

function LinearAlgebra.mul!(y::AbstractVector, op::NearFieldOperator, x::AbstractVector)
    length(y) == length(x) || throw(DimensionMismatch("x length $(length(x)) != $(length(y))"))
    y .= op * x
    return y
end

"""
    NearFieldAdjointOperator

Callable wrapper for the adjoint preconditioner Z_nf⁻ᴴ v.
"""
struct NearFieldAdjointOperator{PType<:AbstractPreconditionerData}
    P::PType
end

Base.size(op::NearFieldAdjointOperator) = (_preconditioner_size(op.P), _preconditioner_size(op.P))
Base.eltype(::NearFieldAdjointOperator) = ComplexF64

function Base.:*(op::NearFieldAdjointOperator, v::AbstractVector)
    y = Vector{ComplexF64}(v)
    _apply_preconditioner_adjoint!(y, op.P)
    return y
end

function LinearAlgebra.mul!(y::StridedVector{ComplexF64}, op::NearFieldAdjointOperator, x::StridedVector{ComplexF64})
    length(y) == length(x) || throw(DimensionMismatch("x length $(length(x)) != $(length(y))"))
    if y !== x
        copyto!(y, x)
    end
    _apply_preconditioner_adjoint!(y, op.P)
    return y
end

function LinearAlgebra.mul!(y::AbstractVector, op::NearFieldAdjointOperator, x::AbstractVector)
    length(y) == length(x) || throw(DimensionMismatch("x length $(length(x)) != $(length(y))"))
    y .= op * x
    return y
end
