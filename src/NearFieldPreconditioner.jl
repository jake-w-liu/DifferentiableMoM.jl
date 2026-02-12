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

export NearFieldPreconditionerData,
       build_nearfield_preconditioner,
       NearFieldOperator,
       NearFieldAdjointOperator,
       rwg_centers

"""
    NearFieldPreconditionerData

Stores the factorized near-field sparse preconditioner.

Fields:
- `Z_nf_fac`: LU factorization of the near-field sparse matrix
- `cutoff`: the distance cutoff used (in meters)
- `nnz_ratio`: fraction of nonzeros retained (nnz(Z_nf)/N²)
"""
struct NearFieldPreconditionerData
    Z_nf_fac::SparseArrays.UMFPACK.UmfpackLU{ComplexF64, Int64}
    cutoff::Float64
    nnz_ratio::Float64
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

# Returns
A `NearFieldPreconditionerData` containing the factorized near-field matrix.
"""
function build_nearfield_preconditioner(Z::Matrix{ComplexF64}, mesh::TriMesh,
                                         rwg::RWGData, cutoff::Float64)
    N = rwg.nedges
    centers = rwg_centers(mesh, rwg)

    # Build sparse near-field matrix
    I_idx = Int[]
    J_idx = Int[]
    V_val = ComplexF64[]

    for m in 1:N
        for n in 1:N
            d = norm(centers[m] - centers[n])
            if d <= cutoff
                push!(I_idx, m)
                push!(J_idx, n)
                push!(V_val, Z[m, n])
            end
        end
    end

    Z_nf = sparse(I_idx, J_idx, V_val, N, N)
    nnz_ratio = length(V_val) / (N * N)

    # LU factorize
    Z_nf_fac = lu(Z_nf)

    return NearFieldPreconditionerData(Z_nf_fac, cutoff, nnz_ratio)
end

"""
    NearFieldOperator

Callable wrapper for use with Krylov.jl as a preconditioner.
Applies Z_nf⁻¹ v.
"""
struct NearFieldOperator
    P::NearFieldPreconditionerData
end

Base.size(op::NearFieldOperator) = (size(op.P.Z_nf_fac, 1), size(op.P.Z_nf_fac, 1))
Base.eltype(::NearFieldOperator) = ComplexF64

function Base.:*(op::NearFieldOperator, v::AbstractVector)
    return Vector{ComplexF64}(op.P.Z_nf_fac \ Vector{ComplexF64}(v))
end

function LinearAlgebra.mul!(y::AbstractVector, op::NearFieldOperator, x::AbstractVector)
    y .= op.P.Z_nf_fac \ Vector{ComplexF64}(x)
    return y
end

"""
    NearFieldAdjointOperator

Callable wrapper for the adjoint preconditioner Z_nf⁻ᴴ v.
"""
struct NearFieldAdjointOperator
    P::NearFieldPreconditionerData
end

Base.size(op::NearFieldAdjointOperator) = (size(op.P.Z_nf_fac, 1), size(op.P.Z_nf_fac, 1))
Base.eltype(::NearFieldAdjointOperator) = ComplexF64

function Base.:*(op::NearFieldAdjointOperator, v::AbstractVector)
    return Vector{ComplexF64}(op.P.Z_nf_fac' \ Vector{ComplexF64}(v))
end

function LinearAlgebra.mul!(y::AbstractVector, op::NearFieldAdjointOperator, x::AbstractVector)
    y .= op.P.Z_nf_fac' \ Vector{ComplexF64}(x)
    return y
end
