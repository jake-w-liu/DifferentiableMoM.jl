# Types.jl — Core data structures for DifferentiableMoM

export Vec3, CVec3, TriMesh, RWGData, PatchPartition, SphGrid, ScatteringResult
export nvertices, ntriangles

const Vec3 = SVector{3,Float64}
const CVec3 = SVector{3,ComplexF64}

"""
Triangle mesh: vertices and triangle connectivity.
"""
struct TriMesh
    xyz::Matrix{Float64}        # (3, Nv) vertex coordinates
    tri::Matrix{Int}            # (3, Nt) 1-based vertex indices per triangle
end

nvertices(m::TriMesh) = size(m.xyz, 2)
ntriangles(m::TriMesh) = size(m.tri, 2)

"""
RWG basis function data for each interior edge.
"""
struct RWGData
    mesh::TriMesh
    nedges::Int
    tplus::Vector{Int}          # T+ triangle index
    tminus::Vector{Int}         # T- triangle index
    evert::Matrix{Int}          # (2, nedges) edge vertex indices
    vplus_opp::Vector{Int}      # opposite vertex in T+
    vminus_opp::Vector{Int}     # opposite vertex in T-
    len::Vector{Float64}        # edge length
    area_plus::Vector{Float64}  # area of T+
    area_minus::Vector{Float64} # area of T-
end

"""
Patch partition: maps each triangle to a design patch.
"""
struct PatchPartition
    tri_patch::Vector{Int}      # length Nt: patch id for each triangle
    P::Int                      # number of patches
end

"""
Spherical far-field sampling grid.
"""
struct SphGrid
    rhat::Matrix{Float64}       # (3, NΩ) unit direction vectors
    theta::Vector{Float64}      # polar angles
    phi::Vector{Float64}        # azimuthal angles
    w::Vector{Float64}          # quadrature weights
end

"""
    ScatteringResult

Result from `solve_scattering`, containing the solution and metadata.
"""
struct ScatteringResult
    I_coeffs::Vector{ComplexF64}
    method::Symbol
    N::Int
    assembly_time_s::Float64
    solve_time_s::Float64
    preconditioner_time_s::Float64
    gmres_iters::Int
    gmres_residual::Float64
    mesh_report::NamedTuple
    warnings::Vector{String}
end
