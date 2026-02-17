module DifferentiableMoM

using LinearAlgebra
using SparseArrays
using StaticArrays
using Random
using Krylov

include("Types.jl")

# Geometry
include("geometry/Mesh.jl")
include("geometry/MeshIO.jl")

# Basis functions & quadrature
include("basis/RWG.jl")
include("basis/Quadrature.jl")
include("basis/Greens.jl")

# Assembly
include("assembly/SingularIntegrals.jl")
include("assembly/EFIE.jl")
include("assembly/Impedance.jl")
include("assembly/Excitation.jl")
include("assembly/CompositeOperator.jl")
include("assembly/SpatialPatches.jl")

# Fast methods
include("fast/ClusterTree.jl")
include("fast/ACA.jl")
include("fast/Octree.jl")
include("fast/MLFMA.jl")

# Post-processing (FarField needed by QMatrix)
include("postprocessing/FarField.jl")

# Optimization objectives
include("optimization/QMatrix.jl")

# Solvers
include("solver/Solve.jl")
include("solver/NearFieldPreconditioner.jl")
include("solver/IterativeSolve.jl")

# Adjoint & optimization
include("optimization/Adjoint.jl")
include("optimization/Verification.jl")
include("optimization/Optimize.jl")
include("optimization/MultiAngleRCS.jl")

# Workflow
include("Workflow.jl")

# Post-processing (remaining)
include("postprocessing/Diagnostics.jl")
include("postprocessing/PhysicalOptics.jl")
include("postprocessing/Mie.jl")
include("postprocessing/Visualization.jl")

end # module
