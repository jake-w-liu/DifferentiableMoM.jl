module DifferentiableMoM

using LinearAlgebra
using SparseArrays
using StaticArrays

include("Types.jl")
include("Mesh.jl")
include("RWG.jl")
include("Quadrature.jl")
include("Greens.jl")
include("SingularIntegrals.jl")
include("EFIE.jl")
include("Impedance.jl")
include("Excitation.jl")
include("FarField.jl")
include("QMatrix.jl")
include("Solve.jl")
include("Adjoint.jl")
include("Verification.jl")
include("Optimize.jl")
include("Diagnostics.jl")
include("Mie.jl")
include("Visualization.jl")

end # module
