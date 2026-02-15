# Quick test: MLFMA matvec accuracy vs dense at different leaf_lambda values
# Uses Level 0 mesh (N=7584) where dense Z is feasible.

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "DifferentiableMoM.jl"))
using .DifferentiableMoM
using LinearAlgebra
using SparseArrays

c0 = 299792458.0
freq = 0.3e9
λ0 = c0 / freq
k = 2π / λ0

obj_path = joinpath(@__DIR__, "demo_aircraft.obj")
mesh_raw = read_obj_mesh(obj_path)
rep = repair_mesh_for_simulation(mesh_raw; allow_boundary=true, auto_drop_nonmanifold=true)
mesh_air = rep.mesh

ref0 = refine_mesh_to_target_edge(mesh_air, 4.0 * λ0; max_iters=2, max_triangles=50_000)
mesh0 = ref0.mesh
rwg0 = build_rwg(mesh0)
N0 = rwg0.nedges
println("N = $N0 RWG unknowns")

# Dense Z for reference
println("Assembling dense Z...")
Z0 = assemble_Z_efie(mesh0, rwg0, k; mesh_precheck=false)
x_test = randn(ComplexF64, N0)
y_dense = Z0 * x_test

# Test different leaf_lambda values
for ll in [3.0, 2.0, 1.5, 1.0, 0.75]
    println("\n--- leaf_lambda = $ll ---")
    A = build_mlfma_operator(mesh0, rwg0, k; leaf_lambda=ll, verbose=true)
    nL = A.octree.nLevels
    n_leaf = length(A.octree.levels[end].boxes)
    nnz_pct = round(nnz(A.Z_near) / N0^2 * 100, digits=1)

    y_mlfma = similar(y_dense)
    mul!(y_mlfma, A, x_test)
    err = norm(y_mlfma - y_dense) / norm(y_dense)
    println("  $nL levels, $n_leaf leaf boxes, NF=$(nnz_pct)%, matvec err=$(round(err*100, digits=4))%")
end

println("\nDone.")
