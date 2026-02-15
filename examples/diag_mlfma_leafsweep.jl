push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "DifferentiableMoM.jl"))
using .DifferentiableMoM
using LinearAlgebra, SparseArrays

c0 = 299792458.0; freq = 0.3e9; λ0 = c0/freq; k = 2π/λ0

obj_path = joinpath(@__DIR__, "demo_aircraft.obj")
mesh_raw = read_obj_mesh(obj_path)
rep = repair_mesh_for_simulation(mesh_raw; allow_boundary=true, auto_drop_nonmanifold=true)
mesh_air = rep.mesh
ref0 = refine_mesh_to_target_edge(mesh_air, 4.0*λ0; max_iters=2, max_triangles=50_000)
mesh0 = ref0.mesh
rwg0 = build_rwg(mesh0)
N0 = rwg0.nedges
println("N = $N0")

Z0 = assemble_Z_efie(mesh0, rwg0, k; mesh_precheck=false)
x_test = randn(ComplexF64, N0)
y_dense = Z0 * x_test

# Sweep finer leaf_lambda values to find best trade-off
for ll in [0.95, 0.9, 0.85, 0.8]
    println("\n--- leaf_lambda=$ll ---")
    A = build_mlfma_operator(mesh0, rwg0, k; leaf_lambda=ll, verbose=true)
    nL = A.octree.nLevels
    nnz_pct = round(nnz(A.Z_near) / N0^2 * 100, digits=1)
    nnz_gb = round(nnz(A.Z_near) * 16 / 1e9, digits=2)
    println("  $nL levels, NF=$nnz_pct% ($nnz_gb GB for N=$N0)")

    # Estimate memory at N=30336
    # NF scales roughly as (N_big/N_small)^2 * same_nnz_pct
    N_big = 30336
    est_nnz_gb = nnz_pct / 100 * N_big^2 * 16 / 1e9
    println("  Est. at N=$(N_big): NF ~ $(round(est_nnz_gb, digits=1)) GB")

    y_mlfma = similar(y_dense)
    mul!(y_mlfma, A, x_test)
    err = norm(y_mlfma - y_dense) / norm(y_dense)
    println("  matvec err = $(round(err*100, digits=4))%")
end
