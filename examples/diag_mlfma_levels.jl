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

# Dense reference
Z0 = assemble_Z_efie(mesh0, rwg0, k; mesh_precheck=false)
x_test = randn(ComplexF64, N0)
y_dense = Z0 * x_test

for ll in [1.0, 0.75]
    println("\n=== leaf_lambda=$ll ===")
    A = build_mlfma_operator(mesh0, rwg0, k; leaf_lambda=ll, verbose=true)
    nL = A.octree.nLevels
    for l in 2:nL
        samp = A.samplings[l-1]
        edge = A.octree.levels[l].edge_length
        nboxes = length(A.octree.levels[l].boxes)
        tf = A.trans_factors[l-1]
        min_r = Inf
        max_T = 0.0
        for (dijk, T) in tf
            r = sqrt(Float64(dijk[1])^2 + Float64(dijk[2])^2 + Float64(dijk[3])^2) * edge
            min_r = min(min_r, r)
            max_T = max(max_T, maximum(abs.(T)))
        end
        kr_min = k * min_r
        diag = sqrt(3.0) * edge
        println("  Level $l: edge=$(round(edge/λ0, digits=3))λ, L=$(samp.L), nθ=$(samp.ntheta), nφ=$(samp.nphi), npts=$(samp.npts), boxes=$nboxes")
        println("           min_kr=$(round(kr_min, digits=2)), kd=$(round(k*diag, digits=2)), L/kr_min=$(round(samp.L/kr_min, digits=2)), max|T|=$(round(max_T, digits=1))")
    end

    # Matvec accuracy
    y_mlfma = similar(y_dense)
    mul!(y_mlfma, A, x_test)
    err = norm(y_mlfma - y_dense) / norm(y_dense)
    println("  Matvec error: $(round(err*100, digits=4))%")
end
