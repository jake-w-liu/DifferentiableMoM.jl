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

# Build dense reference
Z0 = assemble_Z_efie(mesh0, rwg0, k; mesh_precheck=false)
x_test = randn(ComplexF64, N0)
y_dense = Z0 * x_test

# 6-level MLFMA
A = build_mlfma_operator(mesh0, rwg0, k; leaf_lambda=0.75, verbose=true)

# Full matvec
y_full = similar(y_dense)
mul!(y_full, A, x_test)
full_err = norm(y_full - y_dense) / norm(y_dense)
println("\nFull matvec error: $(round(full_err*100, digits=4))%")

# Near-field only matvec
y_nf = A.Z_near * x_test
nf_err = norm(y_nf - y_dense) / norm(y_dense)
println("Near-field only error: $(round(nf_err*100, digits=4))%")

# Far-field contribution
y_ff = y_full - y_nf
y_ff_ref = y_dense - y_nf  # what the far-field SHOULD contribute
ff_err = norm(y_ff - y_ff_ref) / norm(y_ff_ref)
println("Far-field relative error: $(round(ff_err*100, digits=4))%")
println("||y_ff|| / ||y_dense|| = $(round(norm(y_ff)/norm(y_dense)*100, digits=1))%")
println("||y_ff_ref|| / ||y_dense|| = $(round(norm(y_ff_ref)/norm(y_dense)*100, digits=1))%")

# Check spherical Hankel at kr=9.42 (min distance at leaf level)
println("\n--- Spherical Hankel h_l^(2)(kr=9.42) magnitudes ---")
kr = 9.42
h = DifferentiableMoM.spherical_hankel2_all(20, kr)
for l in 0:20
    println("  l=$l: |h_l| = $(round(abs(h[l+1]), sigdigits=4)), (2l+1)|h_l| = $(round((2l+1)*abs(h[l+1]), sigdigits=4))")
end

# Check per-element error distribution
println("\n--- Error distribution per leaf box ---")
nL = A.octree.nLevels
leaf_level = A.octree.levels[nL]
err_vec = abs.(y_full - y_dense)
for (bi, box) in enumerate(leaf_level.boxes)
    box_err = 0.0
    box_norm = 0.0
    for n_perm in box.bf_range
        n = A.octree.perm[n_perm]
        box_err += abs2(y_full[n] - y_dense[n])
        box_norm += abs2(y_dense[n])
    end
    if box_norm > 0
        rel_err = sqrt(box_err / box_norm)
        if rel_err > 0.1  # only print boxes with > 10% error
            println("  Box $bi ($(length(box.bf_range)) BFs): rel_err=$(round(rel_err*100, digits=2))%, center=$(round.(box.center, digits=2))")
        end
    end
end

# Count interaction list sizes at each level
println("\n--- Interaction list statistics ---")
for l in 2:nL
    level = A.octree.levels[l]
    il_sizes = [length(box.interaction_list) for box in level.boxes]
    println("  Level $l: $(length(level.boxes)) boxes, IL sizes: min=$(minimum(il_sizes)) max=$(maximum(il_sizes)) mean=$(round(sum(il_sizes)/length(il_sizes), digits=1))")
end
