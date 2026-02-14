# debug_mlfma_precision.jl — Isolate MLFMA error sources
#
# Tests: (1) precision parameter sensitivity
#        (2) translation operator accuracy alone
#        (3) single-level (no interpolation) matvec accuracy
#        (4) full matvec accuracy
#
# Run: julia -t 4 --project=. examples/debug_mlfma_precision.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "DifferentiableMoM.jl"))
using .DifferentiableMoM
using LinearAlgebra
using SparseArrays

c0 = 299792458.0

# ── Setup: Aircraft at 0.3 GHz, refine to 4λ ──
mesh_raw = read_obj_mesh(joinpath(@__DIR__, "demo_aircraft.obj"))
rep = repair_mesh_for_simulation(mesh_raw; allow_boundary=true, auto_drop_nonmanifold=true)
mesh = rep.mesh

freq = 0.3e9
λ0 = c0 / freq
k  = 2π / λ0

ref = refine_mesh_to_target_edge(mesh, 4.0 * λ0; max_iters=2, max_triangles=50_000)
mesh = ref.mesh

rwg = build_rwg(mesh)
N = rwg.nedges
println("Aircraft 4λ mesh: $(ntriangles(mesh)) tri, N=$N")

# ── Dense Z (ground truth) ──
println("\nAssembling dense Z...")
t = @elapsed Z = assemble_Z_efie(mesh, rwg, k; mesh_precheck=false)
println("  Done in $(round(t, digits=1))s")

# ── Test 1: Precision parameter sweep ──
println("\n" * "="^72)
println("Test 1: Precision parameter sweep (leaf_lambda=3.0)")
println("="^72)

x_test = randn(ComplexF64, N)
y_dense = Z * x_test

for prec in [3, 5, 7, 9]
    A = build_mlfma_operator(mesh, rwg, k; leaf_lambda=3.0, precision=prec, verbose=false)
    y_mlfma = similar(y_dense)
    mul!(y_mlfma, A, x_test)
    err = norm(y_mlfma - y_dense) / norm(y_dense)

    # Far-field error
    y_near = A.Z_near * x_test
    y_far_m = y_mlfma - y_near
    y_far_e = y_dense - y_near
    err_far = norm(y_far_m - y_far_e) / norm(y_far_e)

    nL = A.octree.nLevels
    L_leaf = A.samplings[end].L
    println("  prec=$prec: L_leaf=$L_leaf, npts=$(A.samplings[end].npts), " *
            "matvec err=$(round(err*100, digits=3))%, far-field err=$(round(err_far*100, digits=1))%")
end

# ── Test 2: Leaf lambda sweep (with precision=5) ──
println("\n" * "="^72)
println("Test 2: Leaf lambda sweep (precision=5)")
println("="^72)

for ll in [1.0, 2.0, 3.0, 5.0]
    A = build_mlfma_operator(mesh, rwg, k; leaf_lambda=ll, precision=5, verbose=false)
    y_mlfma = similar(y_dense)
    mul!(y_mlfma, A, x_test)
    err = norm(y_mlfma - y_dense) / norm(y_dense)

    y_near = A.Z_near * x_test
    y_far_m = y_mlfma - y_near
    y_far_e = y_dense - y_near
    err_far = norm(y_far_m - y_far_e) / norm(y_far_e)

    nL = A.octree.nLevels
    n_leaf = length(A.octree.levels[nL].boxes)
    nnz_pct = round(nnz(A.Z_near) / N^2 * 100, digits=1)
    L_leaf = A.samplings[end].L
    println("  leaf_λ=$(ll): $nL levels, $n_leaf leaf boxes, nnz=$(nnz_pct)%, L_leaf=$L_leaf, " *
            "matvec err=$(round(err*100, digits=3))%, far-field err=$(round(err_far*100, digits=1))%")
end

# ── Test 3: Single-level test (leaf only, no interpolation) ──
println("\n" * "="^72)
println("Test 3: 2-level MLFMA (leaf only, no interpolation)")
println("="^72)

# Force a 2-level octree (root + leaf) to eliminate interpolation error
# Use a small leaf_lambda to get more boxes
for ll in [1.0, 2.0, 3.0]
    A = build_mlfma_operator(mesh, rwg, k; leaf_lambda=ll, precision=5, verbose=false)
    nL = A.octree.nLevels
    n_leaf = length(A.octree.levels[nL].boxes)

    if nL == 2
        # Only leaf + root — no interpolation involved
        y_mlfma = similar(y_dense)
        mul!(y_mlfma, A, x_test)
        err = norm(y_mlfma - y_dense) / norm(y_dense)
        y_near = A.Z_near * x_test
        y_far_m = y_mlfma - y_near
        y_far_e = y_dense - y_near
        err_far = norm(y_far_m - y_far_e) / norm(y_far_e)
        nnz_pct = round(nnz(A.Z_near) / N^2 * 100, digits=1)
        L_leaf = A.samplings[end].L
        println("  leaf_λ=$(ll): nL=$nL, $n_leaf leaf, nnz=$(nnz_pct)%, L=$L_leaf, " *
                "matvec err=$(round(err*100, digits=4))%, far err=$(round(err_far*100, digits=2))%")
    else
        println("  leaf_λ=$(ll): nL=$nL (multi-level, skipping)")
    end
end

# ── Test 4: Translation operator accuracy (single pair) ──
println("\n" * "="^72)
println("Test 4: Translation operator accuracy for single box pair")
println("="^72)

# Build MLFMA and extract a single translation
A5 = build_mlfma_operator(mesh, rwg, k; leaf_lambda=3.0, precision=5, verbose=false)
leaf_level = A5.octree.levels[A5.octree.nLevels]
leaf_samp = A5.samplings[end]

# Find a box with interaction list
test_box = nothing
test_il = nothing
for (bi, box) in enumerate(leaf_level.boxes)
    if !isempty(box.interaction_list)
        global test_box = bi
        global test_il = box.interaction_list[1]
        break
    end
end

if test_box !== nothing
    box_a = leaf_level.boxes[test_box]
    box_b = leaf_level.boxes[test_il]
    println("  Box $test_box (ijk=$(box_a.ijk), $(length(box_a.bf_range)) BFs) → " *
            "Box $test_il (ijk=$(box_b.ijk), $(length(box_b.bf_range)) BFs)")

    # Compute Z entries for this box pair from dense Z
    bfs_a = [A5.octree.perm[n] for n in box_a.bf_range]
    bfs_b = [A5.octree.perm[n] for n in box_b.bf_range]
    Z_pair = Z[bfs_a, bfs_b]
    Z_near_pair = A5.Z_near[bfs_a, bfs_b]
    Z_far_pair = Z_pair - Z_near_pair

    println("  |Z_pair|=$(round(norm(Z_pair), sigdigits=3)), " *
            "|Z_near_pair|=$(round(norm(Z_near_pair), sigdigits=3)), " *
            "|Z_far_pair|=$(round(norm(Z_far_pair), sigdigits=3))")

    # Compute far-field via MLFMA (single pair, no interpolation)
    # Z_far[m,n] = prefactor * Σ_q w_q T_q * [Σ_c conj(S_m^c) * S_n^c (vec) - conj(S_m^4)*S_n^4 (scl)]
    dijk = (box_a.ijk[1] - box_b.ijk[1],
            box_a.ijk[2] - box_b.ijk[2],
            box_a.ijk[3] - box_b.ijk[3])
    T = A5.trans_factors[A5.octree.nLevels - 1][dijk]

    Z_far_mlfma = zeros(ComplexF64, length(bfs_a), length(bfs_b))
    for (mi, m) in enumerate(bfs_a)
        for (ni, n) in enumerate(bfs_b)
            val = zero(ComplexF64)
            for q in 1:leaf_samp.npts
                dot4 = zero(ComplexF64)
                for c in 1:3
                    dot4 += conj(A5.bf_patterns[c, q, m]) * T[q] * A5.bf_patterns[c, q, n]
                end
                dot4 -= conj(A5.bf_patterns[4, q, m]) * T[q] * A5.bf_patterns[4, q, n]
                val += leaf_samp.weights[q] * dot4
            end
            Z_far_mlfma[mi, ni] = A5.prefactor * val
        end
    end

    err_pair = norm(Z_far_mlfma - Z_far_pair) / norm(Z_far_pair)
    println("  Single-pair far-field error: $(round(err_pair*100, digits=4))%")

    # Check: what if we flip the prefactor sign?
    Z_far_mlfma_pos = -Z_far_mlfma  # try positive prefactor
    err_pair_pos = norm(Z_far_mlfma_pos - Z_far_pair) / norm(Z_far_pair)
    println("  With flipped sign prefactor: $(round(err_pair_pos*100, digits=4))%")

    # Check individual entry
    println("  Sample entries (first 3×3):")
    for mi in 1:min(3, length(bfs_a))
        for ni in 1:min(3, length(bfs_b))
            z_exact = Z_far_pair[mi, ni]
            z_mlfma = Z_far_mlfma[mi, ni]
            ratio = z_mlfma / z_exact
            println("    Z_far[$mi,$ni]: exact=$(round(z_exact, sigdigits=4)), " *
                    "mlfma=$(round(z_mlfma, sigdigits=4)), ratio=$(round(ratio, sigdigits=4))")
        end
    end
end

println("\nDone.")
