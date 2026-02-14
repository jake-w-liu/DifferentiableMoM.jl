# debug_mlfma_aircraft.jl — Diagnose MLFMA matvec error on 3D aircraft
#
# Compares MLFMA matvec vs dense matvec entry-by-entry to find error sources.

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

# ── Dense Z ──
println("\nAssembling dense Z...")
t = @elapsed Z = assemble_Z_efie(mesh, rwg, k; mesh_precheck=false)
println("  Done in $(round(t, digits=1))s")

# ── MLFMA operator ──
LEAF_LAMBDA = 3.0
println("\nBuilding MLFMA (leaf_lambda=$LEAF_LAMBDA)...")
t = @elapsed A = build_mlfma_operator(mesh, rwg, k; leaf_lambda=LEAF_LAMBDA, verbose=true)
println("  Done in $(round(t, digits=1))s")
println("  $(A.octree.nLevels) levels, $(length(A.octree.levels[end].boxes)) leaf boxes")
println("  NF nnz: $(round(nnz(A.Z_near)/N^2*100, digits=1))%")

# ── Test 1: Random matvec comparison ──
println("\n── Test 1: Random vector matvec ──")
x = randn(ComplexF64, N)
y_dense = Z * x
y_mlfma = similar(y_dense)
mul!(y_mlfma, A, x)

y_near = A.Z_near * x
y_far_mlfma = y_mlfma - y_near
y_far_exact = y_dense - y_near

err_total = norm(y_mlfma - y_dense) / norm(y_dense)
err_far = norm(y_far_mlfma - y_far_exact) / norm(y_far_exact)
println("  Total matvec rel error: $(round(err_total, sigdigits=3))")
println("  Near-field contribution: $(round(norm(y_near)/norm(y_dense)*100, digits=1))%")
println("  Far-field MLFMA rel error: $(round(err_far, sigdigits=3))")
println("  |y_far_mlfma| / |y_far_exact|: $(round(norm(y_far_mlfma)/norm(y_far_exact), digits=3))")

# ── Test 2: Per-BF error distribution ──
println("\n── Test 2: Per-BF error ──")
err_per_bf = abs.(y_mlfma - y_dense)
top10 = sortperm(err_per_bf; rev=true)[1:min(10, N)]
println("  Top 10 error BFs:")
for (i, n) in enumerate(top10)
    # Find which box this BF is in
    box_id = -1
    leaf_level = A.octree.levels[end]
    for (bi, box) in enumerate(leaf_level.boxes)
        if n in [A.octree.perm[j] for j in box.bf_range]
            box_id = bi
            break
        end
    end
    println("    BF $n: |err|=$(round(err_per_bf[n], sigdigits=3)), box=$box_id, |y_d|=$(round(abs(y_dense[n]), sigdigits=3))")
end

# ── Test 3: Per-box error ──
println("\n── Test 3: Per-box error ──")
leaf_level = A.octree.levels[end]
nboxes = length(leaf_level.boxes)
box_err = zeros(nboxes)
box_nbf = zeros(Int, nboxes)
for (bi, box) in enumerate(leaf_level.boxes)
    for n_perm in box.bf_range
        n = A.octree.perm[n_perm]
        box_err[bi] += abs2(y_mlfma[n] - y_dense[n])
        box_nbf[bi] += 1
    end
    box_err[bi] = sqrt(box_err[bi])
end

top_boxes = sortperm(box_err; rev=true)[1:min(10, nboxes)]
println("  Top 10 error boxes:")
for bi in top_boxes
    box = leaf_level.boxes[bi]
    println("    Box $bi: err=$(round(box_err[bi], sigdigits=3)), nBF=$(box_nbf[bi]), " *
            "ijk=$(box.ijk), n_neighbors=$(length(box.neighbors)), n_interact=$(length(box.interaction_list))")
end

# ── Test 4: Check far-field matrix accuracy row by row ──
println("\n── Test 4: Far-field matrix Z_far = Z - Z_near ──")
Z_far = Z - Matrix(A.Z_near)
nz_far = count(abs.(Z_far) .> 1e-14)
println("  Z_far nonzeros: $nz_far / $(N^2) = $(round(nz_far/N^2*100, digits=1))%")
println("  |Z_far|_F / |Z|_F = $(round(norm(Z_far)/norm(Z), digits=3))")

# Check a few specific far-field entries
println("\n  Sampling specific far-field entries (Z_far vs MLFMA far-field)...")
# Use unit vectors to extract MLFMA columns
n_sample = min(5, N)
sample_cols = sort(rand(1:N, n_sample))
max_col_err = 0.0
for n in sample_cols
    e_n = zeros(ComplexF64, N)
    e_n[n] = 1.0
    y_full = zeros(ComplexF64, N)
    mul!(y_full, A, e_n)
    y_nf = A.Z_near * e_n
    y_ff_mlfma = y_full - y_nf
    y_ff_exact = Z_far[:, n]
    col_err = norm(y_ff_mlfma - y_ff_exact) / max(norm(y_ff_exact), 1e-30)
    max_col_err = max(max_col_err, col_err)
    println("    Col $n: MLFMA far-field err = $(round(col_err, sigdigits=3))")
end

# ── Test 5: Check if Z_near matches near-field entries of Z ──
println("\n── Test 5: Near-field matrix accuracy ──")
Z_near_dense = Matrix(A.Z_near)
mask_near = abs.(Z_near_dense) .> 1e-14
Z_near_from_Z = Z .* mask_near
nf_err = norm(Z_near_from_Z - Z_near_dense) / norm(Z_near_dense)
println("  Z_near vs Z (on near-field mask): $(round(nf_err, sigdigits=3))")

# Check if there are entries in Z that are NOT in Z_near but SHOULD be (large Z entries outside near mask)
Z_outside_near = Z .* (.!mask_near)
largest_outside = maximum(abs.(Z_outside_near))
println("  Largest |Z| outside near-field mask: $(round(largest_outside, sigdigits=3))")
# Find which entries these are
println("  Fraction of |Z_far| > |Z_max|/100: $(round(count(abs.(Z_outside_near) .> maximum(abs.(Z))/100) / nz_far * 100, digits=1))%")

println("\nDone.")
