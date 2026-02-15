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

# Build 6-level MLFMA
A = build_mlfma_operator(mesh0, rwg0, k; leaf_lambda=0.75, verbose=true)
nL = A.octree.nLevels

# Check completeness: every pair of leaf boxes must be accounted for
# A pair (i,j) is accounted for if:
# 1. They are neighbors at the leaf level (near-field), OR
# 2. At some ancestor level, their ancestor boxes are in each other's interaction lists

leaf_level = A.octree.levels[nL]
n_leaf = length(leaf_level.boxes)

# For each leaf box, find the path to root
function path_to_root(octree, leaf_idx)
    path = [leaf_idx]
    l = octree.nLevels
    idx = leaf_idx
    while l > 1
        idx = octree.levels[l].boxes[idx].parent
        l -= 1
        push!(path, idx)
    end
    return path  # [leaf_idx, parent_idx, grandparent_idx, ..., root_idx]
end

println("Checking completeness of octree decomposition...")
println("$n_leaf leaf boxes, $(n_leaf * (n_leaf-1) ÷ 2) unique pairs")

# Build neighbor set at leaf level
leaf_neighbor_pairs = Set{Tuple{Int,Int}}()
for (bi, box) in enumerate(leaf_level.boxes)
    # Self is always a neighbor
    for nb in box.neighbors
        if nb > bi  # avoid double counting
            push!(leaf_neighbor_pairs, (bi, nb))
        end
    end
end

# Build interaction list at each level
il_coverage = Set{Tuple{Int,Int}}()  # pairs of LEAF boxes covered by IL at some level

for l in 2:nL
    level = A.octree.levels[l]
    for (bi, box) in enumerate(level.boxes)
        for il_id in box.interaction_list
            # Find all leaf descendants of box bi and il_id at level l
            # Then mark those leaf pairs as covered
        end
    end
end

# Simpler approach: trace each leaf pair and check coverage
missing_pairs = Tuple{Int,Int}[]
uncovered_count = 0
total_pairs = 0

for i in 1:n_leaf
    for j in (i+1):n_leaf
        global total_pairs += 1
        # Check if neighbors at leaf
        is_neighbor = j in leaf_level.boxes[i].neighbors
        if is_neighbor
            continue
        end

        # Check if covered by interaction list at some level
        # Trace up the tree: at each level, check if the ancestor boxes
        # are in each other's interaction lists
        path_i = path_to_root(A.octree, i)
        path_j = path_to_root(A.octree, j)

        covered = false
        for lvl_offset in 0:(nL-1)  # 0 = leaf level itself, 1 = parent, etc.
            if lvl_offset + 1 > length(path_i) || lvl_offset + 1 > length(path_j)
                break
            end
            anc_i = path_i[lvl_offset + 1]
            anc_j = path_j[lvl_offset + 1]
            l = nL - lvl_offset  # actual level
            box_i = A.octree.levels[l].boxes[anc_i]
            if anc_j in box_i.interaction_list
                covered = true
                break
            end
        end

        if !covered
            global uncovered_count += 1
            if uncovered_count <= 20
                push!(missing_pairs, (i, j))
            end
        end
    end
end

println("Total non-neighbor pairs: $(total_pairs - length(leaf_neighbor_pairs))")
println("Uncovered pairs: $uncovered_count")
if uncovered_count > 0
    println("\nFirst $(min(20, uncovered_count)) missing pairs:")
    for (i, j) in missing_pairs
        ci = leaf_level.boxes[i].center
        cj = leaf_level.boxes[j].center
        d = norm(ci - cj)
        path_i = path_to_root(A.octree, i)
        path_j = path_to_root(A.octree, j)
        println("  ($i,$j): dist=$(round(d/λ0, digits=2))λ, " *
                "path_i=$path_i, path_j=$path_j")
    end
end

# Also check: are leaf neighbors correct?
# Two leaf boxes are neighbors iff their ijk indices differ by at most 1 in each dimension
println("\n--- Neighbor check at leaf level ---")
wrong_neighbor = 0
for (bi, box_i) in enumerate(leaf_level.boxes)
    for nb in box_i.neighbors
        if nb == bi; continue; end
        box_j = leaf_level.boxes[nb]
        dijk = abs.(box_i.ijk .- box_j.ijk)
        if maximum(dijk) > 1
            global wrong_neighbor += 1
            if wrong_neighbor <= 5
                println("  Wrong neighbor: box $bi (ijk=$(box_i.ijk)) <-> box $nb (ijk=$(box_j.ijk)), dijk=$dijk")
            end
        end
    end
end
println("Wrong neighbors: $wrong_neighbor")

# Check: missing neighbors (should be neighbors but aren't in list)
missing_neighbor = 0
for bi in 1:n_leaf
    box_i = leaf_level.boxes[bi]
    for bj in (bi+1):n_leaf
        box_j = leaf_level.boxes[bj]
        dijk = abs.(box_i.ijk .- box_j.ijk)
        should_be_neighbor = maximum(dijk) <= 1
        is_neighbor = bj in box_i.neighbors
        if should_be_neighbor && !is_neighbor
            global missing_neighbor += 1
            if missing_neighbor <= 5
                println("  Missing neighbor: box $bi (ijk=$(box_i.ijk)) <-> box $bj (ijk=$(box_j.ijk))")
            end
        end
    end
end
println("Missing neighbors: $missing_neighbor")
