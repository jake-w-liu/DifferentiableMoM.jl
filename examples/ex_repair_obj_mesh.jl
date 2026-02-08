# ex_repair_obj_mesh.jl — Repair OBJ mesh for solver prechecks
#
# Usage:
#   julia --project=. examples/ex_repair_obj_mesh.jl input.obj [output.obj]
#
# Examples:
#   julia --project=. examples/ex_repair_obj_mesh.jl ../Airplane.obj
#   julia --project=. examples/ex_repair_obj_mesh.jl ../Airplane.obj ../Airplane_repaired.obj

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

include(joinpath(@__DIR__, "..", "src", "DifferentiableMoM.jl"))
using .DifferentiableMoM

if isempty(ARGS)
    error("Please provide an input OBJ path. Usage: julia --project=. examples/ex_repair_obj_mesh.jl input.obj [output.obj]")
end

input_path = ARGS[1]
output_path = length(ARGS) >= 2 ? ARGS[2] : replace(input_path, r"(?i)\.obj$" => "_repaired.obj")

println("="^60)
println("OBJ Repair Utility")
println("="^60)
println("Input : $input_path")
println("Output: $output_path")

result = repair_obj_mesh(
    input_path,
    output_path;
    allow_boundary=true,
    require_closed=false,
    drop_invalid=true,
    drop_degenerate=true,
    fix_orientation=true,
    strict_nonmanifold=true,
)

before = result.before
after = result.after

println("\n── Repair summary ──")
println("  Before: boundary=$(before.n_boundary_edges), nonmanifold=$(before.n_nonmanifold_edges), orient_conflicts=$(before.n_orientation_conflicts), degenerate=$(before.n_degenerate_triangles), invalid=$(before.n_invalid_triangles)")
println("  Removed invalid triangles: $(length(result.removed_invalid))")
println("  Removed degenerate triangles: $(length(result.removed_degenerate))")
println("  Flipped triangle orientations: $(length(result.flipped_triangles))")
println("  After : boundary=$(after.n_boundary_edges), nonmanifold=$(after.n_nonmanifold_edges), orient_conflicts=$(after.n_orientation_conflicts), degenerate=$(after.n_degenerate_triangles), invalid=$(after.n_invalid_triangles)")
println("  Repaired OBJ written: $(result.output_path)")

println("\nDone.")
