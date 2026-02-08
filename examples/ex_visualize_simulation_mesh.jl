# ex_visualize_simulation_mesh.jl — visualize repaired/coarsened simulation meshes
#
# Usage:
#   julia --project=. examples/ex_visualize_simulation_mesh.jl
#   julia --project=. examples/ex_visualize_simulation_mesh.jl repaired.obj coarse.obj [output_prefix]
#
# Defaults:
#   repaired.obj = data/airplane_repaired.obj
#   coarse.obj   = data/airplane_coarse.obj
#   output_prefix = figs/airplane_mesh_preview

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

include(joinpath(@__DIR__, "..", "src", "DifferentiableMoM.jl"))
using .DifferentiableMoM

repaired_path = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "data", "airplane_repaired.obj")
coarse_path   = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, "..", "data", "airplane_coarse.obj")
out_prefix    = length(ARGS) >= 3 ? ARGS[3] : joinpath(@__DIR__, "..", "figs", "airplane_mesh_preview")

isfile(repaired_path) || error("Missing repaired mesh OBJ: $repaired_path")
isfile(coarse_path) || error("Missing coarse mesh OBJ: $coarse_path")

mkpath(dirname(out_prefix))

mesh_rep = read_obj_mesh(repaired_path)
mesh_coa = read_obj_mesh(coarse_path)

seg_rep = mesh_wireframe_segments(mesh_rep)
seg_coa = mesh_wireframe_segments(mesh_coa)

preview = save_mesh_preview(
    mesh_rep,
    mesh_coa,
    out_prefix;
    title_a = "Repaired mesh\nV=$(nvertices(mesh_rep)), T=$(ntriangles(mesh_rep)), E=$(seg_rep.n_edges)",
    title_b = "Coarsened simulation mesh\nV=$(nvertices(mesh_coa)), T=$(ntriangles(mesh_coa)), E=$(seg_coa.n_edges)",
    color_a = :steelblue,
    color_b = :darkorange,
    camera = (30, 30),
    size = (1200, 520),
    linewidth = 0.7,
    guidefontsize = 10,
    tickfontsize = 8,
    titlefontsize = 10,
)

println("Saved mesh previews:")
println("  $(preview.png_path)")
println("  $(preview.pdf_path)")
