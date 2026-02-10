# ex_obj_rcs_pipeline.jl — Unified mesh repair + RCS solve + plotting workflow
#
# Commands:
#   full   [input_obj_or_mat] [freq_GHz] [scale_to_m] [target_rwg] [tag]
#   repair [input_obj_or_mat] [output_obj] [scale_to_m]
#   plot   [data_dir] [out_dir] [tag]
#
# Examples:
#   julia --project=. examples/ex_obj_rcs_pipeline.jl
#   julia --project=. examples/ex_obj_rcs_pipeline.jl full ../su27.obj 3.0 0.001 300 su27
#   julia --project=. examples/ex_obj_rcs_pipeline.jl full ../airplane.mat 3.0 1.0 300 airplane_mat
#   julia --project=. examples/ex_obj_rcs_pipeline.jl repair ../su27.obj ../su27_repaired.obj 0.001
#   julia --project=. examples/ex_obj_rcs_pipeline.jl plot ../data ../figs su27

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using LinearAlgebra
using StaticArrays
using CSV
using DataFrames
using Plots

include(joinpath(@__DIR__, "..", "src", "DifferentiableMoM.jl"))
using .DifferentiableMoM

const DATADIR = joinpath(@__DIR__, "..", "data")
const FIGDIR = joinpath(@__DIR__, "..", "figs")

function _sanitize_tag(path::AbstractString)
    stem = splitext(basename(path))[1]
    tag = lowercase(stem)
    tag = replace(tag, r"[^a-z0-9]+" => "_")
    isempty(tag) && return "obj"
    return strip(tag, '_')
end

function _default_demo_obj()
    return joinpath(@__DIR__, "demo_aircraft.obj")
end

function _find_python_for_mat_converter()
    local_python = joinpath(@__DIR__, "..", ".venv", "bin", "python")
    isfile(local_python) && return local_python
    py3 = Sys.which("python3")
    py3 !== nothing && return py3
    py = Sys.which("python")
    py !== nothing && return py
    return nothing
end

function _resolve_mesh_input(input_path::AbstractString; tag::AbstractString=_sanitize_tag(input_path))
    ext = lowercase(splitext(input_path)[2])
    if ext != ".mat"
        return input_path
    end

    converter = joinpath(@__DIR__, "convert_aircraft_mat_to_obj.py")
    isfile(converter) || error("MAT converter script not found: $converter")
    py = _find_python_for_mat_converter()
    py === nothing && error("No Python interpreter found for MAT conversion")

    mkpath(DATADIR)
    converted_obj = joinpath(DATADIR, "$(tag)_from_mat.obj")
    cmd = `$(py) $(converter) $(input_path) $(converted_obj)`
    println("Detected MAT mesh input; converting to OBJ...")
    println("  Command: $cmd")
    run(cmd)
    return converted_obj
end

function _plot_rcs_from_csv(cut_csv::AbstractString, mono_csv::AbstractString, out_prefix::AbstractString)
    isfile(cut_csv) || error("Missing file: $cut_csv")
    isfile(mono_csv) || error("Missing file: $mono_csv")

    cut = CSV.read(cut_csv, DataFrame)
    mono = CSV.read(mono_csv, DataFrame)
    sort!(cut, :theta_deg)

    default(
        linewidth = 2,
        framestyle = :box,
        grid = true,
        legendfontsize = 9,
        guidefontsize = 11,
        tickfontsize = 9,
    )

    p1 = plot(
        cut.theta_deg,
        cut.sigma_dBsm,
        color = :royalblue,
        xlabel = "θ (deg)",
        ylabel = "RCS (dBsm)",
        label = "Bistatic cut (φ ≈ $(round(cut.phi_cut_deg[1], digits=1))°)",
        title = "OBJ PEC RCS — dB scale",
    )
    scatter!(
        p1,
        [mono.theta_obs_deg[1]],
        [mono.sigma_dBsm[1]],
        marker = (:star5, 8, :crimson),
        label = "Monostatic sample",
    )
    annotate!(
        p1,
        mono.theta_obs_deg[1] + 5,
        mono.sigma_dBsm[1] + 1.2,
        text("σ = $(round(mono.sigma_dBsm[1], digits=2)) dBsm", 8, :crimson),
    )

    p2 = plot(
        cut.theta_deg,
        cut.sigma_m2,
        color = :darkorange,
        xlabel = "θ (deg)",
        ylabel = "RCS (m²)",
        label = "Bistatic cut (linear)",
        title = "OBJ PEC RCS — linear scale",
    )
    scatter!(
        p2,
        [mono.theta_obs_deg[1]],
        [mono.sigma_m2[1]],
        marker = (:star5, 8, :crimson),
        label = "Monostatic sample",
    )

    p = plot(p1, p2, layout = (2, 1), size = (800, 700))
    png_path = out_prefix * ".png"
    pdf_path = out_prefix * ".pdf"
    savefig(p, png_path)
    savefig(p, pdf_path)
    return (png_path=png_path, pdf_path=pdf_path)
end

function run_repair(input_path::AbstractString, output_path::AbstractString; scale_to_m::Float64=1.0)
    println("="^68)
    println("OBJ Repair Utility")
    println("="^68)
    println("Input OBJ   : $input_path")
    println("Output OBJ  : $output_path")
    println("Scale to m  : $scale_to_m")

    input_mesh = _resolve_mesh_input(input_path)
    mesh_in = read_obj_mesh(input_mesh)
    mesh_scaled = TriMesh(mesh_in.xyz .* scale_to_m, copy(mesh_in.tri))

    result = repair_mesh_for_simulation(
        mesh_scaled;
        allow_boundary=true,
        require_closed=false,
        drop_invalid=true,
        drop_degenerate=true,
        fix_orientation=true,
        strict_nonmanifold=true,
        auto_drop_nonmanifold=true,
    )
    write_obj_mesh(output_path, result.mesh; header="Repaired from $input_path by DifferentiableMoM")

    println("\n── Repair summary ──")
    println("  Before: boundary=$(result.before.n_boundary_edges), nonmanifold=$(result.before.n_nonmanifold_edges), orient_conflicts=$(result.before.n_orientation_conflicts), degenerate=$(result.before.n_degenerate_triangles), invalid=$(result.before.n_invalid_triangles)")
    println("  Removed non-manifold triangles: $(result.removed_nonmanifold)")
    println("  Removed invalid triangles: $(length(result.removed_invalid))")
    println("  Removed degenerate triangles: $(length(result.removed_degenerate))")
    println("  Flipped triangle orientations: $(length(result.flipped_triangles))")
    println("  After : boundary=$(result.after.n_boundary_edges), nonmanifold=$(result.after.n_nonmanifold_edges), orient_conflicts=$(result.after.n_orientation_conflicts), degenerate=$(result.after.n_degenerate_triangles), invalid=$(result.after.n_invalid_triangles)")
    println("  Repaired OBJ written: $output_path")
    println("\nDone.")
    return (; result..., output_path=output_path)
end

function run_full(input_path::AbstractString;
                  freq_ghz::Float64=3.0,
                  scale_to_m::Float64=1.0,
                  target_rwg::Int=300,
                  tag::AbstractString=_sanitize_tag(input_path))
    mkpath(DATADIR)
    mkpath(FIGDIR)

    println("="^68)
    println("OBJ PEC RCS Demo")
    println("="^68)
    println("Input OBJ   : $input_path")
    println("Frequency   : $(freq_ghz) GHz")
    println("Scale to m  : $scale_to_m")
    println("Target RWG  : $target_rwg")
    println("Tag         : $tag")

    input_mesh = _resolve_mesh_input(input_path; tag=tag)
    mesh_in = read_obj_mesh(input_mesh)
    mesh_scaled = TriMesh(mesh_in.xyz .* scale_to_m, copy(mesh_in.tri))

    repair0 = repair_mesh_for_simulation(
        mesh_scaled;
        allow_boundary=true,
        require_closed=false,
        drop_invalid=true,
        drop_degenerate=true,
        fix_orientation=true,
        strict_nonmanifold=true,
        auto_drop_nonmanifold=true,
    )
    mesh0 = repair0.mesh
    rwg0 = build_rwg(mesh0; precheck=true, allow_boundary=true)

    println("\n── Imported mesh ──")
    println("  Vertices: $(nvertices(mesh_in)) -> $(nvertices(mesh0)) (scaled/repaired)")
    println("  Triangles: $(ntriangles(mesh_in)) -> $(ntriangles(mesh0))")
    println("  RWG (before coarsen): $(rwg0.nedges)")
    println("  Dense matrix size estimate: $(estimate_dense_matrix_gib(rwg0.nedges)) GiB")
    println("  Repaired winding flips: $(length(repair0.flipped_triangles))")
    println("  Non-manifold triangles removed: $(repair0.removed_nonmanifold)")

    mesh_use = mesh0
    rwg_use = rwg0
    if rwg0.nedges > target_rwg
        println("\n── Coarsening mesh for dense solve feasibility ──")
        coarse_result = coarsen_mesh_to_target_rwg(
            mesh0,
            target_rwg;
            max_iters=10,
            allow_boundary=true,
            require_closed=false,
            area_tol_rel=1e-12,
            strict_nonmanifold=true,
        )
        mesh_use = coarse_result.mesh
        rwg_use = build_rwg(mesh_use; precheck=true, allow_boundary=true)
        println("  Coarsened vertices : $(nvertices(mesh_use))")
        println("  Coarsened triangles: $(ntriangles(mesh_use))")
        println("  RWG after coarsen  : $(rwg_use.nedges)")
        println("  Dense matrix estimate: $(estimate_dense_matrix_gib(rwg_use.nedges)) GiB")
        println("  Coarsening iterations: $(coarse_result.iterations), target gap: $(coarse_result.best_gap)")
    else
        println("\nNo coarsening needed.")
    end

    repaired_obj = joinpath(DATADIR, "$(tag)_repaired.obj")
    coarse_obj = joinpath(DATADIR, "$(tag)_coarse.obj")
    write_obj_mesh(repaired_obj, mesh0; header="Scaled+repaired from $input_path")
    write_obj_mesh(coarse_obj, mesh_use; header="Coarsened for RCS demo from $input_path")

    preview_png = ""
    preview_pdf = ""
    try
        seg_rep = mesh_wireframe_segments(mesh0)
        seg_coa = mesh_wireframe_segments(mesh_use)
        preview = save_mesh_preview(
            mesh0,
            mesh_use,
            joinpath(FIGDIR, "$(tag)_mesh_preview");
            title_a="Repaired mesh\nV=$(nvertices(mesh0)), T=$(ntriangles(mesh0)), E=$(seg_rep.n_edges)",
            title_b="Simulation mesh\nV=$(nvertices(mesh_use)), T=$(ntriangles(mesh_use)), E=$(seg_coa.n_edges)",
            color_a=:steelblue,
            color_b=:darkorange,
            camera=(30, 30),
            size=(1200, 520),
            linewidth=0.7,
            guidefontsize=10,
            tickfontsize=8,
            titlefontsize=10,
        )
        preview_png = preview.png_path
        preview_pdf = preview.pdf_path
    catch err
        @warn "Mesh preview generation failed; continuing without preview files." exception=(err, catch_backtrace())
    end

    freq = freq_ghz * 1e9
    c0 = 299792458.0
    lambda0 = c0 / freq
    k = 2π / lambda0
    eta0 = 376.730313668

    println("\n── Solving PEC scattering ──")
    println("  λ0 = $(round(lambda0, digits=5)) m")
    println("  Unknowns = $(rwg_use.nedges)")

    t_asm = @elapsed Z_efie = assemble_Z_efie(mesh_use, rwg_use, k; quad_order=3, eta0=eta0)
    println("  Assembly time: $(round(t_asm, digits=3)) s")

    k_vec = Vec3(0.0, 0.0, -k)
    pol = Vec3(1.0, 0.0, 0.0)
    E0 = 1.0
    v = assemble_v_plane_wave(mesh_use, rwg_use, k_vec, E0, pol; quad_order=3)

    t_solve = @elapsed I = solve_forward(Z_efie, v)
    res = norm(Z_efie * I - v) / max(norm(v), 1e-30)
    println("  Solve time: $(round(t_solve, digits=3)) s")
    println("  Relative residual: $res")

    grid = make_sph_grid(121, 36)
    NΩ = length(grid.w)
    t_ff = @elapsed begin
        G_mat = radiation_vectors(mesh_use, rwg_use, grid, k; quad_order=3, eta0=eta0)
        E_ff = compute_farfield(G_mat, I, NΩ)
    end
    println("  Far-field time: $(round(t_ff, digits=3)) s")

    σ = bistatic_rcs(E_ff; E0=E0)
    khat_inc = k_vec / norm(k_vec)
    bs = backscatter_rcs(E_ff, grid, khat_inc; E0=E0)

    phi_target = grid.phi[argmin(abs.(grid.phi))]
    phi_idx = [q for q in 1:NΩ if abs(grid.phi[q] - phi_target) < 1e-12]
    perm = sortperm(grid.theta[phi_idx])
    phi_sorted = phi_idx[perm]

    df_cut = DataFrame(
        phi_cut_deg = fill(rad2deg(phi_target), length(phi_sorted)),
        theta_deg = rad2deg.(grid.theta[phi_sorted]),
        sigma_m2 = σ[phi_sorted],
        sigma_dBsm = 10 .* log10.(max.(σ[phi_sorted], 1e-30)),
    )
    df_bs = DataFrame(
        sigma_m2 = [bs.sigma],
        sigma_dBsm = [10 * log10(max(bs.sigma, 1e-30))],
        theta_obs_deg = [rad2deg(bs.theta)],
        phi_obs_deg = [rad2deg(bs.phi)],
        sample_angular_error_deg = [bs.angular_error_deg],
    )
    df_summary = DataFrame(
        input_obj = [input_path],
        tag = [tag],
        freq_ghz = [freq_ghz],
        scale_to_m = [scale_to_m],
        rwg_unknowns = [rwg_use.nedges],
        vertices = [nvertices(mesh_use)],
        triangles = [ntriangles(mesh_use)],
        assembly_time_s = [t_asm],
        solve_time_s = [t_solve],
        farfield_time_s = [t_ff],
        residual = [res],
        monostatic_sigma_m2 = [bs.sigma],
        monostatic_sigma_dBsm = [10 * log10(max(bs.sigma, 1e-30))],
    )

    csv_cut = joinpath(DATADIR, "$(tag)_bistatic_rcs_phi0.csv")
    csv_bs = joinpath(DATADIR, "$(tag)_monostatic_rcs.csv")
    csv_summary = joinpath(DATADIR, "$(tag)_rcs_summary.csv")
    CSV.write(csv_cut, df_cut)
    CSV.write(csv_bs, df_bs)
    CSV.write(csv_summary, df_summary)

    plot_prefix = joinpath(FIGDIR, "$(tag)_rcs_heuristic")
    plot_out = _plot_rcs_from_csv(csv_cut, csv_bs, plot_prefix)

    println("\n── Outputs ──")
    println("  Repaired mesh: $repaired_obj")
    println("  Coarsened mesh: $coarse_obj")
    if !isempty(preview_png)
        println("  Mesh preview PNG: $preview_png")
        println("  Mesh preview PDF: $preview_pdf")
    end
    println("  Bistatic φ≈0° cut: $csv_cut")
    println("  Monostatic backscatter: $csv_bs")
    println("  Run summary: $csv_summary")
    println("  Heuristic RCS plot PNG: $(plot_out.png_path)")
    println("  Heuristic RCS plot PDF: $(plot_out.pdf_path)")
    println("  Monostatic σ = $(df_bs.sigma_m2[1]) m² ($(round(df_bs.sigma_dBsm[1], digits=3)) dBsm)")
    println("\nDone.")

    return (
        repaired_obj=repaired_obj,
        coarse_obj=coarse_obj,
        cut_csv=csv_cut,
        mono_csv=csv_bs,
        summary_csv=csv_summary,
        plot_png=plot_out.png_path,
        plot_pdf=plot_out.pdf_path,
    )
end

function run_plot(data_dir::AbstractString, out_dir::AbstractString; tag::AbstractString="demo_aircraft")
    mkpath(out_dir)
    cut_csv = joinpath(data_dir, "$(tag)_bistatic_rcs_phi0.csv")
    mono_csv = joinpath(data_dir, "$(tag)_monostatic_rcs.csv")
    out_prefix = joinpath(out_dir, "$(tag)_rcs_heuristic")
    out = _plot_rcs_from_csv(cut_csv, mono_csv, out_prefix)
    println("Saved:")
    println("  $(out.png_path)")
    println("  $(out.pdf_path)")
    return out
end

function main(args::Vector{String}=ARGS)
    cmd = "full"
    pos = args
    if !isempty(args) && lowercase(args[1]) in ("full", "repair", "plot")
        cmd = lowercase(args[1])
        pos = args[2:end]
    end

    if cmd == "repair"
        input_path = length(pos) >= 1 ? pos[1] : _default_demo_obj()
        output_path = length(pos) >= 2 ? pos[2] : replace(input_path, r"(?i)\.obj$" => "_repaired.obj")
        scale_to_m = length(pos) >= 3 ? parse(Float64, pos[3]) : 1.0
        run_repair(input_path, output_path; scale_to_m=scale_to_m)
        return
    end

    if cmd == "plot"
        data_dir = length(pos) >= 1 ? pos[1] : DATADIR
        out_dir = length(pos) >= 2 ? pos[2] : FIGDIR
        tag = length(pos) >= 3 ? pos[3] : "demo_aircraft"
        run_plot(data_dir, out_dir; tag=tag)
        return
    end

    input_path = length(pos) >= 1 ? pos[1] : _default_demo_obj()
    freq_ghz = length(pos) >= 2 ? parse(Float64, pos[2]) : 3.0
    scale_to_m = length(pos) >= 3 ? parse(Float64, pos[3]) : 1.0
    target_rwg = length(pos) >= 4 ? parse(Int, pos[4]) : 300
    tag = length(pos) >= 5 ? pos[5] : _sanitize_tag(input_path)
    run_full(input_path; freq_ghz=freq_ghz, scale_to_m=scale_to_m, target_rwg=target_rwg, tag=tag)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
