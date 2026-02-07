# ex_auto_preconditioning.jl — recommended auto-preconditioning settings
#
# Shows how :auto decides whether to apply the mass-based left preconditioner,
# and gives a small optimization example with the recommended settings for
# large / iterative runs.
#
# Run:
#   julia --project=. examples/ex_auto_preconditioning.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using LinearAlgebra
using StaticArrays

include(joinpath(@__DIR__, "..", "src", "DifferentiableMoM.jl"))
using .DifferentiableMoM

println("="^60)
println("Auto Preconditioning Example")
println("="^60)

# ------------------------------------------------------------------
# Build a lightweight problem
# ------------------------------------------------------------------
freq = 3e9
c0 = 299792458.0
lambda0 = c0 / freq
k = 2π / lambda0
eta0 = 376.730313668

mesh = make_rect_plate(0.1, 0.1, 3, 3)
rwg = build_rwg(mesh)
N = rwg.nedges
Nt = ntriangles(mesh)

partition = PatchPartition(collect(1:Nt), Nt)
Mp = precompute_patch_mass(mesh, rwg, partition; quad_order=3)

println("\nProblem size: N=$N RWG unknowns, Nt=$Nt impedance patches")

# ------------------------------------------------------------------
# Show :auto behavior
# ------------------------------------------------------------------
println("\n── Auto mode decisions ──")
settings = [
    (name="small/direct solve (default-like)", iterative=false, threshold=256),
    (name="force-on by low threshold", iterative=false, threshold=1),
    (name="iterative large-run recommendation", iterative=true, threshold=256),
]

for setting in settings
    _, enabled, reason = select_preconditioner(
        Mp;
        mode=:auto,
        iterative_solver=setting.iterative,
        n_threshold=setting.threshold,
        eps_rel=1e-6,
    )
    println("  $(setting.name): enabled=$enabled ($reason)")
end

# ------------------------------------------------------------------
# Tiny optimization smoke check
# ------------------------------------------------------------------
k_vec = Vec3(0.0, 0.0, -k)
pol_inc = Vec3(1.0, 0.0, 0.0)
v = assemble_v_plane_wave(mesh, rwg, k_vec, 1.0, pol_inc; quad_order=3)
Z_efie = assemble_Z_efie(mesh, rwg, k; quad_order=3, eta0=eta0)

grid = make_sph_grid(10, 20)
G_mat = radiation_vectors(mesh, rwg, grid, k; quad_order=3, eta0=eta0)
pol_mat = pol_linear_x(grid)
mask = cap_mask(grid; theta_max=30 * π / 180)
Q = build_Q(G_mat, grid, pol_mat; mask=mask)

theta0 = fill(200.0, Nt)

println("\n── Optimization with recommended :auto settings ──")
println("Recommended for large / iterative runs:")
println("  preconditioning=:auto, iterative_solver=true,")
println("  auto_precondition_n_threshold=256, auto_precondition_eps_rel=1e-6")

theta_opt, trace = optimize_lbfgs(
    Z_efie, Mp, v, Q, theta0;
    maxiter=3,
    tol=1e-12,
    alpha0=1e-2,
    verbose=true,
    reactive=true,
    preconditioning=:auto,
    iterative_solver=true,
    auto_precondition_n_threshold=256,
    auto_precondition_eps_rel=1e-6,
)

println("\nFinal objective after $(length(trace)) iterations: J = $(trace[end].J)")
println("Example complete.")

