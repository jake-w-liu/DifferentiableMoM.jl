# 04_beam_steering.jl — Beam steering via reactive impedance optimization
#
# Demonstrates directivity ratio optimization using the quotient-rule
# adjoint gradient:
#   1. Set up a metasurface with reactive (lossless) impedance patches
#   2. Define target and total Q matrices for directivity ratio
#   3. Run optimize_directivity with box constraints
#   4. Compare initial vs optimized far-field patterns
#
# Physics: Z_s = iθ (purely imaginary → lossless), so energy is
# conserved while the surface current distribution is reshaped.
#
# Run: julia --project=. examples/04_beam_steering.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "DifferentiableMoM.jl"))
using .DifferentiableMoM
using LinearAlgebra

println("="^60)
println("Example 04: Beam Steering (Reactive Impedance)")
println("="^60)

# ── 1. Problem setup ────────────────────────────────
freq  = 3e9
c0    = 299792458.0
λ0    = c0 / freq
k     = 2π / λ0

Lx, Ly = 0.2, 0.2                      # 2λ × 2λ plate
Nx, Ny = 10, 10
mesh = make_rect_plate(Lx, Ly, Nx, Ny)
rwg  = build_rwg(mesh)
N    = rwg.nedges

println("\nFrequency: $(freq/1e9) GHz")
println("Plate: $(round(Lx/λ0, digits=1))λ × $(round(Ly/λ0, digits=1))λ, $N RWG unknowns")

# ── 2. Patch partition (grid of patches) ────────────
Nt = ntriangles(mesh)
Px, Py = 5, 5                          # 5×5 patch grid
P = Px * Py
tri_patch = zeros(Int, Nt)
for t in 1:Nt
    cx = sum(mesh.xyz[1, mesh.tri[:, t]]) / 3
    cy = sum(mesh.xyz[2, mesh.tri[:, t]]) / 3
    px = clamp(floor(Int, (cx + Lx/2) / (Lx/Px)) + 1, 1, Px)
    py = clamp(floor(Int, (cy + Ly/2) / (Ly/Py)) + 1, 1, Py)
    tri_patch[t] = (py - 1) * Px + px
end
partition = PatchPartition(tri_patch, P)

println("Patches: $Px × $Py = $P")

# ── 3. Precompute ───────────────────────────────────
Mp     = precompute_patch_mass(mesh, rwg, partition)
Z_efie = assemble_Z_efie(mesh, rwg, k)
k_vec  = Vec3(0.0, 0.0, -k)
v      = assemble_excitation(mesh, rwg, make_plane_wave(k_vec, 1.0, Vec3(1.0, 0.0, 0.0)))

# ── 4. Build Q matrices ────────────────────────────
grid  = make_sph_grid(18, 36)
G_mat = radiation_vectors(mesh, rwg, grid, k)
NΩ    = length(grid.w)
pol   = pol_linear_x(grid)

# Q_target: 10° cone around broadside (z-axis)
mask_target = cap_mask(grid; theta_max=π/18)
Q_target = build_Q(G_mat, grid, pol; mask=mask_target)

# Q_total: all directions (for total radiated power normalization)
Q_total = build_Q(G_mat, grid, pol)

n_target = count(mask_target)
println("Target cone: 10° half-angle ($n_target of $NΩ directions)")

# ── 5. Initial directivity ─────────────────────────
theta0 = zeros(P)                       # Start with PEC (θ=0)

Z0 = assemble_full_Z(Z_efie, Mp, theta0; reactive=true)
I0 = Z0 \ v
f0 = real(dot(I0, Q_target * I0))
g0 = real(dot(I0, Q_total * I0))
D0 = f0 / g0

println("\n── Initial (PEC, θ=0) ──")
println("  Directivity ratio: $(round(D0, sigdigits=4))")

# ── 6. Optimize directivity ────────────────────────
println("\n── Optimizing directivity (reactive impedance, L-BFGS) ──")
theta_opt, trace = optimize_directivity(
    Z_efie, Mp, Vector{ComplexF64}(v),
    Q_target, Q_total, theta0;
    maxiter=40,
    tol=1e-8,
    reactive=true,
    lb=fill(-500.0, P),                 # Capacitive limit
    ub=fill(500.0, P),                  # Inductive limit
    verbose=true,
)

D_opt = trace[end].J

# ── 7. Evaluate optimized pattern ───────────────────
Z_opt = assemble_full_Z(Z_efie, Mp, theta_opt; reactive=true)
I_opt = Z_opt \ v
E_ff_opt = compute_farfield(G_mat, Vector{ComplexF64}(I_opt), NΩ)

# Energy conservation (reactive = lossless, should be ≈ 1)
P_in  = input_power(Vector{ComplexF64}(I_opt), Vector{ComplexF64}(v))
P_rad = radiated_power(E_ff_opt, grid)
println("\n── Results ──")
println("  Initial directivity ratio:   $(round(D0, sigdigits=4))")
println("  Optimized directivity ratio:  $(round(D_opt, sigdigits=4))")
println("  Improvement: $(round(D_opt/D0, digits=2))×")
println("  P_rad/P_in = $(round(P_rad/P_in, digits=4))  (should ≈ 1 for lossless)")
println("  Iterations: $(length(trace))")

# Show impedance distribution
println("\n── Optimized impedance map (Ω/sq, reactive: Z_s = iθ) ──")
for py in 1:Py
    row = [round(theta_opt[(py-1)*Px + px], digits=1) for px in 1:Px]
    println("  row $py: $row")
end

println("\n" * "="^60)
println("Done.")
