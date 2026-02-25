# test_periodic_topology.jl — Tests for periodic MoM + topology optimization modules
#
# Covers: PeriodicGreens (Helmholtz-Ewald), PeriodicEFIE, DensityInterpolation,
#         DensityFiltering, DensityAdjoint, PeriodicMetrics
#
# Test categories:
#   A: Analytical ground truth (independently derived expected values)
#   B: Mathematical properties (symmetry, PSD, reciprocity, adjoint)
#   C: Convergence & asymptotics
#   D: Edge cases & boundaries
#   E: Error handling
#   F: Cross-validation (FD vs analytical, alternative computations)

# ─────────────────────────────────────────────────────────────────
# Test 37: PeriodicGreens — Helmholtz-Ewald summation
# ─────────────────────────────────────────────────────────────────
println("\n── Test 37: PeriodicGreens (Helmholtz-Ewald) ──")

@testset "PeriodicGreens (Helmholtz-Ewald)" begin
    freq = 10e9; c0 = 3e8; lambda = c0 / freq; k = 2π / lambda
    dx = 0.5 * lambda; dy = 0.5 * lambda
    E_opt = sqrt(π / (dx * dy))

    # ── A: PeriodicLattice constructor ──
    @testset "A: PeriodicLattice constructor" begin
        lat = PeriodicLattice(dx, dy, 0.0, 0.0, k)
        # Normal incidence → kx = ky = 0
        @test lat.kx_bloch ≈ 0.0 atol=1e-15
        @test lat.ky_bloch ≈ 0.0 atol=1e-15
        # Optimal splitting: E = sqrt(π/A)
        @test lat.E ≈ E_opt rtol=1e-14
        @test lat.k ≈ k rtol=1e-14
        @test lat.N_spatial == 4
        @test lat.N_spectral == 4

        # Oblique incidence: θ=30°, φ=0° → kx = k sin(30°) = k/2
        lat_obl = PeriodicLattice(dx, dy, π/6, 0.0, k)
        @test lat_obl.kx_bloch ≈ k * 0.5 rtol=1e-14
        @test lat_obl.ky_bloch ≈ 0.0 atol=1e-15
    end

    # ── A: Ewald spatial kernel returns real value ──
    @testset "A: Spatial kernel real-valued" begin
        # K_sp(R) = Re[exp(-ikR) erfc(ER - ik/(2E))] / (4πR)
        # The two erfc terms are complex conjugates → sum is real
        for R in [0.001, 0.005, 0.01, 0.02]
            K_sp = DifferentiableMoM._ewald_spatial_kernel(R, k, E_opt)
            @test isa(K_sp, Float64)  # real type from real() call
        end
    end

    # ── A: Self-correction at R=0 for k=0 (known Laplace result) ──
    @testset "A: Self-correction Laplace limit (k=0)" begin
        # For k=0: erfc(ik/(2E)) = erfc(0) = 1, exp(0) = 1
        # C_self = [2i×0×1 - 4E/√π × 1] / (8π) = -4E/(8π√π) = -E/(2π^{3/2})
        # Source: standard Ewald self-term for 2D Laplace lattice sum
        C_self_k0 = DifferentiableMoM._ewald_self_correction(0.0, 0.0, E_opt)
        expected = -E_opt / (2 * π^(3/2))
        # Tolerance: exact formula, machine precision
        @test real(C_self_k0) ≈ expected rtol=1e-12
        @test abs(imag(C_self_k0)) < 1e-15  # purely real for k=0
    end

    # ── A: Self-correction continuity approaching R=0 ──
    @testset "A: Self-correction smooth at R→0" begin
        C0 = DifferentiableMoM._ewald_self_correction(0.0, k, E_opt)
        # At R=1e-6: K_sp ~ 1/(4πR) ~ 8e4, difference ~ O(E) ~ 100
        # Cancellation: ~3 digits lost → ~12 digits remaining
        CR = DifferentiableMoM._ewald_self_correction(1e-6, k, E_opt)
        @test abs(CR - C0) / abs(C0) < 1e-6
        # At R=1e-10: ~7 digits lost from cancellation → ~8 digits remaining
        CR2 = DifferentiableMoM._ewald_self_correction(1e-10, k, E_opt)
        @test abs(CR2 - C0) / abs(C0) < 1e-3
    end

    # ── A: kz branch cut correctness ──
    @testset "A: Spectral kz branch cut" begin
        # Propagating: kt = 0 → kz = k (real positive)
        kz_prop = DifferentiableMoM._spectral_kz(k, 0.0, 0.0)
        @test real(kz_prop) ≈ k rtol=1e-14
        @test abs(imag(kz_prop)) < 1e-14

        # Evanescent: kt = 2k → kz² = k²-4k² = -3k²
        # kz = sqrt(-3k²) = ik√3 → negate to get Im(kz) ≤ 0 → kz = -ik√3
        kz_evan = DifferentiableMoM._spectral_kz(k, 2k, 0.0)
        @test abs(real(kz_evan)) < 1e-14
        @test imag(kz_evan) ≈ -sqrt(3) * k rtol=1e-14
    end

    # ── B: Reciprocity ΔG(r,rp) = ΔG(rp,r) for normal incidence ──
    @testset "B: Reciprocity (normal incidence)" begin
        lat = PeriodicLattice(dx, dy, 0.0, 0.0, k)
        r1 = SVector(0.002, 0.003, 0.0)
        r2 = SVector(0.007, 0.001, 0.0)
        dG_12 = greens_periodic_correction(r1, r2, k, lat)
        dG_21 = greens_periodic_correction(r2, r1, k, lat)
        # Periodic GF is symmetric for normal incidence (k_∥ = 0)
        # Tolerance: both evaluations use same Ewald code, small roundoff
        @test isapprox(dG_12, dG_21, rtol=1e-12)
    end

    # ── C: Exponential convergence with truncation order ──
    @testset "C: Exponential Ewald convergence" begin
        r  = SVector(0.0, 0.0, 0.0)
        rp = SVector(0.002, 0.003, 0.0)
        # Reference: N=6 (fully converged)
        lat_ref = PeriodicLattice(dx, dy, 0.0, 0.0, k; N_spatial=6, N_spectral=6)
        dG_ref = greens_periodic_correction(r, rp, k, lat_ref)

        errors = Float64[]
        for N in [1, 2, 3, 4]
            lat_N = PeriodicLattice(dx, dy, 0.0, 0.0, k; N_spatial=N, N_spectral=N)
            dG_N = greens_periodic_correction(r, rp, k, lat_N)
            push!(errors, abs(dG_N - dG_ref))
        end
        # Exponential convergence: each step should improve dramatically
        @test errors[2] < errors[1] * 0.01  # N=1→2: 100× improvement
        @test errors[3] < 1e-10 * abs(dG_ref)  # N=3: machine precision
        @test errors[4] < 1e-13 * abs(dG_ref)  # N=4: essentially zero
    end

    # ── D: No NaN/Inf including self-point R=0 ──
    @testset "D: No NaN/Inf" begin
        lat = PeriodicLattice(dx, dy, 0.0, 0.0, k)
        r  = SVector(0.0, 0.0, 0.0)
        rp = SVector(0.002, 0.003, 0.0)
        dG = greens_periodic_correction(r, rp, k, lat)
        @test !isnan(abs(dG)) && !isinf(abs(dG))

        # Self-point (R=0): exercises analytical limit in self-correction
        dG_self = greens_periodic_correction(r, r, k, lat)
        @test !isnan(abs(dG_self)) && !isinf(abs(dG_self))
    end

    # ── D: Oblique incidence ──
    @testset "D: Oblique incidence (θ=30°)" begin
        lat_obl = PeriodicLattice(dx, dy, π/6, 0.0, k)
        r  = SVector(0.0, 0.0, 0.0)
        rp = SVector(0.002, 0.003, 0.0)
        dG = greens_periodic_correction(r, rp, k, lat_obl)
        @test !isnan(abs(dG)) && !isinf(abs(dG))
        @test abs(dG) > 0  # should be nonzero for typical parameters
    end

    # ── C: Large-period Ewald convergence (d up to 100λ) ──
    @testset "C: Large-period Ewald convergence" begin
        # Use non-integer d/λ to avoid exact Wood anomaly (kz=0 singularity).
        # For non-Wood-anomaly periods, the Ewald sum converges and gives
        # finite results for arbitrarily large d/λ.
        # Note: |ΔG| is NOT monotonically decreasing because near-grazing
        # Floquet modes (near Wood anomalies) enhance the correction.
        r  = SVector(0.002, 0.003, 0.0)
        rp = SVector(0.0, 0.0, 0.0)
        for alpha in [2.5, 10.5, 50.5, 100.5]
            d = alpha * lambda
            lat = PeriodicLattice(d, d, 0.0, 0.0, k)
            dG = greens_periodic_correction(r, rp, k, lat)
            @test !isnan(abs(dG)) && !isinf(abs(dG))
            # Correction bounded: |ΔG| ≤ O(k) for any finite period
            # (each image contributes ~1/(4πd), summed over ~d/λ images per shell)
            @test abs(dG) < 100 * k
        end
    end

    # ── F: E-independence (non-Wood-anomaly periods) ──
    @testset "F: E-independence for non-Wood-anomaly periods" begin
        # Ewald splitting parameter E is mathematical; result must be
        # E-independent for non-Wood-anomaly periods. At integer d/λ with
        # normal incidence, Wood anomaly modes (kz=0) break the identity,
        # but non-integer d/λ should give machine-precision agreement.
        r  = SVector(0.002, 0.003, 0.0)
        rp = SVector(0.0, 0.0, 0.0)
        M_erfc = 5.0

        for alpha in [0.5, 2.5, 10.5]
            d = alpha * lambda
            E_min = k / (2 * sqrt(2.0))
            # Reference: E = E_min (α = 2)
            Nf1 = max(8, ceil(Int, d * sqrt(k^2 + 4*E_min^2*M_erfc^2) / (2π)))
            lat1 = PeriodicLattice(d, d, 0.0, 0.0, k, E_min, 8, Nf1)
            dG1 = greens_periodic_correction(r, rp, k, lat1)
            # Compare: E = 3 × E_min (α = 0.22)
            E2 = 3.0 * E_min
            Nf2 = max(8, ceil(Int, d * sqrt(k^2 + 4*E2^2*M_erfc^2) / (2π)))
            lat2 = PeriodicLattice(d, d, 0.0, 0.0, k, E2, 8, Nf2)
            dG2 = greens_periodic_correction(r, rp, k, lat2)
            # Machine-precision agreement for non-Wood-anomaly periods
            @test isapprox(dG1, dG2, rtol=1e-12)
        end
    end
end
println("  PASS ✓")

# ─────────────────────────────────────────────────────────────────
# Test 38: DensityInterpolation — SIMP material model
# ─────────────────────────────────────────────────────────────────
println("\n── Test 38: DensityInterpolation ──")

@testset "DensityInterpolation" begin
    mesh_di = make_rect_plate(0.01, 0.01, 3, 3)
    rwg_di = build_rwg(mesh_di; precheck=false)
    Nt_di = ntriangles(mesh_di)
    N_di = rwg_di.nedges
    Mt = precompute_triangle_mass(mesh_di, rwg_di)
    eta0 = 376.730313668
    config = DensityConfig(; p=3.0, Z_max_factor=1000.0, vf_target=0.5)

    # ── A: DensityConfig ──
    @testset "A: DensityConfig values" begin
        @test real(config.Z_max) ≈ 1000.0 * eta0 rtol=1e-12
        @test config.p ≈ 3.0
        @test config.vf_target ≈ 0.5
    end

    # ── A: Correct count of mass matrices ──
    @testset "A: Mass matrix count" begin
        @test length(Mt) == Nt_di
    end

    # ── B: Mass matrix symmetry ──
    @testset "B: Mass matrix symmetry" begin
        for t in 1:Nt_di
            M = Matrix(Mt[t])
            # M[m,n] = ∫ f_m·f_n dS is symmetric by definition
            # Tolerance: product quadrature, exact up to roundoff
            @test isapprox(M, M', atol=1e-14)
        end
    end

    # ── B: Mass matrix positive semi-definiteness ──
    @testset "B: Mass matrix PSD" begin
        for t in 1:Nt_di
            M = Matrix(Mt[t])
            eigs = eigvals(Symmetric(M))
            # f_m · f_n Gram matrix → all eigenvalues ≥ 0
            @test all(eigs .≥ -1e-14)
        end
    end

    # ── B: Mass matrix sparsity (only supported basis functions) ──
    @testset "B: Mass matrix sparsity" begin
        for t in 1:min(3, Nt_di)  # check a few triangles
            # Identify basis functions with support on triangle t
            supported = Set{Int}()
            for n in 1:N_di
                if rwg_di.tplus[n] == t || rwg_di.tminus[n] == t
                    push!(supported, n)
                end
            end
            for i in 1:N_di, j in 1:N_di
                if !(i ∈ supported) || !(j ∈ supported)
                    @test Mt[t][i, j] == 0.0
                end
            end
        end
    end

    # ── A: Z_penalty vanishes for all-metal (ρ̄ = 1) ──
    @testset "A: Z_penalty = 0 for all-metal" begin
        rho_bar_metal = ones(Nt_di)
        Z_pen = assemble_Z_penalty(Mt, rho_bar_metal, config)
        # (1 - 1^p) = 0 → zero penalty
        @test norm(Z_pen) < 1e-14
    end

    # ── A: Z_penalty for all-void (ρ̄ = 0) ──
    @testset "A: Z_penalty = Z_max × ΣM_t for all-void" begin
        rho_bar_void = zeros(Nt_di)
        Z_pen = assemble_Z_penalty(Mt, rho_bar_void, config)
        # (1 - 0^p) = 1 → Z_penalty = Z_max × Σ M_t
        M_total = sum(Matrix.(Mt))
        Z_expected = config.Z_max .* M_total
        # Tolerance: exact formula, machine precision
        @test isapprox(Z_pen, Z_expected, rtol=1e-12)
    end

    # ── B: Z_penalty linear in Z_max ──
    @testset "B: Z_penalty linear in Z_max" begin
        Random.seed!(77)
        rho_bar = 0.3 .+ 0.4 * rand(Nt_di)
        config_1x = DensityConfig(; p=3.0, Z_max_factor=500.0)
        config_2x = DensityConfig(; p=3.0, Z_max_factor=1000.0)
        Z_1x = assemble_Z_penalty(Mt, rho_bar, config_1x)
        Z_2x = assemble_Z_penalty(Mt, rho_bar, config_2x)
        # Doubling Z_max should double Z_penalty
        @test isapprox(Z_2x, 2.0 .* Z_1x, rtol=1e-12)
    end

    # ── F: dZ/dρ̄ vs finite difference ──
    @testset "F: dZ/dρ̄ vs finite difference" begin
        Random.seed!(78)
        rho_bar = 0.3 .+ 0.4 * rand(Nt_di)
        h = 1e-7
        for t in [1, div(Nt_di, 2), Nt_di]
            dZ_ana = Matrix(assemble_dZ_drhobar(Mt, rho_bar, config, t))
            rho_plus = copy(rho_bar); rho_plus[t] += h
            rho_minus = copy(rho_bar); rho_minus[t] -= h
            Z_plus = assemble_Z_penalty(Mt, rho_plus, config)
            Z_minus = assemble_Z_penalty(Mt, rho_minus, config)
            dZ_fd = (Z_plus - Z_minus) / (2h)
            # Tolerance: O(h²) central FD, h=1e-7 → error ~1e-14 absolute
            # Use rtol since values can span orders of magnitude
            @test isapprox(dZ_ana, dZ_fd, rtol=1e-5)
        end
    end

    # ── E: Dimension mismatch assertion ──
    @testset "E: Dimension mismatch" begin
        @test_throws AssertionError assemble_Z_penalty(Mt, zeros(Nt_di + 1), config)
    end
end
println("  PASS ✓")

# ─────────────────────────────────────────────────────────────────
# Test 39: DensityFiltering — conic filter + Heaviside projection
# ─────────────────────────────────────────────────────────────────
println("\n── Test 39: DensityFiltering ──")

@testset "DensityFiltering" begin
    mesh_df = make_rect_plate(0.015, 0.015, 6, 6)
    Nt_df = ntriangles(mesh_df)
    edge_len_df = 0.015 / 6
    r_min = 2.5 * edge_len_df
    W, w_sum = build_filter_weights(mesh_df, r_min)

    # ── A: Filter weight structure ──
    @testset "A: Filter weight properties" begin
        @test size(W) == (Nt_df, Nt_df)
        # All weights non-negative (conic: max(0, r_min - d))
        @test all(nonzeros(W) .≥ 0)
        # Self-weight positive (distance to self = 0 < r_min)
        for t in 1:Nt_df
            @test W[t, t] > 0
        end
        # Normalization is positive
        @test all(w_sum .> 0)
    end

    # ── A: Filter preserves constant fields (exact) ──
    @testset "A: Filter preserves constants" begin
        for c in [0.0, 0.42, 1.0]
            rho_const = fill(c, Nt_df)
            rho_tilde = apply_filter(W, w_sum, rho_const)
            # ρ̃_t = Σ w_ts c / Σ w_ts = c (exact identity)
            @test isapprox(rho_tilde, fill(c, Nt_df), atol=1e-14)
        end
    end

    # ── A: Heaviside boundary values (exact by construction) ──
    @testset "A: Heaviside H(0)=0, H(1)=1, H(η)=0.5" begin
        for beta in [1.0, 4.0, 16.0, 64.0]
            # H(0) = [tanh(βη) + tanh(-βη)] / denom = 0 (odd function cancels)
            @test heaviside_project([0.0], beta)[1] ≈ 0.0 atol=1e-14
            # H(1) = denom / denom = 1
            @test heaviside_project([1.0], beta)[1] ≈ 1.0 atol=1e-14
            # H(η=0.5) = tanh(β/2) / (2tanh(β/2)) = 0.5 for any β
            @test heaviside_project([0.5], beta)[1] ≈ 0.5 atol=1e-14
        end
    end

    # ── B: Heaviside monotonicity ──
    @testset "B: Heaviside monotonicity" begin
        rho_sorted = collect(range(0.0, 1.0, length=20))
        for beta in [1.0, 4.0, 64.0]
            rho_bar = heaviside_project(rho_sorted, beta)
            @test all(diff(rho_bar) .≥ -1e-15)
        end
    end

    # ── B: Heaviside range [0, 1] ──
    @testset "B: Heaviside output range" begin
        Random.seed!(88)
        rho_rand = rand(100)
        for beta in [1.0, 4.0, 64.0]
            rho_bar = heaviside_project(rho_rand, beta)
            @test all(rho_bar .≥ -1e-14)
            @test all(rho_bar .≤ 1.0 + 1e-14)
        end
    end

    # ── B: Filter adjoint identity: ⟨u, F(v)⟩ = ⟨Fᵀ(u), v⟩ ──
    @testset "B: Filter adjoint identity" begin
        Random.seed!(89)
        for trial in 1:5
            u = randn(Nt_df)
            v = randn(Nt_df)
            Fv = apply_filter(W, w_sum, v)
            Ftu = apply_filter_transpose(W, w_sum, u)
            # ⟨u, W*v/w⟩ = ⟨Wᵀ(u/w), v⟩
            # Tolerance: machine eps × vector norms
            @test isapprox(dot(u, Fv), dot(Ftu, v), rtol=1e-12)
        end
    end

    # ── B: Filter approximately preserves mean density ──
    @testset "B: Filter preserves mean" begin
        Random.seed!(90)
        rho_rand = rand(Nt_df)
        rho_tilde = apply_filter(W, w_sum, rho_rand)
        # Exact for uniform, approximate for random (boundary effects)
        @test abs(mean(rho_rand) - mean(rho_tilde)) < 0.05
    end

    # ── C: Increasing β drives toward binary design ──
    @testset "C: β continuation increases binarization" begin
        Random.seed!(91)
        rho_rand = rand(Nt_df)
        rho_tilde = apply_filter(W, w_sum, rho_rand)
        prev_frac = 0.0
        for beta in [1.0, 4.0, 16.0, 64.0]
            rho_bar = heaviside_project(rho_tilde, beta)
            near_binary = count(x -> x < 0.05 || x > 0.95, rho_bar) / Nt_df
            @test near_binary ≥ prev_frac - 0.01
            prev_frac = near_binary
        end
    end

    # ── F: Heaviside derivative vs central finite difference ──
    @testset "F: Heaviside derivative vs FD" begin
        rho_test = [0.2, 0.4, 0.6, 0.8]
        beta = 8.0
        h = 1e-7
        dH_ana = heaviside_derivative(rho_test, beta)
        for i in eachindex(rho_test)
            rho_plus = copy(rho_test); rho_plus[i] += h
            rho_minus = copy(rho_test); rho_minus[i] -= h
            H_plus = heaviside_project(rho_plus, beta)[i]
            H_minus = heaviside_project(rho_minus, beta)[i]
            dH_fd = (H_plus - H_minus) / (2h)
            # Tolerance: O(h²) FD error ≈ h²/6 × |d³H/dρ̃³|
            # For β=8, derivatives are moderate → rtol=1e-6 conservative
            @test isapprox(dH_ana[i], dH_fd, rtol=1e-6)
        end
    end

    # ── F: Full gradient chain rule vs FD of pipeline ──
    @testset "F: Gradient chain rule vs FD" begin
        Random.seed!(92)
        rho_raw = rand(Nt_df)
        beta = 4.0
        rho_tilde, rho_bar = filter_and_project(W, w_sum, rho_raw, beta)

        # Arbitrary gradient w.r.t. projected density (simulate adjoint output)
        g_rho_bar = randn(Nt_df)
        g_rho = gradient_chain_rule(g_rho_bar, rho_tilde, W, w_sum, beta)

        # Verify via FD: perturb rho_raw[t], measure change in dot(g_rho_bar, rho_bar)
        h = 1e-7
        for t in [1, div(Nt_df, 2), Nt_df]
            rho_plus = copy(rho_raw); rho_plus[t] += h
            rho_minus = copy(rho_raw); rho_minus[t] -= h
            _, rho_bar_plus = filter_and_project(W, w_sum, rho_plus, beta)
            _, rho_bar_minus = filter_and_project(W, w_sum, rho_minus, beta)
            # Directional derivative of the linear functional g_rho_bar' * rho_bar
            fd_val = (dot(g_rho_bar, rho_bar_plus) - dot(g_rho_bar, rho_bar_minus)) / (2h)
            @test isapprox(g_rho[t], fd_val, rtol=1e-5)
        end
    end
end
println("  PASS ✓")

# ─────────────────────────────────────────────────────────────────
# Test 40: DensityAdjoint — gradient computation
# ─────────────────────────────────────────────────────────────────
println("\n── Test 40: DensityAdjoint ──")

@testset "DensityAdjoint" begin
    Random.seed!(42)
    lambda_da = 0.03; k_da = 2π / lambda_da
    mesh_da = make_rect_plate(0.015, 0.015, 5, 5)
    rwg_da = build_rwg(mesh_da; precheck=false)
    Nt_da = ntriangles(mesh_da)
    N_da = rwg_da.nedges

    Z_efie_da = assemble_Z_efie(mesh_da, rwg_da, k_da; mesh_precheck=false)
    pw_da = make_plane_wave(Vec3(0.0, 0.0, -k_da), 1.0, Vec3(1.0, 0.0, 0.0))
    v_da = assemble_excitation(mesh_da, rwg_da, pw_da)
    Mt_da = precompute_triangle_mass(mesh_da, rwg_da)
    grid_da = make_sph_grid(10, 20)
    G_mat_da = radiation_vectors(mesh_da, rwg_da, grid_da, k_da)
    pol_da = pol_linear_x(grid_da)
    Q_da = build_Q(G_mat_da, grid_da, pol_da)
    config_da = DensityConfig(; p=3.0, Z_max_factor=1000.0)

    edge_len_da = 0.015 / 5
    r_min_da = 2.5 * edge_len_da
    W_da, w_sum_da = build_filter_weights(mesh_da, r_min_da)

    rho_da = 0.3 .+ 0.4 * rand(Nt_da)
    beta_da = 4.0
    rho_tilde_da, rho_bar_da = filter_and_project(W_da, w_sum_da, rho_da, beta_da)

    Z_pen_da = assemble_Z_penalty(Mt_da, rho_bar_da, config_da)
    Z_total_da = Z_efie_da + Z_pen_da
    I_sol_da = Z_total_da \ v_da
    J0_da = compute_objective(I_sol_da, Q_da)
    lambda_adj_da = solve_adjoint(Z_total_da, Q_da, I_sol_da)

    # ── B: Gradient is real-valued ──
    @testset "B: Gradient is real-valued" begin
        g = gradient_density(Mt_da, I_sol_da, lambda_adj_da, rho_bar_da, config_da)
        @test eltype(g) == Float64
    end

    # ── A: Zero gradient for zero objective ──
    @testset "A: Zero gradient for zero Q" begin
        # If Q = 0, adjoint λ = 0 → gradient = 0
        Q_zero = zeros(ComplexF64, N_da, N_da)
        lambda_zero = solve_adjoint(Z_total_da, Q_zero, I_sol_da)
        g_zero = gradient_density(Mt_da, I_sol_da, lambda_zero, rho_bar_da, config_da)
        @test norm(g_zero) < 1e-14
    end

    # ── F: Full adjoint gradient vs central finite difference (CRITICAL) ──
    @testset "F: Full gradient vs finite difference" begin
        g_adj = gradient_density_full(Mt_da, I_sol_da, lambda_adj_da,
                                      rho_tilde_da, rho_bar_da, config_da,
                                      W_da, w_sum_da, beta_da)
        h = 1e-5
        n_check = min(5, Nt_da)
        check_indices = sort(randperm(Nt_da)[1:n_check])
        max_rel_err = 0.0

        for t in check_indices
            rho_plus = copy(rho_da); rho_plus[t] += h
            rho_minus = copy(rho_da); rho_minus[t] -= h

            _, rho_bar_plus = filter_and_project(W_da, w_sum_da, rho_plus, beta_da)
            Z_plus = Z_efie_da + assemble_Z_penalty(Mt_da, rho_bar_plus, config_da)
            J_plus = compute_objective(Z_plus \ v_da, Q_da)

            _, rho_bar_minus = filter_and_project(W_da, w_sum_da, rho_minus, beta_da)
            Z_minus = Z_efie_da + assemble_Z_penalty(Mt_da, rho_bar_minus, config_da)
            J_minus = compute_objective(Z_minus \ v_da, Q_da)

            g_fd = (J_plus - J_minus) / (2h)
            rel_err = abs(g_adj[t] - g_fd) / max(abs(g_fd), 1e-20)
            max_rel_err = max(max_rel_err, rel_err)

            # Tolerance: adjoint is exact; FD error = O(h²) ~ 1e-10
            # Divided by |g_fd| ~ 1e-13, gives rtol ~ 1e-3 to 1e-4
            @test rel_err < 1e-4
        end
        println("    Max adjoint vs FD relative error: $(round(max_rel_err, sigdigits=3))")
    end
end
println("  PASS ✓")

# ─────────────────────────────────────────────────────────────────
# Test 41: PeriodicEFIE — periodic EFIE assembly
# ─────────────────────────────────────────────────────────────────
println("\n── Test 41: PeriodicEFIE ──")

@testset "PeriodicEFIE" begin
    lambda_pe = 0.03; k_pe = 2π / lambda_pe
    dx_pe = 0.5 * lambda_pe; dy_pe = 0.5 * lambda_pe
    mesh_pe = make_rect_plate(dx_pe, dy_pe, 4, 4)
    rwg_pe = build_rwg(mesh_pe; precheck=false)
    N_pe = rwg_pe.nedges

    # ── A: Output dimensions ──
    @testset "A: Output dimensions" begin
        lat = PeriodicLattice(dx_pe, dy_pe, 0.0, 0.0, k_pe; N_spatial=2, N_spectral=2)
        Z_per = assemble_Z_efie_periodic(mesh_pe, rwg_pe, k_pe, lat)
        @test size(Z_per) == (N_pe, N_pe)
        @test eltype(Z_per) == ComplexF64
    end

    # ── A: Large period → Z_per approaches Z_free ──
    @testset "A: Large period → Z_per approaches Z_free" begin
        Z_free = assemble_Z_efie(mesh_pe, rwg_pe, k_pe; mesh_precheck=false)

        # Use non-integer d/λ to avoid Wood anomaly (kz=0 singularity).
        # At Wood anomaly (integer d/λ at normal incidence), the periodic
        # Green's function diverges — a physical singularity, not a bug.
        #
        # For non-Wood-anomaly periods, the Ewald sum is exact (E-independent
        # to machine precision) and works for arbitrarily large d/λ.
        #
        # Test: relative difference ‖Z_per - Z_free‖ / ‖Z_free‖ decreases
        # monotonically as d increases (images move farther away).
        prev_rel = Inf
        for alpha in [2.5, 10.5, 50.5]
            d = alpha * lambda_pe
            lat = PeriodicLattice(d, d, 0.0, 0.0, k_pe)
            Z_per = assemble_Z_efie_periodic(mesh_pe, rwg_pe, k_pe, lat)
            @test !any(isnan, Z_per)
            @test !any(isinf, Z_per)
            rel_diff = norm(Z_per - Z_free) / norm(Z_free)
            # Correction shrinks as period grows (images recede)
            @test rel_diff < prev_rel
            prev_rel = rel_diff
        end
        # At d = 50.5λ, correction < 1% of free-space impedance
        @test prev_rel < 0.01
    end

    # ── D: No NaN/Inf in periodic EFIE ──
    @testset "D: No NaN/Inf" begin
        lat = PeriodicLattice(dx_pe, dy_pe, 0.0, 0.0, k_pe; N_spatial=2, N_spectral=2)
        Z_per = assemble_Z_efie_periodic(mesh_pe, rwg_pe, k_pe, lat)
        @test !any(isnan, Z_per)
        @test !any(isinf, Z_per)
    end
end
println("  PASS ✓")

# ─────────────────────────────────────────────────────────────────
# Test 42: PeriodicMetrics — Floquet mode enumeration
# ─────────────────────────────────────────────────────────────────
println("\n── Test 42: PeriodicMetrics ──")

@testset "PeriodicMetrics" begin
    lambda_pm = 0.03; k_pm = 2π / lambda_pm
    dx_pm = 0.5 * lambda_pm; dy_pm = 0.5 * lambda_pm
    lat_pm = PeriodicLattice(dx_pm, dy_pm, 0.0, 0.0, k_pm)

    # ── A: Mode count = (2N_orders+1)² ──
    @testset "A: Mode count" begin
        for N_ord in [1, 2, 3]
            modes = floquet_modes(k_pm, lat_pm; N_orders=N_ord)
            @test length(modes) == (2 * N_ord + 1)^2
        end
    end

    # ── A: Only specular mode propagates for λ/2 cell at normal incidence ──
    @testset "A: Only specular mode for λ/2 cell" begin
        modes = floquet_modes(k_pm, lat_pm; N_orders=3)
        prop_modes = filter(m -> m.propagating, modes)
        # dx=dy=λ/2: next mode (1,0) has kt = 2π/dx = 2k > k → evanescent
        @test length(prop_modes) == 1
        m00 = prop_modes[1]
        @test m00.m == 0 && m00.n == 0
        # kz = k for (0,0) at normal incidence
        @test real(m00.kz) ≈ k_pm rtol=1e-12
        # θ_r = 0 (broadside)
        @test m00.theta_r ≈ 0.0 atol=1e-12
    end

    # ── A: Multiple propagating modes for 2λ cell ──
    @testset "A: Multiple modes for 2λ cell" begin
        dx_big = 2.0 * lambda_pm; dy_big = 2.0 * lambda_pm
        lat_big = PeriodicLattice(dx_big, dy_big, 0.0, 0.0, k_pm)
        modes = floquet_modes(k_pm, lat_big; N_orders=3)
        prop_modes = filter(m -> m.propagating, modes)
        # (m,n) propagating when (m²+n²) < (2dx/λ)² = 4
        # Pairs: (0,0), (±1,0), (0,±1), (±1,±1) → 9 modes
        # (2,0) has kt = k → grazing/evanescent (kz² = 0, strict > check)
        @test length(prop_modes) == 9
    end

    # ── B: All (m,n) pairs present ──
    @testset "B: Complete mode enumeration" begin
        modes = floquet_modes(k_pm, lat_pm; N_orders=1)
        @test length(modes) == 9
        mn_pairs = Set([(m.m, m.n) for m in modes])
        for m in -1:1, n in -1:1
            @test (m, n) ∈ mn_pairs
        end
    end

    # ── B: Evanescent mode properties ──
    @testset "B: Evanescent mode fields" begin
        modes = floquet_modes(k_pm, lat_pm; N_orders=3)
        evan_modes = filter(m -> !m.propagating, modes)
        @test length(evan_modes) > 0
        for mode in evan_modes
            # kz purely imaginary (positive imag from the code: im * sqrt(-kz2))
            @test abs(real(mode.kz)) < 1e-12
            @test imag(mode.kz) > 0
            # Angles undefined for evanescent modes
            @test isnan(mode.theta_r)
            @test isnan(mode.phi_r)
        end
    end

    # ── A: Specular angle for oblique incidence ──
    @testset "A: Specular angle for oblique incidence" begin
        # θ_inc = 30° → specular θ_r = 30° for (0,0) mode
        # With large enough cell to ensure (0,0) propagates at oblique
        dx_obl = 1.0 * lambda_pm; dy_obl = 1.0 * lambda_pm
        lat_obl = PeriodicLattice(dx_obl, dy_obl, π/6, 0.0, k_pm)
        modes = floquet_modes(k_pm, lat_obl; N_orders=3)
        # Find (0,0) mode
        m00 = nothing
        for mode in modes
            if mode.m == 0 && mode.n == 0
                m00 = mode
                break
            end
        end
        @test m00 !== nothing
        @test m00.propagating
        # kx = k sin(30°) = k/2, ky = 0
        # kz = sqrt(k² - (k/2)²) = k√3/2
        @test real(m00.kz) ≈ k_pm * sqrt(3) / 2 rtol=1e-12
        # θ_r = acos(kz/k) = acos(√3/2) = 30° = π/6
        @test m00.theta_r ≈ π / 6 rtol=1e-10
    end
end
println("  PASS ✓")
