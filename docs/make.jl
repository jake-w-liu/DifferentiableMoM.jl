import Pkg
Pkg.activate(@__DIR__)
Pkg.develop(Pkg.PackageSpec(path = joinpath(@__DIR__, "..")))
Pkg.instantiate()

using Documenter
using DifferentiableMoM

makedocs(
    sitename = "DifferentiableMoM.jl",
    authors = "Jake W. Liu",
    modules = [DifferentiableMoM],
    checkdocs = :none,
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        mathengine = Documenter.MathJax3(),
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => [
            "Installation" => "getting-started/installation.md",
            "Quickstart" => "getting-started/quickstart.md",
            "Conventions and Units" => "getting-started/conventions.md",
        ],
        "Part I — Foundations" => [
            "Why Integral Equations" => "fundamentals/01-why-integral-equations.md",
            "EFIE and Boundary Conditions" => "fundamentals/02-efie-boundary-conditions.md",
            "MoM and RWG Discretization" => "fundamentals/03-mom-rwg-discretization.md",
            "Singular Integration" => "fundamentals/04-singular-integration.md",
            "Conditioning and Preconditioning" => "fundamentals/05-conditioning-preconditioning.md",
            "Excitation Theory and Usage" => "fundamentals/06-excitation-theory-and-usage.md",
        ],
        "Part II — Package Fundamentals" => [
            "Mesh Pipeline" => "package-basics/01-mesh-pipeline.md",
            "Forward Pipeline" => "package-basics/02-forward-pipeline.md",
            "Far-Field, Q, and RCS" => "package-basics/03-farfield-q-rcs.md",
            "Visualization" => "package-basics/04-visualization.md",
        ],
        "Part III — Differentiable Design" => [
            "Adjoint Method" => "differentiable-design/01-adjoint-method.md",
            "Impedance Sensitivities" => "differentiable-design/02-impedance-sensitivities.md",
            "Ratio Objectives" => "differentiable-design/03-ratio-objectives.md",
            "Optimization Workflow" => "differentiable-design/04-optimization-workflow.md",
        ],
        "Part IV — Validation" => [
            "Internal Consistency" => "validation/01-internal-consistency.md",
            "Gradient Verification" => "validation/02-gradient-verification.md",
            "Bempp Cross-Validation" => "validation/03-bempp-cross-validation.md",
            "Sphere-vs-Mie Benchmark" => "validation/04-sphere-mie-benchmark.md",
        ],
        "Part V — Advanced Workflows" => [
            "Complex OBJ Platforms" => "advanced/01-complex-obj-platforms.md",
            "Large-Problem Strategy" => "advanced/02-large-problem-strategy.md",
            "Robustness Studies" => "advanced/03-robustness-studies.md",
            "Debugging Playbook" => "advanced/04-debugging-playbook.md",
        ],
        "Tutorials" => [
            "First PEC Plate" => "tutorials/01-first-pec-plate.md",
            "Adjoint Gradient Check" => "tutorials/02-adjoint-gradient-check.md",
            "Beam Steering Design" => "tutorials/03-beam-steering-design.md",
            "Sphere-Mie RCS" => "tutorials/04-sphere-mie-rcs.md",
            "Airplane RCS" => "tutorials/05-airplane-rcs.md",
            "OBJ Repair Workflow" => "tutorials/06-obj-repair-workflow.md",
        ],
        "API Reference" => [
            "Overview" => "api/overview.md",
            "Types" => "api/types.md",
            "Mesh Utilities" => "api/mesh.md",
            "RWG Utilities" => "api/rwg.md",
            "Excitation" => "api/excitation.md",
            "Assembly and Solve" => "api/assembly-solve.md",
            "Far-Field and RCS" => "api/farfield-rcs.md",
            "Adjoint and Optimization" => "api/adjoint-optimize.md",
            "Verification" => "api/verification.md",
            "Visualization" => "api/visualization.md",
        ],
        "Appendices" => [
            "Units and Conventions" => "appendices/units-conventions.md",
            "Mathematical Prerequisites" => "appendices/mathematical-prerequisites.md",
            "Symbol Glossary" => "appendices/symbol-glossary.md",
            "Suggested Reading Paths" => "appendices/reading-path.md",
            "FAQ" => "appendices/faq.md",
        ],
    ],
)

# deploydocs(
#     repo = "github.com/jake-w-liu/DifferentiableMoM.jl.git",
# )
