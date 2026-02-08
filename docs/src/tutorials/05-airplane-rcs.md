# Tutorial: Airplane RCS

## Purpose

Demonstrate an end-to-end complex-platform workflow:
OBJ import ``\rightarrow`` repair ``\rightarrow`` coarsen ``\rightarrow`` PEC RCS
evaluation and visualization.

---

## Learning Goals

After this tutorial, you should be able to:

1. Prepare a large imported mesh for MoM simulation.
2. Run a practical airplane PEC RCS case with manageable unknown count.
3. Interpret heuristic pattern outputs responsibly.

---

## 1) Run the Workflow

```bash
julia --project=. examples/ex_airplane_rcs.jl ../Airplane.obj 3.0 0.001 300
```

Arguments:

1. OBJ path,
2. frequency in GHz,
3. coarsening scale parameter,
4. target RWG count.

---

## 2) What the Script Does

1. imports and repairs OBJ mesh,
2. coarsens to target unknown budget,
3. builds RWG and solves PEC forward problem,
4. computes bistatic cut and monostatic sample,
5. writes artifacts/plots.

---

## 3) Plotting Heuristic Results

Use:

```bash
julia --project=. examples/plot_airplane_rcs.jl
```

Interpretation note: this example is a demonstration workflow, not a certified
high-fidelity platform-RCS campaign. Mesh fidelity and angle resolution control
result quality.

---

## 4) Practical Tuning Knobs

- increase target RWG for higher fidelity,
- reduce frequency or simplify geometry for faster runs,
- always re-check mesh quality after coarsening.

---

## Code Mapping

- Mesh prep/coarsening: `src/Mesh.jl`
- Solve path: `src/EFIE.jl`, `src/Solve.jl`, `src/FarField.jl`
- Visualization helpers: `src/Visualization.jl`
- Main scripts: `examples/ex_airplane_rcs.jl`, `examples/plot_airplane_rcs.jl`

---

## Exercises

- Basic: run two target-RWG values and compare monostatic RCS.
- Challenge: test one additional incidence angle and compare pattern changes.
